#include <iostream>
#include <vector>
#include <cstring>
#include <algorithm>
#include <cstdio>
#include <set>
#include <sys/stat.h>
#include <limits>
#include <boost/sort/spreadsort/spreadsort.hpp>
#include "mpi.h"
using namespace std;
using  ull = unsigned long long;
namespace bs = boost::sort::spreadsort;

static void bcast_str(string& s, int root, MPI_Comm comm) {
	int rank;
	MPI_Comm_rank(comm, &rank);
	int len = (rank == root) ? s.size() : 0;
	MPI_Bcast(&len, 1, MPI_INT, root, comm);
	vector<char> buf(len);

	if (rank == root && len) {
		memcpy(buf.data(), s.data(), len);
	}
	if (len) MPI_Bcast(buf.data(), len, MPI_CHAR, root, comm);
	if (rank != root) {
		s.assign(buf.data(), buf.data() + len);
	}
}

static void separate_interval(ull N, int P, vector<int>& counts, vector<int>& displs) {
	counts.resize(P);
	displs.resize(P);
	ull base = N / P;
	ull rem = N % P;
	long long offset = 0;

	for (int r = 0; r < P; r++) {
		ull c = base + (r < rem ? 1 : 0);
		counts[r] = c;
		displs[r] = offset;
		offset += (long long)c;
	}
}

static bool low_half(const vector<float>& A, const vector<float>& B, vector<float>& out) {
	int needed = A.size();
	out.resize(needed);
	int i = 0, j = 0, k = 0;
	while (k < needed) {
		float v;
		if (j >= B.size() || (i < A.size() && A[i] <= B[j])) v = A[i++];
		else v = B[j++];
		out[k++] = v;
	}

	return j > 0;
}

static bool high_half(const vector<float>& A, const vector<float>& B, vector<float>& out) {
	int needed = A.size();
	out.resize(needed);
	int i = A.size() - 1;
	int j = B.size() - 1;
	int k = needed - 1;

	while (k >= 0) {
		float v;
		if (j < 0 || (i >= 0 && A[i] >= B[j])) v = A[i--];
		else v = B[j--];
		out[k--] = v;
	}

	return A != out;
}

int main(int argc, char** argv){
	MPI_Init(&argc, &argv);

	int rank, size;
	MPI_Comm_rank(MPI_COMM_WORLD, &rank);
	MPI_Comm_size(MPI_COMM_WORLD, &size);

	// ======== Time variables ========
	double t_total = 0.0, t_io = 0.0, t_comm = 0.0, t_comp = 0.0;
	int iter_cnt = 0;
	
	ull N = 0;
	string input_path, output_path;
	if (rank == 0) {
		N = strtoull(argv[1], nullptr, 10);
		input_path = argv[2];
		output_path = argv[3];
	}

	auto tic = []() {return MPI_Wtime();};

	double t0 = tic();

	double ts = tic();
	MPI_Bcast(&N, 1, MPI_UNSIGNED_LONG_LONG, 0, MPI_COMM_WORLD);
	t_comm += tic() - ts;
	ts = tic(); bcast_str(input_path, 0, MPI_COMM_WORLD); t_comm += (tic() - ts);
	ts = tic(); bcast_str(output_path, 0, MPI_COMM_WORLD); t_comm += (tic() - ts);

	// ======== Synchronization ========
	MPI_Barrier(MPI_COMM_WORLD);

	vector<int> counts, displs;
	separate_interval(N, size, counts, displs);

	int local_n = counts[rank];
	long long local_off_elems = displs[rank];
	MPI_File fin, fout;

	vector<float> local(local_n);

	// ===== File in ======
	ts = tic();

	MPI_File_open(MPI_COMM_WORLD, input_path.c_str(), MPI_MODE_RDONLY, MPI_INFO_NULL, &fin);

	MPI_Status st;
	MPI_Offset byte_offset = (MPI_Offset)local_off_elems * sizeof(float);
	MPI_File_read_at_all(fin, byte_offset, local_n ? local.data() : nullptr, local_n, MPI_FLOAT, &st);
	MPI_File_close(&fin);

	t_io += (tic() - ts);
	// ===== Main Computation ======

	ts = tic();
	if(local_n >= 32768) bs::float_sort(local.begin(), local.end());
	else sort(local.begin(), local.end());
	// bs::spreadsort(local.begin(), local.begin() + local_n);
	t_comp += (tic() - ts);

	vector<float> recvbuf;
	vector<float> nextbuf;

	while (true) {
		bool global_change = false;
		//state : 0 -> odd, 1 -> even
		for (int state = 0; state < 2; state++) {
			int neighbor = ((rank % 2) == state) ? (rank + 1) : (rank - 1);
			if (neighbor < 0 || neighbor >= size) {
				// do nothing
			}
			else {
				int neighbor_n = counts[neighbor];
				recvbuf.resize(neighbor_n);

				// Early stop check
				float edges[2];
				if (local_n > 0) {
					edges[0] = local.front();
					edges[1] = local.back();
				} else {
					edges[0] =  std::numeric_limits<float>::infinity();
					edges[1] = -std::numeric_limits<float>::infinity();
				}
				float nb_edges[2];

				double tc = tic();
				MPI_Sendrecv(edges, 2, MPI_FLOAT, neighbor, 100,
							nb_edges, 2, MPI_FLOAT, neighbor, 100,
							MPI_COMM_WORLD, MPI_STATUS_IGNORE);
				t_comm += (tic() - tc);

				bool need_exchange = false;
				if (local_n > 0 && neighbor_n > 0) {
					if (rank < neighbor) {
						need_exchange = !(edges[1] <= nb_edges[0]);
					} else {
						need_exchange = !(edges[0] >= nb_edges[1]);
					}
				}

				if (!need_exchange) continue;


				// Early stop not possible, need change.

				tc = tic();
				MPI_Sendrecv(local_n ? local.data() : nullptr, local_n, MPI_FLOAT, neighbor, 0,
					neighbor_n ? recvbuf.data() : nullptr, neighbor_n, MPI_FLOAT, neighbor, 0,
					MPI_COMM_WORLD, MPI_STATUS_IGNORE);
				t_comm += (tic() - tc);
				
				double tcc = tic();
				bool changed = false;
				if (rank < neighbor) {
					changed = low_half(local, recvbuf, nextbuf);
				}
				else {
					changed = high_half(local, recvbuf, nextbuf);
				}
				t_comp += (tic() - tcc);
				if (changed) {
					local.swap(nextbuf);
					global_change = true;
				}
			}
		}

		int flag = global_change ? 1 : 0;
		int allflag = 0;
		double tc = tic();
		MPI_Allreduce(&flag, &allflag, 1, MPI_INT, MPI_LOR, MPI_COMM_WORLD);
		t_comm += (tic() - tc);
		iter_cnt++;
		if (!allflag) break;
	}

	// ======== File out =======
	ts = tic();
	MPI_File_open(MPI_COMM_WORLD, output_path.c_str(), MPI_MODE_CREATE | MPI_MODE_WRONLY, MPI_INFO_NULL, &fout);

	MPI_File_write_at_all(fout, byte_offset, local_n ? local.data() : nullptr, local_n, MPI_FLOAT, &st);

	MPI_File_close(&fout);

	t_io += (tic() - ts);

	// ======= Finalize ========
	//MPI_Barrier(MPI_COMM_WORLD);
	t_total = tic() - t0;
	/*

	double io_sum, comm_sum, comp_sum, total_max, io_avg, comm_avg, comp_avg;
	MPI_Reduce(&t_io,   &io_sum,   1, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);
	MPI_Reduce(&t_comm, &comm_sum, 1, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);
	MPI_Reduce(&t_comp, &comp_sum, 1, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);
	MPI_Reduce(&t_total,&total_max,1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);


	if(rank == 0){
		io_avg = io_sum / size;
		comm_avg = comm_sum / size;
		comp_avg = comp_sum / size;
		cerr << "==================== Performance ====================\n";
		cerr << "Number of processes : " << size << "\n";
		cerr << "Number of elements : " << N << "\n";
		cerr << "Number of iterations : " << iter_cnt << "\n";
		cerr << "I/O time (avg) : " << io_avg << " sec\n";
		cerr << "Computation time (avg) : " << comp_avg << " sec\n";
		cerr << "Communication time (avg) : " << comm_avg << " sec\n";
		cerr << "Total time (max) : " << total_max << " sec\n";
		cerr << "======================================================\n";

		printf("hw1,%llu,%d,%d,%.6f,%.6f,%.6f,%.6f,%d\n",
           (unsigned long long)N, size, -1,
           io_avg, comm_avg, comp_avg, total_max, iter_cnt);
    	fflush(stdout);
	}
	*/

	MPI_Finalize();
	return 0;
}