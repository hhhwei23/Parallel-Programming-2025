#include <iostream>
#include <vector>
#include <cstring>
#include <algorithm>
#include "mpi.h"
using namespace std;
using  ull = unsigned long long;

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

	return A != out;
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
	
	ull N = 0;
	string input_path, output_path;
	if (rank == 0) {
		N = strtoull(argv[1], nullptr, 10);
		input_path = argv[2];
		output_path = argv[3];
	}

	MPI_Bcast(&N, 1, MPI_UNSIGNED_LONG_LONG, 0, MPI_COMM_WORLD);
	bcast_str(input_path, 0, MPI_COMM_WORLD);
	bcast_str(output_path, 0, MPI_COMM_WORLD);

	vector<int> counts, displs;
	separate_interval(N, size, counts, displs);

	int local_n = counts[rank];
	long long local_off_elems = displs[rank];
	MPI_File fin, fout;

	vector<float> local(local_n);

	// ===== File in ======

	MPI_File_open(MPI_COMM_WORLD, input_path.c_str(), MPI_MODE_RDONLY, MPI_INFO_NULL, &fin);

	MPI_Status st;
	MPI_Offset byte_offset = (MPI_Offset)local_off_elems * sizeof(float);
	MPI_File_read_at_all(fin, byte_offset, local_n ? local.data() : nullptr, local_n, MPI_FLOAT, &st);
	MPI_File_close(&fin);

	// ===== Main Computation ======
	sort(local.begin(), local.end());

	vector<float> recvbuf;
	vector<float> nextbuf;

	while (true) {
		bool global_change = false;
		//state : 0 -> even, 1 -> odd
		for (int state = 0; state < 2; state++) {
			int neighbor = ((rank % 2) == state) ? (rank + 1) : (rank - 1);
			if (neighbor < 0 || neighbor >= size) {
				// do nothing
			}
			else {
				int neighbor_n = counts[neighbor];
				recvbuf.resize(neighbor_n);
				MPI_Sendrecv(local_n ? local.data() : nullptr, local_n, MPI_FLOAT, neighbor, 0,
					neighbor_n ? recvbuf.data() : nullptr, neighbor_n, MPI_FLOAT, neighbor, 0,
					MPI_COMM_WORLD, MPI_STATUS_IGNORE);

				bool changed = false;
				if (rank < neighbor) {
					changed = low_half(local, recvbuf, nextbuf);
				}
				else {
					changed = high_half(local, recvbuf, nextbuf);
				}
				if (changed) {
					local.swap(nextbuf);
					global_change = true;
				}
			}
		}

		int flag = global_change ? 1 : 0;
		int allflag = 0;
		MPI_Allreduce(&flag, &allflag, 1, MPI_INT, MPI_LOR, MPI_COMM_WORLD);
		if (!allflag) break;
	}

	// ======== File out =======
	MPI_File_open(MPI_COMM_WORLD, output_path.c_str(), MPI_MODE_CREATE | MPI_MODE_WRONLY, MPI_INFO_NULL, &fout);

	MPI_File_write_at_all(fout, byte_offset, local_n ? local.data() : nullptr, local_n, MPI_FLOAT, &st);

	MPI_File_close(&fout);


	MPI_Finalize();
	return 0;
}
