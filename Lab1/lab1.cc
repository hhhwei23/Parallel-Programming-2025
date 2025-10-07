#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <stdint.h>

typedef unsigned long long ull;
typedef __uint128_t u128;


int main(int argc, char** argv) {
	MPI_Init(&argc, &argv);
	int rank = 0, size = 1;
	MPI_Comm_rank(MPI_COMM_WORLD, &rank);
	MPI_Comm_size(MPI_COMM_WORLD, &size);

	ull r = strtoull(argv[1], NULL, 10);
	ull k = strtoull(argv[2], NULL, 10);

	ull x_begin = ((u128)r * (u128)rank) / (u128)size;
	ull x_end = ((u128)r * (u128)(rank + 1)) / (u128)size;

	long double R2 = (long double)r * (long double)r;
	ull local_sum = 0;

	for (ull x = x_begin; x < x_end; x++) {
		long double x2 = (long double)x * (long double)x;
		long double d = R2 - x2;

		ull add = (ull)ceill(sqrtl(d));

		local_sum += add;
	}

	ull global_sum = 0;
	MPI_Reduce(&local_sum, &global_sum, 1, MPI_UNSIGNED_LONG_LONG, MPI_SUM, 0, MPI_COMM_WORLD);

	if (rank == 0) {
		ull t = global_sum % k;
		ull ans = (ull)(((u128)t * 4u) % k);
		printf("%llu\n", ans);
	}

	MPI_Finalize();
	return 0;
}