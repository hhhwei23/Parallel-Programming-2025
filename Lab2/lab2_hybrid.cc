#include <mpi.h>
#include <omp.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>

typedef unsigned long long ull;

int main(int argc, char** argv){
    MPI_Init(&argc, &argv);
    int rank = 0, size = 1;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    ull r = strtoull(argv[1], NULL, 10);
    ull k = strtoull(argv[2], NULL, 10);

    ull base = r / size, rem = r % size;
    ull x0 = base * rank + ((rank < rem) ? rank : rem);
    ull len = base + ((rank < rem) ? 1 : 0);
    ull x1 = x0 + len;

    const long double rr = (long double)r;
    ull local_mod = 0;

    #pragma omp parallel for reduction(+:local_mod) schedule(static)
    for(ull x = x0; x< x1; x++){
        long double dx = (long double)x;
        long double val = rr * rr - dx * dx;

        if(val < 0) val = 0;
        ull y = (ull)ceill(sqrtl(val));
        local_mod += y;
    }
    if(k) local_mod %= k;

    ull global_sum = 0;
    MPI_Reduce(&local_mod, &global_sum, 1, MPI_UNSIGNED_LONG_LONG, MPI_SUM, 0, MPI_COMM_WORLD);

    if(rank == 0){
        ull quad = global_sum % k;
        ull ans = (4ull * (quad % k)) % k;
        printf("%llu\n", ans);
    }

    MPI_Finalize();
    return 0;
}