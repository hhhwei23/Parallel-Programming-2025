#include <assert.h>
#include <stdio.h>
#include <math.h>
#include <omp.h>




int main(int argc, char** argv) {
	if (argc != 3) {
		fprintf(stderr, "must provide exactly 2 arguments!\n");
		return 1;
	}
	unsigned long long r = atoll(argv[1]);
	unsigned long long k = atoll(argv[2]);

    const long double rr = (long double) r;
    unsigned long long pixels = 0;

    #pragma omp parallel for schedule(static) reduction(+:pixels) default(none) shared(r, rr)
    for(unsigned long long x = 0; x < r; x++){
        long double dx = (long double)x;
        long double val = rr * rr - dx * dx;
        if(val < 0) val = 0;
        unsigned long long y = (unsigned long long)ceill(sqrtl(val));
        pixels += y;
    }

    unsigned long long ans = (4ull * (pixels % k)) % k;
    printf("%llu\n", ans);

    return 0;    
}