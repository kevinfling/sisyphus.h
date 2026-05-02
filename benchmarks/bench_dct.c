/* SISYPHUS v0.1.0 — Spectral Interpolation System Yielding Precise High-order Univariate Splines
 * Copyright (c) 2026 Kevin Fling
 * Licensed under the MIT License
 * https://github.com/kevinfling/sisyphus.h
 */

#define _POSIX_C_SOURCE 199309L
#define SISYPHUS_IMPLEMENTATION
#include "sisyphus.h"
#include <stdio.h>
#include <stdlib.h>
#include <time.h>

static double now(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return ts.tv_sec + ts.tv_nsec * 1e-9;
}

int main(void) {
    int sizes[] = {64, 128, 256, 512, 1024, 2048, 4096};
    int n_sizes = sizeof(sizes) / sizeof(sizes[0]);
    int repeats = 1000;

    printf("Benchmark: 1D DCT-II + IDCT-II roundtrip\n");
    printf("%-8s %-12s %-12s %-12s\n", "N", "time (us)", "Mops/s", "us/element");

    for (int s = 0; s < n_sizes; ++s) {
        int n = sizes[s];
        double* in = (double*)malloc(n * sizeof(double));
        double* tmp = (double*)malloc(n * sizeof(double));
        double* out = (double*)malloc(n * sizeof(double));
        for (int i = 0; i < n; ++i) in[i] = (double)i;

        /* warm-up */
        for (int r = 0; r < 10; ++r) {
            sis_dct2(in, tmp, n);
            sis_idct2(tmp, out, n);
        }

        double t0 = now();
        for (int r = 0; r < repeats; ++r) {
            sis_dct2(in, tmp, n);
            sis_idct2(tmp, out, n);
        }
        double t1 = now();
        double elapsed = (t1 - t0) / repeats * 1e6; /* us */
        double mops = (double)n / (elapsed);
        double us_per = elapsed / n;

        printf("%-8d %-12.3f %-12.3f %-12.6f\n", n, elapsed, mops, us_per);

        free(in); free(tmp); free(out);
    }

    sis_fftw_cleanup();
    return 0;
}
