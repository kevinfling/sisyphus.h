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
    int n = 4096;
    double* in = (double*)malloc(n * sizeof(double));
    double* out = (double*)malloc(n * sizeof(double));
    for (int i = 0; i < n; ++i) {
        double x = (double)(i + 0.5) / n;
        in[i] = sin(2.0 * M_PI * x);
    }

    int repeats = 1000;
    double alphas[] = {0.5, 1.0, 1.5, 2.0};
    int n_alphas = sizeof(alphas) / sizeof(alphas[0]);

    printf("Benchmark: spectral fractional derivative (n=%d, %d repeats)\n", n, repeats);
    printf("%-8s %-12s\n", "alpha", "time (us)");

    for (int a = 0; a < n_alphas; ++a) {
        double alpha = alphas[a];
        double t0 = now();
        for (int r = 0; r < repeats; ++r) {
            sis_fractional_derivative_1d(in, out, n, alpha, M_PI);
        }
        double t1 = now();
        double elapsed = (t1 - t0) / repeats * 1e6;
        printf("%-8.2f %-12.3f\n", alpha, elapsed);
    }

    free(in); free(out);
    sis_fftw_cleanup();
    return 0;
}
