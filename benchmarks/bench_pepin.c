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
    int n = 256;
    double* x = (double*)malloc(n * sizeof(double));
    double* y = (double*)malloc(n * sizeof(double));
    for (int i = 0; i < n; ++i) {
        x[i] = (double)i / (n - 1);
        y[i] = sin(2.0 * M_PI * x[i]);
    }

    sis_pepin1d_t s;
    sis_pepin1d_build(&s, x, y, n, 7);

    int nq = 1000000;
    double t0 = now();
    double sum = 0.0;
    for (int i = 0; i < nq; ++i) {
        double xq = (double)(i % 1000) / 1000.0;
        sum += sis_pepin1d_eval(&s, xq, 0);
    }
    double t1 = now();
    double elapsed = (t1 - t0) * 1e6; /* us total */
    double ns_per = elapsed * 1000.0 / nq;

    printf("Benchmark: Pepin 1D eval (degree=7, n=%d)\n", n);
    printf("  Queries: %d\n", nq);
    printf("  Total time: %.3f ms\n", elapsed / 1000.0);
    printf("  Time/eval:  %.3f ns\n", ns_per);
    printf("  (checksum: %g)\n", sum);

    sis_pepin1d_free(&s);
    free(x); free(y);
    sis_fftw_cleanup();
    return 0;
}
