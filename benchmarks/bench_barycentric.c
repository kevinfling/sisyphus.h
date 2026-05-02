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
        x[i] = -1.0 + 2.0 * (double)i / (n - 1);
        y[i] = 1.0 / (1.0 + 25.0 * x[i] * x[i]);
    }

    sis_barycentric_t s;
    sis_barycentric_build(&s, x, y, n, 5);

    int nq = 1000000;
    double t0 = now();
    double sum = 0.0;
    for (int i = 0; i < nq; ++i) {
        double xq = (double)(rand() % 2000) / 1000.0 - 1.0;
        sum += sis_barycentric_eval(&s, xq);
    }
    double t1 = now();
    double ns_per = (t1 - t0) * 1e9 / nq;

    printf("Benchmark: Barycentric eval (n=%d, degree=5, Runge)\n", n);
    printf("  Queries: %d\n", nq);
    printf("  Time/eval: %.3f ns\n", ns_per);
    printf("  (checksum: %g)\n", sum);

    sis_barycentric_free(&s);
    free(x); free(y);
    sis_fftw_cleanup();
    return 0;
}
