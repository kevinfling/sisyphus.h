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
    int n = 1024;
    double* y = (double*)malloc(n * sizeof(double));
    for (int i = 0; i < n; ++i) {
        double x = (double)(i + 0.5) / n;
        y[i] = sin(2.0 * M_PI * x) + 0.5 * cos(6.0 * M_PI * x);
    }

    int m = 100000;
    double* xq = (double*)malloc(m * sizeof(double));
    double* out = (double*)malloc(m * sizeof(double));
    for (int i = 0; i < m; ++i) xq[i] = (double)(rand() % 1000) / 1000.0;

    printf("Benchmark: NUFCT (n=%d, m=%d)\n", n, m);

    /* Fast path */
    double t0 = now();
    sis_nufct_eval(n, y, m, xq, out, 1.0, 16);
    double t1 = now();
    printf("  Fast (oversample=16): %.3f ms  (%.3f ns/query)\n",
           (t1 - t0) * 1000.0, (t1 - t0) * 1e9 / m);

    /* Direct path */
    t0 = now();
    sis_nufct_eval(n, y, m, xq, out, 1.0, -1);
    t1 = now();
    printf("  Direct:               %.3f ms  (%.3f ns/query)\n",
           (t1 - t0) * 1000.0, (t1 - t0) * 1e9 / m);

    free(y); free(xq); free(out);
    sis_fftw_cleanup();
    return 0;
}
