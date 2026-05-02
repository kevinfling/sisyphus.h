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
    int n = 100;
    double* xi = (double*)malloc(n * 2 * sizeof(double));
    double* yi = (double*)malloc(n * sizeof(double));
    for (int i = 0; i < n; ++i) {
        xi[i * 2 + 0] = (double)(rand() % 1000) / 1000.0;
        xi[i * 2 + 1] = (double)(rand() % 1000) / 1000.0;
        yi[i] = xi[i * 2 + 0] * xi[i * 2 + 0] - xi[i * 2 + 1] * xi[i * 2 + 1];
    }

    sis_tps_t s;
    double t0 = now();
    sis_tps_fit(&s, xi, yi, n, 2, 1e-6);
    double t1 = now();
    printf("TPS fit (n=%d, 2D): %.3f ms\n", n, (t1 - t0) * 1000.0);

    int nq = 100000;
    t0 = now();
    double sum = 0.0;
    for (int i = 0; i < nq; ++i) {
        double xq[2] = {0.5, 0.5};
        sum += sis_tps_eval(&s, xq);
    }
    t1 = now();
    double ns_per = (t1 - t0) * 1e9 / nq;
    printf("TPS eval: %.3f ns/query (checksum %g)\n", ns_per, sum);

    sis_tps_free(&s);
    free(xi); free(yi);
    sis_fftw_cleanup();
    return 0;
}
