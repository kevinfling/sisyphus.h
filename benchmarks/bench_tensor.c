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
    int n[2] = {32, 32};
    size_t total = 32 * 32;
    double x0[32], x1[32];
    double* y = (double*)malloc(total * sizeof(double));
    for (int i = 0; i < 32; ++i) {
        x0[i] = (double)i / 31.0;
        x1[i] = (double)i / 31.0;
    }
    for (size_t i = 0; i < total; ++i) y[i] = (double)(i % 5);

    const double* xk[SIS_MAX_DIM] = {x0, x1};
    sis_tensor_t s;
    sis_tensor_init(&s, 2, n, 3);

    double t0 = now();
    sis_tensor_build(&s, xk, y, 3);
    double t1 = now();
    printf("Tensor build (32x32, degree=3): %.3f ms\n", (t1 - t0) * 1000.0);

    int nq = 100000;
    t0 = now();
    double sum = 0.0;
    for (int i = 0; i < nq; ++i) {
        double xq[2] = {0.5, 0.5};
        sum += sis_tensor_eval(&s, xq, -1);
    }
    t1 = now();
    double ns_per = (t1 - t0) * 1e9 / nq;
    printf("Tensor eval (degree=3): %.3f ns/query (checksum %g)\n", ns_per, sum);

    sis_tensor_free(&s);
    free(y);
    sis_fftw_cleanup();
    return 0;
}
