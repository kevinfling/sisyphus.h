/* SISYPHUS v0.1.0 — Spectral Interpolation System Yielding Precise High-order Univariate Splines
 * Copyright (c) 2026 Kevin Fling
 * Licensed under the MIT License
 * https://github.com/kevinfling/sisyphus.h
 */

/* example_nufct.c
 * Evaluate a spectrally-denoised signal at 10^4 random non-uniform points
 * using the fast NUFCT.
 */
#define SISYPHUS_IMPLEMENTATION
#include "sisyphus.h"
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>

int main(void) {
    int n = 512;
    double* y = (double*)malloc(n * sizeof(double));
    srand((unsigned)time(NULL));
    for (int i = 0; i < n; ++i) {
        double x = (double)(i + 0.5) / n;
        y[i] = sin(2.0 * M_PI * 3.0 * x) + 0.3 * ((double)rand() / RAND_MAX - 0.5);
    }

    /* Denoise before NUFCT */
    sis_denoise_1d(y, n, 0.08);

    int m = 10000;
    double* xq = (double*)malloc(m * sizeof(double));
    double* out = (double*)malloc(m * sizeof(double));
    for (int i = 0; i < m; ++i) xq[i] = (double)(rand() % 1000) / 1000.0;

    int ret = sis_nufct_eval(n, y, m, xq, out, 1.0, 16);
    if (ret != SIS_OK) {
        fprintf(stderr, "NUFCT failed: %d\n", ret);
        return 1;
    }

    printf("NUFCT example (n=%d uniform -> m=%d random)\n", n, m);
    printf("  sample xq, interpolated\n");
    for (int i = 0; i < 10; ++i) {
        int idx = rand() % m;
        printf("  %.4f  %+.6f\n", xq[idx], out[idx]);
    }

    free(y); free(xq); free(out);
    sis_fftw_cleanup();
    return 0;
}
