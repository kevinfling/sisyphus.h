/* SISYPHUS v0.1.0 — Spectral Interpolation System Yielding Precise High-order Univariate Splines
 * Copyright (c) 2026 Kevin Fling
 * Licensed under the MIT License
 * https://github.com/kevinfling/sisyphus.h
 */

/* example_spline_fit.c
 * Fit a high-degree Pepin spline to noisy samples and evaluate derivatives.
 */
#define SISYPHUS_IMPLEMENTATION
#include "sisyphus.h"
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>

int main(void) {
    int n = 128;
    double* x = (double*)malloc(n * sizeof(double));
    double* y = (double*)malloc(n * sizeof(double));

    srand((unsigned)time(NULL));
    for (int i = 0; i < n; ++i) {
        x[i] = (double)i / (n - 1);
        y[i] = sin(2.0 * M_PI * 2.0 * x[i]) + 0.05 * ((double)rand() / RAND_MAX - 0.5);
    }

    sis_pepin1d_t s;
    int ret = sis_pepin1d_fit_denoised(&s, x, y, n, 7, 0.08);
    if (ret != SIS_OK) {
        fprintf(stderr, "spline fit failed: %d\n", ret);
        return 1;
    }

    printf("Pepin spline fit (n=%d, degree=7, denoised)\n", n);
    printf("  x     f(x)        f'(x)       f''(x)\n");
    for (int i = 0; i <= 20; ++i) {
        double xq = (double)i / 20.0;
        double f  = sis_pepin1d_eval(&s, xq, 0);
        double fp = sis_pepin1d_eval(&s, xq, 1);
        double fpp = sis_pepin1d_eval(&s, xq, 2);
        printf("  %.3f  %+.6f  %+.6f  %+.6f\n", xq, f, fp, fpp);
    }

    double area = sis_pepin1d_integrate(&s, 0.0, 1.0);
    printf("  Integral over [0,1]: %.6f (expected ~0 for full sine periods)\n", area);

    sis_pepin1d_free(&s);
    free(x); free(y);
    sis_fftw_cleanup();
    return 0;
}
