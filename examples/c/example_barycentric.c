/* SISYPHUS v0.1.0 — Spectral Interpolation System Yielding Precise High-order Univariate Splines
 * Copyright (c) 2026 Kevin Fling
 * Licensed under the MIT License
 * https://github.com/kevinfling/sisyphus.h
 */

/* example_barycentric.c
 * Compare Floater-Hormann barycentric rational interpolation with
 * Pepin splines on a uniform knot set.
 */
#define SISYPHUS_IMPLEMENTATION
#include "sisyphus.h"
#include <math.h>
#include <stdio.h>
#include <stdlib.h>

int main(void) {
    int n = 32;
    double* x = (double*)malloc(n * sizeof(double));
    double* y = (double*)malloc(n * sizeof(double));
    for (int i = 0; i < n; ++i) {
        x[i] = (double)i / (n - 1);
        y[i] = 1.0 / (1.0 + 25.0 * (2.0 * x[i] - 1.0) * (2.0 * x[i] - 1.0));
    }

    sis_barycentric_t b;
    sis_barycentric_build(&b, x, y, n, 5);

    sis_pepin1d_t p;
    sis_pepin1d_build(&p, x, y, n, 5);

    printf("Runge function on uniform nodes (n=%d)\n", n);
    printf("  x        exact        barycentric  pepin\n");
    for (int i = 0; i <= 20; ++i) {
        double xq = (double)i / 20.0;
        double exact = 1.0 / (1.0 + 25.0 * (2.0 * xq - 1.0) * (2.0 * xq - 1.0));
        double vb = sis_barycentric_eval(&b, xq);
        double vp = sis_pepin1d_eval(&p, xq, 0);
        printf("  %+.3f  %+.6f  %+.6f  %+.6f\n", xq, exact, vb, vp);
    }

    sis_barycentric_free(&b);
    sis_pepin1d_free(&p);
    free(x); free(y);
    sis_fftw_cleanup();
    return 0;
}
