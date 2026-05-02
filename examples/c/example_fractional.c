/* SISYPHUS v0.1.0 — Spectral Interpolation System Yielding Precise High-order Univariate Splines
 * Copyright (c) 2026 Kevin Fling
 * Licensed under the MIT License
 * https://github.com/kevinfling/sisyphus.h
 */

/* example_fractional.c
 * Demonstrate spectral fractional derivatives of a Gaussian bump.
 */
#define SISYPHUS_IMPLEMENTATION
#include "sisyphus.h"
#include <math.h>
#include <stdio.h>
#include <stdlib.h>

int main(void) {
    int n = 256;
    double* in = (double*)malloc(n * sizeof(double));
    double* out = (double*)malloc(n * sizeof(double));
    for (int i = 0; i < n; ++i) {
        double x = (double)(i + 0.5) / n;
        in[i] = exp(-25.0 * (x - 0.5) * (x - 0.5));
    }

    double alphas[] = {0.0, 0.25, 0.5, 0.75, 1.0, 1.5, 2.0};
    int n_alphas = sizeof(alphas) / sizeof(alphas[0]);

    printf("Fractional derivative of Gaussian bump (n=%d)\n", n);
    printf("  x");
    for (int a = 0; a < n_alphas; ++a) printf("      alpha=%.2f", alphas[a]);
    printf("\n");

    for (int i = 16; i < n; i += 16) {
        double x = (double)(i + 0.5) / n;
        printf("  %.3f", x);
        for (int a = 0; a < n_alphas; ++a) {
            sis_fractional_derivative_1d(in, out, n, alphas[a], M_PI);
            printf("  %+.6f", out[i]);
        }
        printf("\n");
    }

    free(in); free(out);
    sis_fftw_cleanup();
    return 0;
}
