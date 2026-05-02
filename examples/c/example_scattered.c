/* SISYPHUS v0.1.0 — Spectral Interpolation System Yielding Precise High-order Univariate Splines
 * Copyright (c) 2026 Kevin Fling
 * Licensed under the MIT License
 * https://github.com/kevinfling/sisyphus.h
 */

/* example_scattered.c
 * Interpolate scattered 2-D data with a thin-plate spline.
 */
#define SISYPHUS_IMPLEMENTATION
#include "sisyphus.h"
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>

int main(void) {
    int n = 50;
    double* xi = (double*)malloc(n * 2 * sizeof(double));
    double* yi = (double*)malloc(n * sizeof(double));

    srand((unsigned)time(NULL));
    for (int i = 0; i < n; ++i) {
        xi[i * 2 + 0] = (double)(rand() % 1000) / 1000.0;
        xi[i * 2 + 1] = (double)(rand() % 1000) / 1000.0;
        double x = xi[i * 2 + 0];
        double y = xi[i * 2 + 1];
        yi[i] = exp(-(x * x + y * y));
    }

    sis_tps_t s;
    int ret = sis_tps_fit(&s, xi, yi, n, 2, 1e-5);
    if (ret != SIS_OK) {
        fprintf(stderr, "TPS fit failed: %d\n", ret);
        return 1;
    }

    printf("Thin-plate spline scattered interpolation (n=%d centers)\n", n);
    printf("  x     y     interpolated\n");
    for (int i = 0; i <= 10; ++i) {
        for (int j = 0; j <= 10; j += 5) {
            double xq[2] = {(double)i / 10.0, (double)j / 10.0};
            double v = sis_tps_eval(&s, xq);
            printf("  %.2f  %.2f  %+.6f\n", xq[0], xq[1], v);
        }
    }

    sis_tps_free(&s);
    free(xi); free(yi);
    sis_fftw_cleanup();
    return 0;
}
