/* SISYPHUS v0.1.0 — Spectral Interpolation System Yielding Precise High-order Univariate Splines
 * Copyright (c) 2026 Kevin Fling
 * Licensed under the MIT License
 * https://github.com/kevinfling/sisyphus.h
 */

#define SISYPHUS_IMPLEMENTATION
#include "sisyphus.h"
#include "test_harness.h"
#include <math.h>
#include <stdlib.h>

static int test_tensor_2d_reproduces_grid(void) {
    int n[2] = {8, 8};
    size_t total = 64;
    double x0[8], x1[8];
    double* y = (double*)malloc(total * sizeof(double));
    for (int i = 0; i < 8; ++i) {
        x0[i] = (double)i / 7.0;
        x1[i] = (double)i / 7.0;
    }
    for (size_t i = 0; i < total; ++i) {
        int ix = (int)(i / 8);
        int iy = (int)(i % 8);
        y[i] = sin(2.0 * M_PI * x0[ix]) * cos(2.0 * M_PI * x1[iy]);
    }

    const double* xk[SIS_MAX_DIM] = {x0, x1};
    sis_tensor_t s;
    int ret = sis_tensor_init(&s, 2, n, 3);
    TEST_ASSERT(ret == SIS_OK);
    ret = sis_tensor_build(&s, xk, y, 3);
    TEST_ASSERT(ret == SIS_OK);

    for (int i = 0; i < 7; ++i) {
        for (int j = 0; j < 7; ++j) {
            double xq[2] = {x0[i], x1[j]};
            double v = sis_tensor_eval(&s, xq, -1);
            double expected = sin(2.0 * M_PI * x0[i]) * cos(2.0 * M_PI * x1[j]);
            TEST_ASSERT_NEAR(v, expected, 1e-9);
        }
    }

    sis_tensor_free(&s);
    free(y);
    return 0;
}

int main(void) {
    int failures = 0;
    g_tests_run = 0; g_tests_passed = 0;
    failures += test_tensor_2d_reproduces_grid();
    failures += test_report("test_tensor_2d_reproduces_grid");

    sis_fftw_cleanup();
    return failures;
}
