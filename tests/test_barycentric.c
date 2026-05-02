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

static int test_barycentric_reproduces_knots(void) {
    int n = 17;
    double x[17], y[17];
    for (int i = 0; i < n; ++i) {
        x[i] = -1.0 + 2.0 * (double)i / (n - 1);
        y[i] = 1.0 / (1.0 + 25.0 * x[i] * x[i]);
    }
    sis_barycentric_t s;
    int ret = sis_barycentric_build(&s, x, y, n, 3);
    TEST_ASSERT(ret == SIS_OK);
    for (int i = 0; i < n; ++i) {
        double v = sis_barycentric_eval(&s, x[i]);
        TEST_ASSERT_NEAR(v, y[i], 1e-12);
    }
    sis_barycentric_free(&s);
    return 0;
}

static int test_barycentric_runge(void) {
    int n = 65;
    double* x = (double*)malloc(n * sizeof(double));
    double* y = (double*)malloc(n * sizeof(double));
    for (int i = 0; i < n; ++i) {
        x[i] = -1.0 + 2.0 * (double)i / (n - 1);
        y[i] = 1.0 / (1.0 + 25.0 * x[i] * x[i]);
    }
    sis_barycentric_t s;
    int ret = sis_barycentric_build(&s, x, y, n, 5);
    TEST_ASSERT(ret == SIS_OK);
    /* Check accuracy at several interior points */
    double err = 0.0;
    for (int i = 0; i <= 20; ++i) {
        double xq = -0.9 + 1.8 * (double)i / 20.0;
        double v = sis_barycentric_eval(&s, xq);
        double expected = 1.0 / (1.0 + 25.0 * xq * xq);
        err += fabs(v - expected);
    }
    TEST_ASSERT(err < 0.5); /* d=5 should be very accurate for Runge */
    sis_barycentric_free(&s);
    free(x); free(y);
    return 0;
}

static int test_barycentric_irregular(void) {
    int n = 20;
    double x[20], y[20];
    for (int i = 0; i < n; ++i) {
        x[i] = cos(M_PI * (double)i / (n - 1)); /* Chebyshev nodes */
        y[i] = sin(3.0 * x[i]);
    }
    sis_barycentric_t s;
    int ret = sis_barycentric_build(&s, x, y, n, 3);
    TEST_ASSERT(ret == SIS_OK);
    double err = 0.0;
    for (int i = 0; i <= 30; ++i) {
        double xq = -1.0 + 2.0 * (double)i / 30.0;
        double v = sis_barycentric_eval(&s, xq);
        double expected = sin(3.0 * xq);
        err += fabs(v - expected);
    }
    TEST_ASSERT(err < 1e-3);
    sis_barycentric_free(&s);
    return 0;
}

int main(void) {
    int failures = 0;
    g_tests_run = 0; g_tests_passed = 0;
    failures += test_barycentric_reproduces_knots();
    failures += test_report("test_barycentric_reproduces_knots");

    g_tests_run = 0; g_tests_passed = 0;
    failures += test_barycentric_runge();
    failures += test_report("test_barycentric_runge");

    g_tests_run = 0; g_tests_passed = 0;
    failures += test_barycentric_irregular();
    failures += test_report("test_barycentric_irregular");

    sis_fftw_cleanup();
    return failures;
}
