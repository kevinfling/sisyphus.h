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

/* DCT-II nodes on [0,1] are x_j = (j + 0.5) / n */
static int test_derivative_sin(void) {
    int n = 256;
    double* in = (double*)malloc(n * sizeof(double));
    double* out = (double*)malloc(n * sizeof(double));
    double L = 1.0;
    double scale = M_PI / L;
    for (int i = 0; i < n; ++i) {
        double x = (double)(i + 0.5) / n;
        in[i] = sin(2.0 * M_PI * x);
    }

    int ret = sis_derivative_1d(in, out, n, 1, scale);
    TEST_ASSERT(ret == SIS_OK);

    for (int i = 4; i < n - 4; ++i) {
        double x = (double)(i + 0.5) / n;
        double expected = 2.0 * M_PI * cos(2.0 * M_PI * x);
        TEST_ASSERT_NEAR(out[i], expected, 0.3);
    }

    free(in); free(out);
    return 0;
}

static int test_derivative_polynomial(void) {
    int n = 64;
    double* in = (double*)malloc(n * sizeof(double));
    double* out = (double*)malloc(n * sizeof(double));
    double L = 2.0;
    double scale = M_PI / L;
    for (int i = 0; i < n; ++i) {
        double x = L * (double)(i + 0.5) / n;
        in[i] = x * x * x; /* f(x) = x^3, f'(x) = 3x^2 */
    }

    int ret = sis_derivative_1d(in, out, n, 1, scale);
    TEST_ASSERT(ret == SIS_OK);

    for (int i = 4; i < n - 4; ++i) {
        double x = L * (double)(i + 0.5) / n;
        double expected = 3.0 * x * x;
        TEST_ASSERT_NEAR(out[i], expected, 0.5);
    }

    free(in); free(out);
    return 0;
}

static int test_second_derivative_sin(void) {
    int n = 256;
    double* in = (double*)malloc(n * sizeof(double));
    double* out = (double*)malloc(n * sizeof(double));
    double scale = M_PI;
    for (int i = 0; i < n; ++i) {
        double x = (double)(i + 0.5) / n;
        in[i] = sin(2.0 * M_PI * x);
    }

    int ret = sis_derivative_1d(in, out, n, 2, scale);
    TEST_ASSERT(ret == SIS_OK);

    for (int i = 8; i < n - 8; ++i) {
        double x = (double)(i + 0.5) / n;
        double expected = -4.0 * M_PI * M_PI * sin(2.0 * M_PI * x);
        TEST_ASSERT_NEAR(out[i], expected, 3.0);
    }

    free(in); free(out);
    return 0;
}

int main(void) {
    int failures = 0;
    g_tests_run = 0; g_tests_passed = 0;
    failures += test_derivative_sin();
    failures += test_report("test_derivative_sin");

    g_tests_run = 0; g_tests_passed = 0;
    failures += test_derivative_polynomial();
    failures += test_report("test_derivative_polynomial");

    g_tests_run = 0; g_tests_passed = 0;
    failures += test_second_derivative_sin();
    failures += test_report("test_second_derivative_sin");

    sis_fftw_cleanup();
    return failures;
}
