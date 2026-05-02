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
#include <string.h>

/* DCT-II nodes on [0,1] are x_j = (j + 0.5) / n */
static int test_fractional_identity(void) {
    int n = 64;
    double* in = (double*)malloc(n * sizeof(double));
    double* out = (double*)malloc(n * sizeof(double));
    for (int i = 0; i < n; ++i) {
        double x = (double)(i + 0.5) / n;
        in[i] = sin(2.0 * M_PI * x);
    }
    int ret = sis_fractional_derivative_1d(in, out, n, 0.0, M_PI);
    TEST_ASSERT(ret == SIS_OK);
    for (int i = 0; i < n; ++i) TEST_ASSERT_NEAR(out[i], in[i], 1e-12);
    free(in); free(out);
    return 0;
}

static int test_fractional_matches_integer(void) {
    int n = 128;
    double* in = (double*)malloc(n * sizeof(double));
    double* out_int = (double*)malloc(n * sizeof(double));
    double* out_frac = (double*)malloc(n * sizeof(double));
    for (int i = 0; i < n; ++i) {
        double x = (double)(i + 0.5) / n;
        in[i] = sin(2.0 * M_PI * x);
    }
    int ret = sis_derivative_1d(in, out_int, n, 2, M_PI);
    TEST_ASSERT(ret == SIS_OK);
    ret = sis_fractional_derivative_1d(in, out_frac, n, 2.0, M_PI);
    TEST_ASSERT(ret == SIS_OK);
    for (int i = 4; i < n - 4; ++i) {
        TEST_ASSERT_NEAR(out_frac[i], out_int[i], 1e-9);
    }
    free(in); free(out_int); free(out_frac);
    return 0;
}

static int test_fractional_semigroup(void) {
    int n = 128;
    double* in = (double*)malloc(n * sizeof(double));
    double* tmp = (double*)malloc(n * sizeof(double));
    double* out1 = (double*)malloc(n * sizeof(double));
    double* out2 = (double*)malloc(n * sizeof(double));
    for (int i = 0; i < n; ++i) {
        double x = (double)(i + 0.5) / n;
        in[i] = sin(2.0 * M_PI * x);
    }
    /* D^{0.7+0.3} f vs D^{0.7} D^{0.3} f */
    int ret = sis_fractional_derivative_1d(in, tmp, n, 0.3, M_PI);
    TEST_ASSERT(ret == SIS_OK);
    ret = sis_fractional_derivative_1d(tmp, out1, n, 0.7, M_PI);
    TEST_ASSERT(ret == SIS_OK);
    ret = sis_fractional_derivative_1d(in, out2, n, 1.0, M_PI);
    TEST_ASSERT(ret == SIS_OK);
    for (int i = 8; i < n - 8; ++i) {
        TEST_ASSERT_NEAR(out1[i], out2[i], 2.0);
    }
    free(in); free(tmp); free(out1); free(out2);
    return 0;
}

int main(void) {
    int failures = 0;
    g_tests_run = 0; g_tests_passed = 0;
    failures += test_fractional_identity();
    failures += test_report("test_fractional_identity");

    g_tests_run = 0; g_tests_passed = 0;
    failures += test_fractional_matches_integer();
    failures += test_report("test_fractional_matches_integer");

    g_tests_run = 0; g_tests_passed = 0;
    failures += test_fractional_semigroup();
    failures += test_report("test_fractional_semigroup");

    sis_fftw_cleanup();
    return failures;
}
