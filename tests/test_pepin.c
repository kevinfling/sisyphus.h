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

static int test_pepin_eval_reproduces_knots(void) {
    int n = 16;
    double x[16], y[16];
    for (int i = 0; i < n; ++i) {
        x[i] = (double)i / (n - 1);
        y[i] = sin(2.0 * M_PI * x[i]);
    }

    sis_pepin1d_t s;
    int ret = sis_pepin1d_build(&s, x, y, n, 5);
    TEST_ASSERT(ret == SIS_OK);

    for (int i = 0; i < n - 1; ++i) {
        double v = sis_pepin1d_eval(&s, x[i], 0);
        TEST_ASSERT_NEAR(v, y[i], 1e-10);
    }

    sis_pepin1d_free(&s);
    return 0;
}

static int test_pepin_integral_constant(void) {
    int n = 8;
    double x[8], y[8];
    for (int i = 0; i < n; ++i) {
        x[i] = (double)i;
        y[i] = 2.0; /* constant function */
    }

    sis_pepin1d_t s;
    int ret = sis_pepin1d_build(&s, x, y, n, 3);
    TEST_ASSERT(ret == SIS_OK);

    double v = sis_pepin1d_integrate(&s, 0.0, (double)(n - 1));
    TEST_ASSERT_NEAR(v, 2.0 * (n - 1), 1e-9);

    sis_pepin1d_free(&s);
    return 0;
}

static int test_pepin_fit_denoised(void) {
    int n = 64;
    double x[64], y[64];
    for (int i = 0; i < n; ++i) {
        x[i] = (double)i / (n - 1);
        y[i] = sin(2.0 * M_PI * x[i]) + 0.1 * ((double)rand() / RAND_MAX - 0.5);
    }

    sis_pepin1d_t s;
    int ret = sis_pepin1d_fit_denoised(&s, x, y, n, 5, 0.1);
    TEST_ASSERT(ret == SIS_OK);

    double v = sis_pepin1d_eval(&s, 0.25, 0);
    TEST_ASSERT_NEAR(v, 1.0, 0.15); /* sin(pi/2) = 1 */

    sis_pepin1d_free(&s);
    return 0;
}

int main(void) {
    int failures = 0;
    g_tests_run = 0; g_tests_passed = 0;
    failures += test_pepin_eval_reproduces_knots();
    failures += test_report("test_pepin_eval_reproduces_knots");

    g_tests_run = 0; g_tests_passed = 0;
    failures += test_pepin_integral_constant();
    failures += test_report("test_pepin_integral_constant");

    g_tests_run = 0; g_tests_passed = 0;
    failures += test_pepin_fit_denoised();
    failures += test_report("test_pepin_fit_denoised");

    sis_fftw_cleanup();
    return failures;
}
