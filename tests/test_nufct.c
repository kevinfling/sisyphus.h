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

static int test_nufct_reproduces_dct_nodes(void) {
    int n = 64;
    double* y = (double*)malloc(n * sizeof(double));
    double* out = (double*)malloc(n * sizeof(double));
    for (int i = 0; i < n; ++i) {
        double x = (double)(i + 0.5) / n;
        y[i] = sin(2.0 * M_PI * x);
    }
    /* Query at the original DCT-II nodes */
    double* xq = (double*)malloc(n * sizeof(double));
    for (int i = 0; i < n; ++i) xq[i] = (double)(i + 0.5) / n;

    int ret = sis_nufct_eval(n, y, n, xq, out, 1.0, 8);
    TEST_ASSERT(ret == SIS_OK);
    for (int i = 0; i < n; ++i) {
        TEST_ASSERT_NEAR(out[i], y[i], 1e-3); /* cubic interp tolerance */
    }
    free(y); free(out); free(xq);
    return 0;
}

static int test_nufct_accuracy_random(void) {
    int n = 128;
    double* y = (double*)malloc(n * sizeof(double));
    for (int i = 0; i < n; ++i) {
        double x = (double)(i + 0.5) / n;
        y[i] = sin(2.0 * M_PI * x) + 0.5 * cos(4.0 * M_PI * x);
    }
    int m = 1000;
    double* xq = (double*)malloc(m * sizeof(double));
    double* out_fast = (double*)malloc(m * sizeof(double));
    double* out_direct = (double*)malloc(m * sizeof(double));
    for (int i = 0; i < m; ++i) xq[i] = (double)(rand() % 1000) / 1000.0;

    int ret = sis_nufct_eval(n, y, m, xq, out_fast, 1.0, 16);
    TEST_ASSERT(ret == SIS_OK);
    ret = sis_nufct_eval(n, y, m, xq, out_direct, 1.0, -1); /* direct path */
    TEST_ASSERT(ret == SIS_OK);

    double err = 0.0;
    for (int i = 0; i < m; ++i) err += fabs(out_fast[i] - out_direct[i]);
    TEST_ASSERT(err / m < 1e-4);

    free(y); free(xq); free(out_fast); free(out_direct);
    return 0;
}

int main(void) {
    int failures = 0;
    g_tests_run = 0; g_tests_passed = 0;
    failures += test_nufct_reproduces_dct_nodes();
    failures += test_report("test_nufct_reproduces_dct_nodes");

    g_tests_run = 0; g_tests_passed = 0;
    failures += test_nufct_accuracy_random();
    failures += test_report("test_nufct_accuracy_random");

    sis_fftw_cleanup();
    return failures;
}
