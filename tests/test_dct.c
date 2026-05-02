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

static int test_dct_roundtrip_1d(void) {
    int n = 64;
    double* in = (double*)malloc(n * sizeof(double));
    double* tmp = (double*)malloc(n * sizeof(double));
    double* out = (double*)malloc(n * sizeof(double));
    for (int i = 0; i < n; ++i) in[i] = sin(2.0 * M_PI * i / (n - 1));

    int ret = sis_dct2(in, tmp, n);
    TEST_ASSERT(ret == SIS_OK);
    ret = sis_idct2(tmp, out, n);
    TEST_ASSERT(ret == SIS_OK);

    for (int i = 0; i < n; ++i) {
        TEST_ASSERT_NEAR(out[i], in[i], 1e-12);
    }

    free(in); free(tmp); free(out);
    return 0;
}

static int test_dct_nd_roundtrip(void) {
    int dims[2] = {16, 16};
    size_t total = 16 * 16;
    double* in = (double*)malloc(total * sizeof(double));
    double* out = (double*)malloc(total * sizeof(double));
    for (size_t i = 0; i < total; ++i) in[i] = (double)(i % 7) * 0.1;

    int ret = sis_dct2_nd(2, dims, in, out, NULL);
    TEST_ASSERT(ret == SIS_OK);
    ret = sis_idct2_nd(2, dims, out, out, NULL);
    TEST_ASSERT(ret == SIS_OK);

    for (size_t i = 0; i < total; ++i) {
        TEST_ASSERT_NEAR(out[i], in[i], 1e-11);
    }

    free(in); free(out);
    return 0;
}

static int test_dct_preserves_dc(void) {
    int n = 32;
    double* in = (double*)malloc(n * sizeof(double));
    double* out = (double*)malloc(n * sizeof(double));
    for (int i = 0; i < n; ++i) in[i] = 3.0;

    int ret = sis_dct2(in, out, n);
    TEST_ASSERT(ret == SIS_OK);
    /* Raw FFTW REDFT10: DC coefficient = 2 * sum of inputs = 6N */
    TEST_ASSERT_NEAR(out[0], 6.0 * n, 1e-9);
    for (int i = 1; i < n; ++i) TEST_ASSERT_NEAR(out[i], 0.0, 1e-9);

    free(in); free(out);
    return 0;
}

int main(void) {
    int failures = 0;
    g_tests_run = 0; g_tests_passed = 0;
    failures += test_dct_roundtrip_1d();
    failures += test_report("test_dct_roundtrip_1d");

    g_tests_run = 0; g_tests_passed = 0;
    failures += test_dct_nd_roundtrip();
    failures += test_report("test_dct_nd_roundtrip");

    g_tests_run = 0; g_tests_passed = 0;
    failures += test_dct_preserves_dc();
    failures += test_report("test_dct_preserves_dc");

    sis_fftw_cleanup();
    return failures;
}
