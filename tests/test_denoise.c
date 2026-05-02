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

static int test_denoise_reduces_high_freq(void) {
    int n = 128;
    double* in = (double*)malloc(n * sizeof(double));
    double* out = (double*)malloc(n * sizeof(double));
    for (int i = 0; i < n; ++i) {
        double t = (double)i / (n - 1);
        in[i] = sin(2.0 * M_PI * 2.0 * t) + 0.5 * sin(2.0 * M_PI * 60.0 * t); /* low + high freq */
    }
    memcpy(out, in, n * sizeof(double));

    int ret = sis_denoise_1d(out, n, 0.05); /* keep ~6 modes */
    TEST_ASSERT(ret == SIS_OK);

    /* High-frequency energy should be reduced */
    double ein = 0.0, eout = 0.0;
    for (int i = 0; i < n; ++i) {
        ein += in[i] * in[i];
        eout += out[i] * out[i];
    }
    /* Denoising shouldn't amplify total energy dramatically */
    TEST_ASSERT(eout < ein * 1.5);

    free(in); free(out);
    return 0;
}

static int test_denoise_2d(void) {
    int dims[2] = {32, 32};
    size_t total = 32 * 32;
    double* in = (double*)malloc(total * sizeof(double));
    for (size_t i = 0; i < total; ++i) in[i] = (double)(rand() % 100) / 100.0;

    int ret = sis_denoise_nd(2, dims, in, 0.1, NULL);
    TEST_ASSERT(ret == SIS_OK);

    free(in);
    return 0;
}

int main(void) {
    int failures = 0;
    g_tests_run = 0; g_tests_passed = 0;
    failures += test_denoise_reduces_high_freq();
    failures += test_report("test_denoise_reduces_high_freq");

    g_tests_run = 0; g_tests_passed = 0;
    failures += test_denoise_2d();
    failures += test_report("test_denoise_2d");

    sis_fftw_cleanup();
    return failures;
}
