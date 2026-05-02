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

static int test_smoothing_spline_identity(void) {
    int n = 64;
    double x[64], y[64], out[64];
    for (int i = 0; i < n; ++i) {
        x[i] = (double)i / (n - 1);
        y[i] = sin(2.0 * M_PI * x[i]);
    }

    /* lambda = 0 should reproduce input (up to numerical noise) */
    int ret = sis_smoothing_spline_1d(x, y, n, 3, 0.0, out);
    TEST_ASSERT(ret == SIS_OK);

    for (int i = 0; i < n; ++i) {
        TEST_ASSERT_NEAR(out[i], y[i], 1e-11);
    }

    return 0;
}

static int test_smoothing_spline_reduces_roughness(void) {
    int n = 128;
    double x[128], y[128], out[128];
    for (int i = 0; i < n; ++i) {
        x[i] = (double)i / (n - 1);
        y[i] = sin(2.0 * M_PI * x[i]) + 0.5 * ((double)rand() / RAND_MAX - 0.5);
    }

    int ret = sis_smoothing_spline_1d(x, y, n, 3, 1e-2, out);
    TEST_ASSERT(ret == SIS_OK);

    /* Measure roughness as sum of squared second differences */
    double rough_in = 0.0, rough_out = 0.0;
    for (int i = 1; i < n - 1; ++i) {
        double din = y[i+1] - 2.0*y[i] + y[i-1];
        double dout = out[i+1] - 2.0*out[i] + out[i-1];
        rough_in += din * din;
        rough_out += dout * dout;
    }
    TEST_ASSERT(rough_out < rough_in);

    return 0;
}

int main(void) {
    int failures = 0;
    g_tests_run = 0; g_tests_passed = 0;
    failures += test_smoothing_spline_identity();
    failures += test_report("test_smoothing_spline_identity");

    g_tests_run = 0; g_tests_passed = 0;
    failures += test_smoothing_spline_reduces_roughness();
    failures += test_report("test_smoothing_spline_reduces_roughness");

    sis_fftw_cleanup();
    return failures;
}
