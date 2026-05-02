/* SISYPHUS v0.1.0 — Spectral Interpolation System Yielding Precise High-order Univariate Splines
 * Copyright (c) 2026 Kevin Fling
 * Licensed under the MIT License
 * https://github.com/kevinfling/sisyphus.h
 */

/* bench_pepin_convergence.c
 * Measure Pepin eval accuracy and speed as n grows from 256 to 4096.
 * Two sampling strategies:
 *   A) Uniform physical knots (j/(n-1))  -- current API usage
 *   B) DCT-II grid knots ((j+0.5)/n)     -- exact for spectral differentiation
 */
#define _POSIX_C_SOURCE 199309L
#define SISYPHUS_IMPLEMENTATION
#include "sisyphus.h"
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>

static double now(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return ts.tv_sec + ts.tv_nsec * 1e-9;
}

static double dct_interpolant_eval(int n, const double* c, double t) {
    double val = c[0];
    for (int k = 1; k < n; ++k) val += c[k] * cos((double)k * t);
    return val;
}

static void run_config(int n, int degree, int use_dct_grid) {
    double* x = (double*)malloc(n * sizeof(double));
    double* y = (double*)malloc(n * sizeof(double));
    for (int i = 0; i < n; ++i) {
        if (use_dct_grid) x[i] = (double)(i + 0.5) / n;
        else x[i] = (double)i / (n - 1);
        y[i] = sin(2.0 * M_PI * x[i]);
    }

    sis_pepin1d_t s;
    int ret = sis_pepin1d_build(&s, x, y, n, degree);
    if (ret != SIS_OK) {
        fprintf(stderr, "build failed: n=%d deg=%d ret=%d\n", n, degree, ret);
        free(x); free(y);
        return;
    }

    /* Precompute normalized DCT coefficients for ground truth */
    double* c = (double*)malloc(n * sizeof(double));
    sis_dct2(y, c, n);
    double inv_n = 1.0 / (double)n;
    c[0] *= 0.5 * inv_n;
    for (int k = 1; k < n; ++k) c[k] *= inv_n;

    int n_queries = 20000;
    double* xq = (double*)malloc(n_queries * sizeof(double));
    double* exact = (double*)malloc(n_queries * sizeof(double));
    for (int i = 0; i < n_queries; ++i) {
        xq[i] = (double)(rand() % 1000) / 1000.0;
        exact[i] = dct_interpolant_eval(n, c, M_PI * xq[i]);
    }

    /* Warm-up */
    for (int i = 0; i < 1000; ++i) {
        sis_pepin1d_eval(&s, xq[i % n_queries], 0);
    }

    /* Timed eval */
    double t0 = now();
    double sum = 0.0;
    for (int i = 0; i < n_queries; ++i) {
        sum += sis_pepin1d_eval(&s, xq[i], 0);
    }
    double t1 = now();
    double ns_per = (t1 - t0) * 1e9 / n_queries;

    /* Error measurement */
    double l1_err = 0.0, l2_err = 0.0, max_err = 0.0;
    for (int i = 0; i < n_queries; ++i) {
        double v = sis_pepin1d_eval(&s, xq[i], 0);
        double e = fabs(v - exact[i]);
        l1_err += e;
        l2_err += e * e;
        if (e > max_err) max_err = e;
    }
    l1_err /= n_queries;
    l2_err = sqrt(l2_err / n_queries);

    const char* grid_label = use_dct_grid ? "DCT" : "uni";
    printf("%-6d %-6d %-4s %-12.1f %-14.6e %-14.6e %-14.6e  (chk=%.3f)\n",
           n, degree, grid_label, ns_per, l1_err, l2_err, max_err, sum);

    sis_pepin1d_free(&s);
    free(x); free(y);
    free(c); free(xq); free(exact);
}

int main(void) {
    int degrees[] = {5, 7, 9, 11};
    int n_values[] = {256, 512, 1024, 2048, 4096};
    int n_degrees = sizeof(degrees) / sizeof(degrees[0]);
    int n_nvals = sizeof(n_values) / sizeof(n_values[0]);

    printf("Pepin convergence benchmark (20000 random queries per config)\n");
    printf("%-6s %-6s %-4s %-12s %-14s %-14s %-14s\n",
           "n", "deg", "grid", "time(ns)", "L1_err", "L2_err", "max_err");

    for (int d = 0; d < n_degrees; ++d) {
        int degree = degrees[d];
        for (int ni = 0; ni < n_nvals; ++ni) {
            int n = n_values[ni];
            run_config(n, degree, 0); /* uniform grid */
            run_config(n, degree, 1); /* DCT-II grid */
        }
        printf("\n");
    }

    sis_fftw_cleanup();
    return 0;
}
