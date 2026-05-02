/* SISYPHUS C++ wrapper test suite */
#define SISYPHUS_IMPLEMENTATION
#include "sisyphus.hpp"

#include <cmath>
#include <cstdlib>
#include <iostream>
#include <vector>

static int g_tests_run = 0;
static int g_tests_passed = 0;

#define TEST_ASSERT(cond) do { \
    g_tests_run++; \
    if (!(cond)) { \
        std::cerr << "  FAIL: " << __FILE__ << ":" << __LINE__ \
                  << ": " << #cond << '\n'; \
        return 1; \
    } else { \
        g_tests_passed++; \
    } \
} while(0)

#define TEST_ASSERT_NEAR(a, b, tol) do { \
    g_tests_run++; \
    double _a = (a), _b = (b), _tol = (tol); \
    if (std::fabs(_a - _b) > _tol) { \
        std::cerr << "  FAIL: " << __FILE__ << ":" << __LINE__ \
                  << " |" << _a << " - " << _b << "| = " << std::fabs(_a - _b) \
                  << " > " << _tol << '\n'; \
        return 1; \
    } else { \
        g_tests_passed++; \
    } \
} while(0)

static int test_cpp_dct_roundtrip() {
    int n = 64;
    std::vector<double> in(n), fwd(n), back(n);
    for (int i = 0; i < n; ++i) in[i] = std::sin(2.0 * M_PI * i / n);

    sisyphus::dct2(in, fwd);
    sisyphus::idct2(fwd, back);

    for (int i = 0; i < n; ++i)
        TEST_ASSERT_NEAR(back[i], in[i], 1e-12);
    return 0;
}

static int test_cpp_denoise() {
    int n = 128;
    std::vector<double> sig(n);
    for (int i = 0; i < n; ++i)
        sig[i] = std::sin(2.0 * M_PI * i / n) + 0.5 * ((double)std::rand() / RAND_MAX - 0.5);

    sisyphus::denoise(sig, 0.05);
    /* After heavy denoising the signal should be smooth */
    double max_diff = 0.0;
    for (int i = 1; i < n - 1; ++i) {
        double lap = std::fabs(sig[i-1] - 2*sig[i] + sig[i+1]);
        if (lap > max_diff) max_diff = lap;
    }
    TEST_ASSERT(max_diff < 0.5); /* crude smoothness check */
    return 0;
}

static int test_cpp_derivative() {
    int n = 256;
    double L = 1.0;
    std::vector<double> f(n), d1(n);
    /* DCT-II nodes: exact for spectral differentiation */
    for (int i = 0; i < n; ++i) {
        double x = L * (i + 0.5) / n;
        f[i] = std::sin(2.0 * M_PI * x);
    }
    sisyphus::derivative(f, d1, 1, L);

    for (int i = 4; i < n - 4; ++i) {
        double x = L * (i + 0.5) / n;
        double expected = 2.0 * M_PI * std::cos(2.0 * M_PI * x);
        TEST_ASSERT_NEAR(d1[i], expected, 0.3);
    }
    return 0;
}

static int test_cpp_fractional_derivative() {
    int n = 256;
    double L = 1.0;
    std::vector<double> f(n), d05(n);
    for (int i = 0; i < n; ++i) {
        double x = L * i / (n - 1);
        f[i] = std::exp(-10.0 * (x - 0.5) * (x - 0.5));
    }
    sisyphus::fractional_derivative(f, d05, 0.5, L);
    /* Just check it doesn't crash and produces finite values */
    for (int i = 0; i < n; ++i)
        TEST_ASSERT(std::isfinite(d05[i]));
    return 0;
}

static int test_cpp_pepin_eval() {
    int n = 32;
    double L = 1.0;
    std::vector<double> x(n), y(n);
    for (int i = 0; i < n; ++i) {
        x[i] = L * i / (n - 1);
        y[i] = std::sin(2.0 * M_PI * x[i]);
    }
    sisyphus::Pepin1D spline(x, y, 7);

    for (int i = 0; i < n - 1; ++i) {
        double v = spline.eval(x[i], 0);
        TEST_ASSERT_NEAR(v, y[i], 1e-10);
    }
    return 0;
}

static int test_cpp_pepin_derivative() {
    int n = 64;
    double L = 1.0;
    std::vector<double> x(n), y(n);
    for (int i = 0; i < n; ++i) {
        x[i] = L * i / (n - 1);
        y[i] = std::sin(2.0 * M_PI * x[i]);
    }
    sisyphus::Pepin1D spline(x, y, 7);

    double xq = 0.314;
    double d1 = spline.eval(xq, 1);
    double expected = 2.0 * M_PI * std::cos(2.0 * M_PI * xq);
    TEST_ASSERT_NEAR(d1, expected, 0.05);
    return 0;
}

static int test_cpp_pepin_integrate() {
    int n = 16;
    std::vector<double> x(n), y(n, 3.0);
    for (int i = 0; i < n; ++i) x[i] = (double)i;
    sisyphus::Pepin1D spline(x, y, 3);
    double v = spline.integrate(0.0, (double)(n - 1));
    TEST_ASSERT_NEAR(v, 3.0 * (n - 1), 1e-9);
    return 0;
}

static int test_cpp_pepin_move() {
    int n = 16;
    std::vector<double> x(n), y(n, 1.0);
    for (int i = 0; i < n; ++i) x[i] = (double)i / (n - 1);

    sisyphus::Pepin1D s1(x, y, 3);
    sisyphus::Pepin1D s2 = std::move(s1);
    TEST_ASSERT_NEAR(s2.eval(0.5, 0), 1.0, 1e-10);
    return 0;
}

static int test_cpp_tensor_2d() {
    int nx = 16, ny = 16;
    std::vector<std::vector<double>> coords = {
        std::vector<double>(nx), std::vector<double>(ny)
    };
    for (int i = 0; i < nx; ++i) coords[0][i] = (double)i / (nx - 1);
    for (int j = 0; j < ny; ++j) coords[1][j] = (double)j / (ny - 1);

    std::vector<double> y(nx * ny);
    for (int i = 0; i < nx; ++i)
        for (int j = 0; j < ny; ++j)
            y[i * ny + j] = std::sin(2.0 * M_PI * coords[0][i])
                          * std::cos(2.0 * M_PI * coords[1][j]);

    sisyphus::Tensor spline(coords, y, 5);
    /* Evaluate at a grid point (exact reproduction) */
    std::vector<double> xq = {coords[0][4], coords[1][3]};
    double v = spline.eval(xq, -1);
    double expected = std::sin(2.0 * M_PI * coords[0][4])
                    * std::cos(2.0 * M_PI * coords[1][3]);
    TEST_ASSERT_NEAR(v, expected, 1e-9);
    return 0;
}

static int test_cpp_tps_2d() {
    int n = 50;
    int d = 2;
    std::vector<double> xi(n * d), yi(n);
    for (int i = 0; i < n; ++i) {
        xi[i * d + 0] = (double)std::rand() / RAND_MAX;
        xi[i * d + 1] = (double)std::rand() / RAND_MAX;
        yi[i] = std::sin(2.0 * M_PI * xi[i * d + 0])
              * std::cos(2.0 * M_PI * xi[i * d + 1]);
    }
    sisyphus::ThinPlateSpline tps(xi, yi, d, 1e-6);

    std::vector<double> xq = {0.5, 0.5};
    double v = tps.eval(xq);
    double expected = std::sin(M_PI) * std::cos(M_PI); /* = 0 */
    TEST_ASSERT_NEAR(v, expected, 0.05);
    return 0;
}

static int test_cpp_barycentric() {
    int n = 33;
    std::vector<double> x(n), y(n);
    for (int i = 0; i < n; ++i) {
        x[i] = -1.0 + 2.0 * i / (n - 1);
        y[i] = 1.0 / (1.0 + 25.0 * x[i] * x[i]);
    }
    sisyphus::Barycentric bc(x, y, 5);
    double v = bc.eval(0.0);
    TEST_ASSERT_NEAR(v, 1.0, 1e-6);
    return 0;
}

static int test_cpp_nufct() {
    int n = 256;
    double L = 1.0;
    std::vector<double> y(n);
    for (int i = 0; i < n; ++i)
        y[i] = std::sin(2.0 * M_PI * L * i / (n - 1));

    std::vector<double> xq = {0.0, 0.25, 0.5, 0.75};
    auto out = sisyphus::nufct_eval(y, xq, L);

    TEST_ASSERT_NEAR(out[0], 0.0, 0.05);
    TEST_ASSERT_NEAR(out[1], 1.0, 0.05);
    TEST_ASSERT_NEAR(out[2], 0.0, 0.05);
    TEST_ASSERT_NEAR(out[3], -1.0, 0.05);
    return 0;
}

static int test_cpp_smoothing_spline() {
    int n = 128;
    double L = 1.0;
    std::vector<double> x(n), y(n);
    for (int i = 0; i < n; ++i) {
        x[i] = L * i / (n - 1);
        y[i] = std::sin(2.0 * M_PI * x[i]) + 0.5 * ((double)std::rand() / RAND_MAX - 0.5);
    }
    auto smooth = sisyphus::smoothing_spline(x, y, 3, 1e-2);
    TEST_ASSERT(smooth.size() == y.size());
    /* Smoothing should reduce roughness (sum of squared second differences) */
    double rough_in = 0.0, rough_out = 0.0;
    for (int i = 1; i < n - 1; ++i) {
        double din  = y[i+1] - 2.0*y[i] + y[i-1];
        double dout = smooth[i+1] - 2.0*smooth[i] + smooth[i-1];
        rough_in  += din * din;
        rough_out += dout * dout;
    }
    TEST_ASSERT(rough_out < rough_in);
    return 0;
}

static int test_cpp_weak_jacobian_pwc() {
    int k = 3;
    std::vector<int> partition = {0, 0, 1, 1, 1, 2, 2, 2};
    std::vector<double> grad_in = {1.0, 2.0, 3.0};
    auto grad_out = sisyphus::weak_jacobian_pwc(partition, k, grad_in);
    TEST_ASSERT_NEAR(grad_out[0], 1.0, 1e-12);
    TEST_ASSERT_NEAR(grad_out[4], 2.0, 1e-12);
    TEST_ASSERT_NEAR(grad_out[7], 3.0, 1e-12);
    return 0;
}

static int test_report(const char* name) {
    if (g_tests_run == g_tests_passed) {
        std::cout << "[PASS] " << name << " (" << g_tests_passed << "/" << g_tests_run << ")\n";
        return 0;
    } else {
        std::cout << "[FAIL] " << name << " (" << g_tests_passed << "/" << g_tests_run << ")\n";
        return 1;
    }
}

int main() {
    int failures = 0;

#define RUN(t) do { \
    g_tests_run = 0; g_tests_passed = 0; \
    failures += t(); \
    failures += test_report(#t); \
} while(0)

    RUN(test_cpp_dct_roundtrip);
    RUN(test_cpp_denoise);
    RUN(test_cpp_derivative);
    RUN(test_cpp_fractional_derivative);
    RUN(test_cpp_pepin_eval);
    RUN(test_cpp_pepin_derivative);
    RUN(test_cpp_pepin_integrate);
    RUN(test_cpp_pepin_move);
    RUN(test_cpp_tensor_2d);
    RUN(test_cpp_tps_2d);
    RUN(test_cpp_barycentric);
    RUN(test_cpp_nufct);
    RUN(test_cpp_smoothing_spline);
    RUN(test_cpp_weak_jacobian_pwc);

#undef RUN

    sis_fftw_cleanup();
    return failures;
}
