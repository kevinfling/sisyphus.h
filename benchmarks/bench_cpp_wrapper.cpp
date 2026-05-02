/* C++ wrapper overhead benchmark */
#define SISYPHUS_IMPLEMENTATION
#include "sisyphus.hpp"

#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <vector>

static double now_us() {
    auto t = std::chrono::high_resolution_clock::now().time_since_epoch();
    return std::chrono::duration<double, std::micro>(t).count();
}

int main() {
    printf("=== C++ wrapper benchmarks ===\n\n");

    /* Pepin1D evaluation */
    {
        int n = 256;
        std::vector<double> x(n), y(n);
        for (int i = 0; i < n; ++i) {
            x[i] = (double)i / (n - 1);
            y[i] = std::sin(2.0 * M_PI * x[i]);
        }
        sisyphus::Pepin1D spline(x, y, 7);

        int nq = 1000000;
        double sum = 0.0;
        double t0 = now_us();
        for (int i = 0; i < nq; ++i) {
            double xq = (double)(i % 1000) / 1000.0;
            sum += spline.eval(xq, 0);
        }
        double t1 = now_us();
        double ns_per = (t1 - t0) * 1000.0 / nq;
        printf("Pepin1D evaluation (degree 7, N=%d)\n", n);
        printf("  Queries: %d\n", nq);
        printf("  Time / query: %.1f ns\n", ns_per);
        printf("  (checksum: %g)\n\n", sum);
    }

    /* Tensor-product Pepin */
    {
        int nx = 32, ny = 32;
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

        sisyphus::Tensor spline(coords, y, 3);

        int nq = 1000000;
        double sum = 0.0;
        double t0 = now_us();
        for (int i = 0; i < nq; ++i) {
            std::vector<double> xq = {
                (double)(i % nx) / (nx - 1),
                (double)(i % ny) / (ny - 1)
            };
            sum += spline.eval(xq, -1);
        }
        double t1 = now_us();
        double ns_per = (t1 - t0) * 1000.0 / nq;
        printf("Tensor-product Pepin (deg 3, %dx%d)\n", nx, ny);
        printf("  Queries: %d\n", nq);
        printf("  Time / query: %.1f ns\n", ns_per);
        printf("  (checksum: %g)\n\n", sum);
    }

    /* Thin-plate spline eval */
    {
        int ncentres = 100;
        int d = 2;
        std::vector<double> xi(ncentres * d), yi(ncentres);
        for (int i = 0; i < ncentres; ++i) {
            xi[i * d + 0] = (double)std::rand() / RAND_MAX;
            xi[i * d + 1] = (double)std::rand() / RAND_MAX;
            yi[i] = std::sin(2.0 * M_PI * xi[i * d + 0]);
        }
        sisyphus::ThinPlateSpline tps(xi, yi, d, 1e-6);

        int nq = 100000;
        double sum = 0.0;
        double t0 = now_us();
        for (int i = 0; i < nq; ++i) {
            std::vector<double> xq = {
                (double)(i % ncentres) / ncentres,
                (double)(i % ncentres) / ncentres
            };
            sum += tps.eval(xq);
        }
        double t1 = now_us();
        double ns_per = (t1 - t0) * 1000.0 / nq;
        printf("Thin-plate spline eval (n=%d centres, %d-D)\n", ncentres, d);
        printf("  Queries: %d\n", nq);
        printf("  Time / query: %.1f ns\n", ns_per);
        printf("  (checksum: %g)\n\n", sum);
    }

    sis_fftw_cleanup();
    return 0;
}
