/* C++17 example: fit a Pepin spline and evaluate derivatives */
#define SISYPHUS_IMPLEMENTATION
#include "sisyphus.hpp"

#include <cmath>
#include <iomanip>
#include <iostream>
#include <vector>

int main() {
    int n = 256;
    double L = 1.0;
    std::vector<double> x(n), y(n);
    for (int i = 0; i < n; ++i) {
        x[i] = L * i / (n - 1);
        y[i] = std::sin(2.0 * M_PI * x[i]);
    }

    /* Build a degree-7 Pepin spline */
    sisyphus::Pepin1D spline(x, y, 7);

    std::cout << std::setprecision(10);
    std::cout << "Pepin spline (degree " << spline.degree() << ", "
              << spline.n_intervals() << " intervals, "
              << (spline.uniform() ? "uniform" : "non-uniform") << " grid)\n\n";

    double xq = 0.314;
    std::cout << "f(" << xq << ")  = " << spline.eval(xq, 0) << '\n';
    std::cout << "f'(" << xq << ") = " << spline.eval(xq, 1) << '\n';
    std::cout << "f''(" << xq << ") = " << spline.eval(xq, 2) << '\n';

    double integral = spline.integrate(0.0, L);
    std::cout << "\nIntegral over [0, " << L << "] = " << integral << '\n';
    std::cout << "Expected = 0 (sin over full period)\n";

    /* Denoised variant */
    std::vector<double> noisy = y;
    for (auto& v : noisy) v += 0.05 * ((double)std::rand() / RAND_MAX - 0.5);
    auto denoised = sisyphus::Pepin1D::fit_denoised(x, noisy, 7, 0.05);
    std::cout << "\nDenoised f(" << xq << ") = " << denoised.eval(xq, 0) << '\n';

    sis_fftw_cleanup();
    return 0;
}
