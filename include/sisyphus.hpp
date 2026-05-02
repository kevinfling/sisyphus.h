/**===========================================================================
 * sisyphus.hpp
 * C++17 thin convenience wrapper around sisyphus.h
 *
 * Provides RAII containers, std::vector interop, and exception-based error
 * handling for the SISYPHUS spectral toolbox.
 *
 * Usage:
 *   #define SISYPHUS_IMPLEMENTATION   // in exactly one .cpp
 *   #include "sisyphus.hpp"
 *
 *   std::vector<double> x = ... , y = ...;
 *   sisyphus::Pepin1D spline(x, y, 7);
 *   double v = spline.eval(0.5, 0);   // function value
 *   double d = spline.eval(0.5, 1);   // first derivative
 *
 * Version:    v0.1.0
 * License:    MIT License
 * ===========================================================================*/

#ifndef SISYPHUS_HPP
#define SISYPHUS_HPP

#include "sisyphus.h"

#include <array>
#include <cmath>
#include <cstddef>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace sisyphus {

/* ---------------------------------------------------------------------------
 * Error handling
 * ---------------------------------------------------------------------------*/
class Exception : public std::runtime_error {
public:
    explicit Exception(int code, const std::string& msg)
        : std::runtime_error(msg), code_(code) {}
    int code() const noexcept { return code_; }
private:
    int code_;
};

namespace detail {
    inline void check(int code, const char* what) {
        if (code == SIS_OK) return;
        const char* desc = "unknown";
        switch (code) {
            case SIS_ERR_NOMEM: desc = "out of memory"; break;
            case SIS_ERR_SIZE:  desc = "invalid size"; break;
            case SIS_ERR_FFTW:  desc = "FFTW error"; break;
            case SIS_ERR_ARG:   desc = "invalid argument"; break;
            case SIS_ERR_MATH:  desc = "numerical error"; break;
        }
        throw Exception(code, std::string(what) + ": " + desc);
    }
} // namespace detail

/* ---------------------------------------------------------------------------
 * Free functions — 1-D DCT / denoise / differentiate
 * ---------------------------------------------------------------------------*/

/** 1-D DCT-II (FFTW3 REDFT10).  Overwrites \p out. */
inline void dct2(const std::vector<double>& in, std::vector<double>& out) {
    out.resize(in.size());
    detail::check(sis_dct2(in.data(), out.data(), static_cast<int>(in.size())), "dct2");
}

/** 1-D inverse DCT-II (FFTW3 REDFT01, with 1/(2N) normalisation). */
inline void idct2(const std::vector<double>& in, std::vector<double>& out) {
    out.resize(in.size());
    detail::check(sis_idct2(in.data(), out.data(), static_cast<int>(in.size())), "idct2");
}

/** ND separable DCT-II (1-D, 2-D or 3-D). \p dims are row-major. */
inline void dct2_nd(const std::vector<int>& dims,
                    const std::vector<double>& in,
                    std::vector<double>& out) {
    std::size_t total = 1;
    for (int d : dims) total *= static_cast<std::size_t>(d);
    out.resize(total);
    detail::check(sis_dct2_nd(static_cast<int>(dims.size()), dims.data(),
                              in.data(), out.data(), nullptr), "dct2_nd");
}

/** ND separable inverse DCT-II. */
inline void idct2_nd(const std::vector<int>& dims,
                     const std::vector<double>& in,
                     std::vector<double>& out) {
    std::size_t total = 1;
    for (int d : dims) total *= static_cast<std::size_t>(d);
    out.resize(total);
    detail::check(sis_idct2_nd(static_cast<int>(dims.size()), dims.data(),
                               in.data(), out.data(), nullptr), "idct2_nd");
}

/** Hard low-pass filter in the cosine basis (in-place). */
inline void denoise(std::vector<double>& inout, double cutoff_cycles) {
    detail::check(sis_denoise_1d(inout.data(), static_cast<int>(inout.size()),
                                 cutoff_cycles), "denoise");
}

/** ND hard low-pass filter (in-place). */
inline void denoise_nd(const std::vector<int>& dims,
                       std::vector<double>& inout,
                       double cutoff_cycles) {
    detail::check(sis_denoise_nd(static_cast<int>(dims.size()), dims.data(),
                                 inout.data(), cutoff_cycles, nullptr), "denoise_nd");
}

/** Exact mu-th derivative of the DCT interpolant on [0, L]. */
inline void derivative(const std::vector<double>& in,
                       std::vector<double>& out,
                       int mu, double domain_length) {
    out.resize(in.size());
    double scale = (domain_length > 0.0) ? M_PI / domain_length : 1.0;
    detail::check(sis_derivative_1d(in.data(), out.data(),
                                    static_cast<int>(in.size()), mu, scale),
                  "derivative");
}

/** Directional derivative along axis \p dir (0-based). */
inline void derivative_nd(const std::vector<int>& dims,
                          const std::vector<double>& in,
                          std::vector<double>& out,
                          int dir, int mu, double domain_length) {
    std::size_t total = 1;
    for (int d : dims) total *= static_cast<std::size_t>(d);
    out.resize(total);
    double scale = (domain_length > 0.0) ? M_PI / domain_length : 1.0;
    detail::check(sis_derivative_nd(static_cast<int>(dims.size()), dims.data(),
                                    in.data(), out.data(), dir, mu, scale, nullptr),
                  "derivative_nd");
}

/** Fractional derivative of order \p alpha on [0, L]. */
inline void fractional_derivative(const std::vector<double>& in,
                                  std::vector<double>& out,
                                  double alpha, double domain_length) {
    out.resize(in.size());
    double scale = (domain_length > 0.0) ? M_PI / domain_length : 1.0;
    detail::check(sis_fractional_derivative_1d(in.data(), out.data(),
                                               static_cast<int>(in.size()),
                                               alpha, scale),
                  "fractional_derivative");
}

/** Spectral smoothing spline (Demmler–Reinsch). */
inline std::vector<double> smoothing_spline(const std::vector<double>& x,
                                            const std::vector<double>& y,
                                            int order, double lambda) {
    if (x.size() != y.size()) throw Exception(SIS_ERR_ARG, "size mismatch");
    std::vector<double> out(y.size());
    detail::check(sis_smoothing_spline_1d(x.data(), y.data(),
                                          static_cast<int>(x.size()),
                                          order, lambda, out.data()),
                  "smoothing_spline");
    return out;
}

/** Non-uniform fast cosine transform: evaluate DCT interpolant at query points. */
inline std::vector<double> nufct_eval(const std::vector<double>& y,
                                      const std::vector<double>& xq,
                                      double L, int oversample = 0) {
    std::vector<double> out(xq.size());
    detail::check(sis_nufct_eval(static_cast<int>(y.size()), y.data(),
                                 static_cast<int>(xq.size()), xq.data(),
                                 out.data(), L, oversample), "nufct_eval");
    return out;
}

/** Weak Jacobian for piecewise-constant back-prop. */
inline std::vector<double> weak_jacobian_pwc(const std::vector<int>& partition,
                                             int k,
                                             const std::vector<double>& grad_in) {
    std::vector<double> grad_out(partition.size());
    detail::check(sis_weak_jacobian_pwc(static_cast<int>(partition.size()),
                                        partition.data(), k,
                                        grad_in.data(), grad_out.data()),
                  "weak_jacobian_pwc");
    return grad_out;
}

/** Weak Jacobian for piecewise-linear back-prop. */
inline std::vector<double> weak_jacobian_pwl(const std::vector<double>& x,
                                             const std::vector<double>& t,
                                             int k,
                                             const std::vector<double>& grad_in) {
    std::vector<double> grad_out(x.size());
    detail::check(sis_weak_jacobian_pwl(static_cast<int>(x.size()),
                                        x.data(), t.data(), k,
                                        grad_in.data(), grad_out.data()),
                  "weak_jacobian_pwl");
    return grad_out;
}

/* ---------------------------------------------------------------------------
 * RAII wrappers
 * ---------------------------------------------------------------------------*/

/** 1-D Pepin high-degree spline. */
class Pepin1D {
public:
    /** Build from knots and values.  Degree must be ≥ 1 and ≤ 11. */
    Pepin1D(const std::vector<double>& x,
            const std::vector<double>& y,
            int degree)
    {
        if (x.size() != y.size()) throw Exception(SIS_ERR_ARG, "x/y size mismatch");
        detail::check(sis_pepin1d_build(&s_, x.data(), y.data(),
                                        static_cast<int>(x.size()), degree),
                      "Pepin1D::build");
    }

    /** Build from denoised data. */
    static Pepin1D fit_denoised(const std::vector<double>& x,
                                const std::vector<double>& y,
                                int degree,
                                double cutoff_cycles)
    {
        if (x.size() != y.size()) throw Exception(SIS_ERR_ARG, "x/y size mismatch");
        Pepin1D ps;
        detail::check(sis_pepin1d_fit_denoised(&ps.s_, x.data(), y.data(),
                                               static_cast<int>(x.size()),
                                               degree, cutoff_cycles),
                      "Pepin1D::fit_denoised");
        return ps;
    }

    Pepin1D(const Pepin1D&) = delete;
    Pepin1D& operator=(const Pepin1D&) = delete;

    Pepin1D(Pepin1D&& other) noexcept : s_(other.s_) { other.s_ = {}; }
    Pepin1D& operator=(Pepin1D&& other) noexcept {
        if (this != &other) {
            sis_pepin1d_free(&s_);
            s_ = other.s_;
            other.s_ = {};
        }
        return *this;
    }

    ~Pepin1D() { sis_pepin1d_free(&s_); }

    /** Evaluate or differentiate at \p xq.  \p deriv = 0 is the function. */
    double eval(double xq, int deriv = 0) const {
        return sis_pepin1d_eval(&s_, xq, deriv);
    }

    /** Definite integral over [a, b]. */
    double integrate(double a, double b) const {
        return sis_pepin1d_integrate(&s_, a, b);
    }

    int n_intervals() const noexcept { return s_.n; }
    int degree() const noexcept { return s_.degree; }
    bool uniform() const noexcept { return s_.uniform != 0; }

private:
    Pepin1D() = default;
    sis_pepin1d_t s_;
};

/** ND tensor-product Pepin spline (1-D, 2-D or 3-D). */
class Tensor {
public:
    /** Build from rectilinear grid.
     *  \p coords[d] is the 1-D coordinate vector for axis d.
     *  \p y is row-major, size prod(coords[d].size()).
     */
    Tensor(const std::vector<std::vector<double>>& coords,
           const std::vector<double>& y,
           int degree)
    {
        int ndim = static_cast<int>(coords.size());
        if (ndim < 1 || ndim > 3)
            throw Exception(SIS_ERR_ARG, "Tensor: ndim must be 1..3");
        std::vector<int> n(ndim);
        std::size_t total = 1;
        for (int d = 0; d < ndim; ++d) {
            n[d] = static_cast<int>(coords[d].size());
            total *= static_cast<std::size_t>(n[d]);
        }
        if (y.size() != total)
            throw Exception(SIS_ERR_ARG, "Tensor: y size does not match grid");

        detail::check(sis_tensor_init(&s_, ndim, n.data(), degree), "Tensor::init");
        std::vector<const double*> xk(ndim);
        for (int d = 0; d < ndim; ++d) xk[d] = coords[d].data();
        int rc = sis_tensor_build(&s_, xk.data(), y.data(), degree);
        if (rc != SIS_OK) {
            sis_tensor_free(&s_);
            detail::check(rc, "Tensor::build");
        }
    }

    Tensor(const Tensor&) = delete;
    Tensor& operator=(const Tensor&) = delete;

    Tensor(Tensor&& other) noexcept : s_(other.s_) { other.s_ = {}; }
    Tensor& operator=(Tensor&& other) noexcept {
        if (this != &other) {
            sis_tensor_free(&s_);
            s_ = other.s_;
            other.s_ = {};
        }
        return *this;
    }

    ~Tensor() { sis_tensor_free(&s_); }

    /** Evaluate.  \p deriv_dir = -1 is the function; 0,1,2 is d/dx_i. */
    double eval(const std::vector<double>& xq, int deriv_dir = -1) const {
        return sis_tensor_eval(&s_, xq.data(), deriv_dir);
    }

    template <std::size_t N>
    double eval(const std::array<double, N>& xq, int deriv_dir = -1) const {
        static_assert(N >= 1 && N <= 3, "N must be 1..3");
        return sis_tensor_eval(&s_, xq.data(), deriv_dir);
    }

    int ndim() const noexcept { return s_.ndim; }
    int degree() const noexcept { return s_.degree; }

private:
    sis_tensor_t s_;
};

/** Thin-plate spline / polyharmonic RBF for scattered data (2-D or 3-D). */
class ThinPlateSpline {
public:
    /** Fit to scattered centres.
     *  \p xi is row-major, shape [n][d].  \p yi has length n.  d = 2 or 3. */
    ThinPlateSpline(const std::vector<double>& xi,
                    const std::vector<double>& yi,
                    int d,
                    double lambda = 1e-6)
    {
        if (d != 2 && d != 3)
            throw Exception(SIS_ERR_ARG, "ThinPlateSpline: d must be 2 or 3");
        int n = static_cast<int>(yi.size());
        if (xi.size() != static_cast<std::size_t>(n * d))
            throw Exception(SIS_ERR_ARG, "ThinPlateSpline: xi/yi size mismatch");
        detail::check(sis_tps_fit(&s_, xi.data(), yi.data(), n, d, lambda),
                      "ThinPlateSpline::fit");
    }

    ThinPlateSpline(const ThinPlateSpline&) = delete;
    ThinPlateSpline& operator=(const ThinPlateSpline&) = delete;

    ThinPlateSpline(ThinPlateSpline&& other) noexcept : s_(other.s_) { other.s_ = {}; }
    ThinPlateSpline& operator=(ThinPlateSpline&& other) noexcept {
        if (this != &other) {
            sis_tps_free(&s_);
            s_ = other.s_;
            other.s_ = {};
        }
        return *this;
    }

    ~ThinPlateSpline() { sis_tps_free(&s_); }

    double eval(const std::vector<double>& xq) const {
        return sis_tps_eval(&s_, xq.data());
    }

    template <std::size_t N>
    double eval(const std::array<double, N>& xq) const {
        static_assert(N == 2 || N == 3, "N must be 2 or 3");
        return sis_tps_eval(&s_, xq.data());
    }

    int n_centres() const noexcept { return s_.n; }
    int dim() const noexcept { return s_.d; }

private:
    sis_tps_t s_;
};

/** Floater–Hormann barycentric rational interpolator (pole-free). */
class Barycentric {
public:
    /** Build from knots, values and blending degree. */
    Barycentric(const std::vector<double>& x,
                const std::vector<double>& y,
                int degree)
    {
        if (x.size() != y.size()) throw Exception(SIS_ERR_ARG, "x/y size mismatch");
        detail::check(sis_barycentric_build(&s_, x.data(), y.data(),
                                            static_cast<int>(x.size()), degree),
                      "Barycentric::build");
    }

    /** Build from denoised data. */
    static Barycentric fit_denoised(const std::vector<double>& x,
                                    const std::vector<double>& y,
                                    int degree,
                                    double cutoff_cycles)
    {
        if (x.size() != y.size()) throw Exception(SIS_ERR_ARG, "x/y size mismatch");
        Barycentric bc;
        detail::check(sis_barycentric_fit_denoised(&bc.s_, x.data(), y.data(),
                                                   static_cast<int>(x.size()),
                                                   degree, cutoff_cycles),
                      "Barycentric::fit_denoised");
        return bc;
    }

    Barycentric(const Barycentric&) = delete;
    Barycentric& operator=(const Barycentric&) = delete;

    Barycentric(Barycentric&& other) noexcept : s_(other.s_) { other.s_ = {}; }
    Barycentric& operator=(Barycentric&& other) noexcept {
        if (this != &other) {
            sis_barycentric_free(&s_);
            s_ = other.s_;
            other.s_ = {};
        }
        return *this;
    }

    ~Barycentric() { sis_barycentric_free(&s_); }

    double eval(double xq) const {
        return sis_barycentric_eval(&s_, xq);
    }

private:
    Barycentric() = default;
    sis_barycentric_t s_;
};

} // namespace sisyphus

#endif /* SISYPHUS_HPP */
