#ifndef __AUX_H__
#define __AUX_H__

#include <armadillo>
#include <cmath>
#include <complex>
#include <functional>


extern std::complex<double> const I;
extern double const PI;
extern double const DELTA;

//void set_max_real_positive(arma::cx_vec& col);
//arma::cx_mat pure_denmat(arma::uword sz);

// this is a special case: function does not explicitly depend on t
// otherwise the general Runge-Kutta method should take the form
//T rk4_step(double t, T yn, double dt, std::function<T(double,T)> f);
template <typename T>
T rk4_step(T const& yn, double const& dt, std::function<T(T)> const& f) {
	T k1 = dt * f( yn );
	T k2 = dt * f( yn + 0.5*k1 );
	T k3 = dt * f( yn + 0.5*k2 );
	T k4 = dt * f( yn + k3 );
	return yn + ( k1 + 2.0*k2 + 2.0*k3 + k4 ) / 6.0;
}


template <bool is_cplx> struct KeepCplx {
	static std::complex<double> value(std::complex<double> const& z) {return z;}
};

template <> struct KeepCplx<false> {
	static double value(std::complex<double> const& z) {return z.real();}
};

std::function<double(double const&)> diff(std::function<double(double const&)> const& f);
std::function<std::complex<double>(double const&)> diff(std::function<std::complex<double>(double const&)> const& f);


#endif
