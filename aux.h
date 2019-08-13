#ifndef __AUX_H__
#define __AUX_H__

#include <armadillo>
#include <cmath>
#include <complex>
#include <functional>


extern const std::complex<double> I;
extern const double PI;
extern const double DELTA;


//void set_max_real_positive(arma::cx_vec& col);
//arma::cx_mat pure_denmat(arma::uword sz);


// this is a special case: function does not explicitly depend on t
// otherwise the general Runge-Kutta method should take the form
//T rk4_step(double t, T yn, double dt, std::function<T(double,T)> f);
template <typename T>
T rk4_step(const T& yn, const double& dt, const std::function<T(T)>& f) {
	T k1 = dt * f( yn );
	T k2 = dt * f( yn + 0.5*k1 );
	T k3 = dt * f( yn + 0.5*k2 );
	T k4 = dt * f( yn + k3 );
	return yn + ( k1 + 2.0*k2 + 2.0*k3 + k4 ) / 6.0;
}


template <bool is_cplx> struct KeepCplx {
	static std::complex<double> value(const std::complex<double>& z) {return z;}
};

template <> struct KeepCplx<false> {
	static double value(const std::complex<double>& z) {return z.real();}
};


template <size_t ndim, bool is_cplx>
class Diff 
{
	public:
		using Arg			= typename std::conditional< (ndim==1), double, arma::Col<double>::fixed<ndim> >::type;
		using T_ret			= typename std::conditional< is_cplx, std::complex<double>, double >::type;

};

#endif
