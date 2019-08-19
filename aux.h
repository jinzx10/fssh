#ifndef __AUX_H__
#define __AUX_H__

#include <armadillo>
#include <cmath>
#include <complex>
#include <functional>

extern std::complex<double> const I;
extern double const PI;
extern double const DELTA;

template <typename ...> using void_t = void;
template <bool is_cplx = false> using num_t = typename std::conditional<is_cplx, std::complex<double>, double>::type;

template <bool is_cplx> typename std::enable_if<is_cplx, std::complex<double>>::type keep_cplx(std::complex<double> const& z) { return z; }
template <bool is_cplx> typename std::enable_if<!is_cplx, double>::type keep_cplx(std::complex<double> const& z) { return z.real(); }

template <typename T, size_t ndim> typename std::enable_if<(ndim==1), T>::type decay(typename arma::Col<T>::template fixed<1> const& zv) { return zv(0); }
template <typename T, size_t ndim> typename std::enable_if<(ndim!=1), typename arma::Col<T>::template fixed<ndim>>::type decay(typename arma::Col<T>::template fixed<ndim> const& zv) { return zv; }
template <typename T, size_t ndim1, size_t ndim2> typename std::enable_if<ndim1==1 && ndim2==1, T>::type decay(typename arma::Mat<T>::template fixed<1,1> const& zv) { return zv(0,0); }
template <typename T, size_t ndim1, size_t ndim2> typename std::enable_if<ndim1!=1 || ndim2!=1, typename arma::Mat<T>::template fixed<ndim1,ndim2>>::type decay(typename arma::Mat<T>::template fixed<ndim1,ndim2> const& zv) { return zv; }


template <size_t ndim> arma::Col<double>::fixed<ndim> pt(arma::Col<double>::fixed<ndim> x, size_t const& d, double const& dx) { x[d] += dx; return x; }
template <size_t> double pt(double x, size_t const& d, double const& dx) { return x+dx; }

template < size_t ndim = 1, bool is_cplx = false >
struct Diff {
	using Val			= typename std::conditional< is_cplx, std::complex<double>, double >::type;
	using Params		= typename std::conditional< (ndim==1), double, arma::Col<double>::fixed<ndim> >::type;
	using Vals			= typename std::conditional< (ndim==1), Val, typename arma::Col<Val>::template fixed<ndim> >::type;

	using Input			= std::function<Val(Params)>;
	using Output1		= std::function<Val(Params, size_t)>;
	using Output2		= std::function<Vals(Params)>;

	static Output1		pdiff(Input const& f) {
		return [f] (Params const& x, size_t const& d) -> Val {
			return (-f(pt<ndim>(x,d,-3.0*DELTA))/60.0 + 3.0*f(pt<ndim>(x,d,-2.0*DELTA))/20.0 - 3.0*f(pt<ndim>(x,d,-DELTA))/4.0
					+f(pt<ndim>(x,d,+3.0*DELTA))/60.0 - 3.0*f(pt<ndim>(x,d,+2.0*DELTA))/20.0 + 3.0*f(pt<ndim>(x,d,+DELTA))/4.0) / DELTA;
		};
	}

	static Output2		diff(Input const& f) {
		return [f] (Params const& p) -> Vals {
			auto df = pdiff(f);
			typename arma::Col<Val>::template fixed<ndim> vals;
			for (size_t i = 0; i != ndim; ++i)
				vals(i) = df(p,i);
			return decay<Val, ndim>(vals);
		};
	}
};

//void set_max_real_positive(arma::cx_vec& col);
//arma::cx_mat pure_denmat(arma::uword sz);

// this is a special case: function does not explicitly depend on t
// otherwise the general Runge-Kutta method should take the form
//T rk4_step(double t, T yn, double dt, std::function<T(double,T)> f);
//template <typename T>
//T rk4_step(T const& yn, double const& dt, std::function<T(T)> const& f) {
//	T k1 = dt * f( yn );
//	T k2 = dt * f( yn + 0.5*k1 );
//	T k3 = dt * f( yn + 0.5*k2 );
//	T k4 = dt * f( yn + k3 );
//	return yn + ( k1 + 2.0*k2 + 2.0*k3 + k4 ) / 6.0;
//}


#endif
