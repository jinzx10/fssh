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


template <bool is_cplx> num_t<is_cplx> keep_cplx(std::complex<double> const& z) { return z; }
template <>  num_t<false> keep_cplx<false>(std::complex<double> const& z) { return z.real(); }


template <typename T, size_t sz> typename std::enable_if<(sz==1), T>::type squeeze(typename arma::Col<T>::template fixed<1> const& zv) { return zv(0); }
template <typename T, size_t sz> typename std::enable_if<(sz!=1), typename arma::Col<T>::template fixed<sz> >::type squeeze(typename arma::Col<T>::template fixed<sz> const& zv) { return zv; }


template <size_t sz> arma::Col<double>::fixed<sz> pt(arma::Col<double>::fixed<sz> x, size_t const& i, double const& dx) { x(i) += dx; return x; }
template <size_t> double pt(double x, size_t const& i, double const& dx) { return x+dx; }


template < size_t sz_vec = 2, size_t sz_param = 1, bool is_cplx = false >
struct data_type
{
	using elem_t		= num_t<is_cplx>;
	using param_t		= typename std::conditional< (sz_param==1), double, arma::Col<double>::fixed<sz_param> >::type;
	using vec_t			= typename arma::Col<elem_t>::template fixed<sz_vec>;
	using mat_t			= typename arma::Mat<elem_t>::template fixed<sz_vec, sz_vec>;
	using cube_t		= typename arma::Cube<elem_t>::template fixed<sz_vec, sz_vec, sz_vec>;
	using elems_t		= typename std::conditional< (sz_param==1), elem_t, typename arma::Col<elem_t>::template fixed<sz_param> >::type;

	using param2b		= std::function<bool(param_t)>;
	using param2d		= std::function<double(param_t)>;
	using param2e		= std::function<elem_t(param_t)>;
	using param2p		= std::function<param_t(param_t)>;
	using param2es		= std::function<elems_t(param_t)>;
	using paramidx2d	= std::function<double(param_t, size_t)>;
	using paramidx2e	= std::function<elem_t(param_t, size_t)>;
};


template < size_t sz_param = 1, bool is_cplx = false >
struct op : public data_type<0, sz_param, is_cplx>
{
	using typename data_type<0, sz_param, is_cplx>::elem_t;
	using typename data_type<0, sz_param, is_cplx>::param_t;
	using typename data_type<0, sz_param, is_cplx>::elems_t;
	using typename data_type<0, sz_param, is_cplx>::param2e;
	using typename data_type<0, sz_param, is_cplx>::paramidx2e;
	using typename data_type<0, sz_param, is_cplx>::param2es;

	static paramidx2e		pardiff1(param2e const& f) {
		return [f] (param_t const& x, size_t const& i) -> elem_t {
			return (-f(pt<sz_param>(x,i,-3.0*DELTA))/60.0 + 3.0*f(pt<sz_param>(x,i,-2.0*DELTA))/20.0 - 3.0*f(pt<sz_param>(x,i,-DELTA))/4.0
					+f(pt<sz_param>(x,i,+3.0*DELTA))/60.0 - 3.0*f(pt<sz_param>(x,i,+2.0*DELTA))/20.0 + 3.0*f(pt<sz_param>(x,i,+DELTA))/4.0) / DELTA;
		};
	}

	static param2es			diff1(param2e const& f) {
		return [f] (param_t const& x) -> elems_t {
			typename arma::Col<elem_t>::template fixed<sz_param> vals;
			for (size_t i = 0; i != sz_param; ++i)
				vals(i) = pardiff1(f)(x,i);
			return squeeze<elem_t, sz_param>(vals);
		};
	}
};



//template <typename T, size_t ndim1, size_t ndim2> typename std::enable_if<ndim1==1 && ndim2==1, T>::type decay(typename arma::Mat<T>::template fixed<1,1> const& zv) { return zv(0,0); }
//template <typename T, size_t ndim1, size_t ndim2> typename std::enable_if<ndim1!=1 || ndim2!=1, typename arma::Mat<T>::template fixed<ndim1,ndim2>>::type decay(typename arma::Mat<T>::template fixed<ndim1,ndim2> const& zv) { return zv; }


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
