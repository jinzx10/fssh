#ifndef __TWO_LEVEL_SYSTEM_H__
#define __TWO_LEVEL_SYSTEM_H__

#include "aux.h"

template < size_t sz_param = 1, bool is_cplx = false >
class TLS : public data_type<2, sz_param, is_cplx>
{
	public:
		using typename data_type<2, sz_param, is_cplx>::elem_t;
		using typename data_type<2, sz_param, is_cplx>::elems_t;
		using typename data_type<2, sz_param, is_cplx>::param_t;
		using typename data_type<2, sz_param, is_cplx>::vec_t;
		using typename data_type<2, sz_param, is_cplx>::mat_t;
		using typename data_type<2, sz_param, is_cplx>::cube_t;
		using typename data_type<2, sz_param, is_cplx>::param2d;
		using typename data_type<2, sz_param, is_cplx>::param2e;
		using typename data_type<2, sz_param, is_cplx>::param2b;
		using typename data_type<2, sz_param, is_cplx>::paramidx2d;
		using typename data_type<2, sz_param, is_cplx>::param2p;

		TLS(
				param2d			V00_,
				param2d			V11_,
				param2e			V01_,
				param2b			is_within_ = always
		);

		mat_t				H(param_t const&);
		elem_t				H(param_t const&, bool const&, bool const&);
		double				eigval(param_t const&, bool const&);
		arma::vec2			eigval(param_t const&);
		vec_t				eigvec(param_t const&, bool const&); // analytical expression, symmetric phase
		mat_t				eigvec(param_t const&);

		double				F(param_t const&, bool const&, size_t const&); // force from adiabatic PES
		param_t				F(param_t const&, bool const&);
		elem_t				drvcpl(param_t const&, bool const&, bool const&, size_t const&);
		elems_t				drvcpl(param_t const&, bool const&, bool const&);

		param2b				is_within;

	private:
		param2d				V00;
		param2d 			V11;
		param2e 			V01;
		param2e 			V10;

		// V = d0*I + \sum_i d_i * sigma_i
		param2d				d0;
		param2d				dx;
		param2d				dy;
		param2d				dz;

		// spherical coordinates of (dx,dy,dz) in Bloch sphere
		param2d				r;
		param2d				theta;
		param2d				phi;

		paramidx2d			pardiff_d0;
		paramidx2d			pardiff_r;
		paramidx2d			pardiff_theta;
		paramidx2d			pardiff_phi;

		param2p				diff_d0;
		param2p				diff_r;
		param2p				diff_theta;
		param2p				diff_phi;

		static bool			always(param_t const&) { return true; }
};


template <size_t sz_param, bool is_cplx> TLS<sz_param, is_cplx>::TLS( param2d V00_, param2d V11_, param2e V01_, param2b is_within_ ):
	V00(V00_), V11(V11_), V01(V01_), is_within(is_within_) {
	V10 = [this](param_t const& p) -> elem_t { return keep_cplx<is_cplx>( std::conj(V01(p)) ); };
	d0 = [this](param_t const& p) -> double { return 0.5 * ( V00(p) + V11(p) ); };
	dz = [this](param_t const& p) -> double { return 0.5 * ( V00(p) - V11(p) ); };
	dx = [this](param_t const& p) -> double { return std::real(V01(p)); };
	dy = [this](param_t const& p) -> double { return -std::imag(V01(p)); };
	r = [this](param_t const& p) -> double { return std::sqrt( std::pow(dx(p),2) + std::pow(dy(p),2) + std::pow(dz(p),2) ); };
	theta = [this](param_t const& p) -> double { return std::acos( dz(p) / r(p) ); };
	phi = [this](param_t const& p) -> double { double phi0 = std::acos( dx(p) / std::sqrt(std::pow(dx(p),2) + std::pow(dy(p),2)) ); return dy(p) > 0 ? phi0 : (2*PI - phi0); };

	pardiff_d0 = op<sz_param, false>::pardiff1(d0);
	pardiff_r = op<sz_param, false>::pardiff1(r);
	pardiff_theta = op<sz_param, false>::pardiff1(theta);
	pardiff_phi = op<sz_param, false>::pardiff1(phi);
	diff_d0 = op<sz_param, false>::diff1(d0);
	diff_r = op<sz_param, false>::diff1(r);
	diff_theta = op<sz_param, false>::diff1(theta);
	diff_phi = op<sz_param, false>::diff1(phi);
}


template <size_t sz_param, bool is_cplx> typename TLS<sz_param, is_cplx>::mat_t TLS<sz_param, is_cplx>::H(param_t const& p) {
	return mat_t{ {V00(p), V01(p)}, {V10(p), V11(p)} };
}


template <size_t sz_param, bool is_cplx> typename TLS<sz_param, is_cplx>::elem_t TLS<sz_param, is_cplx>::H(param_t const& p, bool const& i, bool const& j) {
	return i ? (j ? V11(p) : V10(p)) : (j ? V01(p) : V00(p));
}


template <size_t sz_param, bool is_cplx> double TLS<sz_param, is_cplx>::eigval(param_t const& p, bool const& state) {
	return d0(p) + pm(state) * r(p);
}


template <size_t sz_param, bool is_cplx> arma::vec2 TLS<sz_param, is_cplx>::eigval(param_t const& p) {
	return d0(p) + r(p) * arma::vec2{-1, 1};
}


template <size_t sz_param, bool is_cplx> typename TLS<sz_param, is_cplx>::vec_t TLS<sz_param, is_cplx>::eigvec(param_t const& p, bool const& state) {
	return state ?
		vec_t{
			std::cos(theta(p)/2.0) * keep_cplx<is_cplx>( std::exp(-I * phi(p) / 2.0) ),
			std::sin(theta(p)/2.0) * keep_cplx<is_cplx>( std::exp(I * phi(p) / 2.0) )
		} :
		vec_t{
			-std::sin(theta(p)/2.0) * keep_cplx<is_cplx>( std::exp(-I * phi(p) / 2.0) ),
			std::cos(theta(p)/2.0) * keep_cplx<is_cplx>( std::exp(I * phi(p) / 2.0) )
		};
}


template <size_t sz_param, bool is_cplx> typename TLS<sz_param, is_cplx>::mat_t TLS<sz_param, is_cplx>::eigvec(param_t const& p) {
	return arma::join_rows(eigvec(p,0), eigvec(p,1));
}


template <size_t sz_param, bool is_cplx> double TLS<sz_param, is_cplx>::F(param_t const& p, bool const& state, size_t const& ip) {
	return - ( pardiff_d0(p, ip) + pm(state) * pardiff_r(p, ip) );
}


template <size_t sz_param, bool is_cplx> typename TLS<sz_param, is_cplx>::param_t TLS<sz_param, is_cplx>::F(param_t const& p, bool const& state) {
	return - ( diff_d0(p) + pm(state) * diff_r(p) );
}


template <size_t sz_param, bool is_cplx> typename TLS<sz_param, is_cplx>::elem_t TLS<sz_param, is_cplx>::drvcpl(param_t const& p, bool const& i, bool const& j, size_t const& ip) {
	return (i==j) ? 
		( is_cplx ? -0.5 * pm(i)  * keep_cplx<is_cplx>(I) * std::cos(theta(p)) * pardiff_phi(p,ip) : 0.0 ) :
		( is_cplx ? -0.5 * ( pm(i) * pardiff_theta(p,ip) + keep_cplx<is_cplx>(I) * std::sin(theta(p)) * pardiff_phi(p,ip) ) : 
					-0.5 * pm(i) * pardiff_theta(p,ip) );
}


template <size_t sz_param, bool is_cplx> typename TLS<sz_param, is_cplx>::elems_t TLS<sz_param, is_cplx>::drvcpl(param_t const& p, bool const& i, bool const& j) {
	return (i==j) ?
		( is_cplx ? -0.5 * pm(i) * keep_cplx<is_cplx>(I) * std::cos(theta(p)) * diff_phi(p) : zeros<elems_t>() ) :
		( is_cplx ? -0.5 * ( pm(i) * diff_theta(p) + keep_cplx<is_cplx>(I) * std::sin(theta(p)) * diff_phi(p) ) :
					-0.5 * pm(i) * diff_theta(p) );
}


#endif
