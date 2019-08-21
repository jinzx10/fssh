#ifndef __TWO_LEVEL_SYSTEM_H__
#define __TWO_LEVEL_SYSTEM_H__

#include "aux.h"
#include <type_traits>
#include <complex>
#include <armadillo>

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

		// constructor
		TLS(
				param2d			V00_,
				param2d			V11_,
				param2e			V01_,
				param2b			is_within_ = always
		);

		// Hamiltonian
		mat_t				H(param_t const&);
		elem_t				H(param_t const&, bool const&, bool const&);
		mat_t				dH(param_t const&, size_t const&);
		cube_t				dH(param_t const&);

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

		double				dV00(param_t const&, size_t const&);
		param_t				dV00(param_t const&);
		double				dV11(param_t const&, size_t const&);
		param_t				dV11(param_t const&);
		elem_t				dV01(param_t const&, size_t const&);
		elems_t				dV01(param_t const&);
		elem_t				dV10(param_t const&, size_t const&);
		elems_t				dV10(param_t const&);

		static bool			always(param_t const&) { return true; }
};


template <size_t ndim, bool is_cplx> TLS<ndim, is_cplx>::TLS( param2d V00_, param2d V11_, param2e V01_, param2b is_within_ ):
	V00(V00_), V11(V11_), V01(V01_), is_within(is_within_) {
	V10 = [this](param_t const& p) -> elem_t { return keep_cplx<is_cplx>( std::conj(V01(p)) ); };
	d0 = [this](param_t const& p) -> double { return 0.5 * ( V00(p) + V11(p) ); };
	dz = [this](param_t const& p) -> double { return 0.5 * ( V00(p) - V11(p) ); };
	dx = [this](param_t const& p) -> double { return std::real(V01(p)); };
	dy = [this](param_t const& p) -> double { return -std::imag(V01(p)); };
	r = [this](param_t const& p) -> double { return std::sqrt( std::pow(dx(p),2) + std::pow(dy(p),2) + std::pow(dz(p),2) ); };
	theta = [this](param_t const& p) -> double { return std::acos( dz(p) / r(p) ); };
	phi = [this](param_t const& p) -> double { return (dy(p) > 0 ? 1 : -1) * std::acos( dx(p) / std::sqrt(std::pow(dx(p),2) + std::pow(dy(p),2)) ); };
}


template <size_t ndim, bool is_cplx> typename TLS<ndim, is_cplx>::mat_t TLS<ndim, is_cplx>::H(TLS::param_t const& p) {
	return mat_t{ {V00(p), V01(p)}, {V10(p), V11(p)} };
}


template <size_t ndim, bool is_cplx> typename TLS<ndim, is_cplx>::elem_t TLS<ndim, is_cplx>::H(TLS::param_t const& p, bool const& i, bool const& j) {
	return i ? (j ? V11(p) : V10(p)) : (j ? V01(p) : V00(p));
}


template <size_t ndim, bool is_cplx> double TLS<ndim, is_cplx>::eigval(TLS::param_t const& p, bool const& state) {
	return d0(p) + (state ? r(p) : -r(p));
}


template <size_t ndim, bool is_cplx> arma::vec2 TLS<ndim, is_cplx>::eigval(TLS::param_t const& p) {
	return d0(p) + r(p) * arma::vec2{-1, 1};
}


template <size_t ndim, bool is_cplx> typename TLS<ndim, is_cplx>::vec_t TLS<ndim, is_cplx>::eigvec(TLS::param_t const& p, bool const& state) {
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


template <size_t ndim, bool is_cplx> typename TLS<ndim, is_cplx>::mat_t TLS<ndim, is_cplx>::eigvec(TLS::param_t const& p) {
	return arma::join_rows(eigvec(p,0), eigvec(p,1));
}


template <size_t ndim, bool is_cplx> double TLS<ndim, is_cplx>::dV00(TLS::param_t const& p, size_t const& d) {
	return op<ndim, false>::pardiff1(V00)(p,d);
}


template <size_t ndim, bool is_cplx> double TLS<ndim, is_cplx>::dV11(TLS::param_t const& p, size_t const& d) {
	return op<ndim, false>::pardiff1(V11)(p,d);
}


template <size_t ndim, bool is_cplx> typename TLS<ndim, is_cplx>::elem_t TLS<ndim, is_cplx>::dV01(TLS::param_t const& p, size_t const& d) {
	return op<ndim, is_cplx>::pardiff1(V01)(p,d);
}


template <size_t ndim, bool is_cplx> typename TLS<ndim, is_cplx>::elem_t TLS<ndim, is_cplx>::dV10(TLS::param_t const& p, size_t const& d) {
	return op<ndim, is_cplx>::pardiff1(V10)(p,d);
}


template <size_t ndim, bool is_cplx> typename TLS<ndim,is_cplx>::param_t TLS<ndim, is_cplx>::dV00(TLS::param_t const& p) {
	return op<ndim, false>::diff1(V00)(p);
}


template <size_t ndim, bool is_cplx> typename TLS<ndim,is_cplx>::param_t TLS<ndim, is_cplx>::dV11(TLS::param_t const& p) {
	return op<ndim, false>::diff1(V11)(p);
}


template <size_t ndim, bool is_cplx> typename TLS<ndim,is_cplx>::elems_t TLS<ndim, is_cplx>::dV01(TLS::param_t const& p) {
	return op<ndim, is_cplx>::diff1(V01)(p);
}


template <size_t ndim, bool is_cplx> typename TLS<ndim,is_cplx>::elems_t TLS<ndim, is_cplx>::dV10(TLS::param_t const& p) {
	return op<ndim, is_cplx>::diff1(V10)(p);
}


template <size_t ndim, bool is_cplx> typename TLS<ndim, is_cplx>::mat_t TLS<ndim, is_cplx>::dH(TLS::param_t const& p, size_t const& d) {
	return mat_t{ {dV00(p,d), dV01(p,d)}, {dV10(p,d), dV11(p,d)} };
}


template <size_t ndim, bool is_cplx> typename TLS<ndim, is_cplx>::cube_t TLS<ndim, is_cplx>::dH(TLS::param_t const& p) {
	cube_t c2;
	for (size_t d = 0; d != ndim; ++d)
		c2.slice(d) = dV(p,d);
	return c2;
}


template <size_t ndim, bool is_cplx> double TLS<ndim, is_cplx>::F(TLS::param_t const& p, bool const& state, size_t const& d) {
	return std::real(eigvec(p, state).t() * dH(p,d) * eigvec(p, state));
}


template <size_t ndim, bool is_cplx> typename TLS<ndim, is_cplx>::param_t TLS<ndim, is_cplx>::F(TLS::param_t const& p, bool const& state) {
	param_t force;
	for (size_t d = 0; d != ndim; ++d)
		force(d) = F(p, state, d);
	return force;
}


template <size_t ndim, bool is_cplx> typename TLS<ndim, is_cplx>::elem_t TLS<ndim, is_cplx>::drvcpl(TLS::param_t const& p, bool const& i, bool const& j, size_t const& d) {
	return keep_cplx<is_cplx>( (i==j) ? 
			( (i ? -1 : 1) * 0.5 * I * std::cos(theta(p)) * op<ndim,false>::pardiff1(phi)(p,d) ) :
			squeeze<elem_t,1,1>(eigvec(p,i).t() * dH(p,d) * eigvec(p,j) / ( eigval(p,j) - eigval(p,i) )) );
}


template <size_t ndim, bool is_cplx> typename TLS<ndim, is_cplx>::elems_t TLS<ndim, is_cplx>::drvcpl(TLS::param_t const& p, bool const& i, bool const& j) {
	elems_t dc;
	for (size_t d = 0; d != ndim; ++d)
		dc(d) = drvcpl(p, i, j, d);
	return dc;
}


#endif
