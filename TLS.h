#ifndef __TWO_LEVEL_SYSTEM_H__
#define __TWO_LEVEL_SYSTEM_H__

#include "aux.h"
#include <type_traits>
#include <complex>
#include <armadillo>

template < size_t ndim = 1, bool is_cplx = false >
class TLS
{
	public:
		using Val			= typename std::conditional< is_cplx, std::complex<double>, double >::type;
		using Params		= typename std::conditional< (ndim==1), double, arma::Col<double>::fixed<ndim> >::type;
		using Vals			= typename std::conditional< (ndim==1), Val, typename arma::Col<Val>::template fixed<ndim> >::type;

		using Param2d		= std::function<double(Params)>;
		using Param2z		= std::function<Val(Params)>;
		using ParamChk		= std::function<bool(Params)>;
		
		using Vec2			= typename arma::Col<Val>::template fixed<2>;
		using Mat2			= typename arma::Mat<Val>::template fixed<2,2>;
		using Cube2			= typename arma::Cube<Val>::template fixed<2,2,ndim>;

		// constructor
		TLS(
				Param2d			V00_,
				Param2d			V11_,
				Param2z			V01_,
				ParamChk		is_within_ = always
		);

		// Hamiltonian
		Mat2				H(Params const&);
		Val					H(Params const&, bool const&, bool const&);
		Mat2				dH(Params const&, size_t const&);
		Cube2				dH(Params const&);

		double				eigval(Params const&, bool const&);
		arma::vec2			eigval(Params const&);
		Vec2				eigvec(Params const&, bool const&); // analytical expression, symmetric phase
		Mat2				eigvec(Params const&);

		double				F(Params const&, bool const&, size_t const&); // force from adiabatic PES
		Params				F(Params const&, bool const&);
		Val					drvcpl(Params const&, bool const&, bool const&, size_t const&);
		Vals				drvcpl(Params const&, bool const&, bool const&);

		ParamChk			is_within;


	private:

		Param2d				V00;
		Param2d 			V11;
		Param2z 			V01;
		Param2z 			V10;

		// V = d0*I + \sum_i d_i * sigma_i
		Param2d				d0;
		Param2d				dx;
		Param2d				dy;
		Param2d				dz;

		// spherical coordinates of (dx,dy,dz) in Bloch sphere
		Param2d				r;
		Param2d				theta;
		Param2d				phi;

		double				dV00(Params const&, size_t const&);
		Params				dV00(Params const&);
		double				dV11(Params const&, size_t const&);
		Params				dV11(Params const&);
		Val					dV01(Params const&, size_t const&);
		Vals				dV01(Params const&);
		Val					dV10(Params const&, size_t const&);
		Vals				dV10(Params const&);

		static bool			always(Params const&) { return true; }
};


template <size_t ndim, bool is_cplx> TLS<ndim, is_cplx>::TLS( Param2d V00_, Param2d V11_, Param2z V01_, ParamChk is_within_ ):
	V00(V00_), V11(V11_), V01(V01_), is_within(is_within_) {
	V10 = [this](Params const& p) -> Val { return keep_cplx<is_cplx>( std::conj(V01(p)) ); };
	d0 = [this](Params const& p) -> double { return 0.5 * ( V00(p) + V11(p) ); };
	dz = [this](Params const& p) -> double { return 0.5 * ( V00(p) - V11(p) ); };
	dx = [this](Params const& p) -> double { return std::real(V01(p)); };
	dy = [this](Params const& p) -> double { return -std::imag(V01(p)); };
	r = [this](Params const& p) -> double { return std::sqrt( std::pow(dx(p),2) + std::pow(dy(p),2) + std::pow(dz(p),2) ); };
	theta = [this](Params const& p) -> double { return std::acos( dz(p) / r(p) ); };
	phi = [this](Params const& p) -> double { return (dy(p) > 0 ? 1 : -1) * std::acos( dx(p) / std::sqrt(std::pow(dx(p),2) + std::pow(dy(p),2)) ); };
}


template <size_t ndim, bool is_cplx> typename TLS<ndim, is_cplx>::Mat2 TLS<ndim, is_cplx>::H(TLS::Params const& p) {
	return Mat2{ {V00(p), V01(p)}, {V10(p), V11(p)} };
}


template <size_t ndim, bool is_cplx> typename TLS<ndim, is_cplx>::Val TLS<ndim, is_cplx>::H(TLS::Params const& p, bool const& i, bool const& j) {
	return i ? (j ? V11(p) : V10(p)) : (j ? V01(p) : V00(p));
}


template <size_t ndim, bool is_cplx> double TLS<ndim, is_cplx>::eigval(TLS::Params const& p, bool const& state) {
	return d0(p) + (state ? r(p) : -r(p));
}


template <size_t ndim, bool is_cplx> arma::vec2 TLS<ndim, is_cplx>::eigval(TLS::Params const& p) {
	return d0(p) + r(p) * arma::vec2{-1, 1};
}


template <size_t ndim, bool is_cplx> typename TLS<ndim, is_cplx>::Vec2 TLS<ndim, is_cplx>::eigvec(TLS::Params const& p, bool const& state) {
	return state ?
		Vec2{
			std::cos(theta(p)/2.0) * keep_cplx<is_cplx>( std::exp(-I * phi(p) / 2.0) ),
			std::sin(theta(p)/2.0) * keep_cplx<is_cplx>( std::exp(I * phi(p) / 2.0) )
		} :
		Vec2{
			-std::sin(theta(p)/2.0) * keep_cplx<is_cplx>( std::exp(-I * phi(p) / 2.0) ),
			std::cos(theta(p)/2.0) * keep_cplx<is_cplx>( std::exp(I * phi(p) / 2.0) )
		};
}


template <size_t ndim, bool is_cplx> typename TLS<ndim, is_cplx>::Mat2 TLS<ndim, is_cplx>::eigvec(TLS::Params const& p) {
	return arma::join_rows(eigvec(p,0), eigvec(p,1));
}


template <size_t ndim, bool is_cplx> double TLS<ndim, is_cplx>::dV00(TLS::Params const& p, size_t const& d) {
	return Diff<ndim, false>::pdiff(V00)(p,d);
}


template <size_t ndim, bool is_cplx> double TLS<ndim, is_cplx>::dV11(TLS::Params const& p, size_t const& d) {
	return Diff<ndim, false>::pdiff(V11)(p,d);
}


template <size_t ndim, bool is_cplx> typename TLS<ndim, is_cplx>::Val TLS<ndim, is_cplx>::dV01(TLS::Params const& p, size_t const& d) {
	return Diff<ndim, is_cplx>::pdiff(V01)(p,d);
}


template <size_t ndim, bool is_cplx> typename TLS<ndim, is_cplx>::Val TLS<ndim, is_cplx>::dV10(TLS::Params const& p, size_t const& d) {
	return Diff<ndim, is_cplx>::pdiff(V10)(p,d);
}


template <size_t ndim, bool is_cplx> typename TLS<ndim,is_cplx>::Params TLS<ndim, is_cplx>::dV00(TLS::Params const& p) {
	return Diff<ndim, false>::diff(V00)(p);
}


template <size_t ndim, bool is_cplx> typename TLS<ndim,is_cplx>::Params TLS<ndim, is_cplx>::dV11(TLS::Params const& p) {
	return Diff<ndim, false>::diff(V11)(p);
}


template <size_t ndim, bool is_cplx> typename TLS<ndim,is_cplx>::Vals TLS<ndim, is_cplx>::dV01(TLS::Params const& p) {
	return Diff<ndim, is_cplx>::diff(V01)(p);
}


template <size_t ndim, bool is_cplx> typename TLS<ndim,is_cplx>::Vals TLS<ndim, is_cplx>::dV10(TLS::Params const& p) {
	return Diff<ndim, is_cplx>::diff(V10)(p);
}


template <size_t ndim, bool is_cplx> typename TLS<ndim, is_cplx>::Mat2 TLS<ndim, is_cplx>::dH(TLS::Params const& p, size_t const& d) {
	return Mat2{ {dV00(p,d), dV01(p,d)}, {dV10(p,d), dV11(p,d)} };
}


template <size_t ndim, bool is_cplx> typename TLS<ndim, is_cplx>::Cube2 TLS<ndim, is_cplx>::dH(TLS::Params const& p) {
	Cube2 c2;
	for (size_t d = 0; d != ndim; ++d)
		c2.slice(d) = dV(p,d);
	return c2;
}


template <size_t ndim, bool is_cplx> double TLS<ndim, is_cplx>::F(TLS::Params const& p, bool const& state, size_t const& d) {
	return std::real(eigvec(p, state).t() * dH(p,d) * eigvec(p, state));
}


template <size_t ndim, bool is_cplx> typename TLS<ndim, is_cplx>::Params TLS<ndim, is_cplx>::F(TLS::Params const& p, bool const& state) {
	Params force;
	for (size_t d = 0; d != ndim; ++d)
		force(d) = F(p, state, d);
	return force;
}


template <size_t ndim, bool is_cplx> typename TLS<ndim, is_cplx>::Val TLS<ndim, is_cplx>::drvcpl(TLS::Params const& p, bool const& i, bool const& j, size_t const& d) {
	return keep_cplx<is_cplx>( (i==j) ? 
			( (i ? -1 : 1) * 0.5 * I * std::cos(theta(p)) * Diff<ndim,false>::pdiff(phi)(p,d) ) :
			decay<Val,1,1>(eigvec(p,i).t() * dH(p,d) * eigvec(p,j) / ( eigval(p,j) - eigval(p,i) )) );
}


template <size_t ndim, bool is_cplx> typename TLS<ndim, is_cplx>::Vals TLS<ndim, is_cplx>::drvcpl(TLS::Params const& p, bool const& i, bool const& j) {
	Vals dc;
	for (size_t d = 0; d != ndim; ++d)
		dc(d) = drvcpl(p, i, j, d);
	return dc;
}


#endif
