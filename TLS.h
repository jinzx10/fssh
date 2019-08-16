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

		using DiabPES		= std::function<double(Params)>;
		using DiabCpl		= std::function<Val(Params)>;
		using ParamsChk		= std::function<bool(Params)>;
		
		using Vec2			= typename arma::Col<Val>::template fixed<2>;
		using Mat2			= typename arma::Mat<Val>::template fixed<2,2>;

		// constructor
		TLS(
				DiabPES			V00_,
				DiabPES			V11_,
				DiabCpl			V01_,
				ParamsChk		is_within_ = always
		);

		// Hamiltonian
		Mat2				V(Params);
		DiabPES				V00;
		DiabPES 			V11;
		DiabCpl 			V01;
		DiabCpl 			V10;

		// V = d0*I + \sum_i d_i * sigma_i
		double				d0(Params);
		double 				dx(Params);
		double 				dy(Params);
		double 				dz(Params);

		// spherical coordinates of (dx,dy,dz) in Bloch sphere
		double				r(Params);
		double				theta(Params);
		double				phi(Params);

		double				eigval(Params, bool);
		arma::vec2			eigval(Params);
		Vec2				eigvec(Params, bool); // analytical expression, symmetric phase
		Mat2				eigvec(Params);

		double				F(Params, bool);
		Params				F(Params);


		double				dV00(Params, size_t);
		Params				dV00(Params);
		double				dV11(Params, size_t);
		Params				dV11(Params);
		Val					dV01(Params, size_t);
		Vals				dV01(Params);
		//Val				dV10(Params, size_t);
		//ParamsVec			dV10(Params);
		//Mat				dV(Params, size_t);
		//ParamsVec			drvcpl(Params, bool, bool);

		ParamsChk				is_within;

	private:
		static bool			always(Params const&) { return true; }

};


template <size_t ndim, bool is_cplx> TLS<ndim, is_cplx>::TLS( DiabPES V00_, DiabPES V11_, DiabCpl V01_, ParamsChk is_within_ ):
	V00(V00_), V11(V11_), V01(V01_), is_within(is_within_) {
	V10 = [this](Params p) { return KeepCplx<is_cplx>::value( std::conj(V01(p)) ); };
}


template <size_t ndim, bool is_cplx> typename TLS<ndim, is_cplx>::Mat TLS<ndim, is_cplx>::V(TLS::Params p) {
	return Mat{ {V00(p), V01(p)}, {V10(p), V11(p)} };
}


template <size_t ndim, bool is_cplx> double TLS<ndim, is_cplx>::d0(TLS::Params p) {
	return 0.5 * (V00(p) + V11(p));
}


template <size_t ndim, bool is_cplx> double TLS<ndim, is_cplx>::dz(TLS::Params p) {
	return 0.5 * (V00(p) - V11(p));
}


template <size_t ndim, bool is_cplx> double TLS<ndim, is_cplx>::dx(TLS::Params p) {
	return std::real(V01(p));
}


template <size_t ndim, bool is_cplx> double TLS<ndim, is_cplx>::dy(TLS::Params p) {
	return -std::imag(V01(p));
}


template <size_t ndim, bool is_cplx> double TLS<ndim, is_cplx>::r(TLS::Params p) {
	return std::sqrt( std::pow(dx(p),2) + std::pow(dy(p),2) + std::pow(dz(p),2) );
}


template <size_t ndim, bool is_cplx> double TLS<ndim, is_cplx>::theta(TLS::Params p) {
	return std::acos( dz(p) / r(p) );
}


template <size_t ndim, bool is_cplx> double TLS<ndim, is_cplx>::phi(TLS::Params p) {
	return (dy(p) > 0 ? 1 : -1) * std::acos( dx(p) / std::sqrt(std::pow(dx(p),2) + std::pow(dy(p),2)) );
}


template <size_t ndim, bool is_cplx> double TLS<ndim, is_cplx>::eigval(TLS::Params p, bool state) {
	return d0(p) + (state ? r(p) : -r(p));
}


template <size_t ndim, bool is_cplx> arma::vec2 TLS<ndim, is_cplx>::eigval(TLS::Params p) {
	return d0(p) + r(p) * arma::vec{-1, 1};
}


template <size_t ndim, bool is_cplx> typename TLS<ndim, is_cplx>::Vec TLS<ndim, is_cplx>::eigvec(TLS::Params p, bool state) {
	return state ?
		Vec{
			std::cos(theta(p)/2.0) * KeepCplx<is_cplx>::value( std::exp(-I * phi(p) / 2.0) ),
			std::sin(theta(p)/2.0) * KeepCplx<is_cplx>::value( std::exp(I * phi(p) / 2.0) )
		} :
		Vec{
			-std::sin(theta(p)/2.0) * KeepCplx<is_cplx>::value( std::exp(-I * phi(p) / 2.0) ),
			std::cos(theta(p)/2.0) * KeepCplx<is_cplx>::value( std::exp(I * phi(p) / 2.0) )
		};
}


template <size_t ndim, bool is_cplx> typename TLS<ndim, is_cplx>::Mat TLS<ndim, is_cplx>::eigvec(TLS::Params p) {
	return arma::join_rows(eigvec(p,0), eigvec(p,1));
}



#endif
