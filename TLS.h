#ifndef __TWO_LEVEL_SYSTEM_H__
#define __TWO_LEVEL_SYSTEM_H__

#include "aux.h"
#include <type_traits>
#include <complex>
#include <armadillo>

template < size_t ndim, bool is_cplx = false >
class TLS
{
	public:
		// Arg is double if ndim == 1, is arma::vec otherwise
		using Arg			= typename std::conditional< (ndim==1), double, arma::Col<double>::fixed<ndim> >::type;

		// T_cpl is std::complex<double> or double, depending on is_cplx
		using T_cpl			= typename std::conditional< is_cplx, std::complex<double>, double >::type;

		// vector types
		using ArgVec		= typename std::conditional< (ndim==1), T_cpl, typename arma::Col<T_cpl>::template fixed<ndim> >::type;
		using ElecVec		= typename arma::Col<T_cpl>::template fixed<2>;
		using ElecMat 		= typename arma::Mat<T_cpl>::template fixed<2,2>;

		// function types
		using PES			= std::function<double(Arg)>;
		using Cpl			= std::function<T_cpl(Arg)>;
		using ArgChk		= std::function<bool(Arg)>;
		
		// constructor
		TLS(
				PES			V00_,
				PES			V11_,
				Cpl			V01_,
				ArgChk		is_within_ = always
		);

		// Hamiltonian
		ElecMat				V(Arg);
		PES					V00;
		PES 				V11;
		Cpl 				V01;
		Cpl 				V10;

		// V = d0*I + \sum_i d_i * sigma_i
		double				d0(Arg);
		double 				dx(Arg);
		double 				dy(Arg);
		double 				dz(Arg);

		// spherical coordinates in Bloch sphere
		double				r(Arg);
		double				theta(Arg);
		double				phi(Arg);

		double				eigval(Arg, bool);
		arma::vec2			eigval(Arg);
		ElecVec				eigvec(Arg, bool); // symmetric analytical expression
		ElecMat				eigvec(Arg);

		//Arg				F(Arg, bool);
		//ArgVec			drvcpl(Arg, bool, bool);

		//double				dV00(Arg, size_t);
		//ArgVec			dV00(Arg);
		//double				dV11(Arg, size_t);
		//ArgVec			dV11(Arg);
		//T_cpl				dV01(Arg, size_t);
		//ArgVec			dV01(Arg);
		//T_cpl				dV10(Arg, size_t);
		//ArgVec			dV10(Arg);
		//ElecMat				dV(Arg, size_t);

		ArgChk				is_within;

	private:
		static bool			always(Arg) {return true;}

};


template <size_t ndim, bool is_cplx> TLS<ndim, is_cplx>::TLS( PES V00_, PES V11_, Cpl V01_, ArgChk is_within_ ):
	V00(V00_), V11(V11_), V01(V01_), is_within(is_within_) {
	V10 = [this](Arg p) { return KeepCplx<is_cplx>::value( std::conj(V01(p)) ); };
}


template <size_t ndim, bool is_cplx> typename TLS<ndim, is_cplx>::ElecMat TLS<ndim, is_cplx>::V(TLS::Arg p) {
	return ElecMat{ {V00(p), V01(p)}, {V10(p), V11(p)} };
}


template <size_t ndim, bool is_cplx> double TLS<ndim, is_cplx>::d0(TLS::Arg p) {
	return 0.5 * (V00(p) + V11(p));
}


template <size_t ndim, bool is_cplx> double TLS<ndim, is_cplx>::dz(TLS::Arg p) {
	return 0.5 * (V00(p) - V11(p));
}


template <size_t ndim, bool is_cplx> double TLS<ndim, is_cplx>::dx(TLS::Arg p) {
	return std::real(V01(p));
}


template <size_t ndim, bool is_cplx> double TLS<ndim, is_cplx>::dy(TLS::Arg p) {
	return -std::imag(V01(p));
}


template <size_t ndim, bool is_cplx> double TLS<ndim, is_cplx>::r(TLS::Arg p) {
	return std::sqrt( std::pow(dx(p),2) + std::pow(dy(p),2) + std::pow(dz(p),2) );
}


template <size_t ndim, bool is_cplx> double TLS<ndim, is_cplx>::theta(TLS::Arg p) {
	return std::acos( dz(p) / r(p) );
}


template <size_t ndim, bool is_cplx> double TLS<ndim, is_cplx>::phi(TLS::Arg p) {
	return (dy(p) > 0 ? 1 : -1) * std::acos( dx(p) / std::sqrt(std::pow(dx(p),2) + std::pow(dy(p),2)) );
}


template <size_t ndim, bool is_cplx> double TLS<ndim, is_cplx>::eigval(TLS::Arg p, bool state) {
	return d0(p) + (state ? r(p) : -r(p));
}


template <size_t ndim, bool is_cplx> arma::vec2 TLS<ndim, is_cplx>::eigval(TLS::Arg p) {
	return d0(p) + r(p) * arma::vec{-1, 1};
}


template <size_t ndim, bool is_cplx> typename TLS<ndim, is_cplx>::ElecVec TLS<ndim, is_cplx>::eigvec(TLS::Arg p, bool state) {
	return state ?
		ElecVec{
			std::cos(theta(p)/2.0) * KeepCplx<is_cplx>::value( std::exp(-I * phi(p) / 2.0) ),
			std::sin(theta(p)/2.0) * KeepCplx<is_cplx>::value( std::exp(I * phi(p) / 2.0) )
		} :
		ElecVec{
			-std::sin(theta(p)/2.0) * KeepCplx<is_cplx>::value( std::exp(-I * phi(p) / 2.0) ),
			std::cos(theta(p)/2.0)*KeepCplx<is_cplx>::value( std::exp(I * phi(p) / 2.0) )
		};
}


template <size_t ndim, bool is_cplx> typename TLS<ndim, is_cplx>::ElecMat TLS<ndim, is_cplx>::eigvec(TLS::Arg p) {
	return arma::join_rows(eigvec(p,0), eigvec(p,1));
}



#endif
