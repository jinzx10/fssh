#ifndef __TWO_LEVEL_SYSTEM_H__
#define __TWO_LEVEL_SYSTEM_H__

#include "aux.h"
#include <type_traits>
#include <complex>
#include <armadillo>

template < arma::uword ndim, bool is_cplx = false >
class TLS
{
	public:
		// Param is double if ndim == 1, is arma::vec otherwise
		typedef typename std::conditional<(ndim==1), double, arma::Col<double>::fixed<ndim>>::type Param;
		// T_cpl is std::complex<double> or double depending on is_cplx
		typedef typename std::conditional<is_cplx, std::complex<double>, double>::type T_cpl;
		typedef double (*PES)(Param);
		typedef T_cpl (*Cpl)(Param);
		typedef bool (*Checker)(Param);
		
		template <typename T>
		using ParamVec = typename arma::Col<T>::template fixed<ndim>;

		template <typename T>
		using ElecVec = typename arma::Col<T>::template fixed<2>;

		template <typename T>
		using ElecMat = typename arma::Mat<T>::template fixed<2,2>;

		TLS(
				PES			V00_,
				PES			V11_,
				Cpl			V01_,
				Checker 	is_within_ = always
		);

		PES					V00;
		PES 				V11;
		Cpl 				V01;
		Cpl 				V10;

		double				dV00(Param, arma::uword);
		ParamVec<double>	dV00(Param);

		double				dV11(Param, arma::uword);
		ParamVec<double>	dV11(Param);

		T_cpl				dV01(Param, arma::uword);
		ParamVec<T_cpl>		dV01(Param);

		T_cpl				dV10(Param, arma::uword);
		ParamVec<T_cpl>		dV10(Param);

		double				eigval(Param, bool);
		ElecVec<double>		eigval(Param);

		double				force(Param, bool);
		ElecVec<double>		force(Param);

		ElecVec<T_cpl>		eigvec(Param, bool);
		ElecMat<T_cpl>		eigvec(Param);

		T_cpl				drvcpl(Param, bool, bool);
		ElecMat<T_cpl>		drvcpl(Param);

		bool				is_within(Param);

	private:
		static bool			always(Param) {return true;}
};

#include "TLS.tpp"

#endif
