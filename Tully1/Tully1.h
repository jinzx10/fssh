#ifndef __TULLY1_H__
#define __TULLY1_H__

#include <complex>
#include <armadillo>

class Tully1
{
	public:
		Tully1();
		Tully1(
				double				A_,
				double 				B_,
				double 				C_,
				double 				D_,
				double 				cpl_phase_ );

		// diabatic potential energy matrix
		double						V00(double x);
		double						V11(double x);
		std::complex<double>		V01(double x);
		std::complex<double> 		V10(double x);
		arma::cx_mat				V(double x);

		// derivative of potential energy matrix
		double						dV00(double x);
		double						dV11(double x);
		std::complex<double>		dV01(double x);
		std::complex<double> 		dV10(double x);
		arma::cx_mat				dV(double x);

		// dV in eigvec basis
		arma::cx_mat				dV_eig(double x);

		// adiabatic energy
		arma::vec					eigval(double x);
		double						eigval0(double x);
		double 						eigval1(double x);
		
		// eigenstates
		arma::cx_mat				eigvec(double x);
		arma::cx_vec 				eigvec0(double x);
		arma::cx_vec 				eigvec1(double x);

		// adiabatic force
		arma::vec					F(double x);
		double						F0(double x);
		double 						F1(double x);

		// derivative coupling
		std::complex<double>		dc01(double x);
		std::complex<double>		dc10(double x);

		// number of potential energy surfaces
		arma::uword					num_elec_dofs();

	private:
		// model parameters
		double						A;
		double 						B;
		double 						C;
		double 						D;

		// phase of diabatic coupling (in unit of pi)
		double						cpl_phase;
};

#endif
