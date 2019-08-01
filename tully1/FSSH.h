#ifndef __FSSH_H__
#define __FSSH_H__

#include <armadillo>

template<typename Model>
class FSSH
{
	public:
		FSSH(
				Model*			model_,
				int				state0_,
				double			x0_,
				double 			v0_,
				double 			dt_			);

		Model*					model;
		double					dt;

		// initial condition
		int						state0;
		double					x0;
		double 					v0;
		arma::cx_mat			eigvec0;
		arma::vec				eigval0;

		// dynamical quantities
		int						state;
		double					x;
		double					v;
		arma::cx_mat			eigvecs;
		arma::vec				eigvals;

		void					init();
		//void					propagate();
		//void 					prob();
		//std::complex<double>	dc(int, int);

	private:

};

#include "FSSH.tpp"

#endif
