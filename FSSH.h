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
		int						init_state;
		double					init_x;
		double 					init_v;
		arma::cx_mat			init_eigvec;
		arma::vec				init_eigval;

		// dynamical quantities
		int						state;
		double					x;
		double					v;
		arma::cx_mat			eigvec;
		arma::vec				eigval;

		void					init();
		//void					propagate();
		//void 					prob();
		std::complex<double>	dc(int, int);

	private:

};

#include "FSSH.tpp"

#endif
