#include <iostream>
#include <cmath>
#include <complex>
#include "Tully1.h"
//#include "../FSSH.h"

//extern const double PI = acos(-1);
//extern const std::complex<double> I(0.0, 1.0);

int main() {
	double phase = 0.5;
	Tully1 model(0.01, 1.6, 0.005, 1.0, phase);

	arma::uword nx = 1000;
	double xmin = -10;
	double xmax = 10;
	arma::vec xgrid = arma::linspace(xmin, xmax, nx);

	arma::mat x_eigval = arma::zeros(nx, 3);
	arma::mat x_dc_real = arma::zeros(nx, 2);
	arma::mat x_dc_imag = arma::zeros(nx, 2);

	x_eigval.col(0) = xgrid;
	x_dc_real.col(0) = xgrid;
	x_dc_imag.col(0) = xgrid;

	//model.eigvec(0).print();
	//std::cout << model.dc01(0) << std::endl;
	
	arma::uvec idx = {1,2};
	for (arma::uword i = 0; i < nx; ++i) {
		x_eigval.row(i).cols(1,2) = model.eigval(xgrid(i)).t();
		std::complex<double> dc = model.dc01(xgrid(i));
		x_dc_real(i,1) = std::real(dc);
		x_dc_imag(i,1) = std::imag(dc);
	}

	x_eigval.save("eigval.txt", arma::raw_ascii);
	x_dc_real.save("dc_real.txt", arma::raw_ascii);
	x_dc_imag.save("dc_imag.txt", arma::raw_ascii);

	//FSSH<Tully1> sys(&model, 0, -0, 0.1, 10);

	//sys.eigvecs.print();

	return 0;
}
