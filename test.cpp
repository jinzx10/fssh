#include "TLS.h"

const std::complex<double> I(0,1);
const double PI = acos(-1);
const double A = 0.01, B = 1.6, C = 0.005, D = 1.0, cpl_phase = 0.1;

auto V00 = [](double x) {
	return (x>0) ? A * (1.0 - exp(-B*x)) : -A * (1 - exp(B*x));
};
auto V11 = [](double x) { return -V00(x);};
auto V01 = [](double x) {
	return C * exp(-D*x*x);
	//return C * exp(-D*x*x) * exp(I*PI*cpl_phase);
};

int main() {

	TLS<1, false> tls(V00, V11, V01);

	std::cout << tls.V01(0.2) << std::endl;
	std::cout << tls.V10(0.2) << std::endl;

	tls.V(0.2).print();

	tls.eigval(0.2).print();

	tls.eigvec(0.2).print();
	return 0;
}
