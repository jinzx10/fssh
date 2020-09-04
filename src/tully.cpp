#include "tully.h"
#include <cmath>

using namespace fssh;

double Tully1::V00(double x) {
    return (x>0) ? A*(1.0-std::exp(-B*x)) : -A*(1.0-std::exp(B*x));
}

double Tully1::V11(double x) {
    return -V00(x);
}

double Tully1::V01(double x) {
    return C*std::exp(-D*x*x);
}

double Tully1::dV00(double x) {
    return A*B*std::exp((x>0 ? -B : B)*x);
}

double Tully1::dV11(double x) {
    return -dV00(x);
}

double Tully1::dV01(double x) {
    return -2.0*D*x*V01(x);
}

double Tully1::energy(double x, bool state) {
    double v0 = V00(x);
    double v1 = V11(x);
    return 0.5*(v0+v1) + (state ? 1 : -1) * std::sqrt(std::pow(0.5*(v0-v1),2) + std::pow(V01(x),2));
}

double Tully1::force(double x, bool state) {
    double v0 = V00(x);
    double v1 = V11(x);
    double w = V01(x);
    double dv0 = dV00(x);
    double dv1 = dV11(x);
    double dw = dV01(x);
    return -0.5*(dv0+dv1) + (state ? -1 : 1) / std::sqrt(std::pow(0.5*(v0-v1), 2) + w*w) * (0.25*(v0-v1)*(dv0-dv1) + w*dw);
}

double Tully1::dc01(double x) {
    double v0 = V00(x);
    double v1 = V11(x);
    double w = V01(x);
    return ( w > 0 ? 1 : -1) * (w*(dV00(x)-dV11(x)) - (v0-v1)*dV01(x)) / (std::pow(v0-v1, 2) + 4*w*w);
}

bool Tully1::out_of_range(double x) {
    return (std::abs(x) > 10);
}


