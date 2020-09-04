#include "fssh1990.h"
#include <random>
#include <algorithm>

using namespace fssh;

Vec fssh::operator*(double x, Vec const& v) {
    Vec w(v.size());
    std::transform(v.begin(), v.end(), w.begin(), [x] (double const& e) { return e*x; });
    return w;
}

Vec fssh::operator+(Vec const& v1, Vec const& v2) {
    Vec w(v1.size());
    std::transform(v1.begin(), v1.end(), v2.begin(), w.begin(), std::plus<double>());
    return w;
}


FSSH1990::FSSH1990(Tully* model_, double mass_): model(model_), mass(mass_) {}

void FSSH1990::run(double x0, double v0, double rho00, double rho01R0, double rho01I0, bool state0, double dt_) {
    clear();
    dt = dt_;

    // initialize a trajectory
    f = {x0, v0, rho00, rho01R0, rho01I0};
    state = state0;

    // propagation
    while (!model->out_of_range(f[0])) {
        evolve();
        update_elec_prop();
        hop();
        collect();
    }
}

void FSSH1990::clear() {
    E_t.clear();
    num_frustrated_hops = 0;
}

void FSSH1990::evolve() { // Fourth-order Runge-Kutta
    Vec df1 = dt * dfdt(f);
    Vec df2 = dt * dfdt(f + 0.5*df1);
    Vec df3 = dt * dfdt(f + 0.5*df2);
    Vec df4 = dt * dfdt(f + df3);
    f = f + (1.0/6.0) * (df1 + 2.0*df2 + 2.0*df3 + df4);
}

void FSSH1990::hop() {
    int v_sign = (f[1] > 0) ? 1 : -1;
    double drho00 = -2*f[1]*dc01*f[3];
    double Ek = 0.5*mass*f[1]*f[1];
    double dE = E1 - E0;

    if (state) { // state == 1
        if ( randu() < dt * drho00 / (1-f[2]) ) {
            f[1] = v_sign * std::sqrt(2*(Ek+dE)/mass);
            state = 0;
        }
    } else { // state == 0
        if ( randu() < dt * (-drho00) / f[2] ) {
            if (Ek > dE) {
                f[1] = v_sign * std::sqrt(2*(Ek-dE)/mass);
                state = 1;
            } else {
                num_frustrated_hops += 1;
            }
        }
    }
}

void FSSH1990::collect() {
    E_t.push_back((state ? E1 : E0) + 0.5*mass*f[1]*f[1]);
}

Vec FSSH1990::dfdt(Vec const& f_) {
    double dc01 = model->dc01(f_[0]);
    double E01 = model->energy(f_[0], 0) - model->energy(f_[0], 1);
    return Vec{ 
        f_[1], 
        model->force(f_[0], state) / mass, 
        -2.0*f_[1]*dc01*f_[3], 
        E01*f_[4] + f_[1]*dc01*(2*f_[2]-1),
        -E01*f_[3]
    };
}

void FSSH1990::update_elec_prop() {
    E0 = model->energy(f[0], 0);
    E1 = model->energy(f[0], 1);
    dc01 = model->dc01(f[0]);
}


