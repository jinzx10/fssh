#ifndef __FEWEST_SWITCHES_SURFACE_HOPPING_TULLY_1990_H__
#define __FEWEST_SWITCHES_SURFACE_HOPPING_TULLY_1990_H__

#include <functional>
#include <vector>
#include <random>
#include "tully.h"

namespace fssh {

    using Vec = std::vector<double>;
    Vec operator*(double x, Vec const& v);
    Vec operator+(Vec const& v1, Vec const& v2);

    struct FSSH1990
    {

        FSSH1990(Tully* model, double mass);
    
        void run(double x0, double v0, double rho00, double rho01R, double rho01I, bool state0, double dt);

        Tully*          model;
        double          mass;

        // dynamic quantities
        Vec             f; // {x, v, rho00, real(rho01), imag(rho01)}
        bool            state;
        double          dt;

        // trajectory info   
        Vec             E_t;
        size_t          num_frustrated_hops;
    
    private:
        void            clear();
        void            evolve();
        void            hop();
        void            collect();

        Vec             dfdt(Vec const& f_);
        void            update_elec_prop(); // update eigval and dc01 

        double          E0;
        double          E1;
        double          dc01;

        double          randu() { return ud(rng); }
        std::mt19937    rng = std::mt19937(std::random_device{}());
        std::uniform_real_distribution<> ud = std::uniform_real_distribution<>(0.0, 1.0);
    };

}
    
#endif
