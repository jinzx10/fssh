#ifndef __TULLY_THREE_MODELS_H__
#define __TULLY_THREE_MODELS_H__

namespace fssh {

    struct Tully
    {
        virtual double energy(double x, bool state) = 0;
        virtual double force(double x, bool state) = 0;
        virtual double dc01(double x) = 0;
        virtual bool out_of_range(double x) = 0;
    };
    
    struct Tully1 : Tully 
    {
        double energy(double x, bool state);
        double force(double x, bool state);
        double dc01(double x);
        bool out_of_range(double x);

    private:
        double V00(double x);
        double V11(double x);
        double V01(double x);
    
        double dV00(double x);
        double dV11(double x);
        double dV01(double x);
    
        double A = 0.01;
        double B = 1.6;
        double C = 0.005;
        double D = 1.0;
    };
}

#endif
