/*
This program computes the ignition condition of the X-ray burst
based on https://github.com/andrewcumming/settle
See DOI: 10.1086/317191 for details.

Usage:
./main.ex <X> <Z> <Fb> <mdot> <COMPRESS>

Fb: Base Flux in MeV/nucleon
X: Hydrogen massfraction
Z: CNO massfraction
mdot: Local accretion rate in Eddington unit
COMPRESS: To include compressional heating or not.
*/

#include <iostream>
#include <AMReX_Real.H>

struct State
{
    // Radius and gravitational acceleration
    amrex::Real R, g;

    // Initial abundance for hydrogen, helium, and metal
    amrex::Real X, Y, Z;

    // Accretion rate
    amrex::Real mdot;

    // Depletion Column Depth
    amrex::Real yd;

    // Top boundary column depth, temperature and flux
    amrex::Real yt, Tt, Ft;

    // Bot boundary column depth, temperature and flux
    // This is assumed to be the ignition condition
    amrex::Real yb, Tb, Fb;
};


void yb_constraint_eq(State& state, amrex::Real yb_0) {
    // Constraint equation to find ignition column depth
    // given the guess ignition column depth, yb_0.
    // Note the guess value is given in log10(yb)

    state.yb = std::pow(10.0, yb_0);

    // Set the upper boundary conditions: yt, Tt, and Ft

    // Pick arbitrary top boundary column depth
    // Follow "settle" version on Github
    state.yt = 1.e3_rt;

    // Compute the top boundary flux
    // Ft = F_b + epsilon_H * (yb or yd)
    // which ever is smaller.
    // If yb < yd, the ignition happens during hydrogen-helium layer
    // If yb > yd, the ignition happens during pure helium layer

    amrex::Real epsilon_H = 5.8e13 * (state.Z / 0.01);
    state.Ft = state.Fb + epsilon_H * std::min(state.yb, state.yd);

    // Compute the top boundary temperature
    // Given by the analytic radiative zero solution for constant flux
    // Tt = (3 κ Ft yt/ [a c])
}


void find_yb() {
    //
}


int main(int argc, char* argv[]) {

    // state contains all information about the ignition conditions
    State state;

    // Gravitational Acceleration for 1.4 solar mass and 11 km NS.
    state.g = 1.5e14_rt;
    state.R = 1.1e6_rt;

    // Check if we have enough arguments passed in
    if (argc < 5) {
        std::cout << "Error" << std::endl;
    }

    // Read in input parameters: X, Z, mdot, and F_b

    state.X = std::stod(argv[0]);
    state.Z = std::stod(argv[1]);

    // Compute the Eddington Accretion Rate
    // mdot_Edd = 2 m_p c / [(1 + X) R σ_th ]
    // They used a value of 8.8e4 g cm^-2 s^-1
    // We can compute it at runtime in the future
    amrex::Real mdot_Edd = 8.8e4_rt;  // Come back later

    state.mdot = std::stod(argv[2]) * m_Edd;

    // Input Fb is MeV/nucleon/mdot.
    // Brown & Bildstein showed that electron capture rates can
    // release ~ 1 MeV/nucleon deep in the crust
    // The compressional term can give additional c_p T ~ 20 T8 keV/nucleon
    // If don't include compressional term, use higher constant Fb
    // to account for compressional heating. They use ~1.5 MeV/nucleon
    // Convert to CGS.
    state.Fb = std::stod(argv[3]) * state.mdot * 14.62_rt * 5.8e21_rt; // Not sure why multiplied by these constants, supposedly it is to convert to CGS...

    // Get the initial helium mass fraction
    state.Y = 1.0_rt - state.X - state.Z;

    // Energy release per gram from hot CNO.
    // This is mass difference between helium and 4 hydrogen
    constexpr amrex::Real E_H = 6.4e18_rt;

    // Specific nuclear energy generation rate for hot CNO
    amrex::Real epsilon_H = 5.8e13 * (state.Z / 0.01);

    // Depletion Column Depth: yd = X mdot E_H / epsilon_H
    state.yd = state.X * state.mdot * E_H / epsilon_H;

    // Now compute the base column depth: yb
    // This is also the ignition column depth
    // Assume that we work with log10(yb)

    state.yb = find_yb();
}
