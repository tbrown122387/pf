#include <iostream>

// from pf
#include <pf/resamplers.h> // for mn_resampler

// from this project
#include "svol_bs.h" // bs filter for svol
#include "forward_sim.h" // function header

// some template parameters
#define dimstate 1
#define dimobs   1
#define FLOATTYPE float // choose float (faster) or double (slower)


void forward_sim()
{
    // "state size vector"
    using ssv = Eigen::Matrix<FLOATTYPE,dimstate,1>;
    // "observation sized vector"
    using osv = Eigen::Matrix<FLOATTYPE,dimobs,1>;    
    // resampler type
    using our_resamp = mn_resampler<1,dimstate,FLOATTYPE>;


    // model parameters that are assumed known
    FLOATTYPE phi = .91;
    FLOATTYPE beta = .5;
    FLOATTYPE sigma = 1.0;

    // this model is a particle filter but it also
    // inherits from the ForwardMod base class
    svol_bs<1,dimstate,dimobs,our_resamp,FLOATTYPE> svol_mod(phi, beta, sigma);
    
    unsigned int length(1000);
    auto xsAndYs = svol_mod.sim_forward(length);
    std::cout << "x, y\n";
    for(size_t i = 0; i < length; ++i){
        std::cout << xsAndYs.first[i] << ", " << xsAndYs.second[i] << "\n";
    }
}
