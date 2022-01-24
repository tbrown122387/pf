#include <iostream>

// from pf
#include <pf/resamplers.h> // for mn_resampler

// from this project
#include "svol_bs.h" // bs filter for svol
#include "forward_sim.h" // function header

// some template parameters
#define dimstate 1
#define dimobs   1
#define numparts 5
#define FLOATTYPE float // choose float (faster) or double (slower)


using namespace pf::resamplers;

void forward_sim()
{
    // "state size vector"
    using ssv = Eigen::Matrix<FLOATTYPE,dimstate,1>;
    // "observation sized vector"
    using osv = Eigen::Matrix<FLOATTYPE,dimobs,1>;    
    // resampler type
    using our_resamp = mn_resampler<numparts,dimstate,FLOATTYPE>;


    // model parameters that are assumed known
    FLOATTYPE phi = .91;
    FLOATTYPE beta = .5;
    FLOATTYPE sigma = 1.0;

    // this model is a particle filter but it also
    // inherits from the ForwardMod base class and the FutureSimulator base class
    svol_bs<numparts,dimstate,dimobs,our_resamp,FLOATTYPE> svol_mod(phi, beta, sigma);
    
    unsigned int length(50);

    // visually assess ForwardMod base class capabilities
    std::cout << "simulating the model without any real data...\n";
    auto xsAndYs = svol_mod.sim_forward(length);
    std::cout << "x, y\n";
    for(size_t i = 0; i < length; ++i){
        std::cout << xsAndYs.first[i] << ", " << xsAndYs.second[i] << "\n";
    }

    // visually assess FutureSimulator base class capabilities
    std::cout << "filter on one piece of data (1.0), and then simulate future trajectories...\n";
    svol_mod.filter(osv(1));
    auto future_obs_paths = svol_mod.sim_future_obs(length);
   
    for(unsigned int time = 0; time < length; ++time){
        for(size_t idx = 0; idx < numparts; ++idx){
            if(idx < numparts - 1){
                std::cout << future_obs_paths[time][idx](0,0) << ", ";
            }else{
                std::cout << future_obs_paths[time][idx](0,0) << "\n";
            }
        }
    }
}
