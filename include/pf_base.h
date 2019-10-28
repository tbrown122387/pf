#ifndef PF_BASE_H
#define PF_BASE_H

#include <map>
#include <string>
#include <vector>


/**
 * @class pf_base
 * @author t
 * @file pf_base.h
 * @brief All particle filters inherit from this. 
 */
class pf_base{
public:
    virtual ~pf_base(){};
};


/**
 * @class homog_forward_model
 * @author t
 * @file pf_base.h
 * @brief inherit from this if you want to simulate from a homogeneous forward/generative model.
 */
template<size_t dimx, size_t dimy, typename float_t>
class homog_forward_model {
public:

    /* state-sized vector  */
    using ssv = Eigen::Matrix<float_t, dimx, 1>;
    
    /* observation-sized vector */
    using osv = Eigen::Matrix<float_t, dimy, 1>;

    /* a pair of paths (xts first, yts second) */
    using aPair = std::pair<std::vector<ssv>, std::vector<osv> >;


    /**
     * @brief simulates forward through time
     */
    aPair sim_forward(unsigned int T);

    /**
     * @brief samples from the first time's state distribution
     * @return a state-sized vector for the x1 sample
     */
    virtual ssv muSamp()                = 0;


    /**
     * @brief returns a sample from the latent Markov transition
     * @return a state-sized vector for the xt sample
     */
    virtual ssv fSamp (const ssv &xtm1) = 0;


    /**
     * @brief returns a sample for the observed series 
     * @return 
     */
    virtual osv gSamp (const ssv &xt)   = 0;
};


template<size_t dimx, size_t dimy, typename float_t>
auto homog_forward_model<dimx,dimy,float_t>::sim_forward(unsigned int T) -> aPair {

    std::vector<ssv> xs;
    std::vector<osv> ys;

    // time 1
    xs.push_back(this->muSamp());
    ys.push_back(this->gSamp(xs[0]));

    // time > 1
    for(size_t i = 1; i < T; ++i) {
        
        auto xt = this->fSamp(xs[i-1]);
        xs.push_back(xt);
        ys.push_back(this->gSamp(xt));
    }

    return aPair(xs, ys); 
}

#endif // PF_BASE_H
