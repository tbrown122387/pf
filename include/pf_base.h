#ifndef PF_BASE_H
#define PF_BASE_H

#include <map>
#include <string>
#include <vector>
#include <Eigen/Dense>


/**
 * @author t
 * @file pf_base.h
 * @brief All non Rao-Blackwellized particle filters inherit from this.
 * @tparam float_t (e.g. double, float, etc.)
 * @tparam dimobs the dimension of each observation
 * @tparam dimstate the dimension of each state
 */
template<typename float_t, size_t dimobs, size_t dimstate>
class pf_base {
public:
    using osv = Eigen::Matrix<float_t,dimobs,1>;
    using ssv = Eigen::Matrix<float_t,dimstate,1>;
    using Mat = Eigen::Matrix<float_t,Eigen::Dynamic,Eigen::Dynamic>;
    using func = std::function<const Mat(const ssv&)>;
    using funcs = std::vector<func>;

    virtual void filter(const osv &data, const funcs& fs = funcs() ) = 0; 
    virtual float_t getLogCondLike() const = 0;
    virtual ~pf_base(){};
};


/**
 * @author t
 * @file pf_base.h
 * @brief All Rao-Blackwellized particle filters inherit from this. 
 * @tparam float_t (e.g. double, float, etc.)
 * @tparam dim_s_state the dimension of the state vector that's sampled
 * @tparam dim_ns_state the dimension of the state vector that isn't sampled
 * @tparam dimobs the dimension of each observation vector
 */
template<typename float_t, size_t dim_s_state, size_t dim_ns_state, size_t dimobs>
class rbpf_base {
public:
    using osv   = Eigen::Matrix<float_t,dimobs,1>;
    using sssv  = Eigen::Matrix<float_t,dim_s_state,1>;
    using nsssv = Eigen::Matrix<float_t,dim_ns_state,1>;
    using Mat   = Eigen::Matrix<float_t,Eigen::Dynamic,Eigen::Dynamic>;
    using func  = std::function<const Mat(const nsssv&, const sssv&)>;
    using funcs = std::vector<func>;

    virtual void filter(const osv &data, const funcs& fs = funcs() ) = 0;
    virtual ~rbpf_base(){};
};


/**
 * @class ForwardMod
 * @author t
 * @file pf_base.h
 * @brief inherit from this if you want to simulate from a homogeneous forward/generative model.
 */
template<size_t dimx, size_t dimy, typename float_t>
class ForwardMod {
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
auto ForwardMod<dimx,dimy,float_t>::sim_forward(unsigned int T) -> aPair {

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
