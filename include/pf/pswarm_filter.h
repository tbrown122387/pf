#ifndef PSWARM_FILTER_H
#define PSWARM_FILTER_H

#include "pf_base.h"

#include <functional>
#include <tuple>
#include <type_traits>


//TODO parallel version
/**
 * @brief object that implements a "particle swarm filter" (see https://arxiv.org/abs/2006.15396)
 * For simplicity, this class assumes model/parameters are drawn from the parameter prior, and that
 * this is implemented outside of this class. Once you sample parameters and construct a model, that
 * model is added to this container class via addModelFuncsPair(). // TODO add prior/paramsample weights
 * to be passed in as well. 
 * @tparam represents a specific parametric model that has inherited from a pf base class template.
 * @tparam number of functions we want to filter. Each filter func represents the h in E[h(xt)|y_1:t]. 
 */
template<typename ModType, size_t n_filt_funcs, size_t nparts, size_t dimy, size_t dimx>
class Swarm {

public:

    /* the floating point number type */
    using float_t    = typename ModType::float_type;

    /* assert that ModType is a proper particle filter model */
    static_assert(std::is_base_of<pf_base<float_t,dimy,dimx>, ModType>::value, 
            "ModType must inherit from a particle filter class.");

    /* the observation sized vector */
    using osv        = typename ModType::osv;

    /* the state sized vector */
    using ssv        = typename ModType::ssv;
    
    /* the Matrix type of the underlying model*/
    using Mat        = typename ModType::Mat;

    /* the function that performs filtering on each model */
    using filt_func  = std::function<const Mat(const ssv&)>;

    /* a collection of observation samples, indexed by param,time, then state particle */ 
    using obsSamples = std::vector<std::vector<std::array<osv, nparts>>>;

private:   


    /* a collection of models each with a randomly chosen parameter and a vector of functions for each model/parameter */
    std::vector<ModType>   m_mods;

    /* a vector of functions for each model (these functions may depend on the model's parameter)  */
    std::vector<std::vector<filt_func>> m_funcs;

    /* log p(y_t+1 | y_{1:t}) */
    float_t m_log_cond_like;
    
    /* E[h(x_t)|y_{1:t}] */
    std::vector<Mat> m_expectations;

    /* keep track of the number of observations seen in time */
    unsigned int m_num_obs;

    // TODO: maybe consider storing the entire evidence because it's a sum of products like IS^2 algorithm

public:

    /**
     * @brief ctor
     * TODO: right now there is no way to get the size of each expectation in the vector.
     * Because we are calling the default constructor on each element, they are 0x0 before
     * any data is seen. Perhaps you can back out the dimension earlier to de-complicate thigns
     */
    Swarm() : m_num_obs(0) { m_expectations.resize(n_filt_funcs); } 


    // can this be done at arbitrary times?
    // TODO: pass b y reference!
    /**
     * @brief adds a model with a randomly sampled parameter to the container.
     * @param mod the randomly-sampled model
     */ 
    void addModelFuncsPair(ModType mod, std::vector<filt_func> funcVec) {
        if(funcVec.size() == n_filt_funcs){
            m_funcs.push_back(funcVec);
            m_mods.push_back(mod);
        }else{
            throw std::invalid_argument("funcVec needs to be the right length.");
        }
    }
    
    
    /**
     * @brief update the model on a new time point's observation
     * @param the most recent observation
     */
    void update(const osv& yt)  {
       

        // TODO: when we average over all parameters/models
        // we are assuming uniform weights because they're being 
        // drawn from the prior...think about generalizing this

        // zero out stuff that will get re-accumulated across parameter samples
        setLogCondLikeToZero();
        setExpecsToZero();

        // iterate over all parameter values/models
        std::vector<Mat> tmp_expecs_given_theta;
        unsigned int num_samples = m_mods.size();
        float_t Ntheta = static_cast<float_t>(num_samples);
        for(size_t i = 0; i < num_samples; ++i) {
            
            // update a model on new data
            // first is the model
            // second is the vector<func>
            m_mods[i].filter(yt, m_funcs[i]);

            // update the conditional likelihood 
            m_log_cond_like += m_mods[i].getLogCondLike();

            // now that we're updated, get the model-specific 
            // filter expectations and then average over all parameters/models
            tmp_expecs_given_theta  = m_mods[i].getExpectations(); 
            if(m_num_obs > 0 || i > 0) {
                    
                for(size_t j = 0; j < n_filt_funcs; ++j) {
                    m_expectations[j] += tmp_expecs_given_theta[j] / Ntheta;
                }
            } else{ // first time point *and* first parameter

                // m_expectations has length zero at this point
                for(size_t j = 0; j < n_filt_funcs; ++j) { 
                    m_expectations[j] = tmp_expecs_given_theta[j] / Ntheta;
                }
            }
        }
        m_log_cond_like /= static_cast<float_t>(num_samples);

        // increment number of observations seen
        ++m_num_obs;
    }
  

    /**
     * @brief simulates future observation paths.
     * The index ordering is param,time,particle
     * @param the number of steps into the future you want to simulate observations
     */
    obsSamples simFutureObs(unsigned int num_future_steps){
        obsSamples returnMe;
        for(size_t paramSamp = 0; paramSamp < m_mods.size(); ++paramSamp){
            returnMe.push_back(this->sim_future_obs(num_future_steps));
        }
        return returnMe; 
    }


    /**
     * @brief get the log of the approx. to the conditional "evidence" log p(y_t+1 | y_0:t, M) 
     * @return the floating point number
     */
    float_t getLogCondLike() const { return m_log_cond_like; }


    /**
     * @brief get the current expectation approx.s E[h(x_t)|y_{1:t}] 
     * @return a vector of Eigen::Mats
     */
    std::vector<Mat> getExpectations() const { return m_expectations; }

private:
   
    /* set the above to zero so it can be re-accumulated  */
    void setExpecsToZero() {
        for(auto& e : m_expectations) {
            e.setZero();
        }
    }

    /* set log of the above to zero so it can be re-accumulated */
    void setLogCondLikeToZero() { m_log_cond_like = 0.0; }

};



#endif // PSWARM_FILTER_H
