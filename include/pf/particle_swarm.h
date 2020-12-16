#ifndef PARTICLE_SWARM_H
#define PARTICLE_SWARM_H

#include <functional>
#include <tuple>


//TODO parallel version
/**
 * @brief object that implements a "particle swarm" (see https://arxiv.org/abs/2006.15396)
 * For simplicity, simulates from the model's parameter prior. 
 * @tparam represents a specific parametric model that has inherited from a pf base class template.
 * @tparam number of functions we want to filter. Each filter func represents the h in E[h(xt)|y_1:t]. 
 */
template<typename ModType, size_t n_filt_funcs>
class Swarm {

public:
    using float_t   = typename ModType::float_type;
    using osv       = typename ModType::osv;
    using ssv       = typename ModType::ssv;
    using Mat       = typename ModType::Mat;
    using filt_func = std::function<const Mat(const ssv&)>;


private:    
    /* a model and a vector of functions for each model/parameter */
    std::vector<ModType>   m_mods;
    std::vector<filt_func> m_funcs;

    /* log p(y_t+1 | y_{1:t}) */
    float_t m_log_cond_like;
    
    /* E[h(x_t)|y_{1:t}] */
    std::vector<Mat> m_expectations;

    /* keep track of the number of observations seen in time */
    unsigned int m_num_obs;

    // TODO: maybe consider storing the entire evidence because it's a sum of products

public:

    /**
     * @brief ctor
     * TODO: right now there is no way to get the size of each expectation in the vector.
     * Because we are calling the default constructor on each element, they are 0x0 before
     * any data is seen. Perhaps you can back out the dimension earlier to de-complicate thigns
     */
    Swarm<ModType, n_filt_funcs>() : m_num_obs(0) {  m_expectations.resize(n_filt_funcs); } 


    // can this be done at arbitrary times?
    // TODO: pass b y reference! 
    void addModelFuncsPair(ModType mod, std::vector<filt_func> funcs) {
        if(funcs.size() == n_filt_funcs) {
            m_mods.push_back(mod);

            // only add functions if it's the first iteration.
            // we are assuming that the same functions apply for all models and all times
            if(m_funcs.empty()) m_funcs = funcs;  
        }else {
            throw std::invalid_argument("number of filter funcs should correspond with template parameter \n");
        }
    }
    
    
    /**
     * @brief update the model on a new time point's observation
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
        unsigned int num_samples = m_mods_and_funcs.size();
        float_t Ntheta = static_cast<float_t>(num_samples);
        for(size_t i = 0; i < num_samples; ++i) {
            
            // update a model on new data
            // first is the model
            // second is the vector<func>
            m_mods[i].filter(yt, m_funcs);

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
   
    /* get log p(y_t+1 | y_0:t, M) */
    float_t getLogCondLike() const { return m_log_cond_like; }

    /* getter of expectations */
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



#endif // PARTICLE_SWARM_H
