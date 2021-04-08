#ifndef PF_BASE_H
#define PF_BASE_H

#include <map>
#include <string>
#include <vector>
#include <Eigen/Dense>


namespace pf {

namespace bases {

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

    /* expose float type to users of this ABCTP */
    using float_type = float_t;

    /* expose observation-sized vector type to users  */
    using observation_sized_vector = Eigen::Matrix<float_t,dimobs,1>;

    /* expose state-sized vector type to users */
    using state_sized_vector = Eigen::Matrix<float_t,dimstate,1>;
    
    /* expose state-sized vector to users */
    using dynamic_matrix = Eigen::Matrix<float_t,Eigen::Dynamic,Eigen::Dynamic>;
    
    /* a function  */
    using func = std::function<const dynamic_matrix(const state_sized_vector&)>;
    
    /* functions  */
    using func_vec = std::vector<func>;

    /* the dimension of each observation vector (allows indirect access to template parameters)*/
    static constexpr unsigned int dim_obs = dimobs;

    /* the dimension of the state vector (allows indirect access to template parameters)*/
    static constexpr unsigned int dim_state = dimstate;


    /**
     * @brief the filtering function that must be defined
     * @param data the most recent observation 
     * @param filter functions whose expected value approx. is computed at each time step
     */ 
    virtual void filter(const observation_sized_vector &data, const func_vec& fs = func_vec() ) = 0;


    /**
     * @brief the getter method that must be defined (for conditional log-likelihood)
     * @return log p(y_t | y_{1:t-1}) approximation
     */ 
    virtual float_t getLogCondLike() const = 0;


    /**
     * @brief virtual destructor
     */
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

    /* expose observation-sized vector type to users  */
    using observation_sized_vector = Eigen::Matrix<float_t,dimobs,1>;
    
    /* expose sampled-state-size vector type to users */
    using sampled_state_sized_vector  = Eigen::Matrix<float_t,dim_s_state,1>;
    
    /* expose not-sampled-state-sized vector type to users */
    using not_sampled_state_sized_vector = Eigen::Matrix<float_t,dim_ns_state,1>;
    
    /* expose state-sized vector to users */
    using dynamic_matrix = Eigen::Matrix<float_t,Eigen::Dynamic,Eigen::Dynamic>;
    
    /* a function */
    using func  = std::function<const dynamic_matrix(const not_sampled_state_sized_vector&, const sampled_state_sized_vector&)>;
    
    /* functions */
    using func_vec = std::vector<func>;

    /* the size of the sampled state portion */
    static constexpr unsigned int dim_sampled_state = dim_s_state;

    /* the size of the not sampled state portion */
    static constexpr unsigned int dim_not_sampled_state = dim_ns_state;

    /* the size of the observations */
    static constexpr unsigned int dim_obs = dimobs;


    /**
     * @brief the filtering function that must be defined
     * @param data the most recent observation 
     * @param filter functions whose expected value approx. is computed at each time step
     */
    virtual void filter(const observation_sized_vector &data, const func_vec& fs = func_vec() ) = 0;


    /**
     * @brief virtual destructor
     */
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

    /* expose observation-sized vector type to users  */
    using observation_sized_vector = Eigen::Matrix<float_t,dimy,1>;

    /* expose state-sized vector type to users */
    using state_sized_vector = Eigen::Matrix<float_t,dimx,1>;

    /* a pair of paths (xts first, yts second) */
    using aPair = std::pair<std::vector<state_sized_vector>, std::vector<observation_sized_vector> >;


    /**
     * @brief simulates once forward through time from p(x_{1:T}, y_{1:T} | theta)
     */
    aPair sim_forward(unsigned int T);


    /**
     * @brief samples from the first time's state distribution
     * @return a state-sized vector for the x1 sample
     */
    virtual state_sized_vector muSamp()                = 0;


    /**
     * @brief returns a sample from the latent Markov transition
     * @return a state-sized vector for the xt sample
     */
    virtual state_sized_vector fSamp (const state_sized_vector &xtm1) = 0;


    /**
     * @brief returns a sample for the observed series 
     * @return 
     */
    virtual observation_sized_vector gSamp (const state_sized_vector &xt)   = 0;
};


template<size_t dimx, size_t dimy, typename float_t>
auto ForwardMod<dimx,dimy,float_t>::sim_forward(unsigned int T) -> aPair {

    std::vector<state_sized_vector> xs;
    std::vector<observation_sized_vector> ys;

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


/**
 * @class FutureSimulator
 * @author t
 * @file pf_base.h
 * @brief inherit from this if, in addition to filtering, you want to simulate future trajectories from the current filtering distribution. 
 * This simulates two future trajectories for each particle you have--one for the state path, and one for the observation path. 
 */
template<size_t dimx, size_t dimy, typename float_t, size_t nparts>
class FutureSimulator {
public:

    /* expose observation-sized vector type to users  */
    using observation_sized_vector = Eigen::Matrix<float_t,dimy,1>;

    /* expose state-sized vector type to users */
    using state_sized_vector = Eigen::Matrix<float_t,dimx,1>;

    /* one time point's pair of of nparts states and nparts observations  */
    using timePair = std::pair<std::array<state_sized_vector, nparts>, std::array<observation_sized_vector, nparts> >;

    /* many path pairs. time, (state/obs), particle */
    using manyPairs = std::vector<timePair>;

    /* observation paths (time, particle) */
    using obsPaths = std::vector<std::array<observation_sized_vector, nparts> >;

    /* many state paths (time, particle) */
    using statePaths = std::vector<std::array<state_sized_vector, nparts> >;


    /**
     * @brief simulates future state and observations paths from p(x_{t+1:T},y_{t+1:T} | y_{1:t}, theta)
     * @param number of time steps into the future you want to simulate
     * @return one state path and one observation path for each state sample 
     */
    manyPairs sim_future(unsigned int num_time_steps);


    /**
     * @brief simulates future observation paths from p(y_{t+1:T} | y_{1:t}, theta)
     * @param number of time steps into the future you want to simulate
     * @return one observation path for each state sample 
     */
    obsPaths sim_future_obs(unsigned int num_time_steps);


    /**
     * @brief simulates future state paths from p(x_{t+1:T} | y_{1:t}, theta)
     * @param number of time steps into the future you want to simulate
     * @return one state path for each state sample
     */
    statePaths sim_future_states(unsigned int num_time_steps);


    /**
     * @brief gets the most recent unweighted samples, to be fed into sim_future()
     */
    virtual std::array<state_sized_vector,nparts> get_uwtd_samps() const = 0;


    /**
     * @brief returns a sample from the latent Markov transition
     * @return a state-sized vector for the xt sample
     */
    virtual state_sized_vector fSamp (const state_sized_vector &xtm1) = 0;


    /**
     * @brief returns a sample for the observed series 
     * @return 
     */
    virtual observation_sized_vector gSamp (const state_sized_vector &xt)   = 0;
};


template<size_t dimx, size_t dimy, typename float_t, size_t nparts>
auto FutureSimulator<dimx,dimy,float_t,nparts>::sim_future(unsigned int num_time_steps) -> manyPairs {

    // this gets returned
    manyPairs allFutures;

    // stuff that gets changed every time loop
    std::array<state_sized_vector, nparts> states;
    std::array<observation_sized_vector, nparts> observations;
    std::array<state_sized_vector,nparts> past_states;
    std::array<state_sized_vector, nparts> first_states = this->get_uwtd_samps();

    // iterate over time
    for(unsigned int i = 0; i < num_time_steps; ++i){

        // create content for a time period
        if(i == 0){ // use xt_samps

            for(size_t j = 0; j < nparts; ++j){
                states[j] = this->fSamp(first_states[j]);
                observations[j] = this->gSamp(states[j]);
            }

        }else{ // use past samples

            for(size_t j = 0; j < nparts; ++j){
                states[j] = this->fSamp(past_states[j]);
                observations[j] = this->gSamp(states[j]);
            }
        }

        // add time period content 
        allFutures.push_back(timePair(states, observations)); //TODO make sure deep copy
        past_states = states;
    }

    return allFutures; 
}


template<size_t dimx, size_t dimy, typename float_t, size_t nparts>
auto FutureSimulator<dimx,dimy,float_t,nparts>::sim_future_obs(unsigned int num_time_steps) -> obsPaths {

    // this gets returned
    obsPaths allFutures;

    // stuff that gets changed every time loop
    std::array<state_sized_vector, nparts> states;
    std::array<observation_sized_vector, nparts> observations;
    std::array<state_sized_vector,nparts> past_states;
    std::array<state_sized_vector, nparts> first_states = this->get_uwtd_samps();

    // iterate over time
    for(unsigned int i = 0; i < num_time_steps; ++i){

        // create content for a time period
        if(i == 0){ // use xt_samps

            for(size_t j = 0; j < nparts; ++j){
                states[j] = this->fSamp(first_states[j]);
                observations[j] = this->gSamp(states[j]);
            }

        }else{ // use past samples

            for(size_t j = 0; j < nparts; ++j){
                states[j] = this->fSamp(past_states[j]);
                observations[j] = this->gSamp(states[j]);
            }
        }

        // add time period content 
        allFutures.push_back(observations); //TODO make sure deep copy
        past_states = states;
    }

    return allFutures; 
}


template<size_t dimx, size_t dimy, typename float_t, size_t nparts>
auto FutureSimulator<dimx,dimy,float_t,nparts>::sim_future_states(unsigned int num_time_steps) -> statePaths {

    // this gets returned
    statePaths allFutures;

    // stuff that gets changed every time loop
    std::array<state_sized_vector, nparts> states;
    std::array<observation_sized_vector, nparts> observations;
    std::array<state_sized_vector,nparts> past_states;
    std::array<state_sized_vector, nparts> first_states = this->get_uwtd_samps();

    // iterate over time
    for(unsigned int i = 0; i < num_time_steps; ++i){

        // create content for a time period
        if(i == 0){ // use xt_samps

            for(size_t j = 0; j < nparts; ++j){
                states[j] = this->fSamp(first_states[j]);
                observations[j] = this->gSamp(states[j]);
            }

        }else{ // use past samples

            for(size_t j = 0; j < nparts; ++j){
                states[j] = this->fSamp(past_states[j]);
                observations[j] = this->gSamp(states[j]);
            }
        }

        // add time period content 
        allFutures.push_back(states); //TODO make sure deep copy
        past_states = states;
    }

    return allFutures; 
}


//! Abstract Base Class for all closed-form filters.
/**
 * @class cf_filter
 * @author taylor
 * @file cf_filters.h
 * @brief forces structure on the closed-form filters.
 */
template<size_t dimstate, size_t dimobs, typename float_t>
class cf_filter{

public:
    
    /* expose observation-sized vector type to users  */
    using observation_sized_vector = Eigen::Matrix<float_t,dimobs,1>;

    /* expose state-sized vector type to users */
    using state_sized_vector = Eigen::Matrix<float_t,dimstate,1>;
    
    /**
     * @brief The (virtual) destructor.
     */
    virtual ~cf_filter();
    
    
    /**
     * @brief returns the log of the most recent conditional likelihood
     * @return log p(y_t | y_{1:t-1}) or log p(y_1)
     */
    virtual float_t getLogCondLike() const = 0;
};


template<size_t dimstate, size_t dimobs, typename float_t>
cf_filter<dimstate,dimobs,float_t>::~cf_filter() {}


} // namespace bases
} // namespace pf
#endif // PF_BASE_H
