#ifndef PF_BASE_H
#define PF_BASE_H

#include <map>
#include <string>
#include <vector>

#ifdef DROPPINGTHISINRPACKAGE
#include <RcppEigen.h>
// [[Rcpp::depends(RcppEigen)]]
#else
#include <Eigen/Dense>
#endif

namespace pf {

namespace bases {

/************************************************************************************************************/

/**
 * @author t
 * @file pf_base.h
 * @brief All non Rao-Blackwellized particle filters without covariates inherit
 * from this.
 * @tparam float_t (e.g. double, float, etc.)
 * @tparam dimobs the dimension of each observation
 * @tparam dimstate the dimension of each state
 */
template <typename float_t, size_t dimobs, size_t dimstate> class pf_base {
public:
  /* expose float type to users of this ABCTP */
  using float_type = float_t;

  /* expose observation-sized vector type to users  */
  using obs_sized_vec = Eigen::Matrix<float_t, dimobs, 1>;

  /* expose state-sized vector type to users */
  using state_sized_vec = Eigen::Matrix<float_t, dimstate, 1>;

  /* expose state-sized vector to users */
  using dynamic_matrix = Eigen::Matrix<float_t, Eigen::Dynamic, Eigen::Dynamic>;

  /* a function  */
  using func = std::function<const dynamic_matrix(const state_sized_vec &)>;

  /* functions  */
  using func_vec = std::vector<func>;

  /* the dimension of each observation vector (allows indirect access to
   * template parameters)*/
  static constexpr unsigned int dim_obs = dimobs;

  /* the dimension of the state vector (allows indirect access to template
   * parameters)*/
  static constexpr unsigned int dim_state = dimstate;

  /**
   * @brief the filtering function that must be defined
   * @param data the most recent observation
   * @param filter functions whose expected value approx. is computed at each
   * time step
   */
  virtual void filter(const obs_sized_vec &data,
                      const func_vec &fs = func_vec()) = 0;

  /**
   * @brief the getter method that must be defined (for conditional
   * log-likelihood)
   * @return log p(y_t | y_{1:t-1}) approximation
   */
  virtual float_t getLogCondLike() const = 0;

  /**
   * @brief virtual destructor
   */
  virtual ~pf_base(){};
};

/************************************************************************************************************/

/**
 * @author t
 * @file pf_base.h
 * @brief All non Rao-Blackwellized particle filters with covariates inherit
 * from this.
 * @tparam float_t (e.g. double, float, etc.)
 * @tparam dimobs the dimension of each observation
 * @tparam dimstate the dimension of each state
 * @tparam dimcov
 */
template <typename float_t, size_t dimobs, size_t dimstate, size_t dimcov>
class pf_withcov_base {
public:
  /* expose float type to users of this ABCTP */
  using float_type = float_t;

  /* expose observation-sized vector type to users  */
  using obs_sized_vec = Eigen::Matrix<float_t, dimobs, 1>;

  /* expose state-sized vector type to users */
  using state_sized_vec = Eigen::Matrix<float_t, dimstate, 1>;

  /** expose covariate-sized vector type to users*/
  using cov_sized_vec = Eigen::Matrix<float_t, dimcov, 1>;

  /* expose state-sized vector to users */
  using dynamic_matrix = Eigen::Matrix<float_t, Eigen::Dynamic, Eigen::Dynamic>;

  /* a function  */
  using func = std::function<const dynamic_matrix(const state_sized_vec &,
                                                  const cov_sized_vec &)>;

  /* functions  */
  using func_vec = std::vector<func>;

  /* the dimension of each observation vector (allows indirect access to
   * template parameters)*/
  static constexpr unsigned int dim_obs = dimobs;

  /* the dimension of the state vector (allows indirect access to template
   * parameters)*/
  static constexpr unsigned int dim_state = dimstate;

  /**
   * @brief the filtering function that must be defined
   * @param data the most recent observation
   * @param filter functions whose expected value approx. is computed at each
   * time step
   */
  virtual void filter(const obs_sized_vec &data, const cov_sized_vec &cov,
                      const func_vec &fs = func_vec()) = 0;

  /**
   * @brief the getter method that must be defined (for conditional
   * log-likelihood)
   * @return log p(y_t | y_{1:t-1}) approximation
   */
  virtual float_t getLogCondLike() const = 0;

  /**
   * @brief virtual destructor
   */
  virtual ~pf_withcov_base(){};
};

/************************************************************************************************************/

/**
 * @author t
 * @file pf_base.h
 * @brief All Rao-Blackwellized particle filters inherit from this.
 * @tparam float_t (e.g. double, float, etc.)
 * @tparam dim_s_state the dimension of the state vector that's sampled
 * @tparam dim_ns_state the dimension of the state vector that isn't sampled
 * @tparam dimobs the dimension of each observation vector
 */
template <typename float_t, size_t dim_s_state, size_t dim_ns_state,
          size_t dimobs>
class rbpf_base {
public:
  /* expose observation-sized vector type to users  */
  using obs_sized_vec = Eigen::Matrix<float_t, dimobs, 1>;

  /* expose sampled-state-size vector type to users */
  using sampled_state_sized_vec = Eigen::Matrix<float_t, dim_s_state, 1>;

  /* expose not-sampled-state-sized vector type to users */
  using not_sampled_state_sized_vec = Eigen::Matrix<float_t, dim_ns_state, 1>;

  /* expose state-sized vector to users */
  using dynamic_matrix = Eigen::Matrix<float_t, Eigen::Dynamic, Eigen::Dynamic>;

  /* a function */
  using func = std::function<const dynamic_matrix(
      const not_sampled_state_sized_vec &, const sampled_state_sized_vec &)>;

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
   * @param filter functions whose expected value approx. is computed at each
   * time step
   */
  virtual void filter(const obs_sized_vec &data,
                      const func_vec &fs = func_vec()) = 0;

  /**
   * @brief virtual destructor
   */
  virtual ~rbpf_base(){};
};

/************************************************************************************************************/

/**
 * @class ForwardMod
 * @author Taylor
 * @file pf_base.h
 * @brief inherit from this if you want to simulate from a homogeneous
 * forward/generative model.
 */
template <size_t dimx, size_t dimy, typename float_t> class ForwardMod {
private:
  /* expose observation-sized vector type to users (private because they clash
   * with pf_base) */
  using obs_sized_vec = Eigen::Matrix<float_t, dimy, 1>;

  /* expose state-sized vector type to users (private because they clash with
   * pf_base) */
  using state_sized_vec = Eigen::Matrix<float_t, dimx, 1>;

public:
  /* a pair of paths (xts first, yts second) */
  using aPair =
      std::pair<std::vector<state_sized_vec>, std::vector<obs_sized_vec>>;

  /**
   * @brief simulates once forward through time from p(x_{1:T}, y_{1:T} | theta)
   */
  aPair sim_forward(unsigned int T);

  /**
   * @brief samples from the first time's state distribution
   * @return a state-sized vector for the x1 sample
   */
  virtual state_sized_vec muSamp() = 0;

  /**
   * @brief returns a sample from the latent Markov transition
   * @param the previous time's state vector
   * @param the previous time's observation
   * @return a state-sized vector for the xt sample
   */
  virtual state_sized_vec fSamp(const state_sized_vec &xtm1) = 0;

  /**
   * @brief returns a sample for the observed series
   * @param the current time's state
   * @param the previous time's observation
   * @return a sample for the observation at this time step
   */
  virtual obs_sized_vec gSamp(const state_sized_vec &xt) = 0;
};

template <size_t dimx, size_t dimy, typename float_t>
auto ForwardMod<dimx, dimy, float_t>::sim_forward(unsigned int T) -> aPair {

  std::vector<state_sized_vec> xs;
  std::vector<obs_sized_vec> ys;

  // time 1
  xs.push_back(this->muSamp());
  ys.push_back(this->gSamp(xs[0]));

  // time > 1
  for (size_t i = 1; i < T; ++i) {

    auto xt = this->fSamp(xs[i - 1]);
    xs.push_back(xt);
    ys.push_back(this->gSamp(xt));
  }

  return aPair(xs, ys);
}

/************************************************************************************************************/

/**
 * @class GenForwardMod
 * @author Taylor
 * @file pf_base.h
 * @brief inherit from this if you want to simulate from a homogeneous
 * forward/generative model. This class is more general than the above because
 * it can simulate future states using past observations.
 */
template <size_t dimx, size_t dimy, typename float_t> class GenForwardMod {
private:
  /* expose observation-sized vector type to users (private because they clash
   * with pf_base) */
  using obs_sized_vec = Eigen::Matrix<float_t, dimy, 1>;

  /* expose state-sized vector type to users (private because they clash with
   * pf_base) */
  using state_sized_vec = Eigen::Matrix<float_t, dimx, 1>;

public:
  /* a pair of paths (xts first, yts second) */
  using aPair =
      std::pair<std::vector<state_sized_vec>, std::vector<obs_sized_vec>>;

  /**
   * @brief simulates once forward through time from p(x_{1:T}, y_{1:T} | theta)
   */
  aPair sim_forward(unsigned int T);

  /**
   * @brief samples from the first time's state distribution
   * @return a state-sized vector for the x1 sample
   */
  virtual state_sized_vec muSamp() = 0;

  /**
   * @brief returns a sample from the latent Markov transition
   * @param the previous time's state vector
   * @param the previous time's observation
   * @return a state-sized vector for the xt sample
   */
  virtual state_sized_vec fSamp(const state_sized_vec &xtm1,
                                const obs_sized_vec &ytm1) = 0;

  /**
   * @brief returns a sample for the observed series
   * @param the current time's state
   * @param the previous time's observation
   * @return a sample for the observation at this time step
   */
  virtual obs_sized_vec gSamp(const state_sized_vec &xt) = 0;
};

template <size_t dimx, size_t dimy, typename float_t>
auto GenForwardMod<dimx, dimy, float_t>::sim_forward(unsigned int T) -> aPair {

  std::vector<state_sized_vec> xs;
  std::vector<obs_sized_vec> ys;

  // time 1
  xs.push_back(this->muSamp());
  ys.push_back(this->gSamp(xs[0]));

  // time > 1
  for (size_t i = 1; i < T; ++i) {

    auto xt = this->fSamp(xs[i - 1], ys[i - 1]);
    xs.push_back(xt);
    ys.push_back(this->gSamp(xt));
  }

  return aPair(xs, ys);
}

/************************************************************************************************************/

/**
 * @class FutureSimulator
 * @author Taylor
 * @file pf_base.h
 * @brief inherit from this if, in addition to filtering, you want to simulate
 * future trajectories from the current filtering distribution. This simulates
 * two future trajectories for each particle you have--one for the state path,
 * and one for the observation path.
 */
template <size_t dimx, size_t dimy, typename float_t, size_t nparts>
class FutureSimulator {
private:
  /* expose observation-sized vector type to users  (private because they clash
   * with pf_base) */
  using obs_sized_vec = Eigen::Matrix<float_t, dimy, 1>;

  /* expose state-sized vector type to users */
  using state_sized_vec = Eigen::Matrix<float_t, dimx, 1>;

public:
  /* one time point's pair of of nparts states and nparts observations  */
  using timePair = std::pair<std::array<state_sized_vec, nparts>,
                             std::array<obs_sized_vec, nparts>>;

  /* many path pairs. time, (state/obs), particle */
  using manyPairs = std::vector<timePair>;

  /* observation paths (time, particle) */
  using obsPaths = std::vector<std::array<obs_sized_vec, nparts>>;

  /* many state paths (time, particle) */
  using statePaths = std::vector<std::array<state_sized_vec, nparts>>;

  /**
   * @brief simulates future state and observations paths from
   * p(x_{t+1:T},y_{t+1:T} | y_{1:t}, theta)
   * @param number of time steps into the future you want to simulate
   * @param the most recent observation that you have available
   * @return one state path and one observation path for each state sample
   */
  manyPairs sim_future(unsigned int num_time_steps);

  /**
   * @brief simulates future observation paths from p(y_{t+1:T} | y_{1:t},
   * theta)
   * @param number of time steps into the future you want to simulate
   * @param the most recent observation that you have available
   * @return one observation path for each state sample
   */
  obsPaths sim_future_obs(unsigned int num_time_steps);

  /**
   * @brief simulates future state paths from p(x_{t+1:T} | y_{1:t}, theta)
   * @param number of time steps into the future you want to simulate
   * @param the most recent observation that you have available
   * @return one state path for each state sample
   */
  statePaths sim_future_states(unsigned int num_time_steps);

  /**
   * @brief gets the most recent unweighted samples, to be fed into sim_future()
   */
  virtual std::array<state_sized_vec, nparts> get_uwtd_samps() const = 0;

  /**
   * @brief returns a sample from the latent Markov transition
   * @param the previous time's state value
   * @param the previous time's observed value
   * @return a state-sized vector for the xt sample
   */
  virtual state_sized_vec fSamp(const state_sized_vec &xtm1) = 0;

  /**
   * @brief returns a sample for the observed series
   * @param the current time step's state value
   * @return an observation sample for the current time step
   */
  virtual obs_sized_vec gSamp(const state_sized_vec &xt) = 0;
};

template <size_t dimx, size_t dimy, typename float_t, size_t nparts>
auto FutureSimulator<dimx, dimy, float_t, nparts>::sim_future(
    unsigned int num_time_steps) -> manyPairs {

  // this gets returned
  manyPairs allFutures;

  // stuff that gets changed every time loop
  std::array<state_sized_vec, nparts> states;
  std::array<obs_sized_vec, nparts> observations;
  std::array<state_sized_vec, nparts> past_states;
  std::array<state_sized_vec, nparts> first_states = this->get_uwtd_samps();

  // iterate over time
  for (unsigned int time = 0; time < num_time_steps; ++time) {

    // use particle filter's most recent information if you're going one step
    // ahead into the future otherwise use output generated in a previous
    // iteration of this time loop
    if (time == 0) {
      past_states = first_states;
    }

    // go particle by particle
    for (size_t j = 0; j < nparts; ++j) {
      states[j] = this->fSamp(past_states[j]);
      observations[j] = this->gSamp(states[j]);
    }

    // add time period content
    allFutures.push_back(
        timePair(states, observations)); // TODO make sure deep copy
    past_states = states;
  }

  return allFutures;
}

template <size_t dimx, size_t dimy, typename float_t, size_t nparts>
auto FutureSimulator<dimx, dimy, float_t, nparts>::sim_future_obs(
    unsigned int num_time_steps) -> obsPaths {

  // this gets returned
  manyPairs obs_and_state_paths = this->sim_future(num_time_steps);
  obsPaths future_obs;
  for (size_t time = 0; time < obs_and_state_paths.size(); ++time) {
    future_obs.push_back(std::get<1>(obs_and_state_paths[time]));
  }

  return future_obs;
}

template <size_t dimx, size_t dimy, typename float_t, size_t nparts>
auto FutureSimulator<dimx, dimy, float_t, nparts>::sim_future_states(
    unsigned int num_time_steps) -> statePaths {

  // this gets returned
  statePaths future_states;
  manyPairs obs_and_state_paths = sim_future(num_time_steps);
  for (size_t time = 0; time < obs_and_state_paths.size(); ++time) {
    future_states.push_back(std::get<0>(obs_and_state_paths[time]));
  }

  return future_states;
}

/************************************************************************************************************/

/**
 * @class GenFutureSimulator
 * @author Taylor
 * @file pf_base.h
 * @brief inherit from this if, in addition to filtering, you want to simulate
 * future trajectories from the current filtering distribution. This simulates
 * two future trajectories for each particle you have--one for the state path,
 * and one for the observation path. Unlike the above class, this one
 */
template <size_t dimx, size_t dimy, typename float_t, size_t nparts>
class GenFutureSimulator {
private:
  /* expose observation-sized vector type to users  (private because they clash
   * with pf_base) */
  using obs_sized_vec = Eigen::Matrix<float_t, dimy, 1>;

  /* expose state-sized vector type to users */
  using state_sized_vec = Eigen::Matrix<float_t, dimx, 1>;

public:
  /* one time point's pair of of nparts states and nparts observations  */
  using timePair = std::pair<std::array<state_sized_vec, nparts>,
                             std::array<obs_sized_vec, nparts>>;

  /* many path pairs. time, (state/obs), particle */
  using manyPairs = std::vector<timePair>;

  /* observation paths (time, particle) */
  using obsPaths = std::vector<std::array<obs_sized_vec, nparts>>;

  /* many state paths (time, particle) */
  using statePaths = std::vector<std::array<state_sized_vec, nparts>>;

  /**
   * @brief simulates future state and observations paths from
   * p(x_{t+1:T},y_{t+1:T} | y_{1:t}, theta)
   * @param number of time steps into the future you want to simulate
   * @param the most recent observation that you have available
   * @return one state path and one observation path for each state sample
   */
  manyPairs sim_future(unsigned int num_time_steps, const obs_sized_vec &yt);

  /**
   * @brief simulates future observation paths from p(y_{t+1:T} | y_{1:t},
   * theta)
   * @param number of time steps into the future you want to simulate
   * @param the most recent observation that you have available
   * @return one observation path for each state sample
   */
  obsPaths sim_future_obs(unsigned int num_time_steps, const obs_sized_vec &yt);

  /**
   * @brief simulates future state paths from p(x_{t+1:T} | y_{1:t}, theta)
   * @param number of time steps into the future you want to simulate
   * @param the most recent observation that you have available
   * @return one state path for each state sample
   */
  statePaths sim_future_states(unsigned int num_time_steps,
                               const obs_sized_vec &yt);

  /**
   * @brief gets the most recent unweighted samples, to be fed into sim_future()
   */
  virtual std::array<state_sized_vec, nparts> get_uwtd_samps() const = 0;

  /**
   * @brief returns a sample from the latent Markov transition
   * @param the previous time's state value
   * @param the previous time's observed value
   * @return a state-sized vector for the xt sample
   */
  virtual state_sized_vec fSamp(const state_sized_vec &xtm1,
                                const obs_sized_vec &ytm1) = 0;

  /**
   * @brief returns a sample for the observed series
   * @param the current time step's state value
   * @return an observation sample for the current time step
   */
  virtual obs_sized_vec gSamp(const state_sized_vec &xt) = 0;
};

template <size_t dimx, size_t dimy, typename float_t, size_t nparts>
auto GenFutureSimulator<dimx, dimy, float_t, nparts>::sim_future(
    unsigned int num_time_steps, const obs_sized_vec &yt) -> manyPairs {

  // this gets returned
  manyPairs allFutures;

  // stuff that gets changed every time loop
  std::array<state_sized_vec, nparts> states;
  std::array<obs_sized_vec, nparts> observations;
  std::array<state_sized_vec, nparts> past_states;
  std::array<obs_sized_vec, nparts> past_obs;
  std::array<state_sized_vec, nparts> first_states = this->get_uwtd_samps();

  // iterate over time
  for (unsigned int time = 0; time < num_time_steps; ++time) {

    // use particle filter's most recent information if you're going one step
    // ahead into the future otherwise use output generated in a previous
    // iteration of this time loop
    if (time == 0) {
      past_states = first_states;
      past_obs.fill(yt);
    }

    // go particle by particle
    for (size_t j = 0; j < nparts; ++j) {
      states[j] = this->fSamp(past_states[j], past_obs[j]);
      observations[j] = this->gSamp(states[j]);
    }

    // add time period content
    allFutures.push_back(
        timePair(states, observations)); // TODO make sure deep copy
    past_states = states;
    past_obs = observations;
  }

  return allFutures;
}

template <size_t dimx, size_t dimy, typename float_t, size_t nparts>
auto GenFutureSimulator<dimx, dimy, float_t, nparts>::sim_future_obs(
    unsigned int num_time_steps, const obs_sized_vec &yt) -> obsPaths {

  // this gets returned
  manyPairs obs_and_state_paths = this->sim_future(num_time_steps, yt);
  obsPaths future_obs;
  for (size_t time = 0; time < obs_and_state_paths.size(); ++time) {
    future_obs.push_back(std::get<1>(obs_and_state_paths[time]));
  }

  return future_obs;
}

template <size_t dimx, size_t dimy, typename float_t, size_t nparts>
auto GenFutureSimulator<dimx, dimy, float_t, nparts>::sim_future_states(
    unsigned int num_time_steps, const obs_sized_vec &yt) -> statePaths {

  // this gets returned
  statePaths future_states;
  manyPairs obs_and_state_paths = sim_future(num_time_steps, yt);
  for (size_t time = 0; time < obs_and_state_paths.size(); ++time) {
    future_states.push_back(std::get<0>(obs_and_state_paths[time]));
  }

  return future_states;
}

/************************************************************************************************************/

//! Abstract Base Class for all closed-form filters.
/**
 * @class cf_filter
 * @author taylor
 * @file cf_filters.h
 * @brief forces structure on the closed-form filters.
 */
template <size_t dimstate, size_t dimobs, typename float_t> class cf_filter {

public:
  /* expose observation-sized vector type to users  */
  using obs_sized_vec = Eigen::Matrix<float_t, dimobs, 1>;

  /* expose state-sized vector type to users */
  using state_sized_vec = Eigen::Matrix<float_t, dimstate, 1>;

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

template <size_t dimstate, size_t dimobs, typename float_t>
cf_filter<dimstate, dimobs, float_t>::~cf_filter() {}

/************************************************************************************************************/

/**
 * @author taylor
 * @file pf_base_crn.h
 * @brief All particle filters that use common random numbers must inherit from
 * this.
 * @tparam float_t (e.g. double, float, etc.)
 * @tparam dimobs the dimension of each observation
 * @tparam dimstate the dimension of each state
 * @tparam dimu the dimension of the common random numbers used for sampling
 * from state proposal
 * @tparam dimur the dimension of the common random numbers used for resampling
 * at a given time point (e.g. 1)
 * @tparam numparts the number of particles for the state samples at each time
 * point
 */
template <typename float_t, size_t dimobs, size_t dimstate, size_t dimu,
          size_t dimur, size_t numparts>
class pf_base_crn {
public:
  /* expose float type to users of this ABCTP */
  using float_type = float_t;

  /* expose observation-sized vector type to users  */
  using obs_sized_vec = Eigen::Matrix<float_t, dimobs, 1>;

  /* expose state-sized vector type to users */
  using state_sized_vec = Eigen::Matrix<float_t, dimstate, 1>;

  /* expose state-sized vector to users */
  using dynamic_matrix = Eigen::Matrix<float_t, Eigen::Dynamic, Eigen::Dynamic>;

  /* a function  */
  using func = std::function<const dynamic_matrix(const state_sized_vec &)>;

  /* functions  */
  using func_vec = std::vector<func>;

  /* type for common random numbers used for sampling state proposals*/
  using usv = Eigen::Matrix<float_t, dimu, 1>;

  /* type for common random numbers used for resampling at a given time point */
  using usvr = Eigen::Matrix<float_t, dimur, 1>;

  /* the dimension of each observation vector (allows indirect access to
   * template parameters)*/
  static constexpr unsigned int dim_obs = dimobs;

  /* the dimension of the state vector (allows indirect access to template
   * parameters)*/
  static constexpr unsigned int dim_state = dimstate;

  /**
   * @brief the filtering function that must be defined
   * @param data the most recent observation
   * @param filter functions whose expected value approx. is computed at each
   * time step
   */
  virtual void filter(const obs_sized_vec &data,
                      const std::array<usv, numparts> &Us, const usvr &Uresamp,
                      const func_vec &fs = func_vec()) = 0;

  /**
   * @brief the getter method that must be defined (for conditional
   * log-likelihood)
   * @return log p(y_t | y_{1:t-1}) approximation
   */
  virtual float_t getLogCondLike() const = 0;

  /**
   * @brief virtual destructor
   */
  virtual ~pf_base_crn(){};
};

} // namespace bases
} // namespace pf
#endif // PF_BASE_H
