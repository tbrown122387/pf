#ifndef RBPF_H
#define RBPF_H

#include <array>
#include <functional>
#include <vector>

#ifdef DROPPINGTHISINRPACKAGE
#include <RcppEigen.h>
// [[Rcpp::depends(RcppEigen)]]
#else
#include <Eigen/Dense>
#endif

#include <algorithm> // std::fill

#include "cf_filters.h" // for closed form filter objects
#include "pf_base.h"

namespace pf {

namespace filters {

//! Rao-Blackwellized/Marginal Particle Filter with inner HMMs
/**
 * @class rbpf_hmm
 * @author t
 * @file rbpf.h
 * @brief Rao-Blackwellized/Marginal Particle Filter with inner HMMs
 * @tparam nparts the number of particles
 * @tparam dimnss dimension of "not sampled state"
 * @tparam dimss dimension of "sampled state"
 * @tparam dimy the dimension of the observations
 * @tparam resamp_t the resampler type (e.g. multinomial, etc.)
 */
template <size_t nparts, size_t dimnss, size_t dimss, size_t dimy,
          typename resamp_t, typename float_t, bool debug = false>
class rbpf_hmm : public bases::rbpf_base<float_t, dimss, dimnss, dimy> {
public:
  /** "sampled state size vector" */
  using sssv = Eigen::Matrix<float_t, dimss, 1>;
  /** "not sampled state size vector" */
  using nsssv = Eigen::Matrix<float_t, dimnss, 1>;
  /** "observation size vector" */
  using osv = Eigen::Matrix<float_t, dimy, 1>;
  /** "not sampled state size matrix" */
  using nsssMat = Eigen::Matrix<float_t, dimnss, dimnss>;
  /** Dynamic size matrix*/
  using Mat = Eigen::Matrix<float_t, Eigen::Dynamic, Eigen::Dynamic>;
  /** array of samples */
  using arrayVec = std::array<sssv, nparts>;
  /** array of weights */
  using arrayfloat_t = std::array<float_t, nparts>;
  /** closed-form model type */
  using cfModType = hmm<dimnss, dimy, float_t, debug>;
  /** array of model objects */
  using arrayMod = std::array<cfModType, nparts>;

  //! The constructor.
  /**
   * @brief constructor.
   * @param resamp_sched how often to resample (e.g. once every resamp_sched
   * time periods)
   */
  rbpf_hmm(const unsigned int &resamp_sched = 1);

  /**
   * @brief The (virtual) destructor.
   */
  virtual ~rbpf_hmm();

  //! Filter.
  /**
   * @brief filters everything based on a new data point.
   * @param data the most recent time series observation.
   * @param fs a vector of functions computing E[h(x_1t, x_2t^i)| x_2t^i,y_1:t]
   * to be averaged to yield E[h(x_1t, x_2t)|,y_1:t]. Will access the
   * probability vector of x_1t
   */
  void filter(
      const osv &data,
      const std::vector<
          std::function<const Mat(const nsssv &x1tLogProbs, const sssv &x2t)>>
          &fs = std::vector<std::function<const Mat(
              const nsssv &, const sssv &)>>()); //, const
                                                 //std::vector<std::function<const
                                                 //Mat(const Vec&)> >& fs);

  //! Get the latest conditional likelihood.
  /**
   * @brief Get the latest conditional likelihood.
   * @return the latest conditional likelihood.
   */
  float_t getLogCondLike() const;

  //!
  /**
   * @brief Get vector of expectations.
   * @return vector of expectations
   */
  std::vector<Mat> getExpectations() const;

  //! Evaluates the first time state density.
  /**
   * @brief evaluates mu.
   * @param x21 component two at time 1
   * @return a float_t evaluation
   */
  virtual float_t logMuEv(const sssv &x21) = 0;

  //! Sample from the first sampler.
  /**
   * @brief samples the second component of the state at time 1.
   * @param y1 most recent datum.
   * @return a sssv sample for x21.
   */
  virtual sssv q1Samp(const osv &y1) = 0;

  //! Provides the initial mean vector for each HMM filter object.
  /**
   * @brief provides the initial probability vector for each HMM filter object.
   * @param x21 the second state componenent at time 1.
   * @return a Vec representing the probability of each state element.
   */
  virtual nsssv initHMMLogProbVec(const sssv &x21) = 0;

  //! Provides the transition matrix for each HMM filter object.
  /**
   * @brief provides the transition matrix for each HMM filter object.
   * @param x21 the second state component at time 1.
   * @return a transition matrix where element (ij) is the probability of
   * transitioning from state i to state j.
   */
  virtual nsssMat initHMMLogTransMat(const sssv &x21) = 0;

  //! Samples the time t second component.
  /**
   * @brief Samples the time t second component.
   * @param x2tm1 the previous time's second state component.
   * @param yt the current observation.
   * @return a Vec sample of the second state component at the current time.
   */
  virtual sssv qSamp(const sssv &x2tm1, const osv &yt) = 0;

  //! Evaluates the proposal density of the second state component at time 1.
  /**
   * @brief Evaluates the proposal density of the second state component at
   * time 1.
   * @param x21 the second state component at time 1 you sampled.
   * @param y1 time 1 observation.
   * @return a float_t evaluation of the density.
   */
  virtual float_t logQ1Ev(const sssv &x21, const osv &y1) = 0;

  //! Evaluates the state transition density for the second state component.
  /**
   * @brief Evaluates the state transition density for the second state
   * component.
   * @param x2t the current second state component.
   * @param x2tm1 the previous second state component.
   * @return a float_t evaluation.
   */
  virtual float_t logFEv(const sssv &x2t, const sssv &x2tm1) = 0;

  //! Evaluates the proposal density at time t > 1.
  /**
   * @brief Evaluates the proposal density at time t > 1.
   * @param x2t the current second state component.
   * @param x2tm1 the previous second state component.
   * @param yt the current time series observation.
   * @return a float_t evaluation.
   */
  virtual float_t logQEv(const sssv &x2t, const sssv &x2tm1, const osv &yt) = 0;

  //! How to update your inner HMM filter object at each time.
  /**
   * @brief How to update your inner HMM filter object at each time.
   * @param aModel a HMM filter object describing the conditional closed-form
   * model.
   * @param yt the current time series observation.
   * @param x2t the current second state component.
   */
  virtual void updateHMM(cfModType &aModel, const osv &yt, const sssv &x2t) = 0;

private:
  /** the current time period */
  unsigned int m_now;
  /** last conditional likelihood */
  float_t m_lastLogCondLike;
  /** resampling schedue */
  unsigned int m_rs;
  /** the array of inner closed-form models */
  arrayMod m_p_innerMods;
  /** the array of samples for the second state portion */
  arrayVec m_p_samps;
  /** the array of unnormalized log-weights */
  arrayfloat_t m_logUnNormWeights;
  /** the resampler object */
  resamp_t m_resampler;
  /** the vector of expectations */
  std::vector<Mat> m_expectations;
};

template <size_t nparts, size_t dimnss, size_t dimss, size_t dimy,
          typename resamp_t, typename float_t, bool debug>
rbpf_hmm<nparts, dimnss, dimss, dimy, resamp_t, float_t, debug>::rbpf_hmm(
    const unsigned int &resamp_sched)
    : m_now(0), m_lastLogCondLike(0.0), m_rs(resamp_sched) {
  std::fill(m_logUnNormWeights.begin(), m_logUnNormWeights.end(), 0.0);
}

template <size_t nparts, size_t dimnss, size_t dimss, size_t dimy,
          typename resamp_t, typename float_t, bool debug>
rbpf_hmm<nparts, dimnss, dimss, dimy, resamp_t, float_t, debug>::~rbpf_hmm() {}

template <size_t nparts, size_t dimnss, size_t dimss, size_t dimy,
          typename resamp_t, typename float_t, bool debug>
void rbpf_hmm<nparts, dimnss, dimss, dimy, resamp_t, float_t, debug>::filter(
    const osv &data,
    const std::vector<std::function<const Mat(const nsssv &x1tLogProbs,
                                              const sssv &x2t)>> &fs) {

  if (m_now > 0) { // m_now > 0

    // update
    sssv newX2Samp;
    float_t sumexpdenom(0.0);
    float_t m1(
        -std::numeric_limits<float_t>::infinity()); // for revised log weights
    float_t m2 =
        *std::max_element(m_logUnNormWeights.begin(), m_logUnNormWeights.end());
    for (size_t ii = 0; ii < nparts; ++ii) {

      newX2Samp = qSamp(m_p_samps[ii], data);
      updateHMM(m_p_innerMods[ii], data, newX2Samp);
      sumexpdenom += std::exp(m_logUnNormWeights[ii] - m2);

      m_logUnNormWeights[ii] += m_p_innerMods[ii].getLogCondLike() +
                                logFEv(newX2Samp, m_p_samps[ii]) -
                                logQEv(newX2Samp, m_p_samps[ii], data);

      // update a max
      if (m_logUnNormWeights[ii] > m1)
        m1 = m_logUnNormWeights[ii];

// print stuff if debug mode is on
#ifndef DROPPINGTHISINRPACKAGE
      if constexpr (debug)
        std::cout << "time: " << m_now
                  << ", transposed x2 sample: " << newX2Samp.transpose()
                  << ", log unnorm weight: " << m_logUnNormWeights[ii] << "\n";
#endif

      m_p_samps[ii] = newX2Samp;
    }

    // calculate log p(y_t | y_{1:t-1})
    float_t sumexpnumer(0.0);
    for (size_t p = 0; p < nparts; ++p)
      sumexpnumer += std::exp(m_logUnNormWeights[p] - m1);
    m_lastLogCondLike = m1 + std::log(sumexpnumer) - m2 - std::log(sumexpdenom);

    // calculate expectations before you resample
    unsigned int fId(0);
    // float_t m = *std::max_element(m_logUnNormWeights.begin(),
    // m_logUnNormWeights.end());
    for (auto &h : fs) {

      Mat testOutput = h(m_p_innerMods[0].getFilterVecLogProbs(), m_p_samps[0]);
      unsigned int rows = testOutput.rows();
      unsigned int cols = testOutput.cols();
      Mat numer = Mat::Zero(rows, cols);
      float_t denom(0.0);
      for (size_t prtcl = 0; prtcl < nparts; ++prtcl) {
        numer +=
            h(m_p_innerMods[prtcl].getFilterVecLogProbs(), m_p_samps[prtcl]) *
            std::exp(m_logUnNormWeights[prtcl] - m1);
        denom += std::exp(m_logUnNormWeights[prtcl] - m1);
      }
      m_expectations[fId] = numer / denom;

// print stuff if debug mode is on
#ifndef DROPPINGTHISINRPACKAGE
      if constexpr (debug)
        std::cout << "transposed expec " << fId << ": "
                  << m_expectations[fId].transpose() << "\n";
#endif

      fId++;
    }

    // resample (unnormalized weights ok)
    if ((m_now + 1) % m_rs == 0)
      m_resampler.resampLogWts(m_p_innerMods, m_p_samps, m_logUnNormWeights);

    // update time step
    m_now++;
  } else //( m_now == 0)
  {      // first data point coming

    // initialize and update the closed-form mods
    nsssv tmpLogProbs;
    nsssMat tmpLogTransMat;
    float_t m1(-std::numeric_limits<float_t>::infinity());
    for (size_t ii = 0; ii < nparts; ++ii) {

      m_p_samps[ii] = q1Samp(data);
      tmpLogProbs = initHMMLogProbVec(m_p_samps[ii]);
      tmpLogTransMat = initHMMLogTransMat(m_p_samps[ii]);
      m_p_innerMods[ii] = cfModType(tmpLogProbs, tmpLogTransMat);
      this->updateHMM(m_p_innerMods[ii], data, m_p_samps[ii]);
      m_logUnNormWeights[ii] = m_p_innerMods[ii].getLogCondLike() +
                               logMuEv(m_p_samps[ii]) -
                               logQ1Ev(m_p_samps[ii], data);

// print stuff if debug mode is on
#ifndef DROPPINGTHISINRPACKAGE
      if constexpr (debug)
        std::cout << "time: " << m_now
                  << ", transposed x2 sample: " << m_p_samps[ii].transpose()
                  << ", log unnorm weight: " << m_logUnNormWeights[ii] << "\n";
#endif

      // maximum to be used in likelihood calc
      if (m_logUnNormWeights[ii] > m1)
        m1 = m_logUnNormWeights[ii];
    }

    // calc log p(y1)
    float_t sumexp(0.0);
    for (size_t p = 0; p < nparts; ++p)
      sumexp += std::exp(m_logUnNormWeights[p] - m1);
    m_lastLogCondLike =
        m1 + std::log(sumexp) - std::log(static_cast<float_t>(nparts));

    // calculate expectations before you resample
    m_expectations.resize(fs.size());
    unsigned int fId(0);
    // float_t m = *std::max_element(m_logUnNormWeights.begin(),
    // m_logUnNormWeights.end());
    for (auto &h : fs) {

      Mat testOutput = h(m_p_innerMods[0].getFilterVecLogProbs(), m_p_samps[0]);
      unsigned int rows = testOutput.rows();
      unsigned int cols = testOutput.cols();
      Mat numer = Mat::Zero(rows, cols);
      float_t denom(0.0);
      for (size_t prtcl = 0; prtcl < nparts; ++prtcl) {
        numer +=
            h(m_p_innerMods[prtcl].getFilterVecLogProbs(), m_p_samps[prtcl]) *
            std::exp(m_logUnNormWeights[prtcl] - m1);
        denom += std::exp(m_logUnNormWeights[prtcl] - m1);
      }
      m_expectations[fId] = numer / denom;

// print stuff if debug mode is on
#ifndef DROPPINGTHISINRPACKAGE
      if constexpr (debug)
        std::cout << "transposed expec " << fId << ": "
                  << m_expectations[fId].transpose() << "\n";
#endif

      fId++;
    }

    // resample (unnormalized weights ok)
    if ((m_now + 1) % m_rs == 0)
      m_resampler.resampLogWts(m_p_innerMods, m_p_samps, m_logUnNormWeights);

    // advance time step
    m_now++;
  }
}

template <size_t nparts, size_t dimnss, size_t dimss, size_t dimy,
          typename resamp_t, typename float_t, bool debug>
float_t rbpf_hmm<nparts, dimnss, dimss, dimy, resamp_t, float_t,
                 debug>::getLogCondLike() const {
  return m_lastLogCondLike;
}

template <size_t nparts, size_t dimnss, size_t dimss, size_t dimy,
          typename resamp_t, typename float_t, bool debug>
auto rbpf_hmm<nparts, dimnss, dimss, dimy, resamp_t, float_t,
              debug>::getExpectations() const -> std::vector<Mat> {
  return m_expectations;
}

//! Rao-Blackwellized/Marginal Bootstrap Filter with inner HMMs
/**
 * @class rbpf_hmm_bs
 * @author t
 * @file rbpf.h
 * @brief Rao-Blackwellized/Marginal Bootstrap Filter with inner HMMs
 * @tparam nparts the number of particles
 * @tparam dimnss dimension of "not sampled state"
 * @tparam dimss dimension of "sampled state"
 * @tparam dimy the dimension of the observations
 * @tparam resamp_t the resampler type (e.g. multinomial, etc.)
 */
template <size_t nparts, size_t dimnss, size_t dimss, size_t dimy,
          typename resamp_t, typename float_t, bool debug = false>
class rbpf_hmm_bs : public bases::rbpf_base<float_t, dimss, dimnss, dimy> {
public:
  /** "sampled state size vector" */
  using sssv = Eigen::Matrix<float_t, dimss, 1>;
  /** "not sampled state size vector" */
  using nsssv = Eigen::Matrix<float_t, dimnss, 1>;
  /** "observation size vector" */
  using osv = Eigen::Matrix<float_t, dimy, 1>;
  /** "not sampled state size matrix" */
  using nsssMat = Eigen::Matrix<float_t, dimnss, dimnss>;
  /** Dynamic size matrix*/
  using Mat = Eigen::Matrix<float_t, Eigen::Dynamic, Eigen::Dynamic>;
  /** closed-form model type*/
  using cfModType = hmm<dimnss, dimy, float_t, debug>;
  /** array of model objects */
  using arrayMod = std::array<cfModType, nparts>;
  /** array of samples */
  using arrayVec = std::array<sssv, nparts>;
  /** array of weights */
  using arrayfloat_t = std::array<float_t, nparts>;

  //! The constructor.
  /**
   * @brief constructor.
   * @param resamp_sched how often to resample (e.g. once every resamp_sched
   * time periods)
   */
  rbpf_hmm_bs(const unsigned int &resamp_sched = 1);

  /**
   * @brief The (virtual) destructor.
   */
  virtual ~rbpf_hmm_bs();

  //! Filter.
  /**
   * @brief filters everything based on a new data point.
   * @param data the most recent time series observation.
   * @param fs a vector of functions computing E[h(x_1t, x_2t^i)| x_2t^i,y_1:t]
   * to be averaged to yield E[h(x_1t, x_2t)|,y_1:t]. Will access the
   * probability vector of x_1t
   */
  void filter(
      const osv &data,
      const std::vector<
          std::function<const Mat(const nsssv &x1tLogProbs, const sssv &x2t)>>
          &fs = std::vector<std::function<const Mat(
              const nsssv &, const sssv &)>>()); //, const
                                                 //std::vector<std::function<const
                                                 //Mat(const Vec&)> >& fs);

  //! Get the latest conditional likelihood.
  /**
   * @brief Get the latest conditional likelihood.
   * @return the latest conditional likelihood.
   */
  float_t getLogCondLike() const;

  //!
  /**
   * @brief Get vector of expectations.
   * @return vector of expectations
   */
  std::vector<Mat> getExpectations() const;

  //! Sample from the first sampler.
  /**
   * @brief samples the second component of the state at time 1.
   * @return a sssv sample for x21.
   */
  virtual sssv muSamp() = 0;

  //! Provides the initial log probability vector for each HMM filter object.
  /**
   * @brief provides the initial log probability vector for each HMM filter
   * object.
   * @param x21 the second state componenent at time 1.
   * @return a Vec representing the log probability of each state element.
   */
  virtual nsssv initHMMLogProbVec(const sssv &x21) = 0;

  //! Provides the log transition matrix for each HMM filter object.
  /**
   * @brief provides the log transition matrix for each HMM filter object.
   * @param x21 the second state component at time 1.
   * @return a (log) transition matrix where element (ij) is the log of the
   * probability of transitioning from state i to state j.
   */
  virtual nsssMat initHMMLogTransMat(const sssv &x21) = 0;

  //! Samples the time t second component.
  /**
   * @brief Samples the time t second component.
   * @param x2tm1 the previous time's second state component.
   * @return a sssv sample of the second state component at the current time.
   */
  virtual sssv fSamp(const sssv &x2tm1) = 0;

  //! How to update your inner HMM filter object at each time.
  /**
   * @brief How to update your inner HMM filter object at each time.
   * @param aModel a HMM filter object describing the conditional closed-form
   * model.
   * @param yt the current time series observation.
   * @param x2t the current second state component.
   */
  virtual void updateHMM(cfModType &aModel, const osv &yt, const sssv &x2t) = 0;

private:
  /** the current time period */
  unsigned int m_now;
  /** last conditional likelihood */
  float_t m_lastLogCondLike;
  /** resampling schedue */
  unsigned int m_rs;
  /** the array of inner closed-form models */
  arrayMod m_p_innerMods;
  /** the array of samples for the second state portion */
  arrayVec m_p_samps;
  /** the array of unnormalized log-weights */
  arrayfloat_t m_logUnNormWeights;
  /** the resampler object */
  resamp_t m_resampler;
  /** the vector of expectations */
  std::vector<Mat> m_expectations;
};

template <size_t nparts, size_t dimnss, size_t dimss, size_t dimy,
          typename resamp_t, typename float_t, bool debug>
rbpf_hmm_bs<nparts, dimnss, dimss, dimy, resamp_t, float_t, debug>::rbpf_hmm_bs(
    const unsigned int &resamp_sched)
    : m_now(0), m_lastLogCondLike(0.0), m_rs(resamp_sched) {
  std::fill(m_logUnNormWeights.begin(), m_logUnNormWeights.end(), 0.0);
}

template <size_t nparts, size_t dimnss, size_t dimss, size_t dimy,
          typename resamp_t, typename float_t, bool debug>
rbpf_hmm_bs<nparts, dimnss, dimss, dimy, resamp_t, float_t,
            debug>::~rbpf_hmm_bs() {}

template <size_t nparts, size_t dimnss, size_t dimss, size_t dimy,
          typename resamp_t, typename float_t, bool debug>
void rbpf_hmm_bs<nparts, dimnss, dimss, dimy, resamp_t, float_t, debug>::filter(
    const osv &data,
    const std::vector<std::function<const Mat(const nsssv &x1tLogProbs,
                                              const sssv &x2t)>> &fs) {

  if (m_now > 0) {
    // update
    sssv newX2Samp;
    float_t sumexpdenom(0.0);
    float_t m1(
        -std::numeric_limits<float_t>::infinity()); // for revised log weights
    float_t m2 =
        *std::max_element(m_logUnNormWeights.begin(), m_logUnNormWeights.end());
    for (size_t ii = 0; ii < nparts; ++ii) {

      newX2Samp = fSamp(m_p_samps[ii]);
      updateHMM(m_p_innerMods[ii], data, newX2Samp);
      sumexpdenom += std::exp(m_logUnNormWeights[ii] - m2);

      m_logUnNormWeights[ii] += m_p_innerMods[ii].getLogCondLike();

// print stuff if debug mode is on
#ifndef DROPPINGTHISINRPACKAGE
      if constexpr (debug)
        std::cout << "time: " << m_now
                  << ", transposed x2 sample: " << newX2Samp.transpose()
                  << ", log unnorm weight: " << m_logUnNormWeights[ii] << "\n";
#endif

      // update a max
      if (m_logUnNormWeights[ii] > m1)
        m1 = m_logUnNormWeights[ii];

      m_p_samps[ii] = newX2Samp;
    }

    // calculate log p(y_t | y_{1:t-1})
    float_t sumexpnumer(0.0);
    for (size_t p = 0; p < nparts; ++p)
      sumexpnumer += std::exp(m_logUnNormWeights[p] - m1);
    m_lastLogCondLike = m1 + std::log(sumexpnumer) - m2 - std::log(sumexpdenom);

    // calculate expectations before you resample
    unsigned int fId(0);
    // float_t m = *std::max_element(m_logUnNormWeights.begin(),
    // m_logUnNormWeights.end());
    for (auto &h : fs) {

      Mat testOutput = h(m_p_innerMods[0].getFilterVecLogProbs(), m_p_samps[0]);
      unsigned int rows = testOutput.rows();
      unsigned int cols = testOutput.cols();
      Mat numer = Mat::Zero(rows, cols);
      float_t denom(0.0);
      for (size_t prtcl = 0; prtcl < nparts; ++prtcl) {
        numer +=
            h(m_p_innerMods[prtcl].getFilterVecLogProbs(), m_p_samps[prtcl]) *
            std::exp(m_logUnNormWeights[prtcl] - m1);
        denom += std::exp(m_logUnNormWeights[prtcl] - m1);
      }
      m_expectations[fId] = numer / denom;

// print stuff if debug mode is on
#ifndef DROPPINGTHISINRPACKAGE
      if constexpr (debug)
        std::cout << "transposed expec " << fId << ": "
                  << m_expectations[fId].transpose() << "\n";
#endif

      fId++;
    }

    // resample (unnormalized weights ok)
    if ((m_now + 1) % m_rs == 0)
      m_resampler.resampLogWts(m_p_innerMods, m_p_samps, m_logUnNormWeights);

    // update time step
    m_now++;
  } else // ( m_now == 0) // first data point coming
  {
    // initialize and update the closed-form mods
    nsssv tmpLogProbs;
    nsssMat tmpLogTransMat;
    float_t m1(-std::numeric_limits<float_t>::infinity());
    for (size_t ii = 0; ii < nparts; ++ii) {

      m_p_samps[ii] = muSamp();
      tmpLogProbs = initHMMLogProbVec(m_p_samps[ii]);
      tmpLogTransMat = initHMMLogTransMat(m_p_samps[ii]);
      m_p_innerMods[ii] = cfModType(tmpLogProbs, tmpLogTransMat);
      this->updateHMM(m_p_innerMods[ii], data, m_p_samps[ii]);
      m_logUnNormWeights[ii] = m_p_innerMods[ii].getLogCondLike();

// print stuff if debug mode is on
#ifndef DROPPINGTHISINRPACKAGE
      if constexpr (debug)
        std::cout << "time: " << m_now
                  << ", transposed x2 sample: " << m_p_samps[ii].transpose()
                  << ", log unnorm weight: " << m_logUnNormWeights[ii] << "\n";
#endif

      // maximum to be used in likelihood calc
      if (m_logUnNormWeights[ii] > m1)
        m1 = m_logUnNormWeights[ii];
    }

    // calc log p(y1)
    float_t sumexp(0.0);
    for (size_t p = 0; p < nparts; ++p)
      sumexp += std::exp(m_logUnNormWeights[p] - m1);
    m_lastLogCondLike =
        m1 + std::log(sumexp) - std::log(static_cast<float_t>(nparts));

    // calculate expectations before you resample
    m_expectations.resize(fs.size());
    unsigned int fId(0);
    // float_t m = *std::max_element(m_logUnNormWeights.begin(),
    // m_logUnNormWeights.end()); /// TODO: can we just use m1?
    for (auto &h : fs) {

      Mat testOutput = h(m_p_innerMods[0].getFilterVecLogProbs(), m_p_samps[0]);
      unsigned int rows = testOutput.rows();
      unsigned int cols = testOutput.cols();
      Mat numer = Mat::Zero(rows, cols);
      float_t denom(0.0);
      for (size_t prtcl = 0; prtcl < nparts; ++prtcl) {
        numer +=
            h(m_p_innerMods[prtcl].getFilterVecLogProbs(), m_p_samps[prtcl]) *
            std::exp(m_logUnNormWeights[prtcl] - m1);
        denom += std::exp(m_logUnNormWeights[prtcl] - m1);
      }
      m_expectations[fId] = numer / denom;

// print stuff if debug mode is on
#ifndef DROPPINGTHISINRPACKAGE
      if constexpr (debug)
        std::cout << "transposed expec " << fId << ": "
                  << m_expectations[fId].transpose() << "\n";
#endif

      fId++;
    }

    // resample (unnormalized weights ok)
    if ((m_now + 1) % m_rs == 0)
      m_resampler.resampLogWts(m_p_innerMods, m_p_samps, m_logUnNormWeights);

    // advance time step
    m_now++;
  }
}

template <size_t nparts, size_t dimnss, size_t dimss, size_t dimy,
          typename resamp_t, typename float_t, bool debug>
float_t rbpf_hmm_bs<nparts, dimnss, dimss, dimy, resamp_t, float_t,
                    debug>::getLogCondLike() const {
  return m_lastLogCondLike;
}

template <size_t nparts, size_t dimnss, size_t dimss, size_t dimy,
          typename resamp_t, typename float_t, bool debug>
auto rbpf_hmm_bs<nparts, dimnss, dimss, dimy, resamp_t, float_t,
                 debug>::getExpectations() const -> std::vector<Mat> {
  return m_expectations;
}

//! Rao-Blackwellized/Marginal Particle Filter with inner Kalman Filter objectss
/**
 * @class rbpf_kalman
 * @author t
 * @file rbpf.h
 * @brief Rao-Blackwellized/Marginal Particle Filter with inner Kalman Filter
 * objectss
 * @tparam nparts the number of particles
 * @tparam dimnss dimension of not-sampled-state vector
 * @tparam dimss dimension of sampled-state vector
 * @tparam dimy the dimension of the observations
 * @tparam resamp_t the resampler type
 */
template <size_t nparts, size_t dimnss, size_t dimss, size_t dimy,
          typename resamp_t, typename float_t, bool debug = false>
class rbpf_kalman : public bases::rbpf_base<float_t, dimss, dimnss, dimy> {

public:
  /** "sampled state size vector" */
  using sssv = Eigen::Matrix<float_t, dimss, 1>;
  /** "not sampled state size vector" */
  using nsssv = Eigen::Matrix<float_t, dimnss, 1>;
  /** "observation size vector" */
  using osv = Eigen::Matrix<float_t, dimy, 1>;
  /** dynamic size matrices */
  using Mat = Eigen::Matrix<float_t, Eigen::Dynamic, Eigen::Dynamic>;
  /** "not sampled state size matrix" */
  using nsssMat = Eigen::Matrix<float_t, dimnss, dimnss>;
  /** closed-form model type*/
  using cfModType = kalman<dimnss, dimy, 0, float_t, debug>;
  /** array of model objects */
  using arrayMod = std::array<cfModType, nparts>;
  /** array of samples */
  using arrayVec = std::array<sssv, nparts>;
  /** array of weights */
  using arrayfloat_t = std::array<float_t, nparts>;

  //! The constructor.
  /**
   \param resamp_sched how often you want to resample (e.g once every
   resamp_sched time points)
   */
  rbpf_kalman(const unsigned int &resamp_sched = 1);

  /**
   * @brief
   */
  virtual ~rbpf_kalman();

  //! Filter!
  /**
   * \brief The workhorse function
   * \param data the most recent observable portion of the time series.
   * \param fs a vector of functions computing E[h(x_1t, x_2t^i)| x_2t^i,y_1:t].
   * to be averaged to yield E[h(x_1t, x_2t)|,y_1:t]
   */
  void filter(
      const osv &data,
      const std::vector<
          std::function<const Mat(const nsssv &x1t, const sssv &x2t)>> &fs =
          std::vector<
              std::function<const Mat(const nsssv &x1t, const sssv &x2t)>>());

  //! Get the latest log conditional likelihood.
  /**
   * \return the latest log conditional likelihood.
   */
  float_t getLogCondLike() const;

  //! Get the latest filtered expectation E[h(x_1t, x_2t) | y_{1:t}]
  /**
   * @brief Get the expectations you're keeping track of.
   * @return a vector of Mats
   */
  std::vector<Mat> getExpectations() const;

  //! Evaluates the first time state density.
  /**
   * @brief evaluates log mu(x21).
   * @param x21 component two at time 1
   * @return a float_t evaluation
   */
  virtual float_t logMuEv(const sssv &x21) = 0;

  //! Sample from the first time's proposal distribution.
  /**
   * @brief samples the second component of the state at time 1.
   * @param y1 most recent datum.
   * @return a Vec sample for x21.
   */
  virtual sssv q1Samp(const osv &y1) = 0;

  //! Provides the initial mean vector for each Kalman filter object.
  /**
   * @brief provides the initial mean vector for each Kalman filter object.
   * @param x21 the second state componenent at time 1.
   * @return a nsssv representing the unconditional mean.
   */
  virtual nsssv initKalmanMean(const sssv &x21) = 0;

  //! Provides the initial covariance matrix for each Kalman filter object.
  /**
   * @brief provides the initial covariance matrix for each Kalman filter
   * object.
   * @param x21 the second state component at time 1.
   * @return a covariance matrix.
   */
  virtual nsssMat initKalmanVar(const sssv &x21) = 0;

  //! Samples the time t second component.
  /**
   * @brief Samples the time t second component.
   * @param x2tm1 the previous time's second state component.
   * @param yt the current observation.
   * @return a sssv sample of the second state component at the current time.
   */
  virtual sssv qSamp(const sssv &x2tm1, const osv &yt) = 0;

  //! Evaluates the proposal density of the second state component at time 1.
  /**
   * @brief Evaluates the proposal density of the second state component at
   * time 1.
   * @param x21 the second state component at time 1 you sampled.
   * @param y1 time 1 observation.
   * @return a float_t evaluation of the density.
   */
  virtual float_t logQ1Ev(const sssv &x21, const osv &y1) = 0;

  //! Evaluates the state transition density for the second state component.
  /**
   * @brief Evaluates the state transition density for the second state
   * component.
   * @param x2t the current second state component.
   * @param x2tm1 the previous second state component.
   * @return a float_t evaluation.
   */
  virtual float_t logFEv(const sssv &x2t, const sssv &x2tm1) = 0;

  //! Evaluates the proposal density at time t > 1.
  /**
   * @brief Evaluates the proposal density at time t > 1.
   * @param x2t the current second state component.
   * @param x2tm1 the previous second state component.
   * @param yt the current time series observation.
   * @return a float_t evaluation.
   */
  virtual float_t logQEv(const sssv &x2t, const sssv &x2tm1, const osv &yt) = 0;

  //! How to update your inner Kalman filter object at each time.
  /**
   * @brief How to update your inner Kalman filter object at each time.
   * @param kMod a Kalman filter object describing the conditional closed-form
   * model.
   * @param yt the current time series observation.
   * @param x2t the current second state component.
   */
  virtual void updateKalman(cfModType &kMod, const osv &yt,
                            const sssv &x2t) = 0;

private:
  /** the resamplign schedule */
  unsigned int m_rs;
  /** the array of inner Kalman filter objects */
  arrayMod m_p_innerMods;
  /** the array of particle samples */
  arrayVec m_p_samps;
  /** the array of the (log of) unnormalized weights */
  arrayfloat_t m_logUnNormWeights;
  /** the current time period */
  unsigned int m_now;
  /** log p(y_t|y_{1:t-1}) or log p(y1) */
  float_t m_lastLogCondLike;
  /** resampler object */
  resamp_t m_resampler;
  /** expectations */
  std::vector<Mat> m_expectations;
};

template <size_t nparts, size_t dimnss, size_t dimss, size_t dimy,
          typename resamp_t, typename float_t, bool debug>
rbpf_kalman<nparts, dimnss, dimss, dimy, resamp_t, float_t, debug>::rbpf_kalman(
    const unsigned int &resamp_sched)
    : m_now(0), m_lastLogCondLike(0.0), m_rs(resamp_sched) {
  std::fill(m_logUnNormWeights.begin(), m_logUnNormWeights.end(), 0.0);
}

template <size_t nparts, size_t dimnss, size_t dimss, size_t dimy,
          typename resamp_t, typename float_t, bool debug>
rbpf_kalman<nparts, dimnss, dimss, dimy, resamp_t, float_t,
            debug>::~rbpf_kalman() {}

template <size_t nparts, size_t dimnss, size_t dimss, size_t dimy,
          typename resamp_t, typename float_t, bool debug>
void rbpf_kalman<nparts, dimnss, dimss, dimy, resamp_t, float_t, debug>::filter(
    const osv &data,
    const std::vector<
        std::function<const Mat(const nsssv &x1t, const sssv &x2t)>> &fs) {

  if (m_now > 0) {

    // update
    sssv newX2Samp;
    float_t m1(
        -std::numeric_limits<float_t>::infinity()); // for updated weights
    float_t m2 =
        *std::max_element(m_logUnNormWeights.begin(), m_logUnNormWeights.end());
    float_t sumexpdenom(0.0);
    for (size_t ii = 0; ii < nparts; ++ii) {
      newX2Samp = qSamp(m_p_samps[ii], data);
      this->updateKalman(m_p_innerMods[ii], data, newX2Samp);

      // before you update the weights
      sumexpdenom += std::exp(m_logUnNormWeights[ii] - m2);

      // update the weights
      m_logUnNormWeights[ii] += m_p_innerMods[ii].getLogCondLike() +
                                logFEv(newX2Samp, m_p_samps[ii]) -
                                logQEv(newX2Samp, m_p_samps[ii], data);

// print stuff if debug mode is on
#ifndef DROPPINGTHISINRPACKAGE
      if constexpr (debug)
        std::cout << "time: " << m_now
                  << ", transposed x2 sample: " << newX2Samp.transpose()
                  << ", log unnorm weight: " << m_logUnNormWeights[ii] << "\n";
#endif

      // update a max
      if (m_logUnNormWeights[ii] > m1)
        m1 = m_logUnNormWeights[ii];

      m_p_samps[ii] = newX2Samp;
    }

    // calc log p(y_t | y_{1:t-1})
    float_t sumexpnumer(0.0);
    for (size_t p = 0; p < nparts; ++p)
      sumexpnumer += std::exp(m_logUnNormWeights[p] - m1);
    m_lastLogCondLike = m1 + std::log(sumexpnumer) - m2 - std::log(sumexpdenom);

    // calculate expectations before you resample
    unsigned int fId(0);
    // float_t m = *std::max_element(m_logUnNormWeights.begin(),
    // m_logUnNormWeights.end());
    for (auto &h : fs) {

      Mat testOutput = h(m_p_innerMods[0].getFilterVec(), m_p_samps[0]);
      unsigned int rows = testOutput.rows();
      unsigned int cols = testOutput.cols();
      Mat numer = Mat::Zero(rows, cols);
      float_t denom(0.0);
      for (size_t prtcl = 0; prtcl < nparts; ++prtcl) {
        numer += h(m_p_innerMods[prtcl].getFilterVec(), m_p_samps[prtcl]) *
                 std::exp(m_logUnNormWeights[prtcl] - m1);
        denom += std::exp(m_logUnNormWeights[prtcl] - m1);
      }
      m_expectations[fId] = numer / denom;

// print stuff if debug mode is on
#ifndef DROPPINGTHISINRPACKAGE
      if constexpr (debug)
        std::cout << "transposed expec " << fId << ": "
                  << m_expectations[fId].transpose() << "\n";
#endif

      fId++;
    }

    // resample (unnormalized weights ok)
    if ((m_now + 1) % m_rs == 0)
      m_resampler.resampLogWts(m_p_innerMods, m_p_samps, m_logUnNormWeights);

    // update time step
    m_now++;
  } else //( m_now == 0) // first data point coming
  {
    // initialize and update the closed-form mods
    nsssv tmpMean;
    nsssMat tmpVar;
    float_t m1(-std::numeric_limits<float_t>::infinity());
    for (size_t ii = 0; ii < nparts; ++ii) {
      m_p_samps[ii] = q1Samp(data);
      tmpMean = initKalmanMean(m_p_samps[ii]);
      tmpVar = initKalmanVar(m_p_samps[ii]);
      m_p_innerMods[ii] =
          cfModType(tmpMean, tmpVar); // TODO: allow for input or check to make
                                      // sure this doesn't break anything else
      this->updateKalman(m_p_innerMods[ii], data, m_p_samps[ii]);

      m_logUnNormWeights[ii] = m_p_innerMods[ii].getLogCondLike() +
                               logMuEv(m_p_samps[ii]) -
                               logQ1Ev(m_p_samps[ii], data);

// print stuff if debug mode is on
#ifndef DROPPINGTHISINRPACKAGE
      if constexpr (debug)
        std::cout << "time: " << m_now
                  << ", transposed x2 sample: " << m_p_samps[ii].transpose()
                  << ", log unnorm weight: " << m_logUnNormWeights[ii] << "\n";
#endif

      // update a max
      if (m_logUnNormWeights[ii] > m1)
        m1 = m_logUnNormWeights[ii];
    }

    // calculate log p(y1)
    float_t sumexp(0.0);
    for (size_t p = 0; p < nparts; ++p)
      sumexp += std::exp(m_logUnNormWeights[p] - m1);
    m_lastLogCondLike =
        m1 + std::log(sumexp) - std::log(static_cast<float_t>(nparts));

    // calculate expectations before you resample
    m_expectations.resize(fs.size());
    unsigned int fId(0);
    // float_t m = *std::max_element(m_logUnNormWeights.begin(),
    // m_logUnNormWeights.end());
    for (auto &h : fs) {

      Mat testOutput = h(m_p_innerMods[0].getFilterVec(), m_p_samps[0]);
      unsigned int rows = testOutput.rows();
      unsigned int cols = testOutput.cols();
      Mat numer = Mat::Zero(rows, cols);
      float_t denom(0.0);
      for (size_t prtcl = 0; prtcl < nparts; ++prtcl) {
        numer += h(m_p_innerMods[prtcl].getFilterVec(), m_p_samps[prtcl]) *
                 std::exp(m_logUnNormWeights[prtcl] - m1);
        denom += std::exp(m_logUnNormWeights[prtcl] - m1);
      }
      m_expectations[fId] = numer / denom;

// print stuff if debug mode is on
#ifndef DROPPINGTHISINRPACKAGE
      if constexpr (debug)
        std::cout << "transposed expec " << fId << ": "
                  << m_expectations[fId].transpose() << "\n";
#endif

      fId++;
    }

    // resample (unnormalized weights ok)
    if ((m_now + 1) % m_rs == 0)
      m_resampler.resampLogWts(m_p_innerMods, m_p_samps, m_logUnNormWeights);

    // advance time step
    m_now++;
  }
}

template <size_t nparts, size_t dimnss, size_t dimss, size_t dimy,
          typename resamp_t, typename float_t, bool debug>
float_t rbpf_kalman<nparts, dimnss, dimss, dimy, resamp_t, float_t,
                    debug>::getLogCondLike() const {
  return m_lastLogCondLike;
}

template <size_t nparts, size_t dimnss, size_t dimss, size_t dimy,
          typename resamp_t, typename float_t, bool debug>
auto rbpf_kalman<nparts, dimnss, dimss, dimy, resamp_t, float_t,
                 debug>::getExpectations() const -> std::vector<Mat> {
  return m_expectations;
}

//! Rao-Blackwellized/Marginal Bootstrap Filter with inner Kalman Filter
//! objectss
/**
 * @class rbpf_kalman_bs
 * @author t
 * @file rbpf.h
 * @brief Rao-Blackwellized/Marginal Bootstrap Filter with inner Kalman Filter
 * objectss
 * @tparam nparts the number of particles
 * @tparam dimnss dimension of not-sampled-state vector
 * @tparam dimss dimension of sampled-state vector
 * @tparam dimy the dimension of the observations
 * @tparam resamp_t the resampler type
 */
template <size_t nparts, size_t dimnss, size_t dimss, size_t dimy,
          typename resamp_t, typename float_t, bool debug = false>
class rbpf_kalman_bs : public bases::rbpf_base<float_t, dimss, dimnss, dimy> {

public:
  /** "sampled state size vector" */
  using sssv = Eigen::Matrix<float_t, dimss, 1>;
  /** "not sampled state size vector" */
  using nsssv = Eigen::Matrix<float_t, dimnss, 1>;
  /** "observation size vector" */
  using osv = Eigen::Matrix<float_t, dimy, 1>;
  /** dynamic size matrices */
  using Mat = Eigen::Matrix<float_t, Eigen::Dynamic, Eigen::Dynamic>;
  /** "not sampled state size matrix" */
  using nsssMat = Eigen::Matrix<float_t, dimnss, dimnss>;
  /** closed-form model type*/
  using cfModType = kalman<dimnss, dimy, 0, float_t, debug>;
  /** array of model objects */
  using arrayMod = std::array<cfModType, nparts>;
  /** array of samples */
  using arrayVec = std::array<sssv, nparts>;
  /** array of weights */
  using arrayfloat_t = std::array<float_t, nparts>;

  //! The constructor.
  /**
   \param resamp_sched how often you want to resample (e.g once every
   resamp_sched time points)
   */
  rbpf_kalman_bs(const unsigned int &resamp_sched = 1);

  /**
   * @brief The (virtual) destructor.
   */
  virtual ~rbpf_kalman_bs();

  //! Filter!
  /**
   * \brief The workhorse function
   * \param data the most recent observable portion of the time series.
   * \param fs a vector of functions computing E[h(x_1t, x_2t^i)| x_2t^i,y_1:t].
   * to be averaged to yield E[h(x_1t, x_2t)|,y_1:t]
   */
  void filter(
      const osv &data,
      const std::vector<
          std::function<const Mat(const nsssv &x1t, const sssv &x2t)>> &fs =
          std::vector<
              std::function<const Mat(const nsssv &x1t, const sssv &x2t)>>());

  //! Get the latest log conditional likelihood.
  /**
   * \return the latest log conditional likelihood.
   */
  float_t getLogCondLike() const;

  //! Get the latest filtered expectation E[h(x_1t, x_2t) | y_{1:t}]
  /**
   * @brief Get the expectations you're keeping track of.
   * @return a vector of Mats
   */
  std::vector<Mat> getExpectations() const;

  //! Sample from the first time's proposal distribution.
  /**
   * @brief samples the second component of the state at time 1.
   * @return a sssv sample for x21.
   */
  virtual sssv muSamp() = 0;

  //! Provides the initial mean vector for each Kalman filter object.
  /**
   * @brief provides the initial mean vector for each Kalman filter object.
   * @param x21 the second state componenent at time 1.
   * @return a nsssv representing the unconditional mean.
   */
  virtual nsssv initKalmanMean(const sssv &x21) = 0;

  //! Provides the initial covariance matrix for each Kalman filter object.
  /**
   * @brief provides the initial covariance matrix for each Kalman filter
   * object.
   * @param x21 the second state component at time 1.
   * @return a covariance matrix.
   */
  virtual nsssMat initKalmanVar(const sssv &x21) = 0;

  //! Samples the time t second component.
  /**
   * @brief Samples the time t second component.
   * @param x2tm1 the previous time's second state component.
   * @return a sssv sample of the second state component at the current time.
   */
  virtual sssv fSamp(const sssv &x2tm1) = 0;

  //! How to update your inner Kalman filter object at each time.
  /**
   * @brief How to update your inner Kalman filter object at each time.
   * @param kMod a Kalman filter object describing the conditional closed-form
   * model.
   * @param yt the current time series observation.
   * @param x2t the current second state component.
   */
  virtual void updateKalman(cfModType &kMod, const osv &yt,
                            const sssv &x2t) = 0;

private:
  /** the resamplign schedule */
  unsigned int m_rs;
  /** the array of inner Kalman filter objects */
  arrayMod m_p_innerMods;
  /** the array of particle samples */
  arrayVec m_p_samps;
  /** the array of the (log of) unnormalized weights */
  arrayfloat_t m_logUnNormWeights;
  /** the current time period */
  unsigned int m_now;
  /** log p(y_t|y_{1:t-1}) or log p(y1) */
  float_t m_lastLogCondLike;
  /** resampler object */
  resamp_t m_resampler;
  /** expectations */
  std::vector<Mat> m_expectations;
};

template <size_t nparts, size_t dimnss, size_t dimss, size_t dimy,
          typename resamp_t, typename float_t, bool debug>
rbpf_kalman_bs<nparts, dimnss, dimss, dimy, resamp_t, float_t,
               debug>::rbpf_kalman_bs(const unsigned int &resamp_sched)
    : m_now(0), m_lastLogCondLike(0.0), m_rs(resamp_sched) {
  std::fill(m_logUnNormWeights.begin(), m_logUnNormWeights.end(), 0.0);
}

template <size_t nparts, size_t dimnss, size_t dimss, size_t dimy,
          typename resamp_t, typename float_t, bool debug>
rbpf_kalman_bs<nparts, dimnss, dimss, dimy, resamp_t, float_t,
               debug>::~rbpf_kalman_bs() {}

template <size_t nparts, size_t dimnss, size_t dimss, size_t dimy,
          typename resamp_t, typename float_t, bool debug>
void rbpf_kalman_bs<nparts, dimnss, dimss, dimy, resamp_t, float_t, debug>::
    filter(
        const osv &data,
        const std::vector<
            std::function<const Mat(const nsssv &x1t, const sssv &x2t)>> &fs) {

  if (m_now > 0) {

    // update
    sssv newX2Samp;
    float_t m1(
        -std::numeric_limits<float_t>::infinity()); // for updated weights
    float_t m2 =
        *std::max_element(m_logUnNormWeights.begin(), m_logUnNormWeights.end());
    float_t sumexpdenom(0.0);
    for (size_t ii = 0; ii < nparts; ++ii) {

      newX2Samp = fSamp(m_p_samps[ii], data);
      this->updateKalman(m_p_innerMods[ii], data, newX2Samp);

      // before you update the weights
      sumexpdenom += std::exp(m_logUnNormWeights[ii] - m2);

      // update the weights
      m_logUnNormWeights[ii] += m_p_innerMods[ii].getLogCondLike();

// print stuff if debug mode is on
#ifndef DROPPINGTHISINRPACKAGE
      if constexpr (debug)
        std::cout << "time: " << m_now
                  << ", transposed x2 sample: " << newX2Samp.transpose()
                  << ", log unnorm weight: " << m_logUnNormWeights[ii] << "\n";
#endif

      // update a max
      if (m_logUnNormWeights[ii] > m1)
        m1 = m_logUnNormWeights[ii];

      m_p_samps[ii] = newX2Samp;
    }

    // calc log p(y_t | y_{1:t-1})
    float_t sumexpnumer(0.0);
    for (size_t p = 0; p < nparts; ++p)
      sumexpnumer += std::exp(m_logUnNormWeights[p] - m1);
    m_lastLogCondLike = m1 + std::log(sumexpnumer) - m2 - std::log(sumexpdenom);

    // calculate expectations before you resample
    unsigned int fId(0);
    // float_t m = *std::max_element(m_logUnNormWeights.begin(),
    // m_logUnNormWeights.end());
    for (auto &h : fs) {

      Mat testOutput = h(m_p_innerMods[0].getFilterVec(), m_p_samps[0]);
      unsigned int rows = testOutput.rows();
      unsigned int cols = testOutput.cols();
      Mat numer = Mat::Zero(rows, cols);
      float_t denom(0.0);
      for (size_t prtcl = 0; prtcl < nparts; ++prtcl) {
        numer += h(m_p_innerMods[prtcl].getFilterVec(), m_p_samps[prtcl]) *
                 std::exp(m_logUnNormWeights[prtcl] - m1);
        denom += std::exp(m_logUnNormWeights[prtcl] - m1);
      }
      m_expectations[fId] = numer / denom;

// print stuff if debug mode is on
#ifndef DROPPINGTHISINRPACKAGE
      if constexpr (debug)
        std::cout << "transposed expec " << fId << ": "
                  << m_expectations[fId].transpose() << "\n";
#endif

      fId++;
    }

    // resample (unnormalized weights ok)
    if ((m_now + 1) % m_rs == 0)
      m_resampler.resampLogWts(m_p_innerMods, m_p_samps, m_logUnNormWeights);

    // update time step
    m_now++;
  } else // ( m_now == 0) // first data point coming
  {
    // initialize and update the closed-form mods
    nsssv tmpMean;
    nsssMat tmpVar;
    float_t m1(-std::numeric_limits<float_t>::infinity());
    for (size_t ii = 0; ii < nparts; ++ii) {
      m_p_samps[ii] = muSamp(data);
      tmpMean = initKalmanMean(m_p_samps[ii]);
      tmpVar = initKalmanVar(m_p_samps[ii]);
      m_p_innerMods[ii] =
          cfModType(tmpMean, tmpVar); // TODO: allow for input or check to make
                                      // sure this doesn't break anything else
      this->updateKalman(m_p_innerMods[ii], data, m_p_samps[ii]);

      m_logUnNormWeights[ii] = m_p_innerMods[ii].getLogCondLike();

// print stuff if debug mode is on
#ifndef DROPPINGTHISINRPACKAGE
      if constexpr (debug)
        std::cout << "time: " << m_now
                  << ", transposed x2 sample: " << m_p_samps[ii].transpose()
                  << ", log unnorm weight: " << m_logUnNormWeights[ii] << "\n";
#endif

      // update a max
      if (m_logUnNormWeights[ii] > m1)
        m1 = m_logUnNormWeights[ii];
    }

    // calculate log p(y1)
    float_t sumexp(0.0);
    for (size_t p = 0; p < nparts; ++p)
      sumexp += std::exp(m_logUnNormWeights[p] - m1);
    m_lastLogCondLike =
        m1 + std::log(sumexp) - std::log(static_cast<float_t>(nparts));

    // calculate expectations before you resample
    m_expectations.resize(fs.size());
    unsigned int fId(0);
    // float_t m = *std::max_element(m_logUnNormWeights.begin(),
    // m_logUnNormWeights.end());
    for (auto &h : fs) {

      Mat testOutput = h(m_p_innerMods[0].getFilterVec(), m_p_samps[0]);
      unsigned int rows = testOutput.rows();
      unsigned int cols = testOutput.cols();
      Mat numer = Mat::Zero(rows, cols);
      float_t denom(0.0);
      for (size_t prtcl = 0; prtcl < nparts; ++prtcl) {
        numer += h(m_p_innerMods[prtcl].getFilterVec(), m_p_samps[prtcl]) *
                 std::exp(m_logUnNormWeights[prtcl] - m1);
        denom += std::exp(m_logUnNormWeights[prtcl] - m1);
      }
      m_expectations[fId] = numer / denom;

// print stuff if debug mode is on
#ifndef DROPPINGTHISINRPACKAGE
      if constexpr (debug)
        std::cout << "transposed expec " << fId << ": "
                  << m_expectations[fId].transpose() << "\n";
#endif

      fId++;
    }

    // resample (unnormalized weights ok)
    if ((m_now + 1) % m_rs == 0)
      m_resampler.resampLogWts(m_p_innerMods, m_p_samps, m_logUnNormWeights);

    // advance time step
    m_now++;
  }
}

template <size_t nparts, size_t dimnss, size_t dimss, size_t dimy,
          typename resamp_t, typename float_t, bool debug>
float_t rbpf_kalman_bs<nparts, dimnss, dimss, dimy, resamp_t, float_t,
                       debug>::getLogCondLike() const {
  return m_lastLogCondLike;
}

template <size_t nparts, size_t dimnss, size_t dimss, size_t dimy,
          typename resamp_t, typename float_t, bool debug>
auto rbpf_kalman_bs<nparts, dimnss, dimss, dimy, resamp_t, float_t,
                    debug>::getExpectations() const -> std::vector<Mat> {
  return m_expectations;
}

} // namespace filters
} // namespace pf

#endif // RBPF_H
