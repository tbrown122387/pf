#ifndef AUXILIARY_PF_H
#define AUXILIARY_PF_H

#include <array>      //array
#include <functional> // function
#include <vector>     // vector

#ifdef DROPPINGTHISINRPACKAGE
#include <RcppEigen.h>
// [[Rcpp::depends(RcppEigen)]]
#else
#include <Eigen/Dense>
#endif

#include <cmath>

#include "pf_base.h"
#include "rv_samp.h" // for k_generator

namespace pf {

namespace filters {

//! A base-class for Auxiliary Particle Filtering. Filtering only, no smoothing.
/**
 * @class APF
 * @author taylor
 * @file auxiliary_pf.h
 * @brief A base class for Auxiliary Particle Filtering.
 * Inherit from this if you want to use an APF for your state space model.
 * Filtering only, no smoothing.
 * @tparam nparts the number of particles
 * @tparam dimx the dimension of the state
 * @tparam dimy the dimension of the observations
 * @tparam resamp_t the resampler type
 */
template <size_t nparts, size_t dimx, size_t dimy, typename resamp_t,
          typename float_t, bool debug = false>
class APF : public bases::pf_base<float_t, dimy, dimx> {
private:
  /** "state size vector" type alias for linear algebra stuff */
  using ssv = Eigen::Matrix<float_t, dimx, 1>;
  /** "observation size vector" type alias for linear algebra stuff */
  using osv = Eigen::Matrix<float_t, dimy, 1>;
  /** type alias for linear algebra stuff (dimension of the state ^2) */
  using Mat = Eigen::Matrix<float_t, Eigen::Dynamic, Eigen::Dynamic>;
  /** type alias for array of float_ts */
  using arrayfloat_t = std::array<float_t, nparts>;
  /** type alias for array of state vectors */
  using arrayVec = std::array<ssv, nparts>;
  /** type alias for array of unsigned ints */
  using arrayUInt = std::array<unsigned int, nparts>;

public:
  /**
   * @brief The constructor.
   * @param rs resampling schedule (e.g. resample every rs time points).
   */
  APF(const unsigned int &rs = 1);

  /**
   * @brief The (virtual) destructor
   */
  virtual ~APF();

  /**
   * @brief Get the latest log conditional likelihood.
   * @return a float_t of the most recent conditional likelihood.
   */
  float_t getLogCondLike() const;

  /**
   * @brief return all stored expectations (taken with respect to
   * $p(x_t|y_{1:t})$
   * @return return a std::vector<Mat> of expectations. How many depends on how
   * many callbacks you gave to
   */
  std::vector<Mat> getExpectations() const;

  /**
   * @brief Use a new datapoint to update the filtering distribution (or
   * smoothing if pathLength > 0).
   * @param data a Eigen::Matrix<float_t,dimy,1> representing the data
   * @param fs a std::vector of callback functions that are used to calculate
   * expectations with respect to the filtering distribution.
   */
  void filter(const osv &data,
              const std::vector<std::function<const Mat(const ssv &)>> &fs =
                  std::vector<std::function<const Mat(const ssv &)>>());

  /**
   * @brief Evaluates the log of mu.
   * @param x1 a Eigen::Matrix<float_t,dimx,1> representing time 1's state.
   * @return a float_t evaluation.
   */
  virtual float_t logMuEv(const ssv &x1) = 0;

  /**
   * @brief Evaluates the proposal distribution taking a
   * Eigen::Matrix<float_t,dimx,1> from the previous time's state, and returning
   * a state for the current time.
   * @param xtm1 a Eigen::Matrix<float_t,dimx,1> representing the previous
   * time's state.
   * @return a Eigen::Matrix<float_t,dimx,1> representing a likely current time
   * state, to be used by the observation density.
   */
  virtual ssv propMu(const ssv &xtm1) = 0;

  /**
   * @brief Samples from q1.
   * @param y1 a Eigen::Matrix<float_t,dimy,1> representing time 1's data point.
   * @return a Eigen::Matrix<float_t,dimx,1> sample for time 1's state.
   */
  virtual ssv q1Samp(const osv &y1) = 0;

  /**
   * @brief Samples from f.
   * @param xtm1 a Eigen::Matrix<float_t,dimx,1> representing the previous
   * time's state.
   * @return a Eigen::Matrix<float_t,dimx,1> state sample for the current time.
   */
  virtual ssv fSamp(const ssv &xtm1) = 0;

  /**
   * @brief Evaluates the log of q1.
   * @param x1 a Eigen::Matrix<float_t,dimx,1> representing time 1's state.
   * @param y1 a Eigen::Matrix<float_t,dimy,1> representing time 1's data
   * observation.
   * @return a float_t evaluation.
   */
  virtual float_t logQ1Ev(const ssv &x1, const osv &y1) = 0;

  /**
   * @brief Evaluates the log of g.
   * @param yt a Eigen::Matrix<float_t,dimy,1> representing time t's data
   * observation.
   * @param xt a Eigen::Matrix<float_t,dimx,1> representing time t's state.
   * @return a float_t evaluation.
   */
  virtual float_t logGEv(const osv &yt, const ssv &xt) = 0;

protected:
  /** @brief particle samples */
  std::array<ssv, nparts> m_particles;

  /** @brief particle unnormalized weights */
  std::array<float_t, nparts> m_logUnNormWeights;

  /** @brief curren time */
  unsigned int m_now;

  /** @brief log p(y_t|y_{1:t-1}) or log p(y1) */
  float_t m_logLastCondLike;

  /** @brief the resampling schedule */
  unsigned int m_rs;

  /** @brief resampler object (default ctor'd)*/
  resamp_t m_resampler;

  /** @brief k generator object (default ctor'd)*/
  rvsamp::k_gen<nparts, float_t> m_kGen;

  /** @brief expectations E[h(x_t) | y_{1:t}] for user defined "h"s */
  std::vector<Mat> m_expectations;
};

template <size_t nparts, size_t dimx, size_t dimy, typename resamp_t,
          typename float_t, bool debug>
APF<nparts, dimx, dimy, resamp_t, float_t, debug>::APF(const unsigned int &rs)
    : m_now(0), m_logLastCondLike(0.0), m_rs(rs) {
  std::fill(m_logUnNormWeights.begin(), m_logUnNormWeights.end(), 0.0);
}

template <size_t nparts, size_t dimx, size_t dimy, typename resamp_t,
          typename float_t, bool debug>
APF<nparts, dimx, dimy, resamp_t, float_t, debug>::~APF() {}

template <size_t nparts, size_t dimx, size_t dimy, typename resamp_t,
          typename float_t, bool debug>
void APF<nparts, dimx, dimy, resamp_t, float_t, debug>::filter(
    const osv &data,
    const std::vector<std::function<const Mat(const ssv &)>> &fs) {

  if (m_now > 0) {

    // set up "first stage weights" to make k index sampler
    arrayfloat_t logFirstStageUnNormWeights = m_logUnNormWeights;
    arrayVec oldPartics = m_particles;
    float_t m3(-std::numeric_limits<float_t>::infinity());
    float_t m2(-std::numeric_limits<float_t>::infinity());
    for (size_t ii = 0; ii < nparts; ++ii) {
      // update m3
      if (m_logUnNormWeights[ii] > m3)
        m3 = m_logUnNormWeights[ii];

      // sample
      ssv xtm1 = oldPartics[ii];
      logFirstStageUnNormWeights[ii] += logGEv(data, propMu(xtm1));

      // accumulate things
      if (logFirstStageUnNormWeights[ii] > m2)
        m2 = logFirstStageUnNormWeights[ii];

// print stuff if debug mode is on
#ifndef DROPPINGTHISINRPACKAGE
      if constexpr (debug) {
        std::cout << "time: " << m_now << ", first stage log unnorm weight: "
                  << logFirstStageUnNormWeights[ii] << "\n";
      }
#endif
    }

    // draw ks (indexes) (handles underflow issues)
    arrayUInt myKs = m_kGen.sample(logFirstStageUnNormWeights);

    // now draw xts
    float_t m1(-std::numeric_limits<float_t>::infinity());
    float_t first_cll_sum(0.0);
    float_t second_cll_sum(0.0);
    float_t third_cll_sum(0.0);
    ssv xtm1k;
    ssv muT;
    for (size_t ii = 0; ii < nparts; ++ii) {
      // calclations for log p(y_t|y_{1:t-1}) (using log-sum-exp trick)
      second_cll_sum += std::exp(logFirstStageUnNormWeights[ii] - m2);
      third_cll_sum += std::exp(m_logUnNormWeights[ii] - m3);

      // sampling and unnormalized weight update
      xtm1k = oldPartics[myKs[ii]];
      m_particles[ii] = fSamp(xtm1k);
      muT = propMu(xtm1k);
      m_logUnNormWeights[ii] +=
          logGEv(data, m_particles[ii]) - logGEv(data, muT);

#ifndef DROPPINGTHISINRPACKAGE
      if constexpr (debug) {
        std::cout << "time: " << m_now
                  << ", transposed sample: " << m_particles[ii].transpose()
                  << ", log unnorm weight: " << m_logUnNormWeights[ii] << "\n";
      }
#endif

      // update m1
      if (m_logUnNormWeights[ii] > m1)
        m1 = m_logUnNormWeights[ii];
    }

    // calculate estimate for log of last conditonal likelihood
    for (size_t p = 0; p < nparts; ++p)
      first_cll_sum += std::exp(m_logUnNormWeights[p] - m1);
    m_logLastCondLike = m1 + std::log(first_cll_sum) + m2 +
                        std::log(second_cll_sum) - 2 * m3 -
                        2 * std::log(third_cll_sum);

#ifndef DROPPINGTHISINRPACKAGE
    if constexpr (debug)
      std::cout << "time: " << m_now << ", log cond like: " << m_logLastCondLike
                << "\n";
#endif

    // calculate expectations before you resample
    unsigned int fId(0);
    for (auto &h : fs) {

      Mat testOutput = h(m_particles[0]);
      unsigned int rows = testOutput.rows();
      unsigned int cols = testOutput.cols();
      Mat numer = Mat::Zero(rows, cols);
      float_t denom(0.0);

      for (size_t prtcl = 0; prtcl < nparts; ++prtcl) {
        numer +=
            h(m_particles[prtcl]) * std::exp(m_logUnNormWeights[prtcl] - m1);
        denom += std::exp(m_logUnNormWeights[prtcl] - m1);
      }
      m_expectations[fId] = numer / denom;

#ifndef DROPPINGTHISINRPACKAGE
      if constexpr (debug)
        std::cout << "transposed expectation " << fId << "; "
                  << m_expectations[fId] << "\n";
#endif

      fId++;
    }

    // if you have to resample
    if ((m_now + 1) % m_rs == 0)
      m_resampler.resampLogWts(m_particles, m_logUnNormWeights);

    // advance time
    m_now += 1;

  } else { // (m_now == 0)

    float_t max(-std::numeric_limits<float_t>::infinity());
    for (size_t ii = 0; ii < nparts; ++ii) {
      // sample particles
      m_particles[ii] = q1Samp(data);
      m_logUnNormWeights[ii] = logMuEv(m_particles[ii]);
      m_logUnNormWeights[ii] += logGEv(data, m_particles[ii]);
      m_logUnNormWeights[ii] -= logQ1Ev(m_particles[ii], data);

// print stuff if debug mode is on
#ifndef DROPPINGTHISINRPACKAGE
      if constexpr (debug) {
        std::cout << "time: " << m_now
                  << ", log unnorm weight: " << m_logUnNormWeights[ii]
                  << ", transposed sample: " << m_particles[ii].transpose()
                  << "\n";
      }
#endif

      // update maximum
      if (m_logUnNormWeights[ii] > max)
        max = m_logUnNormWeights[ii];
    }

    // calculate log-likelihood with log-exp-sum trick
    float_t sumExp(0.0);
    for (size_t i = 0; i < nparts; ++i) {
      sumExp += std::exp(m_logUnNormWeights[i] - max);
    }
    m_logLastCondLike =
        -std::log(static_cast<float_t>(nparts)) + max + std::log(sumExp);

    // calculate expectations before you resample
    m_expectations.resize(fs.size());
    unsigned int fId(0);
    for (auto &h : fs) {

      Mat testOutput = h(m_particles[0]);
      unsigned int rows = testOutput.rows();
      unsigned int cols = testOutput.cols();
      Mat numer = Mat::Zero(rows, cols);
      float_t denom(0.0);
      for (size_t prtcl = 0; prtcl < nparts; ++prtcl) {
        numer +=
            h(m_particles[prtcl]) * std::exp(m_logUnNormWeights[prtcl] - max);
        denom += std::exp(m_logUnNormWeights[prtcl] - max);
      }
      m_expectations[fId] = numer / denom;

#ifndef DROPPINGTHISINRPACKAGE
      if constexpr (debug)
        std::cout << "transposed expectation " << fId << "; "
                  << m_expectations[fId] << "\n";
#endif

      fId++;
    }

    // resample if you should (automatically normalizes)
    if ((m_now + 1) % m_rs == 0)
      m_resampler.resampLogWts(m_particles, m_logUnNormWeights);

    // advance time step
    m_now += 1;
  }
}

template <size_t nparts, size_t dimx, size_t dimy, typename resamp_t,
          typename float_t, bool debug>
float_t
APF<nparts, dimx, dimy, resamp_t, float_t, debug>::getLogCondLike() const {
  return m_logLastCondLike;
}

template <size_t nparts, size_t dimx, size_t dimy, typename resamp_t,
          typename float_t, bool debug>
auto APF<nparts, dimx, dimy, resamp_t, float_t, debug>::getExpectations() const
    -> std::vector<Mat> {
  return m_expectations;
}

} // namespace filters
} // namespace pf
#endif // APF_H
