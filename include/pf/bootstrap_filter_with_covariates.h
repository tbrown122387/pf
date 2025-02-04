#ifndef BOOTSTRAP_FILTER_WC_H
#define BOOTSTRAP_FILTER_WC_H

#include <array>
#include <iostream>
#include <vector>

#ifdef DROPPINGTHISINRPACKAGE
#include <RcppEigen.h>
// [[Rcpp::depends(RcppEigen)]]
#else
#include <Eigen/Dense>
#endif

#include <iostream>

#include "pf_base.h"

namespace pf {

namespace filters {

//! A base class for the bootstrap particle filter with covariates.
/**
 * @class BSFilterWC
 * @author taylor
 * @file bootstrap_filter_with_covariates.h
 * @brief bootstrap particle filter with covariates
 * @tparam nparts the number of particles
 * @tparam dimx the dimension of the state
 * @tparam dimy the dimension of the observations
 * @tparam dimcov the dimension of the covariates
 * @tparam resamp_t the type of resampler
 */
template <size_t nparts, size_t dimx, size_t dimy, size_t dimcov,
          typename resamp_t, typename float_t, bool debug = false>
class BSFilterWC : public bases::pf_withcov_base<float_t, dimy, dimx, dimcov> {
private:
  /** "state size vector" type alias for linear algebra stuff */
  using ssv = Eigen::Matrix<float_t, dimx, 1>;
  /** "obs size vector" type alias for linear algebra stuff */
  using osv = Eigen::Matrix<float_t, dimy, 1>; // obs size vec
  /** covariate size vector" type alias for linear algebra stuff */
  using cvsv = Eigen::Matrix<float_t, dimcov, 1>;
  /** type alias for dynamically sized matrix */
  using Mat = Eigen::Matrix<float_t, Eigen::Dynamic, Eigen::Dynamic>;
  /** type alias for linear algebra stuff */
  using arrayStates = std::array<ssv, nparts>;
  /** type alias for array of float_ts */
  using arrayfloat_t = std::array<float_t, nparts>;

public:
  /**
   * @brief The constructor
   * @param rs the resampling schedule (e.g. every rs time point)
   */
  BSFilterWC(const unsigned int &rs = 1);

  /**
   * @brief The (virtual) destructor
   */
  virtual ~BSFilterWC();

  /**
   * @brief Returns the most recent (log-) conditiona likelihood.
   * @return log p(y_t | y_{1:t-1})
   */
  float_t getLogCondLike() const;

  /**
   * @brief updates filtering distribution on a new datapoint.
   * Optionally stores expectations of functionals.
   * @param data the most recent data point
   * @param covData covariate data
   * @param fs a vector of functions if you want to calculate expectations.
   */
  void filter(
      const osv &ydata, const cvsv &covdata,
      const std::vector<
          std::function<const Mat(const ssv &, const cvsv &)>> &fs =
          std::vector<std::function<const Mat(const ssv &, const cvsv &)>>());

  /**
   * @brief return all stored expectations (taken with respect to
   * $p(x_t|y_{1:t})$
   * @return return a std::vector<Mat> of expectations. How many depends on how
   * many callbacks you gave to
   */
  auto getExpectations() const -> std::vector<Mat>;

  /**
   * @brief  Calculate muEv or logmuEv
   * @param x1 is a const Vec& describing the state sample
   * @param z1 is a const Vec& describing the covariate sample
   * @return the density or log-density evaluation as a float_t
   */
  virtual float_t logMuEv(const ssv &x1, const cvsv &z1) = 0;

  /**
   * @brief Samples from time 1 proposal
   * @param y1 is a const Vec& representing the first observed datum
   * @param z1 is the const Vec& representing the first covariate
   * @return the sample as a Vec
   */
  virtual ssv q1Samp(const osv &y1, const cvsv &z1) = 0;

  /**
   * @brief Calculate q1Ev or log q1Ev
   * @param x1 is a const Vec& describing the time 1 state sample
   * @param y1 is a const Vec& describing the time 1 datum
   * @param z1 is a const Vec& describing the time 1 covariate
   * @return the density or log-density evaluation as a float_t
   */
  virtual float_t logQ1Ev(const ssv &x1, const osv &y1, const cvsv &z1) = 0;

  /**
   * @brief Calculate gEv or logGEv
   * @param yt is a const Vec& describing the time t datum
   * @param xt is a const Vec& describing the time t state
   * @param zt is a const Vec& describing the time t covariate
   * @return the density or log-density evaluation as a float_t
   */
  virtual float_t logGEv(const osv &yt, const ssv &xt, const cvsv &zt) = 0;

  //!
  /**
   * @brief Sample from the state transition distribution.
   * @param xtm1 is a const Vec& describing the time t-1 state
   * @param zt is a const Vec& describing the time t covariate
   * @return the sample as a Vec
   */
  virtual ssv fSamp(const ssv &xtm1, const cvsv &zt) = 0;

protected:
  /** @brief particle samples */
  arrayStates m_particles;

  /** @brief particle unnormalized weights */
  arrayfloat_t m_logUnNormWeights;

  /** @brief time point */
  unsigned int m_now;

  /** @brief log p(y_t|y_{1:t-1}) or log p(y1)  */
  float_t m_logLastCondLike;

  /** @brief resampler object */
  resamp_t m_resampler;

  /** @brief expectations E[h(x_t) | y_{1:t}] for user defined "h"s */
  std::vector<Mat> m_expectations;

  /** @brief resampling schedule (e.g. resample every __ time points) */
  unsigned int m_resampSched;
};

/////////////////////////////////////////////////////////////////////////////////////////////////////////
///////////////////////////////////////// implementations
//////////////////////////////////////////////////
/////////////////////////////////////////////////////////////////////////////////////////////////////////

template <size_t nparts, size_t dimx, size_t dimy, size_t dimcov,
          typename resamp_t, typename float_t, bool debug>
BSFilterWC<nparts, dimx, dimy, dimcov, resamp_t, float_t, debug>::BSFilterWC(
    const unsigned int &rs)
    : m_now(0), m_logLastCondLike(0.0), m_resampSched(rs)

{
  std::fill(m_logUnNormWeights.begin(), m_logUnNormWeights.end(),
            0.0); // log(1) = 0
}

template <size_t nparts, size_t dimx, size_t dimy, size_t dimcov,
          typename resamp_t, typename float_t, bool debug>
BSFilterWC<nparts, dimx, dimy, dimcov, resamp_t, float_t,
           debug>::~BSFilterWC() {}

template <size_t nparts, size_t dimx, size_t dimy, size_t dimcov,
          typename resamp_t, typename float_t, bool debug>
void BSFilterWC<nparts, dimx, dimy, dimcov, resamp_t, float_t, debug>::filter(
    const osv &dat, const cvsv &covData,
    const std::vector<std::function<const Mat(const ssv &, const cvsv &)>>
        &fs) {
  if (m_now > 0) {

    // try to iterate over particles all at once
    ssv newSamp;
    float_t maxOldLogUnNormWts(-std::numeric_limits<float_t>::infinity());
    arrayfloat_t oldLogUnNormWts = m_logUnNormWeights;
    for (size_t ii = 0; ii < nparts; ++ii) {
      // update max of old logUnNormWts
      if (m_logUnNormWeights[ii] > maxOldLogUnNormWts)
        maxOldLogUnNormWts = m_logUnNormWeights[ii];

      // sample and get weight adjustments
      newSamp = fSamp(m_particles[ii], covData);
      m_logUnNormWeights[ii] += logGEv(dat, newSamp, covData);

      // overwrite stuff
      m_particles[ii] = newSamp;

#ifndef DROPPINGTHISINRPACKAGE
      if constexpr (debug)
        std::cout << "time: " << m_now
                  << ", transposed sample: " << m_particles[ii].transpose()
                  << ", log unnorm weight: " << m_logUnNormWeights[ii] << "\n";
#endif
    }

    // compute estimate of log p(y_t|y_{1:t-1}) with log-exp-sum trick
    float_t maxNumer = *std::max_element(
        m_logUnNormWeights.begin(),
        m_logUnNormWeights.end()); // because you added log adjustments
    float_t sumExp1(0.0);
    float_t sumExp2(0.0);
    for (size_t i = 0; i < nparts; ++i) {
      sumExp1 += std::exp(m_logUnNormWeights[i] - maxNumer);
      sumExp2 += std::exp(oldLogUnNormWts[i] - maxOldLogUnNormWts); // 1
    }
    m_logLastCondLike =
        maxNumer + std::log(sumExp1) - maxOldLogUnNormWts - std::log(sumExp2);

    // calculate expectations before you resample
    int fId(0);
    for (auto &h : fs) { // iterate over all functions

      Mat testOutput = h(m_particles[0], covData);
      unsigned int rows = testOutput.rows();
      unsigned int cols = testOutput.cols();
      Mat numer = Mat::Zero(rows, cols);
      float_t weightNormConst(0.0);
      for (size_t prtcl = 0; prtcl < nparts;
           ++prtcl) { // iterate over all particles
        numer += h(m_particles[prtcl], covData) *
                 std::exp(m_logUnNormWeights[prtcl] - maxNumer);
        weightNormConst += std::exp(m_logUnNormWeights[prtcl] - maxNumer);
      }
      m_expectations[fId] = numer / weightNormConst;

#ifndef DROPPINGTHISINRPACKAGE
      if constexpr (debug)
        std::cout << "transposed expectation " << fId << ": "
                  << m_expectations[fId].transpose() << "\n";
#endif

      fId++;
    }

    // resample if you should
    if ((m_now + 1) % m_resampSched == 0)
      m_resampler.resampLogWts(m_particles, m_logUnNormWeights);

    // advance time
    m_now += 1;
  } else { // m_now == 0

    // only need to iterate over particles once
    for (size_t ii = 0; ii < nparts; ++ii) {
      // sample particles
      m_particles[ii] = q1Samp(dat, covData);
      m_logUnNormWeights[ii] = logMuEv(m_particles[ii], covData);
      m_logUnNormWeights[ii] += logGEv(dat, m_particles[ii], covData);
      m_logUnNormWeights[ii] -= logQ1Ev(m_particles[ii], dat, covData);

#ifndef DROPPINGTHISINRPACKAGE
      if constexpr (debug)
        std::cout << "time: " << m_now
                  << ", transposed sample: " << m_particles[ii].transpose()
                  << ", log unnorm weight: " << m_logUnNormWeights[ii] << "\n";
#endif
    }

    // calculate log cond likelihood with log-exp-sum trick
    float_t max =
        *std::max_element(m_logUnNormWeights.begin(), m_logUnNormWeights.end());
    float_t sumExp(0.0);
    for (size_t i = 0; i < nparts; ++i) {
      sumExp += std::exp(m_logUnNormWeights[i] - max);
    }
    m_logLastCondLike = -std::log(nparts) + max + std::log(sumExp);

    // calculate expectations before you resample
    // paying mind to underflow
    m_expectations.resize(fs.size());
    unsigned int fId(0);
    for (auto &h : fs) {

      Mat testOutput = h(m_particles[0], covData);
      unsigned int rows = testOutput.rows();
      unsigned int cols = testOutput.cols();
      Mat numer = Mat::Zero(rows, cols);
      float_t weightNormConst(0.0);
      for (size_t prtcl = 0; prtcl < nparts;
           ++prtcl) { // iterate over all particles
        numer += h(m_particles[prtcl], covData) *
                 std::exp(m_logUnNormWeights[prtcl] - max);
        weightNormConst += std::exp(m_logUnNormWeights[prtcl] - max);
      }
      m_expectations[fId] = numer / weightNormConst;
      fId++;
    }

    // resample if you should
    if ((m_now + 1) % m_resampSched == 0) {
      m_resampler.resampLogWts(m_particles, m_logUnNormWeights);
    }

    // advance time step
    m_now += 1;
  }
}

template <size_t nparts, size_t dimx, size_t dimy, size_t dimcov,
          typename resamp_t, typename float_t, bool debug>
float_t BSFilterWC<nparts, dimx, dimy, dimcov, resamp_t, float_t,
                   debug>::getLogCondLike() const {
  return m_logLastCondLike;
}

template <size_t nparts, size_t dimx, size_t dimy, size_t dimcov,
          typename resamp_t, typename float_t, bool debug>
auto BSFilterWC<nparts, dimx, dimy, dimcov, resamp_t, float_t,
                debug>::getExpectations() const -> std::vector<Mat> {
  return m_expectations;
}

} // namespace filters
} // namespace pf

#endif // BOOTSTRAP_FILTER_WC_H
