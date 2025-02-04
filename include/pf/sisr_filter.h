#ifndef SISR_FILTER_H
#define SISR_FILTER_H

#include <array>

#ifdef DROPPINGTHISINRPACKAGE
#include <RcppEigen.h>
// [[Rcpp::depends(RcppEigen)]]
#else
#include <Eigen/Dense>
#endif

#include "pf_base.h"

namespace pf {

namespace filters {

//! A base class for the Sequential Important Sampling with Resampling (SISR).
/**
 * @class SISRFilter
 * @author taylor
 * @file sisr_filter.h
 * @brief SISR filter.
 * @tparam nparts the number of particles
 * @tparam dimx the size of the state
 * @tparam the size of the observation
 * @tparam resamp_t the type of resampler
 * @tparam float_t the type of floating point numbers used (e.g. float or
 * double)
 * @tparam debug whether to debug or not
 */
template <size_t nparts, size_t dimx, size_t dimy, typename resamp_t,
          typename float_t, bool debug = false>
class SISRFilter : public bases::pf_base<float_t, dimy, dimx> {
private:
  /** "state size vector" type alias for linear algebra stuff */
  using ssv = Eigen::Matrix<float_t, dimx, 1>;
  /** "obs size vector" type alias for linear algebra stuff */
  using osv = Eigen::Matrix<float_t, dimy, 1>; // obs size vec
  /** type alias for linear algebra stuff */
  using Mat = Eigen::Matrix<float_t, Eigen::Dynamic, Eigen::Dynamic>;
  /** type alias for linear algebra stuff */
  using arrayStates = std::array<ssv, nparts>;
  /** type alias for array of float_ts */
  using arrayfloat_t = std::array<float_t, nparts>;

public:
  /**
   * @brief The (one and only) constructor.
   * @param rs the resampling schedule (resample every rs time points).
   */
  SISRFilter(const unsigned int &rs = 1);

  /**
   * @brief The (virtual) destructor.
   */
  virtual ~SISRFilter();

  /**
   * @brief Returns the most recent (log-) conditional likelihood.
   * @return log p(y_t | y_{1:t-1}) or log p(y_1)
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
   * @brief updates filtering distribution on a new datapoint.
   * Optionally stores expectations of functionals.
   * @param data the most recent data point
   * @param fs a vector of functions if you want to calculate expectations.
   */
  void filter(const osv &data,
              const std::vector<std::function<const Mat(const ssv &)>> &fs =
                  std::vector<std::function<const Mat(const ssv &)>>());

  /**
   * @brief  Calculate muEv or logmuEv
   * @param x1 is a const Vec& describing the state sample
   * @return the density or log-density evaluation as a float_t
   */
  virtual float_t logMuEv(const ssv &x1) = 0;

  /**
   * @brief Samples from time 1 proposal
   * @param y1 is a const Vec& representing the first observed datum
   * @return the sample as a Vec
   */
  virtual ssv q1Samp(const osv &y1) = 0;

  /**
   * @brief Calculate q1Ev or log q1Ev
   * @param x1 is a const Vec& describing the time 1 state sample
   * @param y1 is a const Vec& describing the time 1 datum
   * @return the density or log-density evaluation as a float_t
   */
  virtual float_t logQ1Ev(const ssv &x1, const osv &y1) = 0;

  /**
   * @brief Calculate gEv or logGEv
   * @param yt is a const Vec& describing the time t datum
   * @param xt is a const Vec& describing the time t state
   * @return the density or log-density evaluation as a float_t
   */
  virtual float_t logGEv(const osv &yt, const ssv &xt) = 0;

  /**
   * @brief Evaluates the state transition density.
   * @param xt the current state
   * @param xtm1 the previous state
   * @return a float_t evaluaton of the log density/pmf
   */
  virtual float_t logFEv(const ssv &xt, const ssv &xtm1) = 0;

  /**
   * @brief Samples from the proposal/instrumental/importance density at time t
   * @param xtm1 the previous state sample
   * @param yt the current observation
   * @return a state sample for the current time xt
   */
  virtual ssv qSamp(const ssv &xtm1, const osv &yt) = 0;

  /**
   * @brief Evaluates the proposal/instrumental/importance density/pmf
   * @param xt current state
   * @param xtm1 previous state
   * @param yt current observation
   * @return a float_t evaluation of the log density/pmf
   */
  virtual float_t logQEv(const ssv &xt, const ssv &xtm1, const osv &yt) = 0;

protected:
  /** @brief particle samples */
  arrayStates m_particles;

  /** @brief particle weights */
  arrayfloat_t m_logUnNormWeights;

  /** @brief current time point */
  unsigned int m_now;

  /** @brief log p(y_t|y_{1:t-1}) or log p(y1) */
  float_t m_logLastCondLike;

  /** @brief resampling object */
  resamp_t m_resampler;

  /** @brief expectations E[h(x_t) | y_{1:t}] for user defined "h"s */
  std::vector<Mat> m_expectations; // stores any sample averages the user wants

  /** @brief resampling schedule (e.g. resample every __ time points) */
  unsigned int m_resampSched;

  /**
   * @todo implement ESS stuff
   */
};

template <size_t nparts, size_t dimx, size_t dimy, typename resamp_t,
          typename float_t, bool debug>
SISRFilter<nparts, dimx, dimy, resamp_t, float_t, debug>::SISRFilter(
    const unsigned int &rs)
    : m_now(0), m_logLastCondLike(0.0), m_resampSched(rs) {
  std::fill(m_logUnNormWeights.begin(), m_logUnNormWeights.end(),
            0.0); // log(1) = 0
}

template <size_t nparts, size_t dimx, size_t dimy, typename resamp_t,
          typename float_t, bool debug>
SISRFilter<nparts, dimx, dimy, resamp_t, float_t, debug>::~SISRFilter() {}

template <size_t nparts, size_t dimx, size_t dimy, typename resamp_t,
          typename float_t, bool debug>
float_t
SISRFilter<nparts, dimx, dimy, resamp_t, float_t, debug>::getLogCondLike()
    const {
  return m_logLastCondLike;
}

template <size_t nparts, size_t dimx, size_t dimy, typename resamp_t,
          typename float_t, bool debug>
auto SISRFilter<nparts, dimx, dimy, resamp_t, float_t, debug>::getExpectations()
    const -> std::vector<Mat> {
  return m_expectations;
}

template <size_t nparts, size_t dimx, size_t dimy, typename resamp_t,
          typename float_t, bool debug>
void SISRFilter<nparts, dimx, dimy, resamp_t, float_t, debug>::filter(
    const osv &data,
    const std::vector<std::function<const Mat(const ssv &)>> &fs) {

  if (m_now > 0) {

    // try to iterate over particles all at once
    ssv newSamp;
    arrayfloat_t oldLogUnNormWts = m_logUnNormWeights;
    float_t maxOldLogUnNormWts(-std::numeric_limits<float_t>::infinity());
    for (size_t ii = 0; ii < nparts; ++ii) {

      // update max of old logUnNormWts before you change the element
      if (m_logUnNormWeights[ii] > maxOldLogUnNormWts)
        maxOldLogUnNormWts = m_logUnNormWeights[ii];

      // sample and get weight adjustments
      newSamp = qSamp(m_particles[ii], data);
      m_logUnNormWeights[ii] += logFEv(newSamp, m_particles[ii]);
      m_logUnNormWeights[ii] += logGEv(data, newSamp);
      m_logUnNormWeights[ii] -= logQEv(newSamp, m_particles[ii], data);

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
      sumExp2 += std::exp(oldLogUnNormWts[i] - maxOldLogUnNormWts);
    }
    m_logLastCondLike =
        maxNumer + std::log(sumExp1) - maxOldLogUnNormWts - std::log(sumExp2);

    // calculate expectations before you resample
    unsigned int fId(0);
    float_t weightNormConst(0.0);
    for (auto &h : fs) { // iterate over all functions

      Mat testOut = h(m_particles[0]);
      unsigned int rows = testOut.rows();
      unsigned int cols = testOut.cols();
      Mat numer = Mat::Zero(rows, cols);
      float_t denom(0.0);

      for (size_t prtcl = 0; prtcl < nparts;
           ++prtcl) { // iterate over all particles
        numer += h(m_particles[prtcl]) *
                 std::exp(m_logUnNormWeights[prtcl] - maxNumer);
        denom += std::exp(m_logUnNormWeights[prtcl] - maxNumer);
      }
      m_expectations[fId] = numer / denom;

// print stuff if debug mode is on
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
  } else // (m_now == 0) //time 1
  {

    // only need to iterate over particles once
    float_t sumWts(0.0);
    for (size_t ii = 0; ii < nparts; ++ii) {
      // sample particles
      m_particles[ii] = q1Samp(data);
      m_logUnNormWeights[ii] += logMuEv(m_particles[ii]);
      m_logUnNormWeights[ii] += logGEv(data, m_particles[ii]);
      m_logUnNormWeights[ii] -= logQ1Ev(m_particles[ii], data);

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
    m_expectations.resize(fs.size());
    unsigned int fId(0);
    for (auto &h : fs) {

      Mat testOut = h(m_particles[0]);
      unsigned int rows = testOut.rows();
      unsigned int cols = testOut.cols();
      Mat numer = Mat::Zero(rows, cols);
      float_t denom(0.0);

      for (size_t prtcl = 0; prtcl < nparts; ++prtcl) {
        numer +=
            h(m_particles[prtcl]) * std::exp(m_logUnNormWeights[prtcl] - max);
        denom += std::exp(m_logUnNormWeights[prtcl] - max);
      }
      m_expectations[fId] = numer / denom;

// print stuff if debug mode is on
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

    // advance time step
    m_now += 1;
  }
}

//! A base class for the Sequential Important Sampling with Resampling (SISR).
//! Uses normal common random numbers.
/**
 * @class SISRFilterCRN
 * @author taylor
 * @file sisr_filter.h
 * @brief SISR filter but with common random numbers this time.
 * @tparam nparts the number of particles
 * @tparam dimx the size of the state
 * @tparam dimy the size of the observation
 * @tparam dimu the size of the normal random vector
 * @tparam dimur the size of the normal random variables for resampling
 * @tparam resamp_t the type of resampler
 * @tparam float_t the type of floating point numbers used (e.g. float or
 * double)
 * @tparam debug whether to debug or not
 */
template <size_t nparts, size_t dimx, size_t dimy, size_t dimu, size_t dimur,
          typename resamp_t, typename float_t, bool debug = false>
class SISRFilterCRN
    : public bases::pf_base_crn<float_t, dimy, dimx, dimu, dimur, nparts> {
private:
  /** "state size vector" type alias for linear algebra stuff */
  using ssv = Eigen::Matrix<float_t, dimx, 1>;
  /** "obs size vector" type alias for linear algebra stuff */
  using osv = Eigen::Matrix<float_t, dimy, 1>; // obs size vec
  /** "u sized vector" type alias for common random normal vector */
  using usv = Eigen::Matrix<float_t, dimu, 1>;
  /** "u sized vector for resampling"  type alias for common random normal
   * vector that's used in systematic resampling */
  using usvr = Eigen::Matrix<float_t, dimur, 1>;
  /** type alias for linear algebra stuff */
  using Mat = Eigen::Matrix<float_t, Eigen::Dynamic, Eigen::Dynamic>;
  /** type alias for linear algebra stuff */
  using arrayStates = std::array<ssv, nparts>;
  /** type alias for array of common random numbers */
  using arrayUs = std::array<usv, nparts>;
  /** type alias for array of float_ts */
  using arrayfloat_t = std::array<float_t, nparts>;

public:
  /**
   * @brief The (one and only) constructor.
   * @param rs the resampling schedule (resample every rs time points).
   */
  SISRFilterCRN(const unsigned int &rs = 1);

  /**
   * @brief The (virtual) destructor.
   */
  virtual ~SISRFilterCRN();

  /**
   * @brief Returns the most recent (log-) conditional likelihood.
   * @return log p(y_t | y_{1:t-1}) or log p(y_1)
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
   * @brief updates filtering distribution on a new datapoint.
   * Optionally stores expectations of functionals.
   * @param data the most recent data point
   * @param Uarr the U samples that get used to sample from the state proposal
   * @param Uresamp the U sample that is used to resample
   * @param fs a vector of functions if you want to calculate expectations.
   */
  void filter(const osv &data, const arrayUs &Uarr, const usvr &Uresamp,
              const std::vector<std::function<const Mat(const ssv &)>> &fs =
                  std::vector<std::function<const Mat(const ssv &)>>());

  /**
   * @brief  Calculate muEv or logmuEv
   * @param x1 is a const Vec& describing the state sample
   * @return the density or log-density evaluation as a float_t
   */
  virtual float_t logMuEv(const ssv &x1) = 0;

  /**
   * @brief "Samples" from time 1 proposal. Really, it maps the normal random
   * vector into the sample.
   * @param U the normal random vector transformed into X1i
   * @param y1 is a const Vec& representing the first observed datum
   * @return the sample as a Vec
   */
  virtual ssv Xi1(const usv &U, const osv &y1) = 0;

  /**
   * @brief Calculate q1Ev or log q1Ev
   * @param x1 is a const Vec& describing the time 1 state sample
   * @param y1 is a const Vec& describing the time 1 datum
   * @return the density or log-density evaluation as a float_t
   */
  virtual float_t logQ1Ev(const ssv &x1, const osv &y1) = 0;

  /**
   * @brief Calculate gEv or logGEv
   * @param yt is a const Vec& describing the time t datum
   * @param xt is a const Vec& describing the time t state
   * @return the density or log-density evaluation as a float_t
   */
  virtual float_t logGEv(const osv &yt, const ssv &xt) = 0;

  /**
   * @brief Evaluates the state transition density.
   * @param xt the current state
   * @param xtm1 the previous state
   * @return a float_t evaluaton of the log density/pmf
   */
  virtual float_t logFEv(const ssv &xt, const ssv &xtm1) = 0;

  /**
   * @brief "Samples" from the proposal/instrumental/importance density at time
   * t. Really, it maps the normal random vector into the sample.
   * @param xtm1 the previous state sample
   * @param U the normal random vector transformed
   * @param yt the current observation
   * @return a state sample for the current time xt
   */
  virtual ssv Xit(const ssv &xtm1, const usv &U, const osv &yt) = 0;

  /**
   * @brief Evaluates the proposal/instrumental/importance density/pmf
   * @param xt current state
   * @param xtm1 previous state
   * @param yt current observation
   * @return a float_t evaluation of the log density/pmf
   */
  virtual float_t logQEv(const ssv &xt, const ssv &xtm1, const osv &yt) = 0;

protected:
  /** @brief particle samples */
  arrayStates m_particles;

  /** @brief particle weights */
  arrayfloat_t m_logUnNormWeights;

  /** @brief current time point */
  unsigned int m_now;

  /** @brief log p(y_t|y_{1:t-1}) or log p(y1) */
  float_t m_logLastCondLike;

  /** @brief resampling object */
  resamp_t m_resampler;

  /** @brief expectations E[h(x_t) | y_{1:t}] for user defined "h"s */
  std::vector<Mat> m_expectations; // stores any sample averages the user wants

  /** @brief resampling schedule (e.g. resample every __ time points) */
  unsigned int m_resampSched;
};

template <size_t nparts, size_t dimx, size_t dimy, size_t dimu, size_t dimur,
          typename resamp_t, typename float_t, bool debug>
SISRFilterCRN<nparts, dimx, dimy, dimu, dimur, resamp_t, float_t,
              debug>::SISRFilterCRN(const unsigned int &rs)
    : m_now(0), m_logLastCondLike(0.0), m_resampSched(rs) {
  std::fill(m_logUnNormWeights.begin(), m_logUnNormWeights.end(),
            0.0); // log(1) = 0
}

template <size_t nparts, size_t dimx, size_t dimy, size_t dimu, size_t dimur,
          typename resamp_t, typename float_t, bool debug>
SISRFilterCRN<nparts, dimx, dimy, dimu, dimur, resamp_t, float_t,
              debug>::~SISRFilterCRN() {}

template <size_t nparts, size_t dimx, size_t dimy, size_t dimu, size_t dimur,
          typename resamp_t, typename float_t, bool debug>
float_t SISRFilterCRN<nparts, dimx, dimy, dimu, dimur, resamp_t, float_t,
                      debug>::getLogCondLike() const {
  return m_logLastCondLike;
}

template <size_t nparts, size_t dimx, size_t dimy, size_t dimu, size_t dimur,
          typename resamp_t, typename float_t, bool debug>
auto SISRFilterCRN<nparts, dimx, dimy, dimu, dimur, resamp_t, float_t,
                   debug>::getExpectations() const -> std::vector<Mat> {
  return m_expectations;
}

template <size_t nparts, size_t dimx, size_t dimy, size_t dimu, size_t dimur,
          typename resamp_t, typename float_t, bool debug>
void SISRFilterCRN<nparts, dimx, dimy, dimu, dimur, resamp_t, float_t, debug>::
    filter(const osv &data, const arrayUs &Uarr, const usvr &Uresamp,
           const std::vector<std::function<const Mat(const ssv &)>> &fs) {

  if (m_now > 0) {

    // try to iterate over particles all at once
    ssv newSamp;
    arrayfloat_t oldLogUnNormWts = m_logUnNormWeights;
    float_t maxOldLogUnNormWts(-std::numeric_limits<float_t>::infinity());
    for (size_t ii = 0; ii < nparts; ++ii) {

      // update max of old logUnNormWts before you change the element
      if (m_logUnNormWeights[ii] > maxOldLogUnNormWts)
        maxOldLogUnNormWts = m_logUnNormWeights[ii];

      // sample and get weight adjustments
      newSamp = Xit(m_particles[ii], Uarr[ii], data);
      m_logUnNormWeights[ii] += logFEv(newSamp, m_particles[ii]);
      m_logUnNormWeights[ii] += logGEv(data, newSamp);
      m_logUnNormWeights[ii] -= logQEv(newSamp, m_particles[ii], data);

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
      sumExp2 += std::exp(oldLogUnNormWts[i] - maxOldLogUnNormWts);
    }
    m_logLastCondLike =
        maxNumer + std::log(sumExp1) - maxOldLogUnNormWts - std::log(sumExp2);

    // calculate expectations before you resample
    unsigned int fId(0);
    float_t weightNormConst(0.0);
    for (auto &h : fs) { // iterate over all functions

      Mat testOut = h(m_particles[0]);
      unsigned int rows = testOut.rows();
      unsigned int cols = testOut.cols();
      Mat numer = Mat::Zero(rows, cols);
      float_t denom(0.0);

      for (size_t prtcl = 0; prtcl < nparts;
           ++prtcl) { // iterate over all particles
        numer += h(m_particles[prtcl]) *
                 std::exp(m_logUnNormWeights[prtcl] - maxNumer);
        denom += std::exp(m_logUnNormWeights[prtcl] - maxNumer);
      }
      m_expectations[fId] = numer / denom;

// print stuff if debug mode is on
#ifndef DROPPINGTHISINRPACKAGE
      if constexpr (debug)
        std::cout << "transposed expectation " << fId << ": "
                  << m_expectations[fId].transpose() << "\n";
#endif

      fId++;
    }

    // resample if you should
    if ((m_now + 1) % m_resampSched == 0)
      m_resampler.resampLogWts(m_particles, m_logUnNormWeights, Uresamp);

    // advance time
    m_now += 1;
  } else // (m_now == 0) //time 1
  {

    // only need to iterate over particles once
    float_t sumWts(0.0);
    for (size_t ii = 0; ii < nparts; ++ii) {
      // sample particles
      m_particles[ii] = Xi1(Uarr[ii], data);
      m_logUnNormWeights[ii] += logMuEv(m_particles[ii]);
      m_logUnNormWeights[ii] += logGEv(data, m_particles[ii]);
      m_logUnNormWeights[ii] -= logQ1Ev(m_particles[ii], data);

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
    m_expectations.resize(fs.size());
    unsigned int fId(0);
    for (auto &h : fs) {

      Mat testOut = h(m_particles[0]);
      unsigned int rows = testOut.rows();
      unsigned int cols = testOut.cols();
      Mat numer = Mat::Zero(rows, cols);
      float_t denom(0.0);

      for (size_t prtcl = 0; prtcl < nparts; ++prtcl) {
        numer +=
            h(m_particles[prtcl]) * std::exp(m_logUnNormWeights[prtcl] - max);
        denom += std::exp(m_logUnNormWeights[prtcl] - max);
      }
      m_expectations[fId] = numer / denom;

// print stuff if debug mode is on
#ifndef DROPPINGTHISINRPACKAGE
      if constexpr (debug)
        std::cout << "transposed expectation " << fId << ": "
                  << m_expectations[fId].transpose() << "\n";
#endif

      fId++;
    }

    // resample if you should
    if ((m_now + 1) % m_resampSched == 0)
      m_resampler.resampLogWts(m_particles, m_logUnNormWeights, Uresamp);

    // advance time step
    m_now += 1;
  }
}

} // namespace filters
} // namespace pf

#endif // SISR_FILTER_H
