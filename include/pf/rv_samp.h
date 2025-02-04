#ifndef RV_SAMP_H
#define RV_SAMP_H

#include <chrono>

#ifdef DROPPINGTHISINRPACKAGE
#include <RcppEigen.h>
// [[Rcpp::depends(RcppEigen)]]
#else
#include <Eigen/Dense>
#endif

#include <random>

namespace pf {

namespace rvsamp {

//! Base class for all random variable sampler types. Primary benefit is that it
//! sets the seed for you.
/**
 * @class rvsamp_base
 * @author taylor
 * @file rv_samp.h
 * @brief all rv samplers must inherit from this.
 * @tparam dim the dimension of each random vector sample.
 */
class rvsamp_base {
public:
  /**
   * @brief The default constructor. This is the only option available. Sets the
   * seed with the clock.
   */
  inline rvsamp_base()
      : m_rng{
            static_cast<std::uint32_t>(std::chrono::high_resolution_clock::now()
                                           .time_since_epoch()
                                           .count())} {}

protected:
  /** @brief prng */
  std::mt19937 m_rng;
};

//! A class that performs sampling from a univariate Normal distribution.
/**
 * @class UnivNormSampler
 * @author taylor
 * @file rv_samp.h
 * @brief Samples from univariate Normal distribution.
 */
template <typename float_t> class UnivNormSampler : public rvsamp_base {

public:
  /**
   * @brief Default-constructor sets up for standard Normal random variate
   * generation.
   */
  UnivNormSampler();

  /**
   * @brief The user must supply both mean and std. dev.
   * @param mu a float_t for the mean of the sampling distribution.
   * @param sigma a float_t (> 0) representing the standard deviation of the
   * samples.
   */
  UnivNormSampler(float_t mu, float_t sigma);

  /**
   * @brief sets the standard deviation of the sampler.
   * @param sigma the desired standard deviation.
   */
  void setStdDev(float_t sigma);

  /**
   * @brief sets the mean of the sampler.
   * @param mu the desired mean.
   */
  void setMean(float_t mu);

  /**
   * @brief Draws a random number.
   * @return a random sample of type float_t.
   */
  float_t sample();

private:
  /** @brief makes normal random variates */
  std::normal_distribution<float_t> m_z_gen;

  /** @brief the mean */
  float_t m_mu;

  /** @brief the standard deviation */
  float_t m_sigma;
};

template <typename float_t>
UnivNormSampler<float_t>::UnivNormSampler() : rvsamp_base(), m_z_gen(0.0, 1.0) {
  setMean(0.0);
  setStdDev(1.0);
}

template <typename float_t>
UnivNormSampler<float_t>::UnivNormSampler(float_t mu, float_t sigma)
    : rvsamp_base(), m_z_gen(0.0, 1.0) {
  setMean(mu);
  setStdDev(sigma);
}

template <typename float_t> void UnivNormSampler<float_t>::setMean(float_t mu) {
  m_mu = mu;
}

template <typename float_t>
void UnivNormSampler<float_t>::setStdDev(float_t sigma) {
  m_sigma = sigma;
}

template <typename float_t> float_t UnivNormSampler<float_t>::sample() {
  return m_mu + m_sigma * m_z_gen(m_rng);
}

//! A class that performs sampling from Student's T distribution.
/**
 * @class UnivStudTSampler
 * @author taylor
 * @file rv_samp.h
 * @brief Samples from from Student's T distribution.
 */
template <typename float_t> class UnivStudTSampler : public rvsamp_base {

public:
  /**
   * @brief Default-construction is deleted.
   */
  UnivStudTSampler() = delete;

  /**
   * @brief The user must supply the degrees of freedom parameter.
   * @param dof the degrees of freedom that are desired.
   */
  UnivStudTSampler(float_t dof);

  /**
   * @brief Draws a random number.
   * @return a random sample of type float_t.
   */
  float_t sample();

private:
  /** @brief makes t random variates */
  std::student_t_distribution<float_t> m_t_gen;
};

template <typename float_t>
UnivStudTSampler<float_t>::UnivStudTSampler(float_t dof)
    : rvsamp_base(), m_t_gen(dof) {}

template <typename float_t> float_t UnivStudTSampler<float_t>::sample() {
  return m_t_gen(m_rng);
}

//! A class that performs sampling from a univariate Log-Normal distribution.
/**
 * @class UnivLogNormSampler
 * @author taylor
 * @file rv_samp.h
 * @brief Samples from univariate Log-Normal distribution.
 */
template <typename float_t> class UnivLogNormSampler : public rvsamp_base {

public:
  /**
   * @brief Default-constructor sets up for standard Normal random variate
   * generation.
   */
  UnivLogNormSampler();

  /**
   * @brief The user must supply both mu and sigma.
   * @param mu a location parameter for the logarithm of the sample.
   * @param sigma a positive scale parameter for the logarithm of the sample.
   */
  UnivLogNormSampler(float_t mu, float_t sigma);

  /**
   * @brief sets the scale parameter of the logged random variable.
   * @param sigma the desired parameter.
   */
  void setSigma(float_t sigma);

  /**
   * @brief sets the location parameter of the logged random variable.
   * @param mu the desired parameter.
   */
  void setMu(float_t mu);

  /**
   * @brief draws a random number.
   * @return a random sample of type float_t.
   */
  float_t sample();

private:
  /** @brief makes normal random variates */
  std::normal_distribution<float_t> m_z_gen;

  /** @brief mu */
  float_t m_mu;

  /** @brief sigma */
  float_t m_sigma;
};

template <typename float_t>
UnivLogNormSampler<float_t>::UnivLogNormSampler()
    : rvsamp_base(), m_z_gen(0.0, 1.0) {
  setMu(0.0);
  setSigma(1.0);
}

template <typename float_t>
UnivLogNormSampler<float_t>::UnivLogNormSampler(float_t mu, float_t sigma)
    : rvsamp_base(), m_z_gen(0.0, 1.0) {
  setMu(mu);
  setSigma(sigma);
}

template <typename float_t>
void UnivLogNormSampler<float_t>::setMu(float_t mu) {
  m_mu = mu;
}

template <typename float_t>
void UnivLogNormSampler<float_t>::setSigma(float_t sigma) {
  m_sigma = sigma;
}

template <typename float_t> float_t UnivLogNormSampler<float_t>::sample() {
  return std::exp(m_mu + m_sigma * m_z_gen(m_rng));
}

//! A class that performs sampling from a univariate Gamma distribution.
/**
 * @class UnivGammaSampler
 * @author taylor
 * @file rv_samp.h
 * @brief Samples from univariate Gamma distribution.
 */
template <typename float_t> class UnivGammaSampler : public rvsamp_base {

public:
  /**
   * @brief Default-constructor ...
   */
  UnivGammaSampler() = delete;

  /**
   * @param alpha a positive shape parameter.
   * @param beta a positive scale parameter.
   */
  UnivGammaSampler(float_t alpha, float_t beta);

  /**
   * @brief draws a random number.
   * @return a random sample of type float_t.
   */
  float_t sample();

private:
  /** @brief makes gamma random variates */
  std::gamma_distribution<float_t> m_gamma_gen;

  /** @brief mu */
  float_t m_alpha;

  /** @brief sigma */
  float_t m_beta;
};

template <typename float_t>
UnivGammaSampler<float_t>::UnivGammaSampler(float_t alpha, float_t beta)
    : rvsamp_base(), m_gamma_gen(alpha, beta) {}

template <typename float_t> float_t UnivGammaSampler<float_t>::sample() {
  return m_gamma_gen(m_rng);
}

//! A class that performs sampling from a univariate Inverse Gamma distribution.
/**
 * @class UnivInvGammaSampler
 * @author taylor
 * @file rv_samp.h
 * @brief Samples from univariate Inverse Gamma distribution.
 */
template <typename float_t> class UnivInvGammaSampler : public rvsamp_base {

public:
  /**
   * @brief Default-constructor ...
   */
  UnivInvGammaSampler();

  /**
   * @param alpha a positive shape parameter.
   * @param beta a positive scale parameter.
   */
  UnivInvGammaSampler(float_t alpha, float_t beta);

  /**
   * @brief draws a random number.
   * @return a random sample of type float_t.
   */
  float_t sample();

private:
  /** @brief makes gamma random variates that we take the reciprocal of*/
  std::gamma_distribution<float_t> m_gamma_gen;

  /** @brief mu */
  float_t m_alpha;

  /** @brief sigma */
  float_t m_beta;
};

template <typename float_t>
UnivInvGammaSampler<float_t>::UnivInvGammaSampler()
    : rvsamp_base(), m_gamma_gen(1.0, 1.0) {}

template <typename float_t>
UnivInvGammaSampler<float_t>::UnivInvGammaSampler(float_t alpha, float_t beta)
    : rvsamp_base(), m_gamma_gen(alpha, beta) {}

template <typename float_t> float_t UnivInvGammaSampler<float_t>::sample() {
  return 1.0 / m_gamma_gen(m_rng);
}

//! A class that performs sampling from a truncated univariate Normal
//! distribution.
/**
 * @class TruncUnivNormSampler
 * @author taylor
 * @file rv_samp.h
 * @brief Samples from a truncated univariate Normal distribution using the
 * acceptance rejection method. The proposal distribution used is a normal
 * distribution with the same location and scale parameters as the target.
 * As a result, this method will take a long time when the width of the
 * support of the target is narrow.
 */
template <typename float_t> class TruncUnivNormSampler : public rvsamp_base {

public:
  /**
   * @brief The user must supply both mean and std. dev.
   * @param mu a float_t for the location parameter.
   * @param sigma a float_t (> 0) representing the scale of the samples.
   * @param lower the lower bound of the support
   * @param upper the upper bound of the support
   */
  TruncUnivNormSampler(float_t mu, float_t sigma, float_t lower, float_t upper);

  /**
   * @brief Draws a random number.
   * @return a random sample of type float_t.
   */
  float_t sample();

private:
  /** @brief makes normal random variates */
  std::normal_distribution<float_t> m_z_gen;

  /** @brief the mean */
  float_t m_mu;

  /** @brief the standard deviation */
  float_t m_sigma;

  /** @brief the lower bound */
  float_t m_lower;

  /** @brief the upper bound */
  float_t m_upper;
};

template <typename float_t>
TruncUnivNormSampler<float_t>::TruncUnivNormSampler(float_t mu, float_t sigma,
                                                    float_t lower,
                                                    float_t upper)
    : rvsamp_base(), m_z_gen(0.0, 1.0), m_mu(mu), m_sigma(sigma),
      m_lower(lower), m_upper(upper) {}

template <typename float_t> float_t TruncUnivNormSampler<float_t>::sample() {
  float_t proposal;
  bool accepted = false;
  while (!accepted) {
    proposal = m_mu + m_sigma * m_z_gen(this->m_rng);
    if ((m_lower <= proposal) & (proposal <= m_upper))
      accepted = true;
  }
  return proposal;
}

//! A class that performs sampling from a Poisson distribution.
/**
 * @class PoissonSampler
 * @author taylor
 * @file rv_samp.h
 * @brief Samples from univariate Poisson distribution.
 */
template <typename float_t, typename int_t>
class PoissonSampler : public rvsamp_base {

public:
  /**
   * @brief Default-constructor sets up for Poisson random variate generation
   * with lambda = 1.
   */
  PoissonSampler();

  /**
   * @brief Constructs Poisson sampler with user-specified lambda.
   * @param lambda a float_t for the average/variance.
   */
  PoissonSampler(float_t lambda);

  /**
   * @brief sets the parameter lambda.
   * @param lambda (the average and the variance).
   */
  void setLambda(float_t lambda);

  /**
   * @brief Draws a random number.
   * @return a random sample of type int_t.
   */
  int_t sample();

private:
  /** @brief makes normal random variates */
  std::poisson_distribution<int_t> m_p_gen;
};

template <typename float_t, typename int_t>
PoissonSampler<float_t, int_t>::PoissonSampler()
    : rvsamp_base(), m_p_gen(float_t(1.0)) {}

template <typename float_t, typename int_t>
PoissonSampler<float_t, int_t>::PoissonSampler(float_t lambda)
    : rvsamp_base(), m_p_gen(lambda) {}

template <typename float_t, typename int_t>
void PoissonSampler<float_t, int_t>::setLambda(float_t lambda) {
  m_p_gen.param(typename decltype(m_p_gen)::param_type(lambda));
}

template <typename float_t, typename int_t>
int_t PoissonSampler<float_t, int_t>::sample() {
  return m_p_gen(m_rng);
}

//! A class that performs sampling from a univariate Bernoulli distribution.
/**
 * @class BernSampler
 * @author taylor
 * @file rv_samp.h
 * @brief Samples from univariate Bernoulli distribution.
 */
template <typename float_t, typename int_t>
class BernSampler : public rvsamp_base {

public:
  /**
   * @brief Default-constructor sets up for Bernoulli random variate generation
   * with p = .5.
   */
  BernSampler();

  /**
   * @brief Constructs Bernoulli sampler with user-specified p.
   * @param p a float_t for the probability that the rv equals 1.
   */
  BernSampler(float_t p);

  /**
   * @brief sets the parameter p.
   * @param p the p(X=1) = 1-p(X=0).
   */
  void setP(float_t p);

  /**
   * @brief Draws a random number.
   * @return a random sample of type int_t.
   */
  int_t sample();

private:
  /** @brief makes normal random variates */
  std::bernoulli_distribution m_B_gen;

  /** @brief the mean */
  float_t m_p;
};

template <typename float_t, typename int_t>
BernSampler<float_t, int_t>::BernSampler() : rvsamp_base(), m_B_gen(.5) {}

template <typename float_t, typename int_t>
BernSampler<float_t, int_t>::BernSampler(float_t p)
    : rvsamp_base(), m_B_gen(p) {}

template <typename float_t, typename int_t>
void BernSampler<float_t, int_t>::setP(float_t p) {
  m_p = p;
}

template <typename float_t, typename int_t>
int_t BernSampler<float_t, int_t>::sample() {
  return (m_B_gen(m_rng)) ? 1 : 0;
}

//! A class that performs sampling from a multivariate normal distribution.
/**
 * @class MVNSampler
 * @author taylor
 * @file rv_samp.h
 * @brief Can sample from a distribution with fixed mean and covariance, fixed
 * mean only, fixed covariance only, or nothing fixed.
 */
template <size_t dim, typename float_t> class MVNSampler : public rvsamp_base {
public:
  /** type alias for linear algebra stuff */
  using Vec = Eigen::Matrix<float_t, dim, 1>;
  /** type alias for linear algebra stuff */
  using Mat = Eigen::Matrix<float_t, dim, dim>;

  /**
   * @todo: implement move semantics
   */

  /**
   * @brief Default-constructor sets up for multivariate standard Normal random
   * variate generation.
   */
  MVNSampler();

  /**
   * @brief The user must supply both mean and covariance matrix.
   * @param meanVec a Vec for the mean vector of the sampling distribution.
   * @param covMat a Mat representing the covariance matrix of the samples.
   */
  MVNSampler(const Vec &meanVec, const Mat &covMat);

  /**
   * @brief sets the covariance matrix of the sampler.
   * @param covMat the desired covariance matrix.
   */
  void setCovar(const Mat &covMat);

  /**
   * @brief sets the mean vector of the sampler.
   * @param meanVec the desired mean vector.
   */
  void setMean(const Vec &meanVec);

  /**
   * @brief Draws a random vector.
   * @return a Vec random sample.
   */
  auto sample() -> Vec;

private:
  /** @brief makes normal random variates */
  std::normal_distribution<float_t> m_z_gen;

  /** @brief covariance matrix */
  Mat m_scale_mat;

  /** @brief mean vector */
  Vec m_mean;
};

template <size_t dim, typename float_t>
MVNSampler<dim, float_t>::MVNSampler() : rvsamp_base(), m_z_gen(0.0, 1.0) {
  setMean(Vec::Zero());
  setCovar(Mat::Identity());
}

template <size_t dim, typename float_t>
MVNSampler<dim, float_t>::MVNSampler(const Vec &meanVec, const Mat &covMat)
    : rvsamp_base(), m_z_gen(0.0, 1.0) {
  setCovar(covMat);
  setMean(meanVec);
}

template <size_t dim, typename float_t>
void MVNSampler<dim, float_t>::setCovar(const Mat &covMat) {
  Eigen::SelfAdjointEigenSolver<Mat> eigenSolver(covMat);
  m_scale_mat = eigenSolver.eigenvectors() *
                eigenSolver.eigenvalues().cwiseMax(0).cwiseSqrt().asDiagonal();
}

template <size_t dim, typename float_t>
void MVNSampler<dim, float_t>::setMean(const Vec &meanVec) {
  m_mean = meanVec;
}

template <size_t dim, typename float_t>
auto MVNSampler<dim, float_t>::sample() -> Vec {
  Vec Z;
  for (size_t i = 0; i < dim; ++i) {
    Z(i) = m_z_gen(this->m_rng);
  }
  return m_mean + m_scale_mat * Z;
}

//! A class that performs sampling from a continuous uniform distribution.
/**
 * @class UniformSampler
 * @author taylor
 * @file rv_samp.h
 * @brief
 */
template <typename float_t> class UniformSampler : public rvsamp_base {
public:
  /**
   * @brief The default constructor. Gives a lower bound of 0 and upper bound
   * of 1.
   */
  UniformSampler();

  /**
   * @brief The constructor
   * @param lower the lower bound of the PRNG.
   * @param upper the upper bound of the PRNG.
   */
  UniformSampler(float_t lower, float_t upper);

  /**
   * @brief Draws a sample.
   * @return a sample of type float_t.
   */
  float_t sample();

private:
  /** @brief makes uniform random variates */
  std::uniform_real_distribution<float_t> m_unif_gen;
};

template <typename float_t>
UniformSampler<float_t>::UniformSampler()
    : rvsamp_base(), m_unif_gen(0.0, 1.0) {}

template <typename float_t>
UniformSampler<float_t>::UniformSampler(float_t lower, float_t upper)
    : rvsamp_base(), m_unif_gen(lower, upper) {}

template <typename float_t> float_t UniformSampler<float_t>::sample() {
  return m_unif_gen(m_rng);
}

//! A class that performs sampling with replacement (useful for the index
//! sampler in an APF)
/**
 * @class k_gen
 * @author taylor
 * @file rv_samp.h
 * @brief Basically a wrapper for std::discrete_distribution<>
 * outputs are in the rage (0,1,...N-1)
 */
template <size_t N, typename float_t> class k_gen : public rvsamp_base {
public:
  /**
   * @brief default constructor. only one available.
   */
  k_gen();

  /**
   * @brief sample N times from (0,1,...N-1)
   * @param logWts possibly unnormalized type std::array<float_t, N>
   * @return the integers in a std::array<unsigned int, N>
   */
  std::array<unsigned int, N> sample(const std::array<float_t, N> &logWts);
};

template <size_t N, typename float_t>
k_gen<N, float_t>::k_gen() : rvsamp_base() {}

template <size_t N, typename float_t>
std::array<unsigned int, N>
k_gen<N, float_t>::sample(const std::array<float_t, N> &logWts) {
  // these log weights may be very negative. If that's the case, exponentiating
  // them may cause underflow so we use the "log-exp-sum" trick actually not
  // quite...we just shift the log-weights because after they're exponentiated
  // they have the same normalized probabilities

  // Create the distribution with exponentiated log-weights
  // subtract the max first to prevent underflow
  // normalization is taken care of by std::discrete_distribution
  std::array<float_t, N> w;
  float_t m = *std::max_element(logWts.begin(), logWts.end());
  std::transform(logWts.begin(), logWts.end(), w.begin(),
                 [&m](float_t d) -> float_t { return std::exp(d - m); });
  std::discrete_distribution<> kGen(w.begin(), w.end());

  // sample and return ks
  std::array<unsigned int, N> ks;
  for (size_t i = 0; i < N; ++i) {
    ks[i] = kGen(this->m_rng);
  }
  return ks;
}

//! A class that performs sampling from a Beta distribution.
/**
 * @class BetaSampler
 * @author taylor
 * @file rv_samp.h
 * @brief Samples from Beta distribution.
 */
template <typename float_t> class BetaSampler : public rvsamp_base {

public:
  /**
   * @brief Default-constructor sets up for Beta(1,1) random variate generation.
   */
  BetaSampler() = delete;

  /**
   * @brief The user must supply both alpha and beta
   * @param alpha shape 1 parameter (> 0)
   * @param beta shape 2 parameter (> 0)
   */
  BetaSampler(float_t alpha, float_t beta);

  /**
   * @brief Draws a random number.
   * @return a random sample of type float_t.
   */
  float_t sample();

private:
  /** @brief makes gamma random variates */
  std::gamma_distribution<float_t> m_first_gamma_gen;

  /** @brief makes other gamma random variates */
  std::gamma_distribution<float_t> m_second_gamma_gen;

  /** @brief the first shape parameter */
  float_t m_alpha;

  /** @brief the second shape parameter */
  float_t m_beta;
};

template <typename float_t>
BetaSampler<float_t>::BetaSampler(float_t alpha, float_t beta)
    : rvsamp_base(), m_first_gamma_gen(alpha, 1.0),
      m_second_gamma_gen(beta, 1.0) {}

template <typename float_t> float_t BetaSampler<float_t>::sample() {
  float_t first = m_first_gamma_gen(m_rng);
  float_t second = m_second_gamma_gen(m_rng);
  return first / (first + second);
}

} // namespace rvsamp

} // namespace pf

#endif // RV_SAMP_H
