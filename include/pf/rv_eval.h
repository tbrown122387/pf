#ifndef RV_EVAL_H
#define RV_EVAL_H

#include <cstddef> // std::size_t

#ifdef DROPPINGTHISINRPACKAGE
#include <RcppEigen.h>
// [[Rcpp::depends(RcppEigen)]]
#else
#include <Eigen/Dense>
#endif

#include "boost/math/special_functions.hpp"
#include <iostream> // cerr

namespace pf {

namespace rveval {

////////////////////////////////////////////////
/////////         Constants            /////////
////////////////////////////////////////////////

/** (2 pi)^(-1/2) */
template <class T> constexpr T inv_sqrt_2pi = T(0.3989422804014327);

/** (2/pi)^(1/2) */
template <class T> constexpr T sqrt_two_over_pi(0.797884560802865);

/** log(2pi) */
template <class T> constexpr T log_two_pi(1.83787706640935);

/** log(2/pi) */
template <class T> constexpr T log_two_over_pi(-0.451582705289455);

/** log(pi) */
template <class T> constexpr T log_pi(1.1447298858494);

////////////////////////////////////////////////
/////////      Transformations         /////////
////////////////////////////////////////////////

/**
 * @brief Maps (-1, 1) to the reals.
 * @param phi
 * @return psi
 */
template <typename float_t> float_t twiceFisher(float_t phi) {
  if ((phi <= -1.0) || (phi >= 1.0))
    throw std::invalid_argument("error: phi was not between -1 and 1");
  else
    return std::log(1.0 + phi) - std::log(1.0 - phi);
}

/**
 * @brief Maps a real number to the itnerval (-1,1).
 * @param psi
 * @return phi
 */
template <typename float_t> float_t invTwiceFisher(float_t psi) {
  float_t ans = (1.0 - std::exp(psi)) / (-1.0 - std::exp(psi));

  if ((ans <= -1.0) || (ans >= 1.0))
    std::cerr << "error: there was probably overflow for exp(psi) \n";

  return ans;
}

/**
 * @brief Maps (0,1) to the reals.
 * @param p
 * @return logit(p)
 */
template <typename float_t> float_t logit(float_t p) {
  if ((p <= 0.0) || (p >= 1.0))
    std::cerr << "error: p was not between 0 and 1 \n";

  return std::log(p) - std::log(1.0 - p);
}

/**
 * @brief Maps the reals to (0,1)
 * @param r
 * @return p = invlogit(p)
 */
template <typename float_t> float_t inv_logit(float_t r) {
  float_t ans = 1.0 / (1.0 + std::exp(-r));

  if ((ans <= 0.0) || (ans >= 1.0))
    std::cerr << "error: there was probably underflow for exp(-r) \n";

  return ans;
}

/**
 * @brief Maps the reals to the reals
 * @param r
 * @return log(invlogit(p))
 */
template <typename float_t> float_t log_inv_logit(float_t r) {
  if (r < -750.00 || r > 750.00)
    std::cerr << "warning: log_inv_logit might be under/over-flowing\n";
  return -std::log(1.0 + std::exp(-r));
}

/**
 * @brief calculates log-sum-exp in a way that prevents over/under-flow
 * @param a
 * @param b
 * @return a floating point number
 */
template <typename float_t> float_t log_sum_exp(float_t a, float_t b) {
  float_t m = std::max(a, b);
  return m + std::log(std::exp(a - m) + std::exp(b - m));
}

////////////////////////////////////////////////
/////////       float_t evals           /////////
////////////////////////////////////////////////

/**
 * @brief Evaluates the univariate Normal density.
 * @param x the point at which you're evaluating.
 * @param mu the mean.
 * @param sigma the standard deviation.
 * @param log true if you want the log-density. False otherwise.
 * @return a float_t evaluation.
 */
template <typename float_t>
float_t evalUnivNorm(float_t x, float_t mu, float_t sigma, bool log) {
  float_t exponent = -.5 * (x - mu) * (x - mu) / (sigma * sigma);
  if (sigma > 0.0) {
    if (log) {
      return -std::log(sigma) - .5 * log_two_pi<float_t> + exponent;
    } else {
      return inv_sqrt_2pi<float_t> * std::exp(exponent) / sigma;
    }
  } else {
    if (log) {
      return -std::numeric_limits<float_t>::infinity();
    } else {
      return 0.0;
    }
  }
}

/**
 * @brief Evaluates the unnormalized univariate Normal density. Use with care.
 * @param x the point at which you're evaluating.
 * @param mu the mean.
 * @param sigma the standard deviation.
 * @param log true if you want the log-unnormalized density. False otherwise.
 * @return a float_t evaluation.
 */
template <typename float_t>
float_t evalUnivNorm_unnorm(float_t x, float_t mu, float_t sigma, bool log) {
  float_t exponent = -.5 * (x - mu) * (x - mu) / (sigma * sigma);
  if (sigma > 0.0) {
    if (log) {
      return exponent;
    } else {
      return std::exp(exponent);
    }
  } else {
    if (log) {
      return -std::numeric_limits<float_t>::infinity();
    } else {
      return 0.0;
    }
  }
}

/**
 * @brief Evaluates the standard Normal CDF.
 * @param x the quantile.
 * @return the probability Z < x
 */
template <typename float_t>
float_t evalUnivStdNormCDF(float_t x) // john cook code
{
  // constants
  float_t a1 = 0.254829592;
  float_t a2 = -0.284496736;
  float_t a3 = 1.421413741;
  float_t a4 = -1.453152027;
  float_t a5 = 1.061405429;
  float_t p = 0.3275911;

  // Save the sign of x
  int sign = 1;
  if (x < 0)
    sign = -1;
  float_t xt = std::fabs(x) / std::sqrt(2.0);

  // A&S formula 7.1.26
  float_t t = 1.0 / (1.0 + p * xt);
  float_t y = 1.0 - (((((a5 * t + a4) * t) + a3) * t + a2) * t + a1) * t *
                        std::exp(-xt * xt);

  return 0.5 * (1.0 + sign * y);
}

/**
 * @brief Evaluates the univariate Beta density
 * @param x the point
 * @param alpha parameter 1
 * @param beta parameter 2
 * @param log true if you want log density
 * @return float_t evaluation.
 */
template <typename float_t>
float_t evalUnivBeta(float_t x, float_t alpha, float_t beta, bool log) {
  if ((x > 0.0) && (x < 1.0) && (alpha > 0.0) &&
      (beta > 0.0)) { // x in support and parameters acceptable
    if (log) {
      return std::lgamma(alpha + beta) - std::lgamma(alpha) -
             std::lgamma(beta) + (alpha - 1.0) * std::log(x) +
             (beta - 1.0) * std::log(1.0 - x);
    } else {
      return pow(x, alpha - 1.0) * pow(1.0 - x, beta - 1.0) *
             std::tgamma(alpha + beta) /
             (std::tgamma(alpha) * std::tgamma(beta));
    }

  } else { // not ( x in support and parameters acceptable )
    if (log) {
      return -std::numeric_limits<float_t>::infinity();
    } else {
      return 0.0;
    }
  }
}

/**
 * @brief Evaluates the unnormalized univariate Beta density. Use with care.
 * @param x the point
 * @param alpha parameter 1
 * @param beta parameter 2
 * @param log true if you want log unnormalized density
 * @return float_t evaluation.
 */
template <typename float_t>
float_t evalUnivBeta_unnorm(float_t x, float_t alpha, float_t beta, bool log) {
  if ((x > 0.0) && (x < 1.0) && (alpha > 0.0) &&
      (beta > 0.0)) { // x in support and parameters acceptable
    if (log) {
      return (alpha - 1.0) * std::log(x) + (beta - 1.0) * std::log(1.0 - x);
    } else {
      return pow(x, alpha - 1.0) * pow(1.0 - x, beta - 1.0);
    }

  } else { // not ( x in support and parameters acceptable )
    if (log) {
      return -std::numeric_limits<float_t>::infinity();
    } else {
      return 0.0;
    }
  }
}

/**
 * @brief Evaluates the univariate Inverse Gamma density
 * @param x the point
 * @param alpha shape parameter
 * @param beta rate parameter
 * @param log true if you want log density.
 * @return float_t evaluation.
 */
template <typename float_t>
float_t evalUnivInvGamma(float_t x, float_t alpha, float_t beta, bool log) {
  if ((x > 0.0) && (alpha > 0.0) &&
      (beta > 0.0)) { // x in support and acceptable parameters
    if (log) {
      return alpha * std::log(beta) - std::lgamma(alpha) -
             (alpha + 1.0) * std::log(x) - beta / x;
    } else {
      return pow(x, -alpha - 1.0) * exp(-beta / x) * pow(beta, alpha) /
             std::tgamma(alpha);
    }
  } else { // not ( x in support and acceptable parameters )
    if (log) {
      return -std::numeric_limits<float_t>::infinity();
    } else {
      return 0.0;
    }
  }
}

/**
 * @brief Evaluates the unnormalized univariate Inverse Gamma density. Use with
 * care.
 * @param x the point
 * @param alpha shape parameter
 * @param beta rate parameter
 * @param log true if you want log unnormalized density.
 * @return float_t evaluation.
 */
template <typename float_t>
float_t evalUnivInvGamma_unnorm(float_t x, float_t alpha, float_t beta,
                                bool log) {
  if ((x > 0.0) && (alpha > 0.0) &&
      (beta > 0.0)) { // x in support and acceptable parameters
    if (log) {
      return (-alpha - 1.0) * std::log(x) - beta / x;
    } else {
      return pow(x, -alpha - 1.0) * exp(-beta / x);
    }
  } else { // not ( x in support and acceptable parameters )
    if (log) {
      return -std::numeric_limits<float_t>::infinity();
    } else {
      return 0.0;
    }
  }
}

/**
 * @brief Evaluates the half-normal density
 * @param x the point you're evaluating at
 * @param sigmaSqd the scale parameter
 * @param log true if you want log density.
 * @return float_t evaluation.
 */
template <typename float_t>
float_t evalUnivHalfNorm(float_t x, float_t sigmaSqd, bool log) {
  if ((x >= 0.0) && (sigmaSqd > 0.0)) {
    if (log) {
      return .5 * log_two_over_pi<float_t> - .5 * std::log(sigmaSqd) -
             .5 * x * x / sigmaSqd;
    } else {
      return std::exp(-.5 * x * x / sigmaSqd) * sqrt_two_over_pi<float_t> /
             std::sqrt(sigmaSqd);
    }
  } else {
    if (log) {
      return -std::numeric_limits<float_t>::infinity();
    } else {
      return 0.0;
    }
  }
}

/**
 * @brief Evaluates the unnormalized half-normal density. Use with care.
 * @param x the point you're evaluating at
 * @param sigmaSqd the scale parameter
 * @param log true if you want log unnormalized density.
 * @return float_t evaluation.
 */
template <typename float_t>
float_t evalUnivHalfNorm_unnorm(float_t x, float_t sigmaSqd, bool log) {
  if ((x >= 0.0) && (sigmaSqd > 0.0)) {
    if (log) {
      return -.5 * x * x / sigmaSqd;
    } else {
      return std::exp(-.5 * x * x / sigmaSqd);
    }
  } else {
    if (log) {
      return -std::numeric_limits<float_t>::infinity();
    } else {
      return 0.0;
    }
  }
}

/**
 * @brief Evaluates a truncated Normal density.
 * @param x the quantile
 * @param mu the mode
 * @param sigma the scale parameter.
 * @param lower the lower truncation point (may be negative infinity)
 * @param upper the upper truncation point (may be positive infinity).
 * @param log true if you want the log density.
 * @return the floating point number
 */
template <typename float_t>
float_t evalUnivTruncNorm(float_t x, float_t mu, float_t sigma, float_t lower,
                          float_t upper, bool log) {
  if ((sigma > 0.0) && (lower <= x) & (x <= upper)) {
    if (log) {
      return evalUnivNorm(x, mu, sigma, true) -
             std::log(evalUnivStdNormCDF((upper - mu) / sigma) -
                      evalUnivStdNormCDF((lower - mu) / sigma));
    } else {
      return evalUnivNorm(x, mu, sigma, false) /
             (evalUnivStdNormCDF((upper - mu) / sigma) -
              evalUnivStdNormCDF((lower - mu) / sigma));
    }

  } else {
    if (log) {
      return -std::numeric_limits<float_t>::infinity();
    } else {
      return 0.0;
    }
  }
}

/**
 * @brief Evaluates the unnormalized truncated Normal density. Use with care.
 * @param x the quantile
 * @param mu the mode
 * @param sigma the scale parameter.
 * @param lower the lower truncation point (may be negative infinity)
 * @param upper the upper truncation point (may be positive infinity).
 * @param log true if you want the log unnormalized density.
 * @return the floating point number
 */
template <typename float_t>
float_t evalUnivTruncNorm_unnorm(float_t x, float_t mu, float_t sigma,
                                 float_t lower, float_t upper, bool log) {
  if ((sigma > 0.0) && (lower <= x) & (x <= upper)) {
    if (log) {
      return evalUnivNorm_unnorm(x, mu, sigma, true);
    } else {
      return evalUnivNorm_unnorm(x, mu, sigma, false);
    }
  } else {
    if (log) {
      return -std::numeric_limits<float_t>::infinity();
    } else {
      return 0.0;
    }
  }
}

/**
 * @brief Evaluates the logit-Normal distribution (see Wiki for more info)
 * @param x in [0,1] the point you're evaluating at
 * @param mu location parameter that can take any real number
 * @param sigma scale parameter that needs to be positive
 * @param log true if you want to evalute the log-density. False otherwise.
 * @return a float_t evaluation
 */
template <typename float_t>
float_t evalLogitNormal(float_t x, float_t mu, float_t sigma, bool log) {
  if ((x >= 0.0) && (x <= 1.0) && (sigma > 0.0)) {

    float_t exponent =
        -.5 * (logit(x) - mu) * (logit(x) - mu) / (sigma * sigma);
    if (log) {
      return -std::log(sigma) - .5 * log_two_pi<float_t> - std::log(x) -
             std::log(1.0 - x) + exponent;
    } else {
      return inv_sqrt_2pi<float_t> * std::exp(exponent) /
             (x * (1.0 - x) * sigma);
    }
  } else {
    if (log) {
      return -std::numeric_limits<float_t>::infinity();
    } else {
      return 0.0;
    }
  }
}

/**
 * @brief Evaluates the unnormalized logit-Normal distribution. Use with care.
 * @param x in [0,1] the point you're evaluating at
 * @param mu location parameter that can take any real number
 * @param sigma scale parameter that needs to be positive
 * @param log true if you want to evalute the log unnormalized density. False
 * otherwise.
 * @return a float_t evaluation
 */
template <typename float_t>
float_t evalLogitNormal_unnorm(float_t x, float_t mu, float_t sigma, bool log) {
  if ((x >= 0.0) && (x <= 1.0) && (sigma > 0.0)) {

    float_t exponent =
        -.5 * (logit(x) - mu) * (logit(x) - mu) / (sigma * sigma);
    if (log) {
      return -std::log(x) - std::log(1.0 - x) + exponent;
    } else {
      return std::exp(exponent) / x / (1.0 - x);
    }
  } else {
    if (log) {
      return -std::numeric_limits<float_t>::infinity();
    } else {
      return 0.0;
    }
  }
}

/**
 * @brief Evaluates what I call the "twice-fisher-Normal" distribution
 * @param x in [-1,1] the point you are evaluating at
 * @param mu the location parameter (all real numbers)
 * @param sigma the scale parameter (positive)
 * @param log true if you want to evaluate the log-density. False otherwise.
 * @return a float_t evaluation
 */
template <typename float_t>
float_t evalTwiceFisherNormal(float_t x, float_t mu, float_t sigma, bool log) {

  // https://stats.stackexchange.com/questions/321905/what-is-the-name-of-this-random-variable/321907#321907
  if ((x >= -1.0) && (x <= 1.0) && (sigma > 0.0)) {

    float_t exponent = std::log((1.0 + x) / (1.0 - x)) - mu;
    exponent = -.5 * exponent * exponent / sigma / sigma;
    if (log) {
      return -std::log(sigma) - .5 * log_two_pi<float_t> + std::log(2.0) -
             std::log(1.0 + x) - std::log(1.0 - x) + exponent;
    } else {
      return inv_sqrt_2pi<float_t> * 2.0 * std::exp(exponent) /
             ((1.0 - x) * (1.0 + x) * sigma);
    }
  } else {
    if (log) {
      return -std::numeric_limits<float_t>::infinity();
    } else {
      return 0.0;
    }
  }
}

/**
 * @brief Evaluates the unnormalized "twice-fisher-Normal" distribution. Use
 * with care.
 * @param x in [-1,1] the point you are evaluating at
 * @param mu the location parameter (all real numbers)
 * @param sigma the scale parameter (positive)
 * @param log true if you want to evaluate the log unnormalized density. False
 * otherwise.
 * @return a float_t evaluation
 */
template <typename float_t>
float_t evalTwiceFisherNormal_unnorm(float_t x, float_t mu, float_t sigma,
                                     bool log) {

  // https://stats.stackexchange.com/questions/321905/what-is-the-name-of-this-random-variable/321907#321907
  if ((x >= -1.0) && (x <= 1.0) && (sigma > 0.0)) {

    float_t exponent = std::log((1.0 + x) / (1.0 - x)) - mu;
    exponent = -.5 * exponent * exponent / sigma / sigma;
    if (log) {
      return -std::log(1.0 + x) - std::log(1.0 - x) + exponent;
    } else {
      return std::exp(exponent) / (1.0 - x) / (1.0 + x);
    }
  } else {
    if (log) {
      return -std::numeric_limits<float_t>::infinity();
    } else {
      return 0.0;
    }
  }
}

/**
 * @brief Evaluates the lognormal density
 * @param x in (0,infty) the point you are evaluating at
 * @param mu the location parameter
 * @param sigma in (0, infty) the scale parameter
 * @param log true if you want to evaluate the log-density. False otherwise.
 * @return a float_t evaluation
 */
template <typename float_t>
float_t evalLogNormal(float_t x, float_t mu, float_t sigma, bool log) {
  if ((x > 0.0) && (sigma > 0.0)) {

    float_t exponent = std::log(x) - mu;
    exponent = -.5 * exponent * exponent / sigma / sigma;
    if (log) {
      return -std::log(x) - std::log(sigma) - .5 * log_two_pi<float_t> +
             exponent;
    } else {
      return inv_sqrt_2pi<float_t> * std::exp(exponent) / (sigma * x);
    }
  } else {
    if (log) {
      return -std::numeric_limits<float_t>::infinity();
    } else {
      return 0.0;
    }
  }
}

/**
 * @brief Evaluates the unnormalized lognormal density. Use with care.
 * @param x in (0,infty) the point you are evaluating at
 * @param mu the location parameter
 * @param sigma in (0, infty) the scale parameter
 * @param log true if you want to evaluate the log unnormalized density. False
 * otherwise.
 * @return a float_t evaluation
 */
template <typename float_t>
float_t evalLogNormal_unnorm(float_t x, float_t mu, float_t sigma, bool log) {
  if ((x > 0.0) && (sigma > 0.0)) {

    float_t exponent = std::log(x) - mu;
    exponent = -.5 * exponent * exponent / sigma / sigma;
    if (log) {
      return -std::log(x) + exponent;
    } else {
      return std::exp(exponent) / x;
    }
  } else {
    if (log) {
      return -std::numeric_limits<float_t>::infinity();
    } else {
      return 0.0;
    }
  }
}

/**
 * @brief Evaluates the uniform density.
 * @param x in (lower, upper] the point you are evaluating at.
 * @param lower the lower bound of the support for x.
 * @param upper the upper bound for the support of x.
 * @param log true if you want to evaluate the log-density. False otherwise.
 * @return a float_t evaluation.
 */
template <typename float_t>
float_t evalUniform(float_t x, float_t lower, float_t upper, bool log) {

  if ((x > lower) && (x <= upper)) {

    float_t width = upper - lower;
    if (log) {
      return -std::log(width);
    } else {
      return 1.0 / width;
    }
  } else {
    if (log) {
      return -std::numeric_limits<float_t>::infinity();
    } else {
      return 0.0;
    }
  }
}

/**
 * @brief Evaluates the unnormalized uniform density. Use with care.
 * @param x in (lower, upper] the point you are evaluating at.
 * @param lower the lower bound of the support for x.
 * @param upper the upper bound for the support of x.
 * @param log true if you want to evaluate the log unnormalized density. False
 * otherwise.
 * @return a float_t evaluation.
 */
template <typename float_t>
float_t evalUniform_unnorm(float_t x, float_t lower, float_t upper, bool log) {

  if ((x > lower) && (x <= upper)) {

    if (log) {
      return 0.0;
    } else {
      return 1.0;
    }
  } else {
    if (log) {
      return -std::numeric_limits<float_t>::infinity();
    } else {
      return 0.0;
    }
  }
}

/**
 * @brief Evaluates the scaled t distribution.
 * @param x the percentile
 * @param mu the location parameter
 * @param sigma the scale parameter
 * @param dof the degrees of freedom
 * @param log true if you want the log of the unnormalized density. False
 * otherwise.
 * @return a floating point number
 */
template <typename float_t>
float_t evalScaledT(float_t x, float_t mu, float_t sigma, float_t dof,
                    bool log) {

  if ((sigma > 0.0) && (dof > 0.0)) {

    float_t zscore = (x - mu) / sigma;
    float_t lmt = -.5 * (dof + 1.0) * std::log(1.0 + (zscore * zscore) / dof);
    if (log)
      return std::lgamma(.5 * (dof + 1.0)) - std::log(sigma) -
             .5 * std::log(dof) - .5 * log_pi<float_t> - std::lgamma(.5 * dof) +
             lmt;
    else
      return std::exp(std::lgamma(.5 * (dof + 1.0)) - std::log(sigma) -
                      .5 * std::log(dof) - .5 * log_pi<float_t> -
                      std::lgamma(.5 * dof) + lmt);
  } else {
    if (log)
      return -std::numeric_limits<float_t>::infinity();
    else
      return 0.0;
  }
}

/**
 * @brief Evaluates the unnormalized scaled t distribution. Use with care.
 * @param x the percentile
 * @param mu the location parameter
 * @param sigma the scale parameter
 * @param dof the degrees of freedom
 * @param log true if you want the log of the unnormalized density. False
 * otherwise.
 * @return a floating point number
 */
template <typename float_t>
float_t evalScaledT_unnorm(float_t x, float_t mu, float_t sigma, float_t dof,
                           bool log) {
  if ((sigma > 0.0) && (dof > 0.0)) {

    float_t zscore = (x - mu) / sigma;
    float_t lmt = -.5 * (dof + 1.0) * std::log(1.0 + (zscore * zscore) / dof);
    if (log)
      return lmt;
    else
      return std::exp(lmt);
  } else {
    if (log)
      return -std::numeric_limits<float_t>::infinity();
    else
      return 0.0;
  }
}

/**
 * @brief Evaluates the discrete uniform pmf
 * @param x the hypothetical value of a rv
 * @param k the size of the support i.e. (1,2,...k)
 * @param log true if you want log pmf
 * @return P(X=x) probability that X equals x
 */
template <typename int_t, typename float_t>
float_t evalDiscreteUnif(int_t x, int k, bool log) {
  if ((1 <= x) && (x <= k)) {
    if (log) {
      return -std::log(static_cast<float_t>(k));
    } else {
      return 1.0 / static_cast<float_t>(k);
    }
  } else { // x not in support
    if (log) {
      return -std::numeric_limits<float_t>::infinity();
    } else {
      return 0.0;
    }
  }
}

/**
 * @brief Evaluates the unnormalized discrete uniform pmf. Use with care.
 * @param x the hypothetical value of a rv
 * @param k the size of the support i.e. (1,2,...k)
 * @param log true if you want log unnormalized pmf
 * @return P(X=x) probability that X equals x
 */
template <typename int_t, typename float_t>
float_t evalDiscreteUnif_unnorm(int_t x, int_t k, bool log) {
  if ((1 <= x) && (x <= k)) {
    if (log) {
      return 0.0;
    } else {
      return 1.0;
    }
  } else {
    if (log) {
      return -std::numeric_limits<float_t>::infinity();
    } else {
      return 0.0;
    }
  }
}

/**
 * @brief Evaluates the Bernoulli pmf.
 * @param x the hypothetical value of a rv
 * @param p the probability that the rv equals 1
 * @return P(X=x)
 */
template <typename int_t, typename float_t>
float_t evalBernoulli(int_t x, float_t p, bool log) {
  if (((x == 0) || (x == 1)) &&
      ((0.0 <= p) && (p <= 1.0))) { // if valid x and valid p
    if (log) {
      return (x == 1) ? std::log(p) : std::log(1.0 - p);
    } else {
      return (x == 1) ? p : (1.0 - p);
    }
  } else { // either invalid x or invalid p
    if (log) {
      return -std::numeric_limits<float_t>::infinity();
    } else {
      return 0.0;
    }
  }
}

/**
 * @brief Evaluates the Skellam pmf.
 * @param x the point at which you're evaluating.
 * @param mu1.
 * @param mu2.
 * @param log true if you want the log-mass. False otherwise.
 * @return a float_t evaluation.
 */
template <typename int_t, typename float_t>
float_t evalSkellam(int_t x, float_t mu1, float_t mu2, bool log) {
  if ((mu1 > 0) && (mu2 > 0)) {

    // much of this function is adapted from
    // https://github.com/stan-dev/math/blob/9b2e93ba58fa00521275b22a190468ab22f744a3/stan/math/prim/fun/log_modified_bessel_first_kind.hpp

    // step 1: calculate log I_k(2\sqrt{mu_1 mu_2}) using log_sum_exp
    using boost::math::tools::evaluate_polynomial;
    float_t z = 2 * std::sqrt(mu1 * mu2);
    float_t log_I(-std::numeric_limits<float_t>::infinity());

    if (x == 0) {
      // modified from Boost's bessel_i0_imp in the double precision case,
      // which refers to:
      // Modified Bessel function of the first kind of order zero
      // we use the approximating forms derived in:
      // "Rational Approximations for the Modified Bessel Function of the
      // First Kind -- I0(x) for Computations with Double Precision"
      // by Pavel Holoborodko, see
      // http://www.advanpix.com/2015/11/11/rational-approximations-for-the-modified-bessel-function-of-the-first-kind-i0-computations-double-precision
      // The actual coefficients used are [Boost's] own, and extend
      // Pavel's work to precisions other than double.

      if (z < 7.75) {
        // Bessel I0 over[10 ^ -16, 7.75]
        // Max error in interpolated form : 3.042e-18
        // Max Error found at double precision = Poly : 5.106609e-16
        //                                       Cheb : 5.239199e-16
        static const float_t P[] = {
            1.00000000000000000e+00, 2.49999999999999909e-01,
            2.77777777777782257e-02, 1.73611111111023792e-03,
            6.94444444453352521e-05, 1.92901234513219920e-06,
            3.93675991102510739e-08, 6.15118672704439289e-10,
            7.59407002058973446e-12, 7.59389793369836367e-14,
            6.27767773636292611e-16, 4.34709704153272287e-18,
            2.63417742690109154e-20, 1.13943037744822825e-22,
            9.07926920085624812e-25};
        log_I = log_sum_exp<float_t>(
            0.0, 2.0 * std::log(z) - std::log(4.0) +
                     std::log(evaluate_polynomial(P, mu1 * mu2)));

      } else if (z < 500) {
        // Max error in interpolated form : 1.685e-16
        // Max Error found at double precision = Poly : 2.575063e-16
        //                                       Cheb : 2.247615e+00
        static const float_t P[] = {
            3.98942280401425088e-01,  4.98677850604961985e-02,
            2.80506233928312623e-02,  2.92211225166047873e-02,
            4.44207299493659561e-02,  1.30970574605856719e-01,
            -3.35052280231727022e+00, 2.33025711583514727e+02,
            -1.13366350697172355e+04, 4.24057674317867331e+05,
            -1.23157028595698731e+07, 2.80231938155267516e+08,
            -5.01883999713777929e+09, 7.08029243015109113e+10,
            -7.84261082124811106e+11, 6.76825737854096565e+12,
            -4.49034849696138065e+13, 2.24155239966958995e+14,
            -8.13426467865659318e+14, 2.02391097391687777e+15,
            -3.08675715295370878e+15, 2.17587543863819074e+15};

        log_I =
            z + std::log(evaluate_polynomial(P, 1.0 / z)) - 0.5 * std::log(z);

      } else {

        // Max error in interpolated form : 2.437e-18
        // Max Error found at double precision = Poly : 1.216719e-16
        static const float_t P[] = {
            3.98942280401432905e-01, 4.98677850491434560e-02,
            2.80506308916506102e-02, 2.92179096853915176e-02,
            4.53371208762579442e-02};
        log_I =
            z + std::log(evaluate_polynomial(P, 1.0 / z)) - 0.5 * std::log(z);
      }

    } else if (std::abs(x) == 1) {

      // modified from Boost's bessel_i1_imp in the double precision case
      // see credits above in the v == 0 case
      if (z < 7.75) {
        // Bessel I0 over[10 ^ -16, 7.75]
        // Max error in interpolated form: 5.639e-17
        // Max Error found at double precision = Poly: 1.795559e-16

        static const float_t P[] = {
            8.333333333333333803e-02, 6.944444444444341983e-03,
            3.472222222225921045e-04, 1.157407407354987232e-05,
            2.755731926254790268e-07, 4.920949692800671435e-09,
            6.834657311305621830e-11, 7.593969849687574339e-13,
            6.904822652741917551e-15, 5.220157095351373194e-17,
            3.410720494727771276e-19, 1.625212890947171108e-21,
            1.332898928162290861e-23};

        float_t a = mu1 * mu2;
        float_t Q[3] = {1, 0.5, evaluate_polynomial(P, a)};
        log_I =
            std::log(z) + std::log(evaluate_polynomial(Q, a)) - std::log(2.0);

      } else if (z < 500) {
        // Max error in interpolated form: 1.796e-16
        // Max Error found at double precision = Poly: 2.898731e-16

        static const double P[] = {
            3.989422804014406054e-01,  -1.496033551613111533e-01,
            -4.675104253598537322e-02, -4.090895951581637791e-02,
            -5.719036414430205390e-02, -1.528189554374492735e-01,
            3.458284470977172076e+00,  -2.426181371595021021e+02,
            1.178785865993440669e+04,  -4.404655582443487334e+05,
            1.277677779341446497e+07,  -2.903390398236656519e+08,
            5.192386898222206474e+09,  -7.313784438967834057e+10,
            8.087824484994859552e+11,  -6.967602516005787001e+12,
            4.614040809616582764e+13,  -2.298849639457172489e+14,
            8.325554073334618015e+14,  -2.067285045778906105e+15,
            3.146401654361325073e+15,  -2.213318202179221945e+15};
        log_I =
            z + std::log(evaluate_polynomial(P, 1.0 / z)) - 0.5 * std::log(z);
      } else {

        // Max error in interpolated form: 1.320e-19
        // Max Error found at double precision = Poly: 7.065357e-17
        static const double P[] = {
            3.989422804014314820e-01, -1.496033551467584157e-01,
            -4.675105322571775911e-02, -4.090421597376992892e-02,
            -5.843630344778927582e-02};
        log_I =
            z + std::log(evaluate_polynomial(P, 1.0 / z)) - 0.5 * std::log(z);
      }

    } else if (z > 100) {

      // Boost does something like this in asymptotic_bessel_i_large_x
      float_t lim = std::pow((x * x + 2.5) / (2 * z), 3) / 24;

      if (lim < std::numeric_limits<float_t>::epsilon() * 10) {
        float_t s = 1;
        float_t mu = 4 * x * x;
        float_t ex = 8 * z;
        float_t num = mu - 1;
        float_t denom = ex;
        s -= num / denom;
        num *= mu - 9;
        denom *= ex * 2;
        s += num / denom;
        num *= mu - 25;
        denom *= ex * 3;
        s -= num / denom;
        log_I = z - .5 * std::log(z) - .5 * log_two_pi<float_t> + std::log(s);
      }
    } else {

      // just do the sum straightforwardly
      // m=0
      int_t absx = (x < 0) ? -x : x;
      float_t lm1m2 = std::log(mu1) + std::log(mu2);
      float_t first = .5 * absx * lm1m2;
      float_t second = 0.0;
      float_t third = std::lgamma(absx + 1);
      log_I = first - second - third;

      // m > 0
      float_t m = 1.0;
      float_t last_iter_log_I;
      do {
        first += lm1m2;
        second += std::log(m);
        third += std::log(m + absx);
        last_iter_log_I = log_I;
        log_I = log_sum_exp<float_t>(log_I, first - second - third);
        m++;
        if (m > 1000)
          std::cout << "first, second, third, mu1, mu2: " << first << ", "
                    << second << ", " << third << ", " << mu1 << ", " << mu2
                    << "\n";
      } while (log_I != last_iter_log_I);
    }

    // step 2: add the easy parts to get the overall pmf evaluation
    float_t log_mass =
        -mu1 - mu2 + .5 * x * (std::log(mu1) - std::log(mu2)) + log_I;
    // std::cout << "guaranteed: " << std::log(boost::math::cyl_bessel_i<int_t,
    // float_t>(x,2.0*std::sqrt(mu1*mu2)))
    //           << "\nmine: " << log_I << "\n";

    // step 3: handle log/nonlog particulars
    if (log) {
      return log_mass;
    } else {
      return std::exp(log_mass);
    }
  } else {
    if (log) {
      return -std::numeric_limits<float_t>::infinity();
    } else {
      return 0.0;
    }
  }
}

////////////////////////////////////////////////
/////////      Eigen Evals             /////////
////////////////////////////////////////////////

/**
 * @brief Evaluates the multivariate Normal density.
 * If covariance matrix isn't pd, then returns 0
 * (or negative infinity if log is true)
 * @tparam dim the size of the vectors
 * @tparam float_t the floating point type
 * @param x the point you're evaluating at.
 * @param meanVec the mean vector.
 * @param covMat the positive definite, symmetric covariance matrix.
 * @param log true if you want to return the log density. False otherwise.
 * @return a float_t evaluation.
 */
template <std::size_t dim, typename float_t>
float_t evalMultivNorm(const Eigen::Matrix<float_t, dim, 1> &x,
                       const Eigen::Matrix<float_t, dim, 1> &meanVec,
                       const Eigen::Matrix<float_t, dim, dim> &covMat,
                       bool log = false) {
  using Mat = Eigen::Matrix<float_t, dim, dim>;
  Eigen::LLT<Mat> lltM(covMat);
  if (lltM.info() == Eigen::NumericalIssue)
    return log ? -std::numeric_limits<float_t>::infinity()
               : 0.0;     // if not pd return 0 dens
  Mat L = lltM.matrixL(); // the lower diagonal L such that M = LL^T
  float_t quadform = (lltM.solve(x - meanVec)).squaredNorm();
  float_t ld(0.0); // calculate log-determinant using cholesky decomposition too
  // add up log of diagnols of Cholesky L
  for (size_t i = 0; i < dim; ++i) {
    ld += std::log(L(i, i));
  }
  ld *= 2; // covMat = LL^T

  float_t logDens = -.5 * log_two_pi<float_t> * dim - .5 * ld - .5 * quadform;

  if (log) {
    return logDens;
  } else {
    return std::exp(logDens);
  }
}

/**
 * @brief Evaluates the multivariate T density.
 * If covariance matrix isn't pd, then returns 0
 * (or negative infinity if log is true)
 * @tparam dim the size of the vectors
 * @tparam float_t the floating point type
 * @param x the point you're evaluating at.
 * @param locVec the location vector.
 * @param shapeMat the positive definite, symmetric shape matrix.
 * @param log true if you want to return the log density. False otherwise.
 * @return a float_t evaluation.
 */
template <std::size_t dim, typename float_t>
float_t evalMultivT(const Eigen::Matrix<float_t, dim, 1> &x,
                    const Eigen::Matrix<float_t, dim, 1> &locVec,
                    const Eigen::Matrix<float_t, dim, dim> &shapeMat,
                    const float_t &dof, bool log = false) {
  if (dof <= 0.0)
    return log ? -std::numeric_limits<float_t>::infinity()
               : 0.0; // degrees of freedom must be positive
  using Mat = Eigen::Matrix<float_t, dim, dim>;
  Eigen::LLT<Mat> lltM(shapeMat);
  if (lltM.info() == Eigen::NumericalIssue)
    return log ? -std::numeric_limits<float_t>::infinity()
               : 0.0;     // if not pd return 0 dens
  Mat L = lltM.matrixL(); // the lower diagonal L such that M = LL^T
  float_t quadform = (lltM.solve(x - locVec)).squaredNorm();
  float_t ld(0.0); // calculate log-determinant using cholesky decomposition too
  // add up log of diagnols of Cholesky L
  for (size_t i = 0; i < dim; ++i) {
    ld += std::log(L(i, i));
  }
  ld *= 2; // shapeMat = LL^T

  float_t logDens = std::lgamma(.5 * (dof + dim)) - .5 * dim * std::log(dof) -
                    .5 * dim * log_pi<float_t> - std::lgamma(.5 * dof) -
                    .5 * ld - .5 * (dof + dim) * std::log(1.0 + quadform / dof);

  if (log) {
    return logDens;
  } else {
    return std::exp(logDens);
  }
}

/**
 * @brief Evaluates the multivariate Normal density using the Woodbury Matrix
 * Identity to speed up inversion. Sigma = A + UCU'. This function assumes A is
 * diagonal and C is symmetric.
 * @param x the point you're evaluating at.
 * @param meanVec the mean vector.
 * @param A  of A + UCU' in vector form because we explicitly make it diagonal.
 * @param U of A + UCU'
 * @param C of A + UCU'
 * @param log true if you want to return the log density. False otherwise.
 * @return a float_t evaluation.
 */
template <std::size_t bigd, std::size_t smalld, typename float_t>
float_t evalMultivNormWBDA(const Eigen::Matrix<float_t, bigd, 1> &x,
                           const Eigen::Matrix<float_t, bigd, 1> &meanVec,
                           const Eigen::Matrix<float_t, bigd, 1> &A,
                           const Eigen::Matrix<float_t, bigd, smalld> &U,
                           const Eigen::Matrix<float_t, smalld, smalld> &C,
                           bool log = false) {

  using bigmat = Eigen::Matrix<float_t, bigd, bigd>;
  using smallmat = Eigen::Matrix<float_t, smalld, smalld>;

  bigmat Ainv = A.asDiagonal().inverse();
  smallmat Cinv = C.inverse();
  smallmat I = Cinv + U.transpose() * Ainv * U;
  bigmat SigInv = Ainv - Ainv * U * I.ldlt().solve(U.transpose() * Ainv);
  Eigen::LLT<bigmat> lltSigInv(SigInv);
  bigmat L = lltSigInv.matrixL(); // LL' = Sig^{-1}
  float_t quadform = (L * (x - meanVec)).squaredNorm();
  if (log) {

    // calculate log-determinant using cholesky decomposition (assumes symmetric
    // and positive definite)
    float_t halfld(0.0);
    // add up log of diagnols of Cholesky L
    for (size_t i = 0; i < bigd; ++i) {
      halfld += std::log(L(i, i));
    }

    return -.5 * log_two_pi<float_t> * bigd + halfld - .5 * quadform;

  } else { // not the log density
    float_t normConst = std::pow(inv_sqrt_2pi<float_t>, bigd) * L.determinant();
    return normConst * std::exp(-.5 * quadform);
  }
}

/**
 * @brief Evaluates the Wishart density
 * returns 0 if either matrix is not sym pd (
 * or - infinity if log is true)
 * @tparam dim the number of rows of the square matrix
 * @tparam float_t the type of floating point number
 * @param X the matrix you're evaluating at.
 * @param Vinv the INVERSE of the scale matrix.
 * @param n the degrees of freedom.
 * @param log true if you want to return the log density. False otherwise.
 * @return a float_t evaluation.
 */
template <std::size_t dim, typename float_t>
float_t evalWishart(const Eigen::Matrix<float_t, dim, dim> &X,
                    const Eigen::Matrix<float_t, dim, dim> &Vinv,
                    const unsigned int &n, bool log = false) {
  using Mat = Eigen::Matrix<float_t, dim, dim>;
  Eigen::LLT<Mat> lltX(X);
  Eigen::LLT<Mat> lltVinv(Vinv);
  if ((n < dim) | (lltX.info() == Eigen::NumericalIssue) |
      (lltVinv.info() == Eigen::NumericalIssue))
    return log ? -std::numeric_limits<float_t>::infinity() : 0.0;
  // https://stackoverflow.com/questions/35227131/eigen-check-if-matrix-is-positive-semi-definite

  float_t ldx(0.0); // log determinant of X
  float_t ldvinv(0.0);
  Mat Lx = lltX.matrixL(); // the lower diagonal L such that X = LL^T
  Mat Lvi = lltVinv.matrixL();
  float_t logGammaNOver2 =
      .25 * dim * (dim - 1) *
      log_pi<float_t>; // existence guaranteed when n > dim-1

  // add up log of diagonals of each Cholesky L
  for (size_t i = 0; i < dim; ++i) {
    ldx += std::log(Lx(i, i));
    ldvinv += std::log(Lvi(i, i));
    logGammaNOver2 += std::lgamma(.5 * (n - i)); // recall j = i+1
  }
  ldx *= 2.0; // X = LL^T
  ldvinv *= 2.0;

  float_t logDens = .5 * (n - dim - 1) * ldx - .5 * (Vinv * X).trace() -
                    .5 * n * dim * std::log(2.0) + .5 * n * ldvinv -
                    logGammaNOver2;

  if (log) {
    return logDens;
  } else {
    return std::exp(logDens);
  }
}

/**
 * @brief Evaluates the Inverse Wishart density
 * returns 0 if either matrix is not sym pd (
 * or - infinity if log is true)
 * @tparam dim the number of rows of the square matrix
 * @tparam float_t the type of floating point number
 * @param X the matrix you're evaluating at.
 * @param Psi the scale matrix.
 * @param nu the degrees of freedom.
 * @param log true if you want to return the log density. False otherwise.
 * @return a float_t evaluation.
 */
template <std::size_t dim, typename float_t>
float_t evalInvWishart(const Eigen::Matrix<float_t, dim, dim> &X,
                       const Eigen::Matrix<float_t, dim, dim> &Psi,
                       const unsigned int &nu, bool log = false) {
  using Mat = Eigen::Matrix<float_t, dim, dim>;
  Eigen::LLT<Mat> lltX(X);
  Eigen::LLT<Mat> lltPsi(Psi);
  if ((nu < dim) | (lltX.info() == Eigen::NumericalIssue) |
      (lltPsi.info() == Eigen::NumericalIssue))
    return log ? -std::numeric_limits<float_t>::infinity() : 0.0;
  // https://stackoverflow.com/questions/35227131/eigen-check-if-matrix-is-positive-semi-definite

  float_t ldx(0.0); // log determinant of X
  float_t ldPsi(0.0);
  Mat Lx = lltX.matrixL(); // the lower diagonal L such that X = LL^T
  Mat Lpsi = lltPsi.matrixL();
  float_t logGammaNuOver2 =
      .25 * dim * (dim - 1) *
      log_pi<float_t>; // existence guaranteed when n > dim-1

  // add up log of diagonals of each Cholesky L
  for (size_t i = 0; i < dim; ++i) {
    ldx += std::log(Lx(i, i));
    ldPsi += std::log(Lpsi(i, i));
    logGammaNuOver2 += std::lgamma(.5 * (nu - i)); // recall j = i+1
  }
  ldx *= 2.0; // X = LL^T
  ldPsi *= 2.0;

  // TODO: this will probably be faster if you find an analogue...
  // float_t logDens = .5*nu*ldPsi - .5*(nu + dim + 1.0)*ldx -
  // .5*(X.solve(Psi)).trace() - .5*nu*dim*std::log(2.0) - logGammaNuOver2;
  float_t logDens = .5 * nu * ldPsi - .5 * (nu + dim + 1.0) * ldx -
                    .5 * (Psi * X.inverse()).trace() -
                    .5 * nu * dim * std::log(2.0) - logGammaNuOver2;

  if (log) {
    return logDens;
  } else {
    return std::exp(logDens);
  }
}

} // namespace rveval

} // namespace pf

#endif // RV_EVAL_H
