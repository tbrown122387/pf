#ifndef RESAMPLERS_H
#define RESAMPLERS_H

#include <array>
#include <chrono>
#include <cmath>   //floor
#include <numeric> // accumulate, partial_sum
#include <random>

#ifdef DROPPINGTHISINRPACKAGE
#include <RcppEigen.h>
// [[Rcpp::depends(RcppEigen)]]
#else
#include <Eigen/Dense>
#endif

#include <algorithm> // sort
#include <bitset>    // bitset

#include "rv_eval.h" // for rveval::evalUnivStdNormCDF<float_t>()

namespace pf {

namespace resamplers {

//! Base class for all resampler types.
/**
 * @class rbase
 * @author taylor
 * @date 15/04/18
 * @file resamplers.h
 * @brief most resamplers must inherit from this.
 * This will enforce certain structure that are assumed by these pfs.
 * @tparam nparts the number of particles.
 * @tparam dimx the dimension of each state sample.
 * @tparam float_t the type of floating point numbers (e.g. float or double)
 */
template <size_t nparts, size_t dimx, typename float_t> class rbase {
public:
  /** type alias for linear algebra stuff */
  using ssv = Eigen::Matrix<float_t, dimx, 1>;
  /** type alias for array of Eigen Matrices */
  using arrayVec = std::array<ssv, nparts>;
  /** type alias for array of float_ts */
  using arrayFloat = std::array<float_t, nparts>;

  /**
   * @brief The default constructor gets called by default, and it sets the seed
   * with the clock.
   */
  rbase();

  /**
   * @brief The constructor that sets the seed deterministically.
   * @param seed the seed
   */
  rbase(unsigned long seed);

  /**
   * @brief Function to resample from log unnormalized weights
   * @param oldParts
   * @param oldLogUnNormWts
   */
  virtual void resampLogWts(arrayVec &oldParts,
                            arrayFloat &oldLogUnNormWts) = 0;

protected:
  /** @brief prng */
  std::mt19937 m_gen;
};

template <size_t nparts, size_t dimx, typename float_t>
rbase<nparts, dimx, float_t>::rbase()
    : m_gen{static_cast<std::uint32_t>(std::chrono::high_resolution_clock::now()
                                           .time_since_epoch()
                                           .count())} {}

template <size_t nparts, size_t dimx, typename float_t>
rbase<nparts, dimx, float_t>::rbase(unsigned long seed)
    : m_gen{static_cast<std::uint32_t>(seed)} {}

/**
 * @class mn_resampler
 * @author taylor
 * @date 15/04/18
 * @file resamplers.h
 * @brief Class that performs multinomial resampling for "standard" models.
 * @tparam nparts the number of particles.
 * @tparam dimx the dimension of each state sample.
 */
template <size_t nparts, size_t dimx, typename float_t>
class mn_resampler : private rbase<nparts, dimx, float_t> {
public:
  /** type alias for linear algebra stuff */
  using ssv = Eigen::Matrix<float_t, dimx, 1>;
  /** type alias for array of Eigen Matrices */
  using arrayVec = std::array<ssv, nparts>;
  /** type alias for array of float_ts */
  using arrayFloat = std::array<float_t, nparts>;
  /** type alias for array of integers */
  using arrayInt = std::array<unsigned int, nparts>;

  /**
   * @brief Default constructor.
   */
  mn_resampler() = default;

  /**
   * @brief Constructor that sets the seed.
   * @param seed
   */
  mn_resampler(unsigned long seed);

  /**
   * @brief resamples particles.
   * @param oldParts the old particles
   * @param oldLogUnNormWts the old log unnormalized weights
   */
  void resampLogWts(arrayVec &oldParts, arrayFloat &oldLogUnNormWts);
};

template <size_t nparts, size_t dimx, typename float_t>
mn_resampler<nparts, dimx, float_t>::mn_resampler(unsigned long seed)
    : rbase<nparts, dimx, float_t>(seed) {}

template <size_t nparts, size_t dimx, typename float_t>
void mn_resampler<nparts, dimx, float_t>::resampLogWts(
    arrayVec &oldParts, arrayFloat &oldLogUnNormWts) {
  // these log weights may be very negative. If that's the case, exponentiating
  // them may cause underflow so we use the "log-exp-sum" trick actually not
  // quite...we just shift the log-weights because after they're exponentiated
  // they have the same normalized probabilities

  // Create the distribution with exponentiated log-weights
  arrayFloat w;
  float_t m = *std::max_element(oldLogUnNormWts.begin(), oldLogUnNormWts.end());
  std::transform(oldLogUnNormWts.begin(), oldLogUnNormWts.end(), w.begin(),
                 [&m](float_t &d) -> float_t { return std::exp(d - m); });
  std::discrete_distribution<> idxSampler(w.begin(), w.end());

  // create temporary particle vector and weight vector
  arrayVec tmpPartics = oldParts;

  // sample from the original parts and store in tmpParts
  unsigned int whichPart;
  for (size_t part = 0; part < nparts; ++part) {
    whichPart = idxSampler(this->m_gen);
    tmpPartics[part] = oldParts[whichPart];
  }

  // overwrite olds with news
  oldParts = std::move(tmpPartics);
  std::fill(oldLogUnNormWts.begin(), oldLogUnNormWts.end(), 0.0); // change back
}

/**
 * @class mn_resampler_rbpf
 * @author taylor
 * @file resamplers.h
 * @brief Class that performs multinomial resampling for RBPFs.
 * @tparam nparts the number of particles.
 * @tparam dimsampledx the dimension of each state sample.
 * @tparam cfModT the type of closed form model
 * @tparam float_t the type of floating point number
 */
template <size_t nparts, size_t dimsampledx, typename cfModT, typename float_t>
class mn_resampler_rbpf {
public:
  /** type alias for linear algebra stuff */
  using ssv = Eigen::Matrix<float_t, dimsampledx, 1>;
  /** type alias for linear algebra stuff */
  using arrayVec = std::array<ssv, nparts>;
  /** type alias for array of float_ts */
  using arrayFloat = std::array<float_t, nparts>;
  /** type alias for array of closed-form models */
  using arrayMod = std::array<cfModT, nparts>;

  /**
   * @brief Default constructor.
   */
  mn_resampler_rbpf();

  /**
   * @brief Default constructor.
   */
  mn_resampler_rbpf(unsigned long seed);

  /**
   * @brief resamples particles.
   * @param oldMods the old closed-form models
   * @param oldParts the old particles
   * @param oldLogUnNormWts the old log unnormalized weights
   */
  void resampLogWts(arrayMod &oldMods, arrayVec &oldParts,
                    arrayFloat &oldLogUnNormWts);

private:
  /** @brief prng */
  std::mt19937 m_gen;
};

template <size_t nparts, size_t dimsampledx, typename cfModT, typename float_t>
mn_resampler_rbpf<nparts, dimsampledx, cfModT, float_t>::mn_resampler_rbpf()
    : m_gen{static_cast<std::uint32_t>(std::chrono::high_resolution_clock::now()
                                           .time_since_epoch()
                                           .count())} {}

template <size_t nparts, size_t dimsampledx, typename cfModT, typename float_t>
mn_resampler_rbpf<nparts, dimsampledx, cfModT, float_t>::mn_resampler_rbpf(
    unsigned long seed)
    : m_gen{static_cast<std::uint32_t>(seed)} {}

template <size_t nparts, size_t dimsampledx, typename cfModT, typename float_t>
void mn_resampler_rbpf<nparts, dimsampledx, cfModT, float_t>::resampLogWts(
    arrayMod &oldMods, arrayVec &oldSamps, arrayFloat &oldLogUnNormWts) {
  // Create the distribution with exponentiated log-weights
  arrayFloat w;
  float_t m = *std::max_element(oldLogUnNormWts.begin(), oldLogUnNormWts.end());
  std::transform(oldLogUnNormWts.begin(), oldLogUnNormWts.end(), w.begin(),
                 [&m](float_t &d) -> float_t { return std::exp(d - m); });
  std::discrete_distribution<> idxSampler(w.begin(), w.end());

  // create temporary vectors for samps and mods
  arrayVec tmpSamps;
  arrayMod tmpMods;

  // sample from the original parts and store in temporary
  unsigned int whichPart;
  for (size_t part = 0; part < nparts; ++part) {
    whichPart = idxSampler(m_gen);
    tmpSamps[part] = oldSamps[whichPart];
    tmpMods[part] = oldMods[whichPart];
  }

  // overwrite olds with news
  oldMods = std::move(tmpMods);
  oldSamps = std::move(tmpSamps);
  std::fill(oldLogUnNormWts.begin(), oldLogUnNormWts.end(), 0.0);
}

/**
 * @class resid_resampler
 * @author taylor
 * @date 10/25/19
 * @file resamplers.h
 * @brief Class that performs residual resampling on "standard" models.
 * @tparam nparts the number of particles.
 * @tparam dimx the dimension of each state sample.
 * @tparam float_t the floating point for samples
 */
template <size_t nparts, size_t dimx, typename float_t>
class resid_resampler : private rbase<nparts, dimx, float_t> {
public:
  /** type alias for linear algebra stuff */
  using ssv = Eigen::Matrix<float_t, dimx, 1>;
  /** type alias for array of Eigen Matrices */
  using arrayVec = std::array<ssv, nparts>;
  /** type alias for array of float_ts */
  using arrayFloat = std::array<float_t, nparts>;
  /** type alias for array of integers */
  using arrayInt = std::array<unsigned int, nparts>;

  /**
   * @brief Default constructor.
   */
  resid_resampler() = default;

  /**
   * @brief Constructor that sets the seed.
   * @param seed
   */
  resid_resampler(unsigned long seed);

  /**
   * @brief resamples particles.
   * @param oldParts the old particles
   * @param oldLogUnNormWts the old log unnormalized weights
   */
  void resampLogWts(arrayVec &oldParts, arrayFloat &oldLogUnNormWts);
};

template <size_t nparts, size_t dimx, typename float_t>
resid_resampler<nparts, dimx, float_t>::resid_resampler(unsigned long seed)
    : rbase<nparts, dimx, float_t>(seed) {}

template <size_t nparts, size_t dimx, typename float_t>
void resid_resampler<nparts, dimx, float_t>::resampLogWts(
    arrayVec &oldParts, arrayFloat &oldLogUnNormWts) {

  // calculate normalized weights
  arrayFloat w;
  float_t m = *std::max_element(oldLogUnNormWts.begin(), oldLogUnNormWts.end());
  std::transform(oldLogUnNormWts.begin(), oldLogUnNormWts.end(), w.begin(),
                 [&m](const float_t &d) -> float_t { return std::exp(d - m); });
  float_t norm_const(0.0);
  norm_const = std::accumulate(w.begin(), w.end(), norm_const);
  for (auto &weight : w)
    weight = weight / norm_const;

  // calc unNormWBars and numRandomSamples (N-R using IIHMM notation)
  size_t i;
  arrayFloat unNormWBar;
  float_t numRandomSamples(0.0);
  for (i = 0; i < nparts; ++i) {
    unNormWBar[i] = w[i] * nparts - std::floor(w[i] * nparts);
    numRandomSamples += unNormWBar[i];
  }

  // make multinomial distribution for residuals
  std::discrete_distribution<> idxSampler(unNormWBar.begin(), unNormWBar.end());

  // start resampling by producing a count vector
  arrayInt sampleCounts;
  for (i = 0; i < nparts; ++i) {
    sampleCounts[i] =
        static_cast<unsigned int>(std::floor(nparts * w[i])); // initial
  }
  for (i = 0; i < static_cast<unsigned int>(numRandomSamples); ++i) {
    sampleCounts[idxSampler(this->m_gen)]++;
  }

  // now resample the particles using the counts
  arrayVec tmpPartics;
  unsigned int c(0);
  for (i = 0; i < nparts; ++i) { // over count container
    unsigned int num_replicants = sampleCounts[i];
    if (num_replicants > 0) {
      for (size_t j = 0; j < num_replicants;
           ++j) { // assign the same thing several times
        tmpPartics[c] = oldParts[i];
        c++;
      }
    }
  }

  // overwrite olds with news
  oldParts = std::move(tmpPartics);
  std::fill(oldLogUnNormWts.begin(), oldLogUnNormWts.end(), 0.0); // change back
}

/**
 * @class stratif_resampler
 * @author taylor
 * @date 10/25/19
 * @file resamplers.h
 * @brief Class that performs stratified resampling on "standard" models.
 * @tparam nparts the number of particles.
 * @tparam dimx the dimension of each state sample.
 * @tparam float_t the floating point for samples
 */
template <size_t nparts, size_t dimx, typename float_t>
class stratif_resampler : private rbase<nparts, dimx, float_t> {
public:
  /** type alias for linear algebra stuff */
  using ssv = Eigen::Matrix<float_t, dimx, 1>;
  /** type alias for array of Eigen Matrices */
  using arrayVec = std::array<ssv, nparts>;
  /** type alias for array of float_ts */
  using arrayFloat = std::array<float_t, nparts>;
  /** type alias for array of integers */
  using arrayInt = std::array<unsigned int, nparts>;

  /**
   * @brief Default constructor.
   */
  stratif_resampler() = default;

  /**
   * @brief Constructor that sets the seed
   * @param seed
   */
  stratif_resampler(unsigned long seed);

  /**
   * @brief resamples particles.
   * @param oldParts the old particles
   * @param oldLogUnNormWts the old log unnormalized weights
   */
  void resampLogWts(arrayVec &oldParts, arrayFloat &oldLogUnNormWts);
};

template <size_t nparts, size_t dimx, typename float_t>
stratif_resampler<nparts, dimx, float_t>::stratif_resampler(unsigned long seed)
    : rbase<nparts, dimx, float_t>(seed) {}

template <size_t nparts, size_t dimx, typename float_t>
void stratif_resampler<nparts, dimx, float_t>::resampLogWts(
    arrayVec &oldParts, arrayFloat &oldLogUnNormWts) {

  // calculate normalized weights
  arrayFloat w;
  float_t m = *std::max_element(oldLogUnNormWts.begin(), oldLogUnNormWts.end());
  std::transform(oldLogUnNormWts.begin(), oldLogUnNormWts.end(), w.begin(),
                 [&m](const float_t &d) -> float_t { return std::exp(d - m); });
  float_t norm_const(0.0);
  norm_const = std::accumulate(w.begin(), w.end(), norm_const);
  for (auto &weight : w)
    weight = weight / norm_const;

  // calculate the cumulative sums of the weights
  arrayFloat cumsums;
  std::partial_sum(w.begin(), w.end(), cumsums.begin());

  // samplethe Uis
  std::uniform_real_distribution<float_t> u_sampler(0.0, 1.0 / nparts);
  arrayFloat u_samples;
  for (size_t i = 0; i < nparts; ++i) {
    u_samples[i] = i / nparts + u_sampler(this->m_gen);
  }

  // resample
  arrayVec tmpPartics;
  for (size_t i = 0; i < nparts; ++i) { // tmpPartics, Uis

    // find which index
    unsigned int idx;
    for (unsigned int j = 0; j < nparts; ++j) {

      // get the first time it gets covered by a cumsum
      if (cumsums[j] >= u_samples[i]) {
        idx = j;
        break;
      }
    }

    // assign
    tmpPartics[i] = oldParts[idx];
  }

  // overwrite olds with news
  oldParts = std::move(tmpPartics);
  std::fill(oldLogUnNormWts.begin(), oldLogUnNormWts.end(), 0.0); // change back
}

/**
 * @class systematic_resampler
 * @author taylor
 * @date 10/25/19
 * @file resamplers.h
 * @brief Class that performs systematic resampling on "standard" models.
 * @tparam nparts the number of particles.
 * @tparam dimx the dimension of each state sample.
 * @tparam float_t the floating point for samples
 */
template <size_t nparts, size_t dimx, typename float_t>
class systematic_resampler : private rbase<nparts, dimx, float_t> {
public:
  /** type alias for linear algebra stuff */
  using ssv = Eigen::Matrix<float_t, dimx, 1>;
  /** type alias for array of Eigen Matrices */
  using arrayVec = std::array<ssv, nparts>;
  /** type alias for array of float_ts */
  using arrayFloat = std::array<float_t, nparts>;
  /** type alias for array of integers */
  using arrayInt = std::array<unsigned int, nparts>;

  /**
   * @brief Default constructor.
   */
  systematic_resampler() = default;

  /**
   * @brief Constructor that sets the seed.
   * @param seed.
   */
  systematic_resampler(unsigned long seed);

  /**
   * @brief resamples particles.
   * @param oldParts the old particles
   * @param oldLogUnNormWts the old log unnormalized weights
   */
  void resampLogWts(arrayVec &oldParts, arrayFloat &oldLogUnNormWts);
};

template <size_t nparts, size_t dimx, typename float_t>
systematic_resampler<nparts, dimx, float_t>::systematic_resampler(
    unsigned long seed)
    : rbase<nparts, dimx, float_t>(seed) {}

template <size_t nparts, size_t dimx, typename float_t>
void systematic_resampler<nparts, dimx, float_t>::resampLogWts(
    arrayVec &oldParts, arrayFloat &oldLogUnNormWts) {

  // calculate normalized weights
  arrayFloat w;
  float_t m = *std::max_element(oldLogUnNormWts.begin(), oldLogUnNormWts.end());
  std::transform(oldLogUnNormWts.begin(), oldLogUnNormWts.end(), w.begin(),
                 [&m](const float_t &d) -> float_t { return std::exp(d - m); });
  float_t norm_const(0.0);
  norm_const = std::accumulate(w.begin(), w.end(), norm_const);
  for (auto &weight : w)
    weight = weight / norm_const;

  // calculate the cumulative sums of the weights
  arrayFloat cumsums;
  std::partial_sum(w.begin(), w.end(), cumsums.begin());

  // samplethe Uis
  std::uniform_real_distribution<float_t> u_sampler(0.0, 1.0 / nparts);
  arrayFloat u_samples;
  u_samples[0] = u_sampler(this->m_gen);
  for (size_t i = 1; i < nparts; ++i) {
    u_samples[i] = u_samples[i - 1] + 1.0 / nparts;
  }

  // resample
  // unlike stratified, take advantage of U's being sorted
  arrayVec tmpPartics;
  unsigned idx;
  unsigned int j = 0;
  for (size_t i = 0; i < nparts; ++i) { // tmpPartics, Uis

    // find which index
    while (j < nparts) {

      // get the first time it gets covered by a cumsum
      if (cumsums[j] >= u_samples[i]) {
        idx = j;
        break;
      }

      j++;
    }

    // assign
    tmpPartics[i] = oldParts[idx];
  }

  // overwrite olds with news
  oldParts = std::move(tmpPartics);
  std::fill(oldLogUnNormWts.begin(), oldLogUnNormWts.end(), 0.0); // change back
}

/**
 * @class mn_resamp_fast1
 * @author taylor
 * @file resamplers.h
 * @brief Class that performs multinomial resampling for "standard" models.
 * For justification, see page 244 of "Inference in Hidden Markov Models"
 * @tparam nparts the number of particles.
 * @tparam dimx the dimension of each state sample.
 */
template <size_t nparts, size_t dimx, typename float_t>
class mn_resamp_fast1 : private rbase<nparts, dimx, float_t> {
public:
  /** type alias for linear algebra stuff */
  using ssv = Eigen::Matrix<float_t, dimx, 1>;
  /** type alias for array of Eigen Matrices */
  using arrayVec = std::array<ssv, nparts>;
  /** type alias for array of float_ts */
  using arrayFloat = std::array<float_t, nparts>;
  /** type alias for array of integers */
  using arrayInt = std::array<unsigned int, nparts>;

  /**
   * @brief Default constructor.
   */
  mn_resamp_fast1() = default;

  /**
   * @brief Default constructor.
   */
  mn_resamp_fast1(unsigned long seed);

  /**
   * @brief resamples particles.
   * @param oldParts the old particles
   * @param oldLogUnNormWts the old log unnormalized weights
   */
  void resampLogWts(arrayVec &oldParts, arrayFloat &oldLogUnNormWts);
};

template <size_t nparts, size_t dimx, typename float_t>
mn_resamp_fast1<nparts, dimx, float_t>::mn_resamp_fast1(unsigned long seed)
    : rbase<nparts, dimx, float_t>(seed) {}

template <size_t nparts, size_t dimx, typename float_t>
void mn_resamp_fast1<nparts, dimx, float_t>::resampLogWts(
    arrayVec &oldParts, arrayFloat &oldLogUnNormWts) {
  // these log weights may be very negative. If that's the case, exponentiating
  // them may cause underflow so we use the "log-exp-sum" trick actually not
  // quite...we just shift the log-weights because after they're exponentiated
  // they have the same normalized probabilities

  // Also, we're using a fancier algorthm detailed on page 244 of IHMM

  // Create unnormalized weights
  arrayFloat unnorm_weights;
  float_t m = *std::max_element(oldLogUnNormWts.begin(), oldLogUnNormWts.end());
  std::transform(oldLogUnNormWts.begin(), oldLogUnNormWts.end(),
                 unnorm_weights.begin(),
                 [&m](float_t &d) -> float_t { return std::exp(d - m); });

  // get a uniform rv sampler
  std::uniform_real_distribution<float_t> u_sampler(0.0, 1.0);

  // two things:
  // 1.) calculate normalizing constant for weights, and
  // 2.) generate all these exponentials to help with getting order statistics
  // NB: you never need to store E_{N+1}! (this is subtle)
  float_t weight_norm_const(0.0);
  arrayFloat exponentials;
  float_t G(0.0);
  for (size_t i = 0; i < nparts; ++i) {
    weight_norm_const += unnorm_weights[i];
    exponentials[i] = -std::log(u_sampler(this->m_gen));
    G += exponentials[i];
  }
  G -= std::log(u_sampler(this->m_gen)); // E_{N+1}

  // see Fig 7.15 in IHMM on page 243
  arrayVec tmpPartics = oldParts;  // the new particles
  float_t uniform_order_stat(0.0); // U_{(i)} in the notation of IHMM
  float_t running_sum_normalized_weights(
      unnorm_weights[0] /
      weight_norm_const); // \sum_{j=1}^I \omega^j in the notation of IHMM
  float_t one_less_summand(0.0); // \sum_{j=1}^{I-1} \omega^j
  unsigned int idx = 0;
  for (size_t i = 0; i < nparts; ++i) {
    uniform_order_stat += exponentials[i] / G; // add a spacing E_i/G
    do {
      if ((one_less_summand < uniform_order_stat) &&
          (uniform_order_stat <= running_sum_normalized_weights)) {
        // select index idx
        tmpPartics[i] = oldParts[idx];
        break;
      } else {
        // increment idx because it will never be chosen (all the other order
        // statistics are even higher)
        idx++;
        running_sum_normalized_weights +=
            unnorm_weights[idx] / weight_norm_const;
        one_less_summand += unnorm_weights[idx - 1] / weight_norm_const;
      }
    } while (true);
  }

  // overwrite olds with news
  oldParts = std::move(tmpPartics);
  std::fill(oldLogUnNormWts.begin(), oldLogUnNormWts.end(), 0.0);
}

/**
 * @brief converts an integer in a transpose form to a position on the Hilbert
 * Curve. Code is based off of John Skilling , "Programming the Hilbert curve",
 * AIP Conference Proceedings 707, 381-387 (2004)
 * https://doi.org/10.1063/1.1751381
 * @file resamplers.h
 * @tparam num_bits how "accurate/fine/squiggly" you want the Hilbert curve
 * @tparam num_dims the number of dimensions the curve is in
 * @param X an unsigned integer in a "Transpose" form.
 * @return a position on the hilbert curve
 */
template <size_t num_bits, size_t num_dims>
std::array<std::bitset<num_bits>, num_dims>
TransposeToAxes(std::array<std::bitset<num_bits>, num_dims> X) {

  using coord_t = std::bitset<num_bits>;

  // Gray decode by H ^ (H/2)
  coord_t t = X[num_dims - 1] >> 1;
  for (int i = num_dims - 1; i > 0; i--) // https://stackoverflow.com/a/10384110
    X[i] ^= X[i - 1];
  X[0] ^= t;

  // Undo excess work
  coord_t N = 2 << (num_bits - 1);
  for (coord_t Q = 2; Q != N; Q <<= 1) {
    coord_t P = Q.to_ulong() - 1;
    for (int i = num_dims - 1; i >= 0; i--) {
      if ((X[i] & Q).any()) { // invert low bits of X[0]
        X[0] ^= P;
      } else { // exchange low bits of X[i] and X[0]
        t = (X[0] ^ X[i]) & P;
        X[0] ^= t;
        X[i] ^= t;
      }
    }
  }

  return X;
}

/**
 * @brief converts a position on the Hilbert curve into an integer in a
 * "transpose" form. Code is based off of John Skilling , "Programming the
 * Hilbert curve", AIP Conference Proceedings 707, 381-387 (2004)
 * https://doi.org/10.1063/1.1751381
 * @file resamplers.h
 * @tparam num_bits how "accurate/fine/squiggly" you want the Hilbert curve
 * @tparam num_dims the number of dimensions the curve is in
 * @param X a position on the hilbert curve (each dimension coordinate is in
 * base 2)
 * @return a position on the real line (in a "Transpose" form)
 */
template <size_t num_bits, size_t num_dims>
std::array<std::bitset<num_bits>, num_dims>
AxesToTranspose(std::array<std::bitset<num_bits>, num_dims> X) {
  using coord_t = std::bitset<num_bits>;

  // Inverse undo
  coord_t M = 1 << (num_bits - 1);
  for (coord_t Q = M; Q.to_ulong() > 1; Q >>= 1) {
    coord_t P = Q.to_ulong() - 1;
    for (size_t i = 0; i < num_dims; i++) {
      if ((X[i] & Q).any())
        X[0] ^= P;
      else {
        coord_t t = (X[0] ^ X[i]) & P;
        X[0] ^= t;
        X[i] ^= t;
      }
    }
  } // exchange

  // Gray encode
  for (size_t i = 1; i < num_dims; i++)
    X[i] ^= X[i - 1];

  coord_t t = 0;
  for (coord_t Q = M; Q.to_ulong() > 1; Q >>= 1) {
    if ((X[num_dims - 1] & Q).any())
      t ^= Q.to_ulong() - 1;
  }

  for (size_t i = 0; i < num_dims; i++)
    X[i] ^= t;

  return X;
}

/**
 * @brief converts an integer on the positive integers into its "Transpose"
 * representation.. This code supplements the above two functions that are based
 * off of John Skilling , "Programming the Hilbert curve", AIP Conference
 * Proceedings 707, 381-387 (2004) https://doi.org/10.1063/1.1751381
 * @file resamplers.h
 * @tparam num_bits how "accurate/fine/squiggly" you want the Hilbert curve
 * @tparam num_dims the number of dimensions the curve is in
 * @param H a position on the hilbert curve (0,1,..,2^(num_dims * num_bits) )
 * @return a position on the real line (in a "Transpose" form)
 */
template <size_t num_bits, size_t num_dims>
std::array<std::bitset<num_bits>, num_dims> makeHTranspose(unsigned int H) {
  using coord_t = std::bitset<num_bits>;
  using coords_t = std::array<coord_t, num_dims>;
  using big_coord_t = std::bitset<num_bits * num_dims>;

  big_coord_t Hb = H;
  coords_t X;
  for (size_t dim = 0; dim < num_dims; ++dim) {

    coord_t dim_coord_tmp;
    unsigned start_bit = num_bits * num_dims - 1 - dim;
    unsigned int c = num_bits - 1;
    for (int bit = start_bit; bit >= 0; bit -= num_dims) {
      dim_coord_tmp[c] = Hb[bit];
      c--;
    }
    X[dim] = dim_coord_tmp;
  }
  return X;
}

/**
 * @brief converts an integer in its "Transpose" representation into a positive
 * integer.. This code supplements two functions above that are based off of
 * John Skilling , "Programming the Hilbert curve", AIP Conference Proceedings
 * 707, 381-387 (2004) https://doi.org/10.1063/1.1751381
 * @file resamplers.h
 * @tparam num_bits how "accurate/fine/squiggly" you want the Hilbert curve
 * @tparam num_dims the number of dimensions the curve is in
 * @param Htrans a position on the real line (in a "Transpose" form)
 * @return a position on the hilbert curve (0,1,..,2^(num_dims * num_bits) )
 */
template <size_t num_bits, size_t num_dims>
unsigned int makeH(std::array<std::bitset<num_bits>, num_dims> Htrans) {
  using big_coord_t = std::bitset<num_bits * num_dims>;

  big_coord_t H;
  unsigned int which_dim = 0;
  unsigned which_bit;
  for (int i = num_bits * num_dims - 1; i >= 0; i--) {
    which_bit = i / num_dims;
    H[i] = Htrans[which_dim][which_bit];
    which_dim = (which_dim + 1) % num_dims;
  }
  return H.to_ulong();
}

//! Base class for resampler types that use a Hilbert curve sorting technique.
/**
 * @class rbase_hcs
 * @author taylor
 * @date 08/06/21
 * @file resamplers.h
 * @brief resamplers that use a Hilbert curve sorting procedure must inherit
 * from this. Unlike rbase, this does not hold onto a random number generator
 * object, so no seed-setting is necessary.
 * @tparam nparts the number of particles.
 * @tparam dimx the dimension of each state sample.
 * @tparam float_t the type of floating point numbers (e.g. float or double)
 */
template <size_t nparts, size_t dimx, size_t dimur, size_t num_hilb_bits,
          typename float_t>
class rbase_hcs {
public:
  /** type alias for linear algebra stuff */
  using ssv = Eigen::Matrix<float_t, dimx, 1>;
  /** type alias for array of Eigen Matrices */
  using arrayVec = std::array<ssv, nparts>;
  /** type alias for array of float_ts */
  using arrayFloat = std::array<float_t, nparts>;
  /** type alias for common normal random variable */
  using usvr = Eigen::Matrix<float_t, dimur, 1>;

  /**
   * @brief The default constructor. There is no seed-setting.
   */
  rbase_hcs() = default;

  /**
   * @brief Function to resample from log unnormalized weights
   * @param oldParts
   * @param oldLogUnNormWts
   * @param ur common random number used to resample.
   */
  virtual void resampLogWts(arrayVec &oldParts, arrayFloat &oldLogUnNormWts,
                            const usvr &ur) = 0;

private:
  /**
   * @brief Function that "sorts" multidimensional vectors using an (inverse)
   * Hilbert-curve map. For more information see
   * https://arxiv.org/pdf/1511.04992.pdf
   */
  static bool hilbertComparison(const ssv &first, const ssv &second);

public:
  /**
   * @brief get a permutation based on unsorted particle samples (not their
   * weights)
   * @param unsortedParts the particle samples
   * @return the permutation as a std::array
   */
  std::array<unsigned, nparts> get_permutation(const arrayVec &unsortedParts);
};

template <size_t nparts, size_t dimx, size_t dimur, size_t num_hilb_bits,
          typename float_t>
bool rbase_hcs<nparts, dimx, dimur, num_hilb_bits, float_t>::hilbertComparison(
    const ssv &first, const ssv &second) {
  // return true if first "<" second
  // three intermediate steps to do that:
  // 1.
  // squash each vector's elements from
  // (-infty,infty) -> [0, 2^num_hilb_bits)
  // with the function
  // f(x) = 2^num_bits/(1 + e^{-x}) = 2^{num_bits - 1} + 2^{num_bits -
  // 1}*tanh(x/2)
  float_t c = std::pow(2, num_hilb_bits - 1);
  ssv squashed_first = first * .5;
  squashed_first = squashed_first.array().tanh() * c + c;
  ssv squashed_second = second * .5;
  squashed_second = squashed_second.array().tanh() * c + c;

  // 2.
  // convert the squashed matrix into bitset type obj"
  std::array<std::bitset<num_hilb_bits>, dimx> axes_first;
  std::array<std::bitset<num_hilb_bits>, dimx> axes_second;
  for (size_t dim = 0; dim < dimx; ++dim) {
    axes_first[dim] =
        static_cast<unsigned int>(std::floor(squashed_first(dim)));
    axes_second[dim] =
        static_cast<unsigned int>(std::floor(squashed_second(dim)));
  }

  // 3.
  // convert to one dimensional unsigned using "AxesToTranspose" and "makeH"
  return makeH(AxesToTranspose(axes_first)) <
         makeH(AxesToTranspose(axes_second));
}

template <size_t nparts, size_t dimx, size_t dimur, size_t num_hilb_bits,
          typename float_t>
std::array<unsigned, nparts>
rbase_hcs<nparts, dimx, dimur, num_hilb_bits, float_t>::get_permutation(
    const arrayVec &unsortedParts) {
  // create unsorted index
  std::array<unsigned, nparts> indexes;
  for (unsigned i = 0; i < nparts; ++i)
    indexes[i] = i;

  // create functor to help sort indexes based on particle samples (not weights)
  struct sort_indexes {

    arrayVec m_unsorted_parts;

    sort_indexes(const arrayVec &prts) : m_unsorted_parts(prts){};

    bool operator()(unsigned i, unsigned j) const {
      return rbase_hcs<nparts, dimx, dimur, num_hilb_bits,
                       float_t>::hilbertComparison(m_unsorted_parts[i],
                                                   m_unsorted_parts[j]);
    }
  };

  // sort the indexes and return them
  std::sort(indexes.begin(), indexes.end(), sort_indexes(unsortedParts));
  return indexes;
}

/**
 * @class sys_hilb_resampler
 * @author taylor
 * @date 08/06/21
 * @file resamplers.h
 * @brief Class that performs systematic resampling with a Hilbert curve sorting
 * scheme.
 * @tparam nparts the number of particles.
 * @tparam dimx the dimension of each state sample.
 * @tparam float_t the floating point for samples
 */
template <size_t nparts, size_t dimx, size_t num_hilb_bits, typename float_t>
class sys_hilb_resampler
    : private rbase_hcs<nparts, dimx, 1, num_hilb_bits, float_t> {
public:
  /** type alias for linear algebra stuff */
  using ssv = Eigen::Matrix<float_t, dimx, 1>;
  /** type alias for array of Eigen Matrices */
  using arrayVec = std::array<ssv, nparts>;
  /** type alias for array of float_ts */
  using arrayFloat = std::array<float_t, nparts>;
  /** type alias for array of integers */
  using arrayInt = std::array<unsigned int, nparts>;
  /** type alias for resampling normal random variable */
  using usvr = Eigen::Matrix<float_t, 1, 1>;

  /**
   * @brief Default constructor. This class does not handle random numbers, so
   * there is no seed-setting.
   */
  sys_hilb_resampler() = default;

  /**
   * @brief resamples particles.
   * @param oldParts the old particles that get changed in place
   * @param oldLogUnNormWts the old log unnormalized weights that get changed in
   * place
   * @param ur standard normal random variable used to resample indexes
   */
  void resampLogWts(arrayVec &oldParts, arrayFloat &oldLogUnNormWts,
                    const usvr &ur);
};

template <size_t nparts, size_t dimx, size_t num_hilb_bits, typename float_t>
void sys_hilb_resampler<nparts, dimx, num_hilb_bits, float_t>::resampLogWts(
    arrayVec &oldParts, arrayFloat &oldLogUnNormWts, const usvr &ur) {
  // calculate normalized weights
  arrayFloat w;
  float_t m = *std::max_element(oldLogUnNormWts.begin(), oldLogUnNormWts.end());
  std::transform(oldLogUnNormWts.begin(), oldLogUnNormWts.end(), w.begin(),
                 [&m](const float_t &d) -> float_t { return std::exp(d - m); });
  float_t norm_const(0.0);
  norm_const = std::accumulate(w.begin(), w.end(), norm_const);
  for (auto &weight : w)
    weight = weight / norm_const;

  // samplethe Ubar_tis
  arrayFloat ubar_samples;
  ubar_samples[0] = rveval::evalUnivStdNormCDF<float_t>(ur(0)) / nparts;
  for (size_t i = 1; i < nparts; ++i) {
    ubar_samples[i] = ubar_samples[i - 1] + 1.0 / nparts;
  }

  // calculate the cumulative sums of the sorted weights
  // calculate sorted particles while you're at it
  auto sigmaPermutation = this->get_permutation(oldParts);
  arrayFloat sortedWeights;
  arrayVec sortedParts;
  for (size_t i = 0; i < nparts; ++i) {
    unsigned sigma_i = sigmaPermutation[i];
    sortedWeights[i] = w[sigma_i];
    sortedParts[i] = oldParts[sigma_i];
  }
  arrayFloat cumsums;
  std::partial_sum(sortedWeights.begin(), sortedWeights.end(), cumsums.begin());

  // resample
  // unlike stratified, take advantage of U's being sorted
  arrayVec tmpPartics;
  unsigned idx;
  unsigned int j = 0;
  for (size_t i = 0; i < nparts; ++i) { // tmpPartics, Uis

    // find which index
    while (j < nparts) {

      // get the first time it gets covered by a cumsum
      if (cumsums[j] >= ubar_samples[i]) {
        idx = j;
        break;
      }

      j++;
    }

    // assign
    tmpPartics[i] = sortedParts[idx];
  }

  // overwrite olds with news
  oldParts = std::move(tmpPartics);
  std::fill(oldLogUnNormWts.begin(), oldLogUnNormWts.end(), 0.0); // change back
}

} // namespace resamplers
} // namespace pf

#endif // RESAMPLERS_H
