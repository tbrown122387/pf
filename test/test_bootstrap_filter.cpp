#include <catch2/catch_all.hpp>

#include <pf/bootstrap_filter.h>
#include <pf/cf_filters.h>
#include <pf/resamplers.h>
#include <pf/rv_eval.h>
#include <pf/rv_samp.h>

using namespace pf::filters;
using namespace pf::resamplers;
using namespace pf::rvsamp;
using Catch::Approx;

namespace {

constexpr size_t NPARTS = 10000;
constexpr size_t DIMX = 1;
constexpr size_t DIMY = 1;

// A local-level (random walk + noise) linear-Gaussian model:
//   x_1 ~ N(mu0, sigma0^2)
//   x_t = x_{t-1} + w_t,  w_t ~ N(0, sigmaW^2)
//   y_t = x_t + v_t,      v_t ~ N(0, sigmaV^2)
// This has a well-known closed-form Kalman filter solution (pf::filters::
// kalman, in cf_filters.h), used below as ground truth for the bootstrap
// particle filter.
template <typename float_t, typename resamp_t>
class LocalLevelBS : public BSFilter<NPARTS, DIMX, DIMY, resamp_t, float_t> {
public:
  using ssv = Eigen::Matrix<float_t, DIMX, 1>;
  using osv = Eigen::Matrix<float_t, DIMY, 1>;

  LocalLevelBS(float_t mu0, float_t sigma0, float_t sigmaW, float_t sigmaV)
      : m_mu0(mu0), m_sigma0(sigma0), m_sigmaW(sigmaW), m_sigmaV(sigmaV) {}

  float_t logMuEv(const ssv &x1) override {
    return pf::rveval::evalUnivNorm<float_t>(x1(0), m_mu0, m_sigma0, true);
  }

  ssv q1Samp(const osv & /*y1*/) override {
    ssv x1;
    x1(0) = m_mu0 + m_sigma0 * m_normSampler.sample();
    return x1;
  }

  float_t logQ1Ev(const ssv &x1, const osv & /*y1*/) override {
    // bootstrap proposal at time 1: same as the prior, logMuEv
    return pf::rveval::evalUnivNorm<float_t>(x1(0), m_mu0, m_sigma0, true);
  }

  float_t logGEv(const osv &yt, const ssv &xt) override {
    return pf::rveval::evalUnivNorm<float_t>(yt(0), xt(0), m_sigmaV, true);
  }

  ssv fSamp(const ssv &xtm1) override {
    ssv xt;
    xt(0) = xtm1(0) + m_sigmaW * m_normSampler.sample();
    return xt;
  }

private:
  float_t m_mu0;
  float_t m_sigma0;
  float_t m_sigmaW;
  float_t m_sigmaV;
  UnivNormSampler<float_t> m_normSampler;
};

} // namespace

TEST_CASE("bootstrap filter matches closed-form Kalman filter on a "
          "linear-Gaussian local-level model",
          "[bootstrap_filter]") {
  using float_t = double;
  using resamp_t = mn_resampler<NPARTS, DIMX, float_t>;
  using ssv = Eigen::Matrix<float_t, DIMX, 1>;
  using osv = Eigen::Matrix<float_t, DIMY, 1>;
  using Mat = Eigen::Matrix<float_t, Eigen::Dynamic, Eigen::Dynamic>;
  using ssMat = Eigen::Matrix<float_t, DIMX, DIMX>;
  using osMat = Eigen::Matrix<float_t, DIMY, DIMY>;
  using obsStateSizeMat = Eigen::Matrix<float_t, DIMY, DIMX>;
  using siMat = Eigen::Matrix<float_t, DIMX, 0>;
  using oiMat = Eigen::Matrix<float_t, DIMY, 0>;
  using isv = Eigen::Matrix<float_t, 0, 1>;

  float_t mu0 = 0.0;
  float_t sigma0 = 2.0;
  float_t sigmaW = 0.5;
  float_t sigmaV = 1.0;

  // fixed synthetic data: doesn't need to be simulated from the model, just
  // needs to be a plausible observed sequence to filter on
  std::vector<float_t> ys = {0.3,  0.9,  1.4,  1.1, 0.4,  -0.2, -0.9,
                             -1.3, -1.0, -0.3, 0.5, 1.2,  1.6,  1.0,
                             0.2};

  LocalLevelBS<float_t, resamp_t> pf_filter(mu0, sigma0, sigmaW, sigmaV);
  kalman<DIMX, DIMY, 0, float_t> kf(ssv::Constant(mu0),
                                    ssMat::Constant(sigma0 * sigma0));

  ssMat stateTrans = ssMat::Identity();
  ssMat cholStateVar = ssMat::Constant(sigmaW);
  obsStateSizeMat obsMat = obsStateSizeMat::Identity();
  osMat cholObsVar = osMat::Constant(sigmaV);
  siMat stateInptAffector;
  oiMat obsInptAffector;
  isv inputData;

  auto meanFn = [](const ssv &x) -> Mat { return x; };
  auto sqFn = [](const ssv &x) -> Mat {
    Mat m(1, 1);
    m(0, 0) = x(0) * x(0);
    return m;
  };
  std::vector<std::function<const Mat(const ssv &)>> fs = {meanFn, sqFn};

  for (float_t y : ys) {
    osv yt;
    yt(0) = y;

    pf_filter.filter(yt, fs);
    kf.update(yt, stateTrans, cholStateVar, stateInptAffector, inputData,
              obsMat, obsInptAffector, cholObsVar);

    std::vector<Mat> expectations = pf_filter.getExpectations();
    float_t pfMean = expectations[0](0, 0);
    float_t pfSecondMoment = expectations[1](0, 0);
    float_t pfVar = pfSecondMoment - pfMean * pfMean;

    float_t kfMean = kf.getFiltMean()(0);
    float_t kfVar = kf.getFiltVar()(0, 0);

    INFO("y=" << y << " pfMean=" << pfMean << " kfMean=" << kfMean
              << " pfVar=" << pfVar << " kfVar=" << kfVar);
    REQUIRE(pfMean == Approx(kfMean).margin(0.15));
    REQUIRE(pfVar == Approx(kfVar).margin(0.15));
  }
}
