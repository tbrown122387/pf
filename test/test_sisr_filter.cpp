#include <catch2/catch_all.hpp>

#include <pf/cf_filters.h>
#include <pf/resamplers.h>
#include <pf/rv_eval.h>
#include <pf/rv_samp.h>
#include <pf/sisr_filter.h>

using namespace pf::filters;
using namespace pf::resamplers;
using namespace pf::rvsamp;
using Catch::Approx;

namespace {

constexpr size_t NPARTS = 10000;
constexpr size_t DIMX = 1;
constexpr size_t DIMY = 1;

// Same local-level model as test_bootstrap_filter.cpp, but run through the
// more general SISRFilter with its proposal set equal to the prior
// (qSamp == fSamp, logQEv == logFEv). That makes this mathematically the
// same algorithm as the bootstrap filter, just exercising SISRFilter's own
// (more general) code path against the same closed-form Kalman ground
// truth, rather than testing a different model.
template <typename float_t, typename resamp_t>
class LocalLevelSISR
    : public SISRFilter<NPARTS, DIMX, DIMY, resamp_t, float_t> {
public:
  using ssv = Eigen::Matrix<float_t, DIMX, 1>;
  using osv = Eigen::Matrix<float_t, DIMY, 1>;

  LocalLevelSISR(float_t mu0, float_t sigma0, float_t sigmaW, float_t sigmaV)
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
    return pf::rveval::evalUnivNorm<float_t>(x1(0), m_mu0, m_sigma0, true);
  }

  float_t logGEv(const osv &yt, const ssv &xt) override {
    return pf::rveval::evalUnivNorm<float_t>(yt(0), xt(0), m_sigmaV, true);
  }

  float_t logFEv(const ssv &xt, const ssv &xtm1) override {
    return pf::rveval::evalUnivNorm<float_t>(xt(0), xtm1(0), m_sigmaW, true);
  }

  ssv qSamp(const ssv &xtm1, const osv & /*yt*/) override {
    ssv xt;
    xt(0) = xtm1(0) + m_sigmaW * m_normSampler.sample();
    return xt;
  }

  float_t logQEv(const ssv &xt, const ssv &xtm1, const osv & /*yt*/) override {
    return pf::rveval::evalUnivNorm<float_t>(xt(0), xtm1(0), m_sigmaW, true);
  }

private:
  float_t m_mu0;
  float_t m_sigma0;
  float_t m_sigmaW;
  float_t m_sigmaV;
  UnivNormSampler<float_t> m_normSampler;
};

} // namespace

TEST_CASE("SISR filter matches closed-form Kalman filter on a "
          "linear-Gaussian local-level model",
          "[sisr_filter]") {
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

  std::vector<float_t> ys = {0.3,  0.9,  1.4,  1.1, 0.4,  -0.2, -0.9,
                             -1.3, -1.0, -0.3, 0.5, 1.2,  1.6,  1.0,
                             0.2};

  LocalLevelSISR<float_t, resamp_t> pf_filter(mu0, sigma0, sigmaW, sigmaV);
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
