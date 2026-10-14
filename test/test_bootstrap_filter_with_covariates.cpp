#include <catch2/catch_all.hpp>

#include <pf/bootstrap_filter_with_covariates.h>
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
constexpr size_t DIMCOV = 1;

// A local-level model extended with a known exogenous regressor affecting
// the observation equation (a dynamic linear regression):
//   x_1 ~ N(mu0, sigma0^2)
//   x_t = x_{t-1} + w_t,            w_t ~ N(0, sigmaW^2)
//   y_t = x_t + beta * z_t + v_t,   v_t ~ N(0, sigmaV^2)
// cf_filters.h's kalman class supports this directly via its
// obsInptAffector/inputData machinery, giving the same kind of exact
// closed-form ground truth used in the other filter tests.
template <typename float_t, typename resamp_t>
class LocalLevelBSWC
    : public BSFilterWC<NPARTS, DIMX, DIMY, DIMCOV, resamp_t, float_t> {
public:
  using ssv = Eigen::Matrix<float_t, DIMX, 1>;
  using osv = Eigen::Matrix<float_t, DIMY, 1>;
  using cvsv = Eigen::Matrix<float_t, DIMCOV, 1>;

  LocalLevelBSWC(float_t mu0, float_t sigma0, float_t sigmaW, float_t sigmaV,
                 float_t beta)
      : m_mu0(mu0), m_sigma0(sigma0), m_sigmaW(sigmaW), m_sigmaV(sigmaV),
        m_beta(beta) {}

  float_t logMuEv(const ssv &x1, const cvsv & /*z1*/) override {
    return pf::rveval::evalUnivNorm<float_t>(x1(0), m_mu0, m_sigma0, true);
  }

  ssv q1Samp(const osv & /*y1*/, const cvsv & /*z1*/) override {
    ssv x1;
    x1(0) = m_mu0 + m_sigma0 * m_normSampler.sample();
    return x1;
  }

  float_t logQ1Ev(const ssv &x1, const osv & /*y1*/,
                   const cvsv & /*z1*/) override {
    return pf::rveval::evalUnivNorm<float_t>(x1(0), m_mu0, m_sigma0, true);
  }

  float_t logGEv(const osv &yt, const ssv &xt, const cvsv &zt) override {
    return pf::rveval::evalUnivNorm<float_t>(yt(0), xt(0) + m_beta * zt(0),
                                             m_sigmaV, true);
  }

  ssv fSamp(const ssv &xtm1, const cvsv & /*zt*/) override {
    ssv xt;
    xt(0) = xtm1(0) + m_sigmaW * m_normSampler.sample();
    return xt;
  }

private:
  float_t m_mu0;
  float_t m_sigma0;
  float_t m_sigmaW;
  float_t m_sigmaV;
  float_t m_beta;
  UnivNormSampler<float_t> m_normSampler;
};

} // namespace

TEST_CASE("bootstrap filter with covariates matches closed-form Kalman "
          "filter on a dynamic linear regression",
          "[bootstrap_filter_with_covariates]") {
  using float_t = double;
  using resamp_t = mn_resampler<NPARTS, DIMX, float_t>;
  using ssv = Eigen::Matrix<float_t, DIMX, 1>;
  using osv = Eigen::Matrix<float_t, DIMY, 1>;
  using cvsv = Eigen::Matrix<float_t, DIMCOV, 1>;
  using Mat = Eigen::Matrix<float_t, Eigen::Dynamic, Eigen::Dynamic>;
  using ssMat = Eigen::Matrix<float_t, DIMX, DIMX>;
  using osMat = Eigen::Matrix<float_t, DIMY, DIMY>;
  using obsStateSizeMat = Eigen::Matrix<float_t, DIMY, DIMX>;
  using siMat = Eigen::Matrix<float_t, DIMX, DIMCOV>;
  using oiMat = Eigen::Matrix<float_t, DIMY, DIMCOV>;
  using isv = Eigen::Matrix<float_t, DIMCOV, 1>;

  float_t mu0 = 0.0;
  float_t sigma0 = 2.0;
  float_t sigmaW = 0.5;
  float_t sigmaV = 1.0;
  float_t beta = 1.5;

  std::vector<float_t> ys = {0.9,  2.1,  2.3,  1.5, 0.5,  -0.6, -1.5,
                             -2.0, -1.4, -0.1, 1.1, 2.0,  2.5,  1.4,
                             0.5};
  std::vector<float_t> zs = {0.4,  0.8,  0.6,  0.2, -0.1, -0.3, -0.4,
                             -0.5, -0.3, 0.1,  0.4, 0.5,  0.6,  0.3,
                             0.2};

  LocalLevelBSWC<float_t, resamp_t> pf_filter(mu0, sigma0, sigmaW, sigmaV,
                                               beta);
  kalman<DIMX, DIMY, DIMCOV, float_t> kf(ssv::Constant(mu0),
                                         ssMat::Constant(sigma0 * sigma0));

  ssMat stateTrans = ssMat::Identity();
  ssMat cholStateVar = ssMat::Constant(sigmaW);
  obsStateSizeMat obsMat = obsStateSizeMat::Identity();
  osMat cholObsVar = osMat::Constant(sigmaV);
  siMat stateInptAffector = siMat::Zero();
  oiMat obsInptAffector = oiMat::Constant(beta);

  auto meanFn = [](const ssv &x, const cvsv & /*z*/) -> Mat { return x; };
  auto sqFn = [](const ssv &x, const cvsv & /*z*/) -> Mat {
    Mat m(1, 1);
    m(0, 0) = x(0) * x(0);
    return m;
  };
  std::vector<std::function<const Mat(const ssv &, const cvsv &)>> fs = {
      meanFn, sqFn};

  for (size_t t = 0; t < ys.size(); ++t) {
    osv yt;
    yt(0) = ys[t];
    cvsv zt;
    zt(0) = zs[t];
    isv inputData;
    inputData(0) = zs[t];

    pf_filter.filter(yt, zt, fs);
    kf.update(yt, stateTrans, cholStateVar, stateInptAffector, inputData,
              obsMat, obsInptAffector, cholObsVar);

    std::vector<Mat> expectations = pf_filter.getExpectations();
    float_t pfMean = expectations[0](0, 0);
    float_t pfSecondMoment = expectations[1](0, 0);
    float_t pfVar = pfSecondMoment - pfMean * pfMean;

    float_t kfMean = kf.getFiltMean()(0);
    float_t kfVar = kf.getFiltVar()(0, 0);

    INFO("t=" << t << " pfMean=" << pfMean << " kfMean=" << kfMean
              << " pfVar=" << pfVar << " kfVar=" << kfVar);
    REQUIRE(pfMean == Approx(kfMean).margin(0.15));
    REQUIRE(pfVar == Approx(kfVar).margin(0.15));
  }
}
