#include <catch2/catch_all.hpp>

#include <pf/cf_filters.h>
#include <pf/rbpf.h>
#include <pf/resamplers.h>
#include <pf/rv_eval.h>
#include <pf/rv_samp.h>

using namespace pf::filters;
using namespace pf::resamplers;
using namespace pf::rvsamp;
using Catch::Approx;

namespace {

constexpr size_t NPARTS = 500;
constexpr size_t DIMNSS = 1; // "not sampled state": the Kalman-filtered x1
constexpr size_t DIMSS = 1;  // "sampled state": x2, intentionally degenerate
constexpr size_t DIMY = 1;

// An RBPF where the sampled component x2 is given a near-degenerate
// (tiny-variance) distribution that the inner Kalman model never actually
// depends on. Since the proposal for x2 is the bootstrap proposal (qSamp ==
// fSamp, logQEv == logFEv, so those terms cancel exactly in the weight
// update) and the Kalman model's matrices don't depend on x2 at all, every
// particle carries an identical weight and runs an identical inner Kalman
// recursion at every time step. So, unlike the Monte Carlo comparisons for
// the other filters, this should match the closed-form Kalman filter to
// near machine precision, regardless of nparts.
template <typename float_t, typename resamp_t>
class DegenerateRBPFKalman
    : public rbpf_kalman<NPARTS, DIMNSS, DIMSS, DIMY, resamp_t, float_t> {
public:
  using nsssv = Eigen::Matrix<float_t, DIMNSS, 1>;
  using nsssMat = Eigen::Matrix<float_t, DIMNSS, DIMNSS>;
  using sssv = Eigen::Matrix<float_t, DIMSS, 1>;
  using osv = Eigen::Matrix<float_t, DIMY, 1>;
  using cfModType = kalman<DIMNSS, DIMY, 0, float_t>;
  using ssMat = Eigen::Matrix<float_t, DIMNSS, DIMNSS>;
  using osMat = Eigen::Matrix<float_t, DIMY, DIMY>;
  using obsStateSizeMat = Eigen::Matrix<float_t, DIMY, DIMNSS>;
  using siMat = Eigen::Matrix<float_t, DIMNSS, 0>;
  using oiMat = Eigen::Matrix<float_t, DIMY, 0>;
  using isv = Eigen::Matrix<float_t, 0, 1>;

  DegenerateRBPFKalman(float_t mu0, float_t sigma0, float_t sigmaW,
                       float_t sigmaV, float_t x2Sigma)
      : m_mu0(mu0), m_sigma0(sigma0), m_x2Sigma(x2Sigma) {
    m_stateTrans = ssMat::Identity();
    m_cholStateVar = ssMat::Constant(sigmaW);
    m_obsMat = obsStateSizeMat::Identity();
    m_cholObsVar = osMat::Constant(sigmaV);
  }

  float_t logMuEv(const sssv &x21) override {
    return pf::rveval::evalUnivNorm<float_t>(x21(0), 0.0, m_x2Sigma, true);
  }

  sssv q1Samp(const osv & /*y1*/) override {
    sssv x21;
    x21(0) = m_x2Sigma * m_normSampler.sample();
    return x21;
  }

  nsssv initKalmanMean(const sssv & /*x21*/) override {
    return nsssv::Constant(m_mu0);
  }

  nsssMat initKalmanVar(const sssv & /*x21*/) override {
    return nsssMat::Constant(m_sigma0 * m_sigma0);
  }

  sssv qSamp(const sssv &x2tm1, const osv & /*yt*/) override {
    sssv x2t;
    x2t(0) = x2tm1(0) + m_x2Sigma * m_normSampler.sample();
    return x2t;
  }

  float_t logQ1Ev(const sssv &x21, const osv & /*y1*/) override {
    return pf::rveval::evalUnivNorm<float_t>(x21(0), 0.0, m_x2Sigma, true);
  }

  float_t logFEv(const sssv &x2t, const sssv &x2tm1) override {
    return pf::rveval::evalUnivNorm<float_t>(x2t(0), x2tm1(0), m_x2Sigma,
                                             true);
  }

  float_t logQEv(const sssv &x2t, const sssv &x2tm1,
                 const osv & /*yt*/) override {
    return pf::rveval::evalUnivNorm<float_t>(x2t(0), x2tm1(0), m_x2Sigma,
                                             true);
  }

  void updateKalman(cfModType &kMod, const osv &yt,
                    const sssv & /*x2t*/) override {
    siMat stateInptAffector;
    oiMat obsInptAffector;
    isv inputData;
    kMod.update(yt, m_stateTrans, m_cholStateVar, stateInptAffector,
                inputData, m_obsMat, obsInptAffector, m_cholObsVar);
  }

private:
  float_t m_mu0;
  float_t m_sigma0;
  float_t m_x2Sigma;
  ssMat m_stateTrans;
  ssMat m_cholStateVar;
  obsStateSizeMat m_obsMat;
  osMat m_cholObsVar;
  UnivNormSampler<float_t> m_normSampler;
};

} // namespace

TEST_CASE("Rao-Blackwellized (Kalman) particle filter exactly matches "
          "closed-form Kalman filter when the sampled component carries no "
          "real information",
          "[rbpf_kalman]") {
  using float_t = double;
  using resamp_t =
      mn_resampler_rbpf<NPARTS, DIMSS, kalman<DIMNSS, DIMY, 0, float_t>,
                        float_t>;
  using nsssv = Eigen::Matrix<float_t, DIMNSS, 1>;
  using sssv = Eigen::Matrix<float_t, DIMSS, 1>;
  using osv = Eigen::Matrix<float_t, DIMY, 1>;
  using Mat = Eigen::Matrix<float_t, Eigen::Dynamic, Eigen::Dynamic>;
  using ssMat = Eigen::Matrix<float_t, DIMNSS, DIMNSS>;
  using osMat = Eigen::Matrix<float_t, DIMY, DIMY>;
  using obsStateSizeMat = Eigen::Matrix<float_t, DIMY, DIMNSS>;
  using siMat = Eigen::Matrix<float_t, DIMNSS, 0>;
  using oiMat = Eigen::Matrix<float_t, DIMY, 0>;
  using isv = Eigen::Matrix<float_t, 0, 1>;

  float_t mu0 = 0.0;
  float_t sigma0 = 2.0;
  float_t sigmaW = 0.5;
  float_t sigmaV = 1.0;
  float_t x2Sigma = 1e-6; // near-degenerate "sampled" component

  std::vector<float_t> ys = {0.3,  0.9,  1.4,  1.1, 0.4,  -0.2, -0.9,
                             -1.3, -1.0, -0.3, 0.5, 1.2,  1.6,  1.0,
                             0.2};

  DegenerateRBPFKalman<float_t, resamp_t> rbpf(mu0, sigma0, sigmaW, sigmaV,
                                               x2Sigma);
  kalman<DIMNSS, DIMY, 0, float_t> kf(nsssv::Constant(mu0),
                                      ssMat::Constant(sigma0 * sigma0));

  ssMat stateTrans = ssMat::Identity();
  ssMat cholStateVar = ssMat::Constant(sigmaW);
  obsStateSizeMat obsMat = obsStateSizeMat::Identity();
  osMat cholObsVar = osMat::Constant(sigmaV);
  siMat stateInptAffector;
  oiMat obsInptAffector;
  isv inputData;

  auto meanFn = [](const nsssv &x1t, const sssv & /*x2t*/) -> Mat {
    return x1t;
  };
  std::vector<std::function<const Mat(const nsssv &, const sssv &)>> fs = {
      meanFn};

  for (float_t y : ys) {
    osv yt;
    yt(0) = y;

    rbpf.filter(yt, fs);
    kf.update(yt, stateTrans, cholStateVar, stateInptAffector, inputData,
              obsMat, obsInptAffector, cholObsVar);

    std::vector<Mat> expectations = rbpf.getExpectations();
    float_t rbpfMean = expectations[0](0, 0);
    float_t kfMean = kf.getFiltMean()(0);

    INFO("y=" << y << " rbpfMean=" << rbpfMean << " kfMean=" << kfMean);
    REQUIRE(rbpfMean == Approx(kfMean).margin(1e-6));
  }
}
