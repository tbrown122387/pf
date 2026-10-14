#include <catch2/catch_all.hpp>

#include <pf/pf_base.h>
#include <pf/rv_samp.h>

using namespace pf::bases;
using namespace pf::rvsamp;

namespace {

constexpr size_t DIMX = 1;
constexpr size_t DIMY = 1;
constexpr size_t NPARTS = 5;

// pf_base.h's ForwardMod/FutureSimulator mixins aren't filters themselves -
// they're reusable building blocks for simulating forward from a
// generative model. Since nothing in the test suite exercised them before,
// this is a standalone smoke test: a simple local-level model implementing
// just the sampling primitives these mixins need, checking that the
// simulated paths have the right shape and that repeated calls actually
// produce different (genuinely random) draws rather than silently
// returning constant/stale data.
template <typename float_t>
class LocalLevelForwardSim : public ForwardMod<DIMX, DIMY, float_t>,
                             public FutureSimulator<DIMX, DIMY, float_t,
                                                     NPARTS> {
public:
  using ssv = Eigen::Matrix<float_t, DIMX, 1>;
  using osv = Eigen::Matrix<float_t, DIMY, 1>;

  LocalLevelForwardSim(float_t mu0, float_t sigma0, float_t sigmaW,
                       float_t sigmaV)
      : m_mu0(mu0), m_sigma0(sigma0), m_sigmaW(sigmaW), m_sigmaV(sigmaV) {
    for (auto &s : m_currentSamps)
      s(0) = 0.0;
  }

  ssv muSamp() override {
    ssv x1;
    x1(0) = m_mu0 + m_sigma0 * m_normSampler.sample();
    return x1;
  }

  ssv fSamp(const ssv &xtm1) override {
    ssv xt;
    xt(0) = xtm1(0) + m_sigmaW * m_normSampler.sample();
    return xt;
  }

  osv gSamp(const ssv &xt) override {
    osv yt;
    yt(0) = xt(0) + m_sigmaV * m_normSampler.sample();
    return yt;
  }

  std::array<ssv, NPARTS> get_uwtd_samps() const override {
    return m_currentSamps;
  }

private:
  float_t m_mu0;
  float_t m_sigma0;
  float_t m_sigmaW;
  float_t m_sigmaV;
  std::array<ssv, NPARTS> m_currentSamps;
  UnivNormSampler<float_t> m_normSampler;
};

} // namespace

TEST_CASE("ForwardMod::sim_forward produces correctly-shaped, genuinely "
          "random paths",
          "[pf_base][forward_mod]") {
  using float_t = double;
  constexpr unsigned int T = 20;

  LocalLevelForwardSim<float_t> model(0.0, 2.0, 0.5, 1.0);

  auto [xs1, ys1] = model.sim_forward(T);
  REQUIRE(xs1.size() == T);
  REQUIRE(ys1.size() == T);

  auto [xs2, ys2] = model.sim_forward(T);
  REQUIRE(xs2.size() == T);
  REQUIRE(ys2.size() == T);

  // two independent simulated paths from the same model should not be
  // identical - if they were, that would indicate the sampler is silently
  // returning constant/stale values instead of actually drawing randomly
  bool anyStateDiffers = false;
  bool anyObsDiffers = false;
  for (unsigned int t = 0; t < T; ++t) {
    if (xs1[t](0) != xs2[t](0))
      anyStateDiffers = true;
    if (ys1[t](0) != ys2[t](0))
      anyObsDiffers = true;
  }
  REQUIRE(anyStateDiffers);
  REQUIRE(anyObsDiffers);
}

TEST_CASE("FutureSimulator::sim_future/_obs/_states produce correctly-shaped, "
          "genuinely random paths",
          "[pf_base][future_simulator]") {
  using float_t = double;
  constexpr unsigned int T = 15;

  LocalLevelForwardSim<float_t> model(0.0, 2.0, 0.5, 1.0);

  auto manyPairs1 = model.sim_future(T);
  REQUIRE(manyPairs1.size() == T);
  for (const auto &timePair : manyPairs1) {
    REQUIRE(std::get<0>(timePair).size() == NPARTS);
    REQUIRE(std::get<1>(timePair).size() == NPARTS);
  }

  auto obsPaths = model.sim_future_obs(T);
  REQUIRE(obsPaths.size() == T);
  for (const auto &obsAtT : obsPaths)
    REQUIRE(obsAtT.size() == NPARTS);

  auto statePaths = model.sim_future_states(T);
  REQUIRE(statePaths.size() == T);
  for (const auto &stateAtT : statePaths)
    REQUIRE(stateAtT.size() == NPARTS);

  // different particles within the same simulated time step should not all
  // be identical (each should be an independent draw)
  const auto &firstTimeStates = std::get<0>(manyPairs1[0]);
  bool anyParticleDiffers = false;
  for (size_t p = 1; p < NPARTS; ++p) {
    if (firstTimeStates[p](0) != firstTimeStates[0](0))
      anyParticleDiffers = true;
  }
  REQUIRE(anyParticleDiffers);

  // two independent calls to sim_future from the same starting samples
  // should not produce identical results
  auto manyPairs2 = model.sim_future(T);
  bool anyCallDiffers = false;
  for (unsigned int t = 0; t < T; ++t) {
    const auto &states1 = std::get<0>(manyPairs1[t]);
    const auto &states2 = std::get<0>(manyPairs2[t]);
    for (size_t p = 0; p < NPARTS; ++p) {
      if (states1[p](0) != states2[p](0))
        anyCallDiffers = true;
    }
  }
  REQUIRE(anyCallDiffers);
}
