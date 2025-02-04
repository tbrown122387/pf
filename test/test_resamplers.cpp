#include <catch2/catch_all.hpp>

#include <pf/cf_filters.h>
#include <pf/resamplers.h>

#define NUMPARTICLES 20
#define DIMSTATE 3
#define DIMINNERMOD 2
#define DIMOBS 1

using namespace pf::resamplers;
using namespace pf::filters;
using Catch::Approx;

class MRFixture {
public:
  // types
  using ssv = Eigen::Matrix<double, DIMSTATE, 1>;
  using innerVec = Eigen::Matrix<double, DIMINNERMOD, 1>;
  using transMat = Eigen::Matrix<double, DIMINNERMOD, DIMINNERMOD>;
  using arrayVec = std::array<ssv, NUMPARTICLES>;
  using arrayDouble = std::array<double, NUMPARTICLES>;
  using arrayHMMMods =
      std::array<hmm<DIMINNERMOD, DIMOBS, double>, NUMPARTICLES>;
  using arrayInnerVec = std::array<innerVec, NUMPARTICLES>;

  // make the resampling object(s)
  mn_resampler<NUMPARTICLES, DIMSTATE, double> m_mr;
  mn_resampler_rbpf<NUMPARTICLES, DIMSTATE, hmm<DIMINNERMOD, DIMOBS, double>,
                    double>
      m_mr_rbpf_hmm;
  resid_resampler<NUMPARTICLES, DIMSTATE, double> m_residr;
  stratif_resampler<NUMPARTICLES, DIMSTATE, double> m_stratifr;
  systematic_resampler<NUMPARTICLES, DIMSTATE, double> m_systematicr;

  // for Test_resampLogWts and systematic, residual, stratified, etc.
  double m_good_value;
  arrayVec m_vparts;
  arrayDouble m_vw;
  arrayVec m_vparts2;
  arrayDouble m_vw2;
  arrayVec m_vparts3;
  arrayDouble m_vw3;
  arrayVec m_vparts4;
  arrayDouble m_vw4;

  // for Test_resampLogWts_RBPF
  innerVec m_initLogProbDistn1;
  innerVec m_initLogProbDistn2;
  transMat m_initLogTransMat1;
  transMat m_initLogTransMat2;
  arrayHMMMods m_hmms;
  arrayVec m_rbpf_samps;
  arrayDouble m_rbpf_logwts;

  MRFixture()
      : m_initLogTransMat1(transMat::Zero()),
        m_initLogTransMat2(transMat::Zero()) {
    // make the first particle have good value for everything (samples and
    // weights) make all other particles have 0 for samples and -INF for weights
    m_good_value = 42.42;
    for (size_t i = 0; i < NUMPARTICLES; ++i) {
      if (i == 0) {
        m_vparts[i] = ssv::Constant(m_good_value);
        m_vw[i] = 0.0;
        m_vparts2[i] = ssv::Constant(m_good_value);
        m_vw2[i] = 0.0;
        m_vparts3[i] = ssv::Constant(m_good_value);
        m_vw3[i] = 0.0;
        m_vparts4[i] = ssv::Constant(m_good_value);
        m_vw4[i] = 0.0;
      } else {
        m_vparts[i] = ssv::Constant(0.0);
        m_vw[i] = -std::numeric_limits<float_t>::infinity();
        m_vparts2[i] = ssv::Constant(0.0);
        m_vw2[i] = -std::numeric_limits<float_t>::infinity();
        m_vparts3[i] = ssv::Constant(0.0);
        m_vw3[i] = -std::numeric_limits<float_t>::infinity();
        m_vparts4[i] = ssv::Constant(0.0);
        m_vw4[i] = -std::numeric_limits<float_t>::infinity();
        ;
      }
    }

    // for Test_resampLogWts_RBPF
    // make the first particle have 0 for samples and weights,
    //      and will have a model with all weight in the first spot,
    //      and a transition matrix that keeps everything in that first spot

    // make everything BUT the first particle have 1 for samples and -INF for
    // weights
    //      as well as models that have models with all weight in the last spot
    //      and transition matrices that keep it in that last spot
    for (size_t i = 0; i < DIMINNERMOD; ++i) {

      m_initLogTransMat1(i, 0) = std::log(1.0);
      m_initLogTransMat2(i, DIMINNERMOD - 1) = std::log(1.0);
      m_initLogTransMat1.block(i, 1, 1, DIMINNERMOD - 1)
          .fill(std::log(
              0.0)); // = Eigen::Matrix<double,1,DIMINNERMOD-1>::Zero();
      m_initLogTransMat2.block(i, 0, 1, DIMINNERMOD - 1)
          .fill(std::log(
              0.0)); // = Eigen::Matrix<double,1,DIMINNERMOD-1>::Zero();

      if (i == 0) {
        m_initLogProbDistn1(i) = std::log(1.0);
        m_initLogProbDistn2(i) = std::log(0.0);
      } else if (i == DIMINNERMOD - 1) {
        m_initLogProbDistn1(i) = std::log(0.0);
        m_initLogProbDistn2(i) = std::log(1.0);
      } else {
        m_initLogProbDistn1(i) = std::log(0.0);
        m_initLogProbDistn2(i) = std::log(0.0);
      }
    }

    for (size_t i = 0; i < NUMPARTICLES; ++i) {
      if (i == 0) {
        m_hmms[i] = hmm<DIMINNERMOD, DIMOBS, double>(m_initLogProbDistn1,
                                                     m_initLogTransMat1);
        m_rbpf_samps[i] = ssv::Constant(0.0);
        m_rbpf_logwts[i] = 0.0;
      } else {
        m_hmms[i] = hmm<DIMINNERMOD, DIMOBS, double>(m_initLogProbDistn2,
                                                     m_initLogTransMat2);
        m_rbpf_samps[i] = ssv::Constant(1.0);
        m_rbpf_logwts[i] = -std::numeric_limits<float_t>::infinity();
      }
    }
  }
};

TEST_CASE_METHOD(MRFixture, "test resampLogWts", "[resamplers]") {

  m_mr.resampLogWts(m_vparts, m_vw);
  for (unsigned int p = 0; p < NUMPARTICLES; ++p) {

    REQUIRE(m_vw[p] == 0.0);
    for (unsigned int i = 0; i < DIMSTATE; ++i) {
      REQUIRE(m_vparts[p](i) == m_good_value);
    }
  }
}

TEST_CASE_METHOD(MRFixture, "test resampLogWts 2", "[resamplers]") {

  arrayVec oneGoodSamp;
  arrayDouble oneGoodWeight;
  for (size_t i = 0; i < NUMPARTICLES; ++i) {
    oneGoodWeight[i] = -std::numeric_limits<float_t>::infinity();
    ssv xt;
    xt(0) = i;
    oneGoodSamp[i] = xt;
  }
  oneGoodWeight[3] = 0.0;
  m_mr.resampLogWts(oneGoodSamp, oneGoodWeight);
  for (unsigned int p = 0; p < NUMPARTICLES; ++p) {

    REQUIRE(oneGoodSamp[p](0) == Approx(3.0));
    REQUIRE(oneGoodWeight[p] == 0.0);
  }
}

TEST_CASE_METHOD(MRFixture, "test resampLogWts_RBPF", "[resamplers]") {

  // resample... you should only have the first set of things repeated a bunch
  // of times
  m_mr_rbpf_hmm.resampLogWts(m_hmms, m_rbpf_samps, m_rbpf_logwts);

  // check all the particles
  for (unsigned int p = 0; p < NUMPARTICLES; ++p) {

    // log 1 = 0
    REQUIRE(m_rbpf_logwts[p] == 0.0);

    // everything should be 0
    for (unsigned int i = 0; i < DIMSTATE; ++i) {
      REQUIRE(m_rbpf_samps[p](i) == 0.0);
    }

    // the model that keeps everything in the first spot should remain
    innerVec before = m_hmms[p].getFilterVecLogProbs();
    for (unsigned int i = 0; i < DIMINNERMOD; ++i) {
      if (i == 0) {
        REQUIRE(before(i) == std::log(1.0)); // all weight in the first spot
      } else {
        REQUIRE(before(i) == std::log(0.0)); // all weight in the first spot
      }
    }
    m_hmms[p].update(innerVec::Constant(1.0));
    innerVec after = m_hmms[p].getFilterVecLogProbs();
    for (unsigned int i = 0; i < DIMINNERMOD; ++i) {
      if (i == 0) {
        REQUIRE(after(i) == std::log(1.0)); // all weight in the first spot
      } else {
        REQUIRE(after(i) == std::log(0.0)); // all weight in the first spot
      }
    }
  }

  /**
   * @TODO complete Kalman testing as well
   */
}

TEST_CASE_METHOD(MRFixture, "test resampLogWts_resid"
                            "[resamplers]") {

  m_residr.resampLogWts(m_vparts2, m_vw2);
  for (unsigned int p = 0; p < NUMPARTICLES; ++p) {

    REQUIRE(m_vw2[p] == Approx(0.0));
    for (unsigned int i = 0; i < DIMSTATE; ++i) {
      REQUIRE(m_vparts2[p](i) == m_good_value);
    }
  }
}

TEST_CASE_METHOD(MRFixture, "test resampLogWts_resid2"
                            "[resamplers]") {

  arrayVec oneGoodSamp;
  arrayDouble oneGoodWeight;
  for (size_t i = 0; i < NUMPARTICLES; ++i) {
    oneGoodWeight[i] = -std::numeric_limits<float_t>::infinity();
    ssv xt;
    xt(0) = i;
    oneGoodSamp[i] = xt;
  }
  oneGoodWeight[3] = 0.0;
  m_residr.resampLogWts(oneGoodSamp, oneGoodWeight);
  for (unsigned int p = 0; p < NUMPARTICLES; ++p) {

    REQUIRE(oneGoodSamp[p](0) == Approx(3.0));
    REQUIRE(oneGoodWeight[p] == 0.0);
  }
}

TEST_CASE_METHOD(MRFixture, "test resampLogWts_stratif", "[resamplers]") {

  m_stratifr.resampLogWts(m_vparts3, m_vw3);
  for (unsigned int p = 0; p < NUMPARTICLES; ++p) {

    REQUIRE(m_vw3[p] == 0.0);
    for (unsigned int i = 0; i < DIMSTATE; ++i) {
      REQUIRE(m_vparts3[p](i) == m_good_value);
    }
  }
}

TEST_CASE_METHOD(MRFixture, "test resampLogWts_stratif2", "[resamplers]") {

  arrayVec oneGoodSamp;
  arrayDouble oneGoodWeight;
  for (size_t i = 0; i < NUMPARTICLES; ++i) {
    oneGoodWeight[i] = -std::numeric_limits<float_t>::infinity();
    ssv xt;
    xt(0) = i;
    oneGoodSamp[i] = xt;
  }
  oneGoodWeight[3] = 0.0;
  m_stratifr.resampLogWts(oneGoodSamp, oneGoodWeight);
  for (unsigned int p = 0; p < NUMPARTICLES; ++p) {

    REQUIRE(oneGoodSamp[p](0) == Approx(3.0));
    REQUIRE(oneGoodWeight[p] == 0.0);
  }
}

TEST_CASE_METHOD(MRFixture, "test resampLogWts_systematic", "[resamplers]") {

  m_systematicr.resampLogWts(m_vparts4, m_vw4);
  for (unsigned int p = 0; p < NUMPARTICLES; ++p) {

    REQUIRE(m_vw4[p] == 0.0);
    for (unsigned int i = 0; i < DIMSTATE; ++i) {
      REQUIRE(m_vparts4[p](i) == m_good_value);
    }
  }
}

TEST_CASE_METHOD(MRFixture, "test resampLogWts_systematic2", "[resamplers]") {

  arrayVec oneGoodSamp;
  arrayDouble oneGoodWeight;
  for (size_t i = 0; i < NUMPARTICLES; ++i) {
    oneGoodWeight[i] = -std::numeric_limits<float_t>::infinity();
    ssv xt;
    xt(0) = i;
    oneGoodSamp[i] = xt;
  }
  oneGoodWeight[3] = 0.0;
  m_systematicr.resampLogWts(oneGoodSamp, oneGoodWeight);
  for (unsigned int p = 0; p < NUMPARTICLES; ++p) {

    REQUIRE(oneGoodSamp[p](0) == Approx(3.0));
    REQUIRE(oneGoodWeight[p] == 0.0);
  }
}

TEST_CASE_METHOD(MRFixture, "test auxiliary hilbert functions",
                 "[resamplers]") {
  using namespace pf::resamplers;

  constexpr unsigned nb = 3;
  constexpr unsigned nd = 2;
  unsigned total_matches = 0;
  unsigned recov_H;
  for (unsigned H = 0; H < pow(2, nb * nd); ++H) {

    recov_H = makeH<nb, nd>(makeHTranspose<nb, nd>(H));
    if (recov_H == H)
      total_matches++;
  }
  REQUIRE(total_matches == pow(2, nb * nd));
}

TEST_CASE_METHOD(MRFixture, "test auxiliary hilbert functions 2",
                 "[resamplers]") {
  using namespace pf::resamplers;

  constexpr unsigned nb = 2;
  constexpr unsigned nd = 2;
  unsigned total_matches = 0;
  unsigned recov_H;
  for (unsigned H = 0; H < pow(2, nb * nd); ++H) {

    recov_H = makeH<nb, nd>(makeHTranspose<nb, nd>(H));
    if (recov_H == H)
      total_matches++;
  }
  REQUIRE(total_matches == pow(2, nb * nd));
}

TEST_CASE_METHOD(MRFixture, "test auxiliary hilbert functions 3",
                 "[resamplers]") {
  using namespace pf::resamplers;

  constexpr unsigned nb = 1;
  constexpr unsigned nd = 2;
  unsigned total_matches = 0;
  unsigned recov_H;
  for (unsigned H = 0; H < pow(2, nb * nd); ++H) {

    recov_H = makeH<nb, nd>(makeHTranspose<nb, nd>(H));
    if (recov_H == H)
      total_matches++;
  }
  REQUIRE(total_matches == pow(2, nb * nd));
}

TEST_CASE_METHOD(MRFixture, "test auxiliary hilbert functions 4",
                 "[resamplers]") {
  using namespace pf::resamplers;

  constexpr unsigned nb = 3;
  constexpr unsigned nd = 3;
  unsigned total_matches = 0;
  unsigned recov_H;
  for (unsigned H = 0; H < pow(2, nb * nd); ++H) {

    recov_H = makeH<nb, nd>(makeHTranspose<nb, nd>(H));
    if (recov_H == H)
      total_matches++;
  }
  REQUIRE(total_matches == pow(2, nb * nd));
}

TEST_CASE_METHOD(MRFixture, "test auxiliary hilbert functions 5",
                 "[resamplers]") {
  using namespace pf::resamplers;

  constexpr unsigned nb = 2;
  constexpr unsigned nd = 3;
  unsigned total_matches = 0;
  unsigned recov_H;
  for (unsigned H = 0; H < pow(2, nb * nd); ++H) {

    recov_H = makeH<nb, nd>(makeHTranspose<nb, nd>(H));
    if (recov_H == H)
      total_matches++;
  }
  REQUIRE(total_matches == pow(2, nb * nd));
}

TEST_CASE_METHOD(MRFixture, "test auxiliary hilbert functions 6",
                 "[resamplers]") {
  using namespace pf::resamplers;

  constexpr unsigned nb = 1;
  constexpr unsigned nd = 3;
  unsigned total_matches = 0;
  unsigned recov_H;
  for (unsigned H = 0; H < pow(2, nb * nd); ++H) {

    recov_H = makeH<nb, nd>(makeHTranspose<nb, nd>(H));
    if (recov_H == H)
      total_matches++;
  }
  REQUIRE(total_matches == pow(2, nb * nd));
}

TEST_CASE_METHOD(MRFixture, "test hilbert inverses 1", "[resamplers]") {

  constexpr unsigned nb = 1;
  constexpr unsigned nd = 2;
  using namespace pf::resamplers;

  unsigned total_matches = 0;
  unsigned recov_H;
  for (unsigned H = 0; H < pow(2, nb * nd); ++H) {

    recov_H = makeH<nb, nd>(AxesToTranspose<nb, nd>(
        TransposeToAxes<nb, nd>(makeHTranspose<nb, nd>(H))));
    if (recov_H == H)
      total_matches++;
    // std::cout << H << ", " << recov_H << "\n";

    // this is just for printing if you want it
    // auto H_image = TransposeToAxes<nb,nd>(makeHTranspose<nb,nd>(H));
    // std::cout << H << ", ";
    // for(size_t dim = 0; dim < nd; ++dim)
    //    std::cout << H_image[dim].to_ulong() << ", ";
    // std::cout << "\n";
  }
  REQUIRE(total_matches == pow(2, nb * nd));
}

TEST_CASE_METHOD(MRFixture, "test hilbert inverses 2", "[resamplers]") {

  constexpr unsigned nb = 2;
  constexpr unsigned nd = 2;
  using namespace pf::resamplers;

  unsigned total_matches = 0;
  unsigned recov_H;
  for (unsigned H = 0; H < pow(2, nb * nd); ++H) {

    recov_H = makeH<nb, nd>(AxesToTranspose<nb, nd>(
        TransposeToAxes<nb, nd>(makeHTranspose<nb, nd>(H))));
    if (recov_H == H)
      total_matches++;
  }
  REQUIRE(total_matches == pow(2, nb * nd));
}

TEST_CASE_METHOD(MRFixture, "test hilbert inverses 3", "[resamplers]") {

  constexpr unsigned nb = 3;
  constexpr unsigned nd = 2;
  using namespace pf::resamplers;

  unsigned total_matches = 0;
  unsigned recov_H;
  for (unsigned H = 0; H < pow(2, nb * nd); ++H) {

    recov_H = makeH<nb, nd>(AxesToTranspose<nb, nd>(
        TransposeToAxes<nb, nd>(makeHTranspose<nb, nd>(H))));
    if (recov_H == H)
      total_matches++;
  }
  REQUIRE(total_matches == pow(2, nb * nd));
}

TEST_CASE_METHOD(MRFixture, "test hilbert inverses 4", "[resamplers]") {

  constexpr unsigned nb = 4;
  constexpr unsigned nd = 2;
  using namespace pf::resamplers;

  unsigned total_matches = 0;
  unsigned recov_H;
  for (unsigned H = 0; H < pow(2, nb * nd); ++H) {

    recov_H = makeH<nb, nd>(AxesToTranspose<nb, nd>(
        TransposeToAxes<nb, nd>(makeHTranspose<nb, nd>(H))));
    if (recov_H == H)
      total_matches++;
  }
  REQUIRE(total_matches == pow(2, nb * nd));
}

TEST_CASE_METHOD(MRFixture, "test hilbert inverses 5", "[resamplers]") {

  constexpr unsigned nb = 1;
  constexpr unsigned nd = 3;
  using namespace pf::resamplers;

  unsigned total_matches = 0;
  unsigned recov_H;
  for (unsigned H = 0; H < pow(2, nb * nd); ++H) {

    recov_H = makeH<nb, nd>(AxesToTranspose<nb, nd>(
        TransposeToAxes<nb, nd>(makeHTranspose<nb, nd>(H))));
    if (recov_H == H)
      total_matches++;
  }
  REQUIRE(total_matches == pow(2, nb * nd));
}

TEST_CASE_METHOD(MRFixture, "test hilbert inverses 6", "[resamplers]") {

  constexpr unsigned nb = 2;
  constexpr unsigned nd = 3;
  using namespace pf::resamplers;

  unsigned total_matches = 0;
  unsigned recov_H;
  for (unsigned H = 0; H < pow(2, nb * nd); ++H) {

    recov_H = makeH<nb, nd>(AxesToTranspose<nb, nd>(
        TransposeToAxes<nb, nd>(makeHTranspose<nb, nd>(H))));
    if (recov_H == H)
      total_matches++;
  }
  REQUIRE(total_matches == pow(2, nb * nd));
}

TEST_CASE_METHOD(MRFixture, "test hilbert inverses 7", "[resamplers]") {

  constexpr unsigned nb = 3;
  constexpr unsigned nd = 3;
  using namespace pf::resamplers;

  unsigned total_matches = 0;
  unsigned recov_H;
  for (unsigned H = 0; H < pow(2, nb * nd); ++H) {

    recov_H = makeH<nb, nd>(AxesToTranspose<nb, nd>(
        TransposeToAxes<nb, nd>(makeHTranspose<nb, nd>(H))));
    if (recov_H == H)
      total_matches++;
  }
  REQUIRE(total_matches == pow(2, nb * nd));
}

TEST_CASE_METHOD(MRFixture, "test hilbert inverses 8", "[resamplers]") {

  constexpr unsigned nb = 4;
  constexpr unsigned nd = 3;
  using namespace pf::resamplers;

  unsigned total_matches = 0;
  unsigned recov_H;
  for (unsigned H = 0; H < pow(2, nb * nd); ++H) {

    recov_H = makeH<nb, nd>(AxesToTranspose<nb, nd>(
        TransposeToAxes<nb, nd>(makeHTranspose<nb, nd>(H))));
    if (recov_H == H)
      total_matches++;
  }
  REQUIRE(total_matches == pow(2, nb * nd));
}
