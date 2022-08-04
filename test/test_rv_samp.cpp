#include <catch2/catch_all.hpp>

#include <pf/rv_samp.h>

#define bigdim 2
#define smalldim 1


using namespace pf;
using Catch::Approx;


class SampFixture
{
public:

    using Vec = Eigen::Matrix<double,bigdim,1>;
    using Mat = Eigen::Matrix<double,bigdim,bigdim>;

    // data members    
    rvsamp::UnivNormSampler<double> m_ns;
//    UnivNormSampler m_ns2; // nondefault construction TODO add
    rvsamp::MVNSampler<bigdim,double> m_mns;
//    MVNSampler<bigdim> m_mns2; // nondefault construction  TODO add
    rvsamp::UniformSampler<double> m_us;
    rvsamp::UniformSampler<double> m_us2; // nondefault construction
    rvsamp::UnivStudTSampler<double> m_t_sampler;
    rvsamp::BetaSampler<double> m_beta_sampler;   
 
  
  
   SampFixture() 
        : m_us2(-2.0, -1.0) // weirder upper and lower bounds
        , m_t_sampler(2.0)
        , m_beta_sampler(30.0, 10.0)
    {
        m_ns.setMean(2.0);
        m_ns.setStdDev(1.5);
        m_mns.setMean(Vec::Zero());
        m_mns.setCovar(Mat::Identity());
    }


};


TEST_CASE_METHOD(SampFixture, "univNormalTest", "[samplers]")
{
    // TODO test correctness
    m_ns.sample();
}


TEST_CASE_METHOD(SampFixture, "MultivNormalTest", "[samplers]")
{
    m_mns.sample();
}


TEST_CASE_METHOD(SampFixture, "UniformTest", "[samplers]")
{
    REQUIRE( 0.0 < m_us.sample());
    REQUIRE( m_us.sample() < 1.0);
    REQUIRE(-2.0 < m_us2.sample());
    REQUIRE( m_us2.sample() < -1.0);
}


TEST_CASE_METHOD(SampFixture, "Student T Test", "[samplers]")
{

    REQUIRE( -std::numeric_limits<double>::infinity() < m_t_sampler.sample());
}


TEST_CASE_METHOD(SampFixture, "Beta Test", "[samplers]")
{
    float_t ave = 0;
    unsigned num_sims = 1000;
    for(unsigned i = 0; i < num_sims; i++){
        float_t s = m_beta_sampler.sample();
        REQUIRE( 0.0 < s);
        REQUIRE( s < 1.0);
        ave += s/static_cast<float_t>(num_sims);
    }
    REQUIRE( std::abs(ave - .75) < .01);
}



