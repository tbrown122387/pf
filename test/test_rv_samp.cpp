//#include "UnitTest++.h"
#include <UnitTest++/UnitTest++.h>
#include "rv_samp.h"

#define bigdim 2
#define smalldim 1

class SampFixture
{
public:

    using Vec = Eigen::Matrix<double,bigdim,1>;
    using Mat = Eigen::Matrix<double,bigdim,bigdim>;

    // data members    
    rvsamp::UnivNormSampler m_ns;
//    UnivNormSampler m_ns2; // nondefault construction TODO add
    rvsamp::MVNSampler<bigdim> m_mns;
//    MVNSampler<bigdim> m_mns2; // nondefault construction  TODO add
    rvsamp::UniformSampler m_us;
    rvsamp::UniformSampler m_us2; // nondefault construction
    
    SampFixture() : m_us2(-2.0, -1.0) // weirder upper and lower bounds
    {
        m_ns.setMean(2.0);
        m_ns.setStdDev(1.5);
        m_mns.setMean(Vec::Zero());
        m_mns.setCovar(Mat::Identity());
    }
    
};


TEST_FIXTURE(SampFixture, univNormalTest)
{
    // TODO test correctness
    m_ns.sample();
}


TEST_FIXTURE(SampFixture, MultivNormalTest)
{
    m_mns.sample();
}


TEST_FIXTURE(SampFixture, UniformTest)
{
    CHECK( (0.0 < m_us.sample()) && (m_us.sample() < 1.0));
    CHECK( (-2.0 < m_us2.sample()) && (m_us2.sample() < -1.0));
}
