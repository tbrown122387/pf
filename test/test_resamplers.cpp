#include "UnitTest++.h"
#include "resamplers.h"

#define NUMPARTICLES 2
#define DIMSTATE     3

class MRFixture
{
public:

    // types
    using ssv = Eigen::Matrix<double,DIMSTATE,1>;
    using arrayVec = std::array<ssv,NUMPARTICLES>;
    using arrayDouble = std::array<double,NUMPARTICLES>;
    
    
    // make the resampling object(s)
    mn_resampler<NUMPARTICLES, DIMSTATE> m_mr;
    
    // for Test_resampLogWts
    arrayVec    m_vparts;
    arrayDouble m_vw;

    MRFixture()
    {
        // for Test_resampLogWts
        m_vparts[0] = ssv::Constant(0.0);
        m_vparts[1] = ssv::Constant(1.0);
        m_vw[0] = 0.0;
        m_vw[1] = -1.0/0.0;
    }
    
};


TEST_FIXTURE(MRFixture, Test_resampLogWts)
{
    
    m_mr.resampLogWts(m_vparts, m_vw);
    for(unsigned int p = 0; p < NUMPARTICLES; ++p){
        
        CHECK_EQUAL(m_vw[p], 0.0);
        for(unsigned int i = 0; i < DIMSTATE; ++i){
            CHECK_EQUAL(m_vparts[p](i), 0.0);
        }
    }
}
