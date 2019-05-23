#include "rv_samp.h"


///////////////////////////////////////////////
// only implements the non-templated classes from the header file
using namespace rvsamp;


rvsamp_base::rvsamp_base() 
    : m_rng{static_cast<std::uint32_t>(std::chrono::high_resolution_clock::now().time_since_epoch().count())} 
{
}


///////////////////////////////////////////////

BernSampler::BernSampler() : rvsamp_base()
                           , m_B_gen(.5)
{
}


BernSampler::BernSampler(const double& p) : rvsamp_base()
                                          , m_B_gen(p)
{
}


BernSampler::BernSampler(const float& p) : rvsamp_base()
                                          , m_B_gen(p)
{
}


void BernSampler::setP(const double& p)
{
    m_p = p;
}


void BernSampler::setP(const float_t& p)
{
    m_p = p;
}


int BernSampler::sample()
{
    return (m_B_gen(m_rng)) ? 1 : 0;
}


///////////////////////////////////////////////




