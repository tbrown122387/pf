#include "rv_samp.h"

// only implements the non-templated classes from the header file

using namespace pf;

UnivNormSampler::UnivNormSampler()
    : rvsamp_base<1>::rvsamp_base()
    , m_z_gen(0.0, 1.0)
{
    setMean(0.0);
    setStdDev(1.0);
}


UnivNormSampler::UnivNormSampler(const double &mu, const double &sigma)
    : rvsamp_base<1>::rvsamp_base()
    , m_z_gen(0.0, 1.0)
{
    setMean(mu); 
    setStdDev(sigma);
}


void UnivNormSampler::setMean(const double &mu)
{
    m_mu = mu;
}


void UnivNormSampler::setStdDev(const double &sigma)
{
    m_sigma = sigma;
}


double UnivNormSampler::sample()
{
    return m_mu + m_sigma * m_z_gen(m_rng);
}

///////////////////////////////////////////////

UniformSampler::UniformSampler() 
        : rvsamp_base<1>()
        , m_unif_gen(0.0, 1.0)
{
}


UniformSampler::UniformSampler(const double &lower, const double &upper) 
        : rvsamp_base<1>()
        , m_unif_gen(lower, upper)
{    
}


double UniformSampler::sample()
{
    return m_unif_gen(m_rng);
}


