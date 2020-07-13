#ifndef SVOL_SISR_H
#define SVOL_SISR_H

#include <pf/sisr_filter.h> // inherit the right particle filter
#include <pf/rv_eval.h> // for evaluating densities and pmfs
#include <pf/rv_samp.h> // for sampling random numbers

template<size_t nparts, size_t dimx, size_t dimy, typename resampT, typename float_t>
class svol_sisr : public SISRFilter<nparts, dimx, dimy, resampT,float_t>
{
public:
    using ssv = Eigen::Matrix<float_t, dimx, 1>;
    using osv = Eigen::Matrix<float_t, dimy, 1>;

    // parameters
    float_t m_phi;
    float_t m_beta;
    float_t m_sigma;

    // use this for samplign
    rvsamp::UnivNormSampler<float_t> m_stdNormSampler; // for sampling 

    // ctor
    svol_sisr(const float_t &phi, const float_t &beta, const float_t &sigma);

    // functions tha twe need to define
    float_t logMuEv (const ssv &x1);
    ssv q1Samp (const osv &y1);    
    float_t logQ1Ev (const ssv &x1, const osv &y1 );
    float_t logGEv (const osv &yt, const ssv &xt );
    float_t logFEv (const ssv &xt, const ssv &xtm1 );
    ssv qSamp (const ssv &xtm1, const osv &yt );
    float_t logQEv (const ssv &xt, const ssv &xtm1, const osv &yt );    
};


template<size_t nparts, size_t dimx, size_t dimy, typename resampT, typename float_t>
svol_sisr<nparts,dimx,dimy,resampT,float_t>::svol_sisr(const float_t &phi, const float_t &beta, const float_t &sigma)
    : SISRFilter<nparts, dimx, dimy, resampT,float_t>()
    , m_phi(phi), m_beta(beta), m_sigma(sigma) 
{
}


template<size_t nparts, size_t dimx, size_t dimy, typename resampT, typename float_t>
float_t svol_sisr<nparts,dimx,dimy,resampT,float_t>::logMuEv (const ssv &x1 )
{
    return rveval::evalUnivNorm<float_t>(x1(0), 0.0, m_sigma/std::sqrt(1.0 - m_phi*m_phi), true);
}


template<size_t nparts, size_t dimx, size_t dimy, typename resampT, typename float_t>
auto svol_sisr<nparts,dimx,dimy,resampT,float_t>::q1Samp(const osv &y1) -> ssv
{
    ssv x1samp;
    x1samp(0) = m_stdNormSampler.sample() * m_sigma / std::sqrt(1.-m_phi*m_phi);
    return x1samp;
}

template<size_t nparts, size_t dimx, size_t dimy, typename resampT, typename float_t>
float_t svol_sisr<nparts,dimx,dimy,resampT,float_t>::logQ1Ev (const ssv &x1, const osv &y1)
{
    return rveval::evalUnivNorm<float_t>(x1(0), 0.0, m_sigma/std::sqrt(1.0 - m_phi*m_phi), true);
}


template<size_t nparts, size_t dimx, size_t dimy, typename resampT, typename float_t>
float_t svol_sisr<nparts,dimx,dimy,resampT,float_t>::logGEv (const osv &yt, const ssv &xt)
{
    return rveval::evalUnivNorm<float_t>(yt(0), 0.0, m_beta * std::exp(.5*xt(0)), true);
}


template<size_t nparts, size_t dimx, size_t dimy, typename resampT, typename float_t>
float_t svol_sisr<nparts,dimx,dimy,resampT,float_t>::logFEv(const ssv &xt, const ssv &xtm1)
{
    return rveval::evalUnivNorm<float_t>(xt(0), m_phi * xtm1(0), m_sigma, true);
}


template<size_t nparts, size_t dimx, size_t dimy, typename resampT, typename float_t>
auto svol_sisr<nparts,dimx,dimy,resampT,float_t>::qSamp(const ssv &xtm1, const osv &yt) -> ssv
{
    ssv xtsamp;
    xtsamp(0) = m_phi * xtm1(0) + m_stdNormSampler.sample()*m_sigma;
    return xtsamp;
}


template<size_t nparts, size_t dimx, size_t dimy, typename resampT, typename float_t>
float_t svol_sisr<nparts,dimx,dimy,resampT,float_t>::logQEv(const ssv &xt, const ssv &xtm1, const osv &yt)
{
    return rveval::evalUnivNorm<float_t>(xt(0), m_phi * xtm1(0), m_sigma, true);
}





#endif //SVOL_SISR_H
