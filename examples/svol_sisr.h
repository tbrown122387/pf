#ifndef SVOL_SISR_H
#define SVOL_SISR_H

#include "sisr_filter.h" // inherit the right particle filter
#include "rv_eval.h" // for evaluating densities and pmfs
#include "rv_samp.h" // for sampling random numbers

template<size_t nparts, size_t dimx, size_t dimy, typename resampT>
class svol_sisr : public SISRFilter<nparts, dimx, dimy, resampT>
{
public:
    using ssv = Eigen::Matrix<double, dimx, 1>;
    using osv = Eigen::Matrix<double, dimy, 1>;

    // parameters
    double m_phi;
    double m_beta;
    double m_sigma;

    // use this for samplign
    rvsamp::UnivNormSampler m_stdNormSampler; // for sampling 

    // ctor
    svol_sisr(const double &phi, const double &beta, const double &sigma);

    // functions tha twe need to define
    double logMuEv (const ssv &x1);
    ssv q1Samp (const osv &y1);    
    double logQ1Ev (const ssv &x1, const osv &y1 );
    double logGEv (const osv &yt, const ssv &xt );
    double logFEv (const ssv &xt, const ssv &xtm1 );
    ssv qSamp (const ssv &xtm1, const osv &yt );
    double logQEv (const ssv &xt, const ssv &xtm1, const osv &yt );    
};


template<size_t nparts, size_t dimx, size_t dimy, typename resampT>
svol_sisr<nparts,dimx,dimy,resampT>::svol_sisr(const double &phi, const double &beta, const double &sigma)
    : SISRFilter<nparts, dimx, dimy, resampT>()
    , m_phi(phi), m_beta(beta), m_sigma(sigma) 
{
}


template<size_t nparts, size_t dimx, size_t dimy, typename resampT>
double svol_sisr<nparts,dimx,dimy,resampT>::logMuEv (const ssv &x1 )
{
    return rveval::evalUnivNorm(x1(0), 0.0, m_sigma/std::sqrt(1.0 - m_phi*m_phi), true);
}


template<size_t nparts, size_t dimx, size_t dimy, typename resampT>
auto svol_sisr<nparts,dimx,dimy,resampT>::q1Samp(const osv &y1) -> ssv
{
    ssv x1samp;
    x1samp(0) = m_stdNormSampler.sample() * m_sigma / std::sqrt(1.-m_phi*m_phi);
    return x1samp;
}

template<size_t nparts, size_t dimx, size_t dimy, typename resampT>
double svol_sisr<nparts,dimx,dimy,resampT>::logQ1Ev (const ssv &x1, const osv &y1)
{
    return rveval::evalUnivNorm(x1(0), 0.0, m_sigma/std::sqrt(1.0 - m_phi*m_phi), true);
}


template<size_t nparts, size_t dimx, size_t dimy, typename resampT>
double svol_sisr<nparts,dimx,dimy,resampT>::logGEv (const osv &yt, const ssv &xt)
{
    return rveval::evalUnivNorm(yt(0), 0.0, m_beta * std::exp(.5*xt(0)), true);
}


template<size_t nparts, size_t dimx, size_t dimy, typename resampT>
double svol_sisr<nparts,dimx,dimy,resampT>::logFEv(const ssv &xt, const ssv &xtm1)
{
    return rveval::evalUnivNorm(xt(0), m_phi * xtm1(0), m_sigma, true);
}


template<size_t nparts, size_t dimx, size_t dimy, typename resampT>
auto svol_sisr<nparts,dimx,dimy,resampT>::qSamp(const ssv &xtm1, const osv &yt) -> ssv
{
    ssv xtsamp;
    xtsamp(0) = m_phi * xtm1(0) + m_stdNormSampler.sample()*m_sigma;
    return xtsamp;
}


template<size_t nparts, size_t dimx, size_t dimy, typename resampT>
double svol_sisr<nparts,dimx,dimy,resampT>::logQEv(const ssv &xt, const ssv &xtm1, const osv &yt)
{
    return rveval::evalUnivNorm(xt(0), m_phi * xtm1(0), m_sigma, true);
}





#endif //SVOL_SISR_H