#ifndef SVOL_APF_H
#define SVOL_APF_H

#include <Eigen/Dense>
#include "auxiliary_pf.h" // the auxiliary parrticle filter
#include "rv_eval.h" // for evaluating densities and pmfs
#include "rv_samp.h" // for sampling random numbers

template<size_t nparts, size_t dimx, size_t dimy, typename resampT>
class svol_apf : public APF<nparts,dimx,dimy,resampT>
{
public:
    using ssv = Eigen::Matrix<double, dimx, 1>;
    using osv = Eigen::Matrix<double, dimy, 1>;
    
    // parameters
    double m_phi;
    double m_beta;
    double m_sigma;
  
    // use this for samplign
    UnivNormSampler m_stdNormSampler; // for sampling 
    
    svol_apf(const double &phi, const double &beta, const double &sigma);
    
    // functions that we need to define
    double logMuEv (const ssv &x1 );
    ssv propMu (const ssv &xtm1 );
    ssv q1Samp (const osv &y1);
    ssv fSamp (const ssv &xtm1);
    double logQ1Ev (const ssv &x1, const osv &y1);
    double logGEv (const osv &yt, const ssv &xt);
  
};


template<size_t nparts, size_t dimx, size_t dimy, typename resampT>
svol_apf<nparts,dimx,dimy,resampT>::svol_apf(const double &phi, const double &beta, const double &sigma)
    : APF<nparts, dimx, dimy, resampT>()
    , m_phi(phi), m_beta(beta), m_sigma(sigma) 
{
}

    
template<size_t nparts, size_t dimx, size_t dimy, typename resampT>
double svol_apf<nparts,dimx,dimy,resampT>::logMuEv (const ssv &x1 )
{
    return rveval::evalUnivNorm(x1(0),
                                   0.0,
                                   m_sigma/std::sqrt(1.0 - m_phi*m_phi),
                                   true);
}


template<size_t nparts, size_t dimx, size_t dimy, typename resampT>
auto svol_apf<nparts,dimx,dimy,resampT>::propMu (const ssv &xtm1 ) -> ssv
{
    return m_phi*xtm1;
}


template<size_t nparts, size_t dimx, size_t dimy, typename resampT>
auto svol_apf<nparts,dimx,dimy,resampT>::q1Samp (const osv &y1) -> ssv
{
    ssv x1samp(1);
    x1samp(0) = m_stdNormSampler.sample() * m_sigma / std::sqrt(1.-m_phi*m_phi);
    return x1samp;
}


template<size_t nparts, size_t dimx, size_t dimy, typename resampT>
auto svol_apf<nparts,dimx,dimy,resampT>::fSamp (const ssv &xtm1) -> ssv
{
    ssv xtsamp(1);
    xtsamp(0) = m_phi * xtm1(0) + m_stdNormSampler.sample() * m_sigma;
    return xtsamp;
}


template<size_t nparts, size_t dimx, size_t dimy, typename resampT>
double svol_apf<nparts,dimx,dimy,resampT>::logQ1Ev (const ssv &x1, const osv &y1)
{
    return rveval::evalUnivNorm(x1(0), 0.0, m_sigma/std::sqrt(1.0 - m_phi*m_phi), true);
}


template<size_t nparts, size_t dimx, size_t dimy, typename resampT>
double svol_apf<nparts,dimx,dimy,resampT>::logGEv (const osv &yt, const ssv &xt)
{
    return rveval::evalUnivNorm(yt(0),
                                   0.0,
                                   m_beta * std::exp(.5*xt(0)),
                                   true);
}

    

#endif // SVOL_APF_H