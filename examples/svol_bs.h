#ifndef SVOL_BS_H
#define SVOL_BS_H


#include <Eigen/Dense>

#include "bootstrap_filter.h" // the boostrap particle filter
#include "rv_samp.h" // for sampling random numbers
#include "rv_eval.h" // for evaluating densities and pmfs

template<size_t nparts, size_t dimx, size_t dimy, typename resampT>
class svol_bs : public BSFilter<nparts, dimx, dimy, resampT>
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

    // ctor
    svol_bs(const double &phi, const double &beta, const double &sigma);
    
    // required functions defined by the state space model
    double logQ1Ev(const ssv &x1, const osv &y1);
    double logMuEv(const ssv &x1);
    double logGEv(const osv &yt, const ssv &xt);
    auto fSamp(const ssv &xtm1) -> ssv;
    auto q1Samp(const osv &y1) -> ssv;
    
};


template<size_t nparts, size_t dimx, size_t dimy, typename resampT>
svol_bs<nparts, dimx, dimy, resampT>::svol_bs(const double &phi, const double &beta, const double &sigma) 
    : BSFilter<nparts, dimx, dimy, resampT>()
    , m_phi(phi), m_beta(beta), m_sigma(sigma)
{
}


template<size_t nparts, size_t dimx, size_t dimy, typename resampT>
auto svol_bs<nparts, dimx, dimy, resampT>::q1Samp(const osv &y1) -> ssv
{
    ssv x1samp(1);
    x1samp(0) = m_stdNormSampler.sample() * m_sigma / std::sqrt(1.-m_phi*m_phi);
    return x1samp;
}


template<size_t nparts, size_t dimx, size_t dimy, typename resampT>
auto svol_bs<nparts, dimx, dimy, resampT>::fSamp(const ssv &xtm1) -> ssv
{
    ssv xtsamp(1);
    xtsamp(0) = m_phi * xtm1(0) + m_stdNormSampler.sample() * m_sigma;
    return xtsamp;
}


template<size_t nparts, size_t dimx, size_t dimy, typename resampT>
double svol_bs<nparts, dimx, dimy, resampT>::logGEv(const osv &yt, const ssv &xt)
{
    return rveval::evalUnivNorm(yt(0),
                                   0.0,
                                   m_beta * std::exp(.5*xt(0)),
                                   true);
}


template<size_t nparts, size_t dimx, size_t dimy, typename resampT>
double svol_bs<nparts, dimx, dimy, resampT>::logMuEv(const ssv &x1)
{
    return rveval::evalUnivNorm(x1(0),
                                   0.0,
                                   m_sigma/std::sqrt(1.0 - m_phi*m_phi),
                                   true);
}


template<size_t nparts, size_t dimx, size_t dimy, typename resampT>
double svol_bs<nparts, dimx, dimy, resampT>::logQ1Ev(const ssv &x1samp, const osv &y1)
{
    return rveval::evalUnivNorm(x1samp(0), 0.0, m_sigma/std::sqrt(1.0 - m_phi*m_phi), true);
}


#endif //SVOL_BS_H