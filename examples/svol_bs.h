#ifndef SVOL_BS_H
#define SVOL_BS_H


#include <Eigen/Dense>

#include <pf/bootstrap_filter.h> // the boostrap particle filter
#include <pf/rv_samp.h> // for sampling random numbers
#include <pf/rv_eval.h> // for evaluating densities and pmfs


template<size_t nparts, size_t dimx, size_t dimy, typename resampT, typename float_t>
class svol_bs : public BSFilter<nparts, dimx, dimy, resampT, float_t>
              , public ForwardMod<dimx,dimy,float_t>
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
    svol_bs(const float_t &phi, const float_t &beta, const float_t &sigma);
    
    // required by bootstrap filter base class
    float_t logQ1Ev(const ssv &x1, const osv &y1);
    float_t logMuEv(const ssv &x1);
    float_t logGEv(const osv &yt, const ssv &xt);
    auto fSamp(const ssv &xtm1) -> ssv;
    auto q1Samp(const osv &y1) -> ssv;

    // required by ForwardMod<> base class
    auto muSamp() -> ssv;
    auto gSamp(const ssv &xt) -> osv;
    
};


template<size_t nparts, size_t dimx, size_t dimy, typename resampT, typename float_t>
svol_bs<nparts, dimx, dimy, resampT, float_t>::svol_bs(const float_t &phi, const float_t &beta, const float_t &sigma) 
    : m_phi(phi), m_beta(beta), m_sigma(sigma)
{
}


template<size_t nparts, size_t dimx, size_t dimy, typename resampT, typename float_t>
auto svol_bs<nparts, dimx, dimy, resampT, float_t>::q1Samp(const osv &y1) -> ssv
{
    ssv x1samp;
    x1samp(0) = m_stdNormSampler.sample() * m_sigma / std::sqrt(1.-m_phi*m_phi);
    return x1samp;
}


template<size_t nparts, size_t dimx, size_t dimy, typename resampT, typename float_t>
auto svol_bs<nparts, dimx, dimy, resampT, float_t>::fSamp(const ssv &xtm1) -> ssv
{
    ssv xtsamp;
    xtsamp(0) = m_phi * xtm1(0) + m_stdNormSampler.sample() * m_sigma;
    return xtsamp;
}


template<size_t nparts, size_t dimx, size_t dimy, typename resampT, typename float_t>
float_t svol_bs<nparts, dimx, dimy, resampT, float_t>::logGEv(const osv &yt, const ssv &xt)
{
    return rveval::evalUnivNorm<float_t>(
				   yt(0),
                                   0.0,
                                   m_beta * std::exp(.5*xt(0)),
                                   true);
}


template<size_t nparts, size_t dimx, size_t dimy, typename resampT, typename float_t>
auto svol_bs<nparts, dimx, dimy, resampT, float_t>::gSamp(const ssv &xt) -> osv {
    osv yt;
    yt(0) = m_stdNormSampler.sample() * m_beta * std::exp(.5*xt(0));
    return yt;
}


template<size_t nparts, size_t dimx, size_t dimy, typename resampT, typename float_t>
float_t svol_bs<nparts, dimx, dimy, resampT, float_t>::logMuEv(const ssv &x1)
{
    return rveval::evalUnivNorm<float_t>(x1(0),
                                   0.0,
                                   m_sigma/std::sqrt(1.0 - m_phi*m_phi),
                                   true);
}


template<size_t nparts, size_t dimx, size_t dimy, typename resampT, typename float_t>
auto svol_bs<nparts, dimx, dimy, resampT, float_t>::muSamp() -> ssv {

    ssv x1; 
    x1(0) = m_stdNormSampler.sample() * m_sigma/std::sqrt(1.0 - m_phi*m_phi);
    return x1;
}


template<size_t nparts, size_t dimx, size_t dimy, typename resampT, typename float_t>
float_t svol_bs<nparts, dimx, dimy, resampT, float_t>::logQ1Ev(const ssv &x1samp, const osv &y1)
{
    return rveval::evalUnivNorm<float_t>(x1samp(0), 0.0, m_sigma/std::sqrt(1.0 - m_phi*m_phi), true);
}


#endif //SVOL_BS_H
