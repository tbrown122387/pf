#ifndef SVOL_SISR_HILB_H
#define SVOL_SISR_HILB_H

#include <pf/sisr_filter.h> // inherit the right particle filter
#include <pf/rv_eval.h> // for evaluating densities and pmfs
#include <pf/rv_samp.h> // for sampling random numbers


using namespace pf;
using namespace pf::filters;


template<size_t nparts, size_t num_bits, typename resamp_t, typename float_t>
class svol_sisr_hilb : public SISRFilterCRN<nparts, 1, 1, 1, 1, resamp_t, float_t>
{
public:
    using ssv = Eigen::Matrix<float_t, 1, 1>;
    using osv = Eigen::Matrix<float_t, 1, 1>;
    using usv = Eigen::Matrix<float_t, 1, 1>;

    // parameters
    float_t m_phi;
    float_t m_beta;
    float_t m_sigma;

    // ctor
    svol_sisr_hilb(const float_t &phi, const float_t &beta, const float_t &sigma);

    // functions tha twe need to define
    float_t logMuEv (const ssv &x1);
    ssv Xi1(const usv& U, const osv &y1);   // changed 
    float_t logQ1Ev (const ssv &x1, const osv &y1 );
    float_t logGEv (const osv &yt, const ssv &xt );
    float_t logFEv (const ssv &xt, const ssv &xtm1 );
    ssv Xit(const ssv &xtm1, const usv& U, const osv &yt ); // changed;
    float_t logQEv (const ssv &xt, const ssv &xtm1, const osv &yt );    
};


template<size_t nparts, size_t num_bits, typename resamp_t, typename float_t>
svol_sisr_hilb<nparts,num_bits,resamp_t,float_t>::svol_sisr_hilb(const float_t &phi, const float_t &beta, const float_t &sigma)
    : SISRFilterCRN<nparts, 1, 1, 1, 1, resamp_t, float_t>()
    , m_phi(phi), m_beta(beta), m_sigma(sigma) 
{
}


template<size_t nparts, size_t num_bits, typename resamp_t, typename float_t>
float_t svol_sisr_hilb<nparts,num_bits,resamp_t,float_t>::logMuEv (const ssv &x1 )
{
    return rveval::evalUnivNorm<float_t>(x1(0), 0.0, m_sigma/std::sqrt(1.0 - m_phi*m_phi), true);
}


template<size_t nparts, size_t num_bits, typename resamp_t, typename float_t>
auto svol_sisr_hilb<nparts,num_bits,resamp_t,float_t>::Xi1(const usv& U, const osv &y1) -> ssv
{
    return U * m_sigma / std::sqrt(1.-m_phi*m_phi);
}


template<size_t nparts, size_t num_bits, typename resamp_t, typename float_t>
float_t svol_sisr_hilb<nparts,num_bits,resamp_t,float_t>::logQ1Ev (const ssv &x1, const osv &y1)
{
    return rveval::evalUnivNorm<float_t>(x1(0), 0.0, m_sigma/std::sqrt(1.0 - m_phi*m_phi), true);
}


template<size_t nparts, size_t num_bits, typename resamp_t, typename float_t>
float_t svol_sisr_hilb<nparts,num_bits,resamp_t,float_t>::logGEv (const osv &yt, const ssv &xt)
{
    return rveval::evalUnivNorm<float_t>(yt(0), 0.0, m_beta * std::exp(.5*xt(0)), true);
}


template<size_t nparts, size_t num_bits, typename resamp_t, typename float_t>
float_t svol_sisr_hilb<nparts,num_bits,resamp_t,float_t>::logFEv(const ssv &xt, const ssv &xtm1)
{
    return rveval::evalUnivNorm<float_t>(xt(0), m_phi * xtm1(0), m_sigma, true);
}


template<size_t nparts, size_t num_bits, typename resamp_t, typename float_t>
auto svol_sisr_hilb<nparts,num_bits,resamp_t,float_t>::Xit(const ssv &xtm1, const usv& U, const osv &yt) -> ssv
{
    return m_phi * xtm1 + U*m_sigma;
}


template<size_t nparts, size_t num_bits, typename resamp_t, typename float_t>
float_t svol_sisr_hilb<nparts,num_bits,resamp_t,float_t>::logQEv(const ssv &xt, const ssv &xtm1, const osv &yt)
{
    return rveval::evalUnivNorm<float_t>(xt(0), m_phi * xtm1(0), m_sigma, true);
}





#endif //SVOL_SISR_HILB_H
