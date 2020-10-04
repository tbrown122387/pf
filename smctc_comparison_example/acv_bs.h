#ifndef ACV_BS_H
#define ACV_BS_H


#include <Eigen/Dense>

#include <pf/bootstrap_filter.h> // the boostrap particle filter
#include <pf/rv_samp.h> // for sampling random numbers
#include <pf/rv_eval.h> // for evaluating densities and pmfs


template<size_t nparts, size_t dimx, size_t dimy, typename resampT, typename float_t>
class acv_bs : public BSFilter<nparts, dimx, dimy, resampT, float_t>
{
public:
    using ssv = Eigen::Matrix<float_t, dimx, 1>;
    using osv = Eigen::Matrix<float_t, dimy, 1>;
    using ssm = Eigen::Matrix<float_t, dimx, dimx>;
    using osm = Eigen::Matrix<float_t, dimy, dimy>;
    using esm = Eigen::Matrix<float_t, dimy,dimx>; // emission sized matrix

    // parameters that need to be stored 
    float_t m_var_s0;
    float_t m_var_u0;
    float_t m_nu_y;

    // higher-level parameters that need to be stored
    ssm m_A;
    esm m_B;
    osm m_obs_shape_mat;

    // use these objects for sampling 
    rvsamp::UnivNormSampler<float_t> m_stdNormSampler;  
    rvsamp::UnivStudTSampler<float_t> m_t_sampler;
    rvsamp::MVNSampler<dimx,float_t> m_state_error_sampler;

    // the constructor
    acv_bs(float_t var_s0, float_t var_u0, float_t var_s, float_t var_u, float_t scale_y, float_t nu_y, float_t Delta);
    
    // methods that are required by this algorithm's abstract base class (template)
    float_t logQ1Ev(const ssv &x1, const osv &y1);
    float_t logMuEv(const ssv &x1);
    float_t logGEv(const osv &yt, const ssv &xt);
    auto fSamp(const ssv &xtm1) -> ssv;
    auto q1Samp(const osv &y1) -> ssv;
};


template<size_t nparts, size_t dimx, size_t dimy, typename resampT, typename float_t>
acv_bs<nparts, dimx, dimy, resampT, float_t>::acv_bs( float_t var_s0, float_t var_u0, float_t var_s, float_t var_u, float_t scale_y, float_t nu_y, float_t Delta)
    : m_var_s0(var_s0), m_var_u0(var_u0), m_nu_y(nu_y)
    , m_A(ssm::Identity()), m_B(esm::Zero()), m_obs_shape_mat(scale_y * osm::Identity())
    , m_t_sampler(nu_y), m_state_error_sampler(ssv::Zero(), ssm::Identity()) 
{
    ssm state_error_cov(ssm::Zero()); 
    state_error_cov(0,0) = var_s;
    state_error_cov(1,1) = var_u;
    state_error_cov(2,2) = var_s;
    state_error_cov(3,3) = var_u;
    m_state_error_sampler.setCovar(state_error_cov);

    m_A(0,1) = Delta;
    m_A(2,3) = Delta;

    m_B(0,0) = 1.0;
    m_B(1,2) = 1.0;
}


template<size_t nparts, size_t dimx, size_t dimy, typename resampT, typename float_t>
auto acv_bs<nparts, dimx, dimy, resampT, float_t>::q1Samp(const osv &y1) -> ssv
{
    ssv x1samp;
    x1samp(0) = m_stdNormSampler.sample() * std::sqrt(m_var_s0);
    x1samp(1) = m_stdNormSampler.sample() * std::sqrt(m_var_u0);
    x1samp(2) = m_stdNormSampler.sample() * std::sqrt(m_var_s0);
    x1samp(3) = m_stdNormSampler.sample() * std::sqrt(m_var_u0);
    return x1samp;
}


template<size_t nparts, size_t dimx, size_t dimy, typename resampT, typename float_t>
auto acv_bs<nparts, dimx, dimy, resampT, float_t>::fSamp(const ssv &xtm1) -> ssv
{
    return m_A * xtm1 + m_state_error_sampler.sample(); 
}


template<size_t nparts, size_t dimx, size_t dimy, typename resampT, typename float_t>
float_t acv_bs<nparts, dimx, dimy, resampT, float_t>::logGEv(const osv &yt, const ssv &xt)
{
    return rveval::evalMultivT<dimy,float_t>(yt, m_B * xt, m_obs_shape_mat, m_nu_y, true);
}


template<size_t nparts, size_t dimx, size_t dimy, typename resampT, typename float_t>
float_t acv_bs<nparts, dimx, dimy, resampT, float_t>::logMuEv(const ssv &x1)
{
    return rveval::evalUnivNorm<float_t>(x1(0),0.0,std::sqrt(m_var_s0), true) + 
           rveval::evalUnivNorm<float_t>(x1(1),0.0,std::sqrt(m_var_u0), true) + 
           rveval::evalUnivNorm<float_t>(x1(2),0.0,std::sqrt(m_var_s0), true) + 
           rveval::evalUnivNorm<float_t>(x1(3),0.0,std::sqrt(m_var_u0), true); 
}


template<size_t nparts, size_t dimx, size_t dimy, typename resampT, typename float_t>
float_t acv_bs<nparts, dimx, dimy, resampT, float_t>::logQ1Ev(const ssv &x1samp, const osv &y1)
{
    return rveval::evalUnivNorm<float_t>(x1samp(0),0.0,std::sqrt(m_var_s0), true) + 
           rveval::evalUnivNorm<float_t>(x1samp(1),0.0,std::sqrt(m_var_u0), true) + 
           rveval::evalUnivNorm<float_t>(x1samp(2),0.0,std::sqrt(m_var_s0), true) + 
           rveval::evalUnivNorm<float_t>(x1samp(3),0.0,std::sqrt(m_var_u0), true); 
}


#endif //ACV_BS_H
