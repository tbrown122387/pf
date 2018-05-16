#ifndef SISR_FILTER_H
#define SISR_FILTER_H

#include <array>
#include <Eigen/Dense>


//! A base class for the Sequential Important Sampling with Resampling (SISR).
/**
 * @class SISRFilter
 * @author taylor
 * @file sisr_filter.h
 * @brief SISR filter.
 * @tparam nparts the number of particles
 * @tparam dimx the size of the state
 * @tparam the size of the observation
 * @tparam resampT the type of resampler
 */
template<size_t nparts, size_t dimx, size_t dimy, typename resampT>
class SISRFilter
{
public:

    /** "state size vector" type alias for linear algebra stuff */
    using ssv         = Eigen::Matrix<double, dimx, 1>; 
    /** "obs size vector" type alias for linear algebra stuff */
    using osv         = Eigen::Matrix<double, dimy, 1>; // obs size vec
    /** type alias for linear algebra stuff */
    using Mat         = Eigen::MatrixXd;
    /** type alias for linear algebra stuff */
    using arrayStates = std::array<ssv, nparts>;
    /** type alias for array of doubles */
    using arrayDouble = std::array<double, nparts>;
    

    /**
     * @brief The (one and only) constructor.
     * @param rs the resampling schedule (resample every rs time points). 
     */
    SISRFilter(const unsigned int &rs=1);
    
    
    /**
     * @brief The (virtual) destructor.
     */
    virtual ~SISRFilter();
    
    
    /**
     * @brief Returns the most recent (log-) conditiona likelihood.
     * @return log p(y_t | y_{1:t-1}) or log p(y_1)
     */
    double getLogCondLike() const; 
    
    
    /**
     * @brief return all stored expectations (taken with respect to $p(x_t|y_{1:t})$
     * @return return a std::vector<Mat> of expectations. How many depends on how many callbacks you gave to 
     */
    std::vector<Mat> getExpectations() const;
    
    
    /**
     * @brief updates filtering distribution on a new datapoint. 
     * Optionally stores expectations of functionals.
     * @param data the most recent data point
     * @param fs a vector of functions if you want to calculate expectations.
     */
    void filter(const osv &data, const std::vector<std::function<const Mat(const ssv&)> >& fs = std::vector<std::function<const Mat(const ssv&)> >());
    
    
    /**
     * @brief  Calculate muEv or logmuEv
     * @param x1 is a const Vec& describing the state sample
     * @return the density or log-density evaluation as a double
     */
    virtual double logMuEv (const ssv &x1) = 0;
    
    
    /**
     * @brief Samples from time 1 proposal 
     * @param y1 is a const Vec& representing the first observed datum 
     * @return the sample as a Vec
     */
    virtual ssv q1Samp (const osv &y1) = 0;    
    
    
    /**
     * @brief Calculate q1Ev or log q1Ev
     * @param x1 is a const Vec& describing the time 1 state sample
     * @param y1 is a const Vec& describing the time 1 datum
     * @return the density or log-density evaluation as a double
     */
    virtual double logQ1Ev (const ssv &x1, const osv &y1 ) = 0;
    
    
    /**
     * @brief Calculate gEv or logGEv
     * @param yt is a const Vec& describing the time t datum
     * @param xt is a const Vec& describing the time t state
     * @return the density or log-density evaluation as a double
     */
    virtual double logGEv (const osv &yt, const ssv &xt ) = 0;
    
    
    /**
     * @brief Evaluates the state transition density.
     * @param xt the current state
     * @param xtm1 the previous state
     * @return a double evaluaton of the log density/pmf
     */
    virtual double logFEv (const ssv &xt, const ssv &xtm1 ) = 0;
    
    
    /**
     * @brief Samples from the proposal/instrumental/importance density at time t
     * @param xtm1 the previous state sample
     * @param yt the current observation
     * @return a state sample for the current time xt
     */
    virtual ssv qSamp (const ssv &xtm1, const osv &yt ) = 0;
    
    
    /**
     * @brief Evaluates the proposal/instrumental/importance density/pmf
     * @param xt current state
     * @param xtm1 previous state
     * @param yt current observation
     * @return a double evaluation of the log density/pmf
     */
    virtual double logQEv (const ssv &xt, const ssv &xtm1, const osv &yt ) = 0;    
    
private:

    /** @brief particle samples */
    arrayStates m_particles;

    /** @brief particle weights */
    arrayDouble m_logUnNormWeights;
    
    /** @brief current time point */
    unsigned int m_now; 

    /** @brief log p(y_t|y_{1:t-1}) or log p(y1) */
    double m_logLastCondLike;  
    
    /** @brief resampling object */
    resampT m_resampler;
    
    /** @brief expectations E[h(x_t) | y_{1:t}] for user defined "h"s */
    std::vector<Mat> m_expectations; // stores any sample averages the user wants
    
    /** @brief resampling schedule (e.g. resample every __ time points) */
    unsigned int m_resampSched;
    
    
    /**
     * @todo implement ESS stuff
     */
  
};



template<size_t nparts, size_t dimx, size_t dimy, typename resampT>
SISRFilter<nparts,dimx,dimy,resampT>::SISRFilter(const unsigned int &rs)
                : m_now(0)
                , m_logLastCondLike(0.0)
                , m_resampSched(rs) 
{
    std::fill(m_logUnNormWeights.begin(), m_logUnNormWeights.end(), 0.0); // log(1) = 0
}


template<size_t nparts, size_t dimx, size_t dimy, typename resampT>
SISRFilter<nparts,dimx,dimy,resampT>::~SISRFilter() {}

    
template<size_t nparts, size_t dimx, size_t dimy, typename resampT>
double SISRFilter<nparts,dimx,dimy,resampT>::getLogCondLike() const
{
    return m_logLastCondLike;
}
    

template<size_t nparts, size_t dimx, size_t dimy, typename resampT>    
auto SISRFilter<nparts,dimx,dimy,resampT>::getExpectations() const -> std::vector<Mat> 
{
    return m_expectations;
}


template<size_t nparts, size_t dimx, size_t dimy, typename resampT>
void SISRFilter<nparts,dimx,dimy,resampT>::filter(const osv &data, const std::vector<std::function<const Mat(const ssv&)> >& fs)
{

    if (m_now == 0) //time 1
    {
       
        // only need to iterate over particles once
        double sumWts(0.0);
        for(size_t ii = 0; ii < nparts; ++ii)
        {
            // sample particles
            m_particles[ii] = q1Samp(data);
            m_logUnNormWeights[ii] = logMuEv(m_particles[ii]);
            m_logUnNormWeights[ii] += logGEv(data, m_particles[ii]);
            m_logUnNormWeights[ii] -= logQ1Ev(m_particles[ii], data);
        }
       
        // calculate log cond likelihood with log-exp-sum trick
        double max = *std::max_element(m_logUnNormWeights.begin(), m_logUnNormWeights.end());
        double sumExp(0.0);
        for(size_t i = 0; i < nparts; ++i){
            sumExp += std::exp(m_logUnNormWeights[i] - max);
        }
        m_logLastCondLike = -std::log(nparts) + max + std::log(sumExp);
   
        // calculate expectations before you resample
        m_expectations.resize(fs.size());
        //std::fill(m_expectations.begin(), m_expectations.end(), ssv::Zero()); 
        unsigned int fId(0);
        for(auto & h : fs){
            
            Mat testOut = h(m_particles[0]);
            unsigned int rows = testOut.rows();
            unsigned int cols = testOut.cols();
            Mat numer = Mat::Zero(rows,cols);
            double denom(0.0);

            for(size_t prtcl = 0; prtcl < nparts; ++prtcl){ // iterate over all particles
                numer += h(m_particles[prtcl]) * std::exp(m_logUnNormWeights[prtcl]);
                denom += std::exp(m_logUnNormWeights[prtcl]);
            }
            m_expectations[fId] = numer/denom;
            fId++;
        }
   
        // resample if you should
        if( (m_now + 1) % m_resampSched == 0)
            m_resampler.resampLogWts(m_particles, m_logUnNormWeights);
   
        // advance time step
        m_now += 1;   
    }
    else // m_now > 0
    {

        // try to iterate over particles all at once
        ssv newSamp;
        arrayDouble oldLogUnNormWts = m_logUnNormWeights;
        double maxOldLogUnNormWts(-1.0/0.0);
        for(size_t ii = 0; ii < nparts; ++ii)
        {

            // update max of old logUnNormWts before you change the element
            if (m_logUnNormWeights[ii] > maxOldLogUnNormWts)
                maxOldLogUnNormWts = m_logUnNormWeights[ii];
            
            // sample and get weight adjustments
            newSamp = qSamp(m_particles[ii], data);
            m_logUnNormWeights[ii]  = logFEv(newSamp, m_particles[ii]);
            m_logUnNormWeights[ii] += logGEv(data, newSamp);
            m_logUnNormWeights[ii] -= logQEv(newSamp, m_particles[ii], data);
 
            // overwrite stuff
            m_particles[ii] = newSamp;
        }
       
        // compute estimate of log p(y_t|y_{1:t-1}) with log-exp-sum trick
        double maxNumer = *std::max_element(m_logUnNormWeights.begin(), m_logUnNormWeights.end()); //because you added log adjustments
        double sumExp1(0.0);
        double sumExp2(0.0);
        for(size_t i = 0; i < nparts; ++i){
            sumExp1 += std::exp(m_logUnNormWeights[i] - maxNumer);
            sumExp2 += std::exp(oldLogUnNormWts[i] - maxOldLogUnNormWts);
        }
        m_logLastCondLike = maxNumer + std::log(sumExp1) - maxOldLogUnNormWts - std::log(sumExp2);

        // calculate expectations before you resample
        //m_expectations.resize(fs.size());
        //std::fill(m_expectations.begin(), m_expectations.end(), ssv::Zero()); // TODO: should this be Mat::Zero(m_dimState, m_dimState)?
        unsigned int fId(0);
        double weightNormConst(0.0);
        for(auto & h : fs){ // iterate over all functions

            Mat testOut = h(m_particles[0]);
            unsigned int rows = testOut.rows();
            unsigned int cols = testOut.cols();
            Mat numer = Mat::Zero(rows,cols);
            double denom(0.0);

            for(size_t prtcl = 0; prtcl < nparts; ++prtcl){ // iterate over all particles
                numer += h(m_particles[prtcl]) * std::exp(m_logUnNormWeights[prtcl]);
                denom += std::exp(m_logUnNormWeights[prtcl]);
            }
            m_expectations[fId] = numer/denom;
            fId++;
        }
 
        // resample if you should
        if( (m_now + 1) % m_resampSched == 0)
            m_resampler.resampLogWts(m_particles, m_logUnNormWeights);

        // advance time
        m_now += 1;       
    }
}








#endif //SISR_FILTER_H