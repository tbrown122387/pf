#ifndef BOOTSTRAP_FILTER_H
#define BOOTSTRAP_FILTER_H

#include <array>
#include <Eigen/Dense>

#include "pf_base.h"
    

//! A base class for the boostrap particle filter.
/**
 * @class BSFilter
 * @author taylor
 * @file bootstrap_filter.h
 * @brief bootstrap particle filter
 * @tparam nparts the number of particles
 * @tparam dimx the dimension of the state
 * @tparam dimy the dimension of the observations
 * @tparam resampT the type of resampler
 */
template<size_t nparts, size_t dimx, size_t dimy, typename resampT>
class BSFilter : public pf_base
{
public:
    
    /** "state size vector" type alias for linear algebra stuff */
    using ssv         = Eigen::Matrix<double, dimx, 1>; 
    /** "obs size vector" type alias for linear algebra stuff */
    using osv         = Eigen::Matrix<double, dimy, 1>; // obs size vec
    /** type alias for dynamically sized matrix */
    using Mat         = Eigen::MatrixXd;
    /** type alias for linear algebra stuff */
    using arrayStates = std::array<ssv, nparts>;
    /** type alias for array of doubles */
    using arrayDouble = std::array<double, nparts>;


    /**
     * @brief The constructor
     * @param rs the resampling schedule (e.g. every rs time point) 
     */
    BSFilter(const unsigned int &rs = 1);
    
    
    /**
     * @brief The (virtual) destructor
     */
    virtual ~BSFilter();
    
    
    /**
     * @brief Returns the most recent (log-) conditiona likelihood.
     * @return log p(y_t | y_{1:t-1})
     */
    double getLogCondLike() const; 
    
    
    /**
     * @brief updates filtering distribution on a new datapoint. 
     * Optionally stores expectations of functionals.
     * @param data the most recent data point
     * @param fs a vector of functions if you want to calculate expectations.
     */
    void filter(const osv &data, const std::vector<std::function<const Mat(const ssv&)> >& fs = std::vector<std::function<const Mat(const ssv&)> >()); 


    /**
     * @brief return all stored expectations (taken with respect to $p(x_t|y_{1:t})$
     * @return return a std::vector<Mat> of expectations. How many depends on how many callbacks you gave to 
     */
    auto getExpectations () const -> std::vector<Mat>;
    
    
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
    
    
    //!
    /**
     * @brief Sample from the state transition distribution
     * @param xtm1 is a const Vec& describing the time t-1 state
     * @return the sample as a Vec
     */
    virtual ssv fSamp (const ssv &xtm1) = 0;
    
protected:
    /** @brief particle samples */
    arrayStates      m_particles;
    
    /** @brief particle unnormalized weights */
    arrayDouble      m_logUnNormWeights;
    
    /** @brief time point */
    unsigned int     m_now;         
    
    /** @brief log p(y_t|y_{1:t-1}) or log p(y1)  */
    double           m_logLastCondLike;

    /** @brief resampler object */
    resampT          m_resampler;
    
    /** @brief expectations E[h(x_t) | y_{1:t}] for user defined "h"s */
    std::vector<Mat> m_expectations; 
    
    /** @brief resampling schedule (e.g. resample every __ time points) */
    unsigned int     m_resampSched;
};

    
    
/////////////////////////////////////////////////////////////////////////////////////////////////////////
///////////////////////////////////////// implementations ///////////////////////////////////////////////
/////////////////////////////////////////////////////////////////////////////////////////////////////////

template<size_t nparts, size_t dimx, size_t dimy, typename resampT>
BSFilter<nparts, dimx, dimy, resampT>::BSFilter(const unsigned int &rs)
                : m_now(0)
                , m_logLastCondLike(0.0)
                , m_resampSched(rs)
                  
{
    std::fill(m_logUnNormWeights.begin(), m_logUnNormWeights.end(), 0.0); // log(1) = 0
}


template<size_t nparts, size_t dimx, size_t dimy, typename resampT>
BSFilter<nparts, dimx, dimy, resampT>::~BSFilter() {}


template<size_t nparts, size_t dimx, size_t dimy, typename resampT>
void BSFilter<nparts, dimx, dimy, resampT>::filter(const osv &dat, const std::vector<std::function<const Mat(const ssv&)> >& fs) 
{

    /**
     * @todo: work in support for effective sample size stuff. 
     */
    if (m_now == 0) //time 1
    {  
        // only need to iterate over particles once
        for(size_t ii = 0; ii < nparts; ++ii)
        {
            // sample particles
            m_particles[ii] = q1Samp(dat);
            m_logUnNormWeights[ii] = logMuEv(m_particles[ii]);
            m_logUnNormWeights[ii] += logGEv(dat, m_particles[ii]);
            m_logUnNormWeights[ii] -= logQ1Ev(m_particles[ii], dat);
        }
       
        // calculate log cond likelihood with log-exp-sum trick
        double max = *std::max_element(m_logUnNormWeights.begin(), m_logUnNormWeights.end());
        double sumExp(0.0);
        for(size_t i = 0; i < nparts; ++i){
            sumExp += std::exp(m_logUnNormWeights[i] - max);
        }
        m_logLastCondLike = -std::log(nparts) + (max) + std::log(sumExp);
   
        // calculate expectations before you resample
        // paying mind to underflow
        m_expectations.resize(fs.size());
        unsigned int fId(0);
        for(auto & h : fs){

            Mat testOutput = h(m_particles[0]);
            unsigned int rows = testOutput.rows();
            unsigned int cols = testOutput.cols();
            Mat numer = Mat::Zero(rows,cols);
            double weightNormConst (0.0);
            for(size_t prtcl = 0; prtcl < nparts; ++prtcl){ // iterate over all particles
                numer += h(m_particles[prtcl]) * std::exp( m_logUnNormWeights[prtcl] - (max) );
                weightNormConst += std::exp( m_logUnNormWeights[prtcl] - (max) );
            }
            m_expectations[fId] = numer/weightNormConst;
            fId++;
        }
   
        // resample if you should
        if ( (m_now+1) % m_resampSched == 0){
            m_resampler.resampLogWts(m_particles, m_logUnNormWeights);
        }
        
        // advance time step
        m_now += 1;   
    }
    else // m_now > 0
    {
       
        // try to iterate over particles all at once
        ssv newSamp;
        double maxOldLogUnNormWts(-1.0/0.0);
        arrayDouble oldLogUnNormWts = m_logUnNormWeights;
        for(size_t ii = 0; ii < nparts; ++ii)
        {
            // update max of old logUnNormWts
            if (m_logUnNormWeights[ii] > maxOldLogUnNormWts)
                maxOldLogUnNormWts = m_logUnNormWeights[ii];
            
            // sample and get weight adjustments
            newSamp = fSamp(m_particles[ii]);
            m_logUnNormWeights[ii] = logGEv(dat, newSamp);
 
            // overwrite stuff
            m_particles[ii] = newSamp;
        }
        
        // compute estimate of log p(y_t|y_{1:t-1}) with log-exp-sum trick
        double maxNumer = *std::max_element(m_logUnNormWeights.begin(), m_logUnNormWeights.end()); //because you added log adjustments
        double sumExp1(0.0);
        double sumExp2(0.0);
        for(size_t i = 0; i < nparts; ++i){
            sumExp1 += std::exp(m_logUnNormWeights[i] - maxNumer);
            sumExp2 += std::exp(oldLogUnNormWts[i] - maxOldLogUnNormWts);  //1
        }
        m_logLastCondLike = maxNumer + std::log(sumExp1) - maxOldLogUnNormWts - std::log(sumExp2);

        // calculate expectations before you resample
//        m_expectations.resize(fs.size());
        int fId(0);
        for(auto & h : fs){ // iterate over all functions
        
            Mat testOutput = h(m_particles[0]);
            unsigned int rows = testOutput.rows();
            unsigned int cols = testOutput.cols();
            Eigen::MatrixXd numer = Eigen::MatrixXd::Zero(rows,cols);
            double weightNormConst (0.0);
            for(size_t prtcl = 0; prtcl < nparts; ++prtcl){ // iterate over all particles
                numer += h(m_particles[prtcl]) * std::exp(m_logUnNormWeights[prtcl] - maxNumer);
                weightNormConst += std::exp(m_logUnNormWeights[prtcl] - maxNumer);
            }
            m_expectations[fId] = numer/weightNormConst;
            fId++;
        }

        // resample if you should
        if ( (m_now+1) % m_resampSched == 0)
            m_resampler.resampLogWts(m_particles, m_logUnNormWeights);

        // advance time
        m_now += 1;       
    }
}


template<size_t nparts, size_t dimx, size_t dimy, typename resampT>
double BSFilter<nparts, dimx, dimy, resampT>::getLogCondLike() const
{
    return m_logLastCondLike;
}


template<size_t nparts, size_t dimx, size_t dimy, typename resampT>
auto BSFilter<nparts, dimx, dimy, resampT>::getExpectations() const -> std::vector<Mat>
{
    return m_expectations;
}



#endif // BOOTSTRAP_FILTER_H
