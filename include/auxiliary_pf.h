#ifndef AUXILIARY_PF_H
#define AUXILIARY_PF_H
 
#include <array> //array
#include <vector> // vector
#include <functional> // function
#include <Eigen/Dense>
#include <cmath>

#include "pf_base.h"
#include "rv_samp.h" // for k_generator


//! A base-class for Auxiliary Particle Filtering. Filtering only, no smoothing.
 /**
  * @class APF
  * @author taylor
  * @file auxiliary_pf.h
  * @brief A base class for Auxiliary Particle Filtering.
  * Inherit from this if you want to use an APF for your state space model. 
  * Filtering only, no smoothing. 
  * @tparam nparts the number of particles
  * @tparam dimx the dimension of the state
  * @tparam dimy the dimension of the observations
  * @tparam resampT the resampler type
  */
template<size_t nparts, size_t dimx, size_t dimy, typename resampT>
class APF : public pf_base
{
public:
    /** "state size vector" type alias for linear algebra stuff */
    using ssv = Eigen::Matrix<double,dimx,1>;
    /** "observation size vector" type alias for linear algebra stuff */
    using osv = Eigen::Matrix<double,dimy,1>;
    /** type alias for linear algebra stuff (dimension of the state ^2) */
    using Mat = Eigen::MatrixXd;
    /** type alias for array of doubles */
    using arrayDouble = std::array<double, nparts>;
    /** type alias for array of state vectors */
    using arrayVec = std::array<ssv, nparts>;
    /** type alias for array of unsigned ints */
    using arrayUInt = std::array<unsigned int, nparts>;
    
public:

     /**
      * @brief The constructor.
      * @param rs resampling schedule (e.g. resample every rs time points).
      */
    APF(const unsigned int &rs=1);
    
    
    /**
     * @brief The (virtual) destructor
     */
    virtual ~APF();
    
     /**
      * @brief Get the latest log conditional likelihood.
      * @return a double of the most recent conditional likelihood.
      */
    double getLogCondLike () const; 
    
    
    /**
     * @brief return all stored expectations (taken with respect to $p(x_t|y_{1:t})$
     * @return return a std::vector<Mat> of expectations. How many depends on how many callbacks you gave to 
     */
    std::vector<Mat> getExpectations () const;
    

     /**
      * @brief Use a new datapoint to update the filtering distribution (or smoothing if pathLength > 0).
      * @param data a Eigen::Matrix<double,dimy,1> representing the data
      * @param fs a std::vector of callback functions that are used to calculate expectations with respect to the filtering distribution.
      */
    void filter(const osv &data, const std::vector<std::function<const Mat(const ssv&)> >& fs = std::vector<std::function<const Mat(const ssv&)> >());
    

    /**
     * @brief Evaluates the log of mu.
     * @param x1 a Eigen::Matrix<double,dimx,1> representing time 1's state.
     * @return a double evaluation.
     */
    virtual double logMuEv (const ssv &x1 ) = 0;
    
    
    /**
     * @brief Evaluates the proposal distribution taking a Eigen::Matrix<double,dimx,1> from the previous time's state, and returning a state for the current time.
     * @param xtm1 a Eigen::Matrix<double,dimx,1> representing the previous time's state.
     * @return a Eigen::Matrix<double,dimx,1> representing a likely current time state, to be used by the observation density.
     */
    virtual ssv propMu (const ssv &xtm1 ) = 0;
    
    
    /**
     * @brief Samples from q1.
     * @param y1 a Eigen::Matrix<double,dimy,1> representing time 1's data point.
     * @return a Eigen::Matrix<double,dimx,1> sample for time 1's state.
     */
    virtual ssv q1Samp (const osv &y1) = 0;
    
    
    /**
     * @brief Samples from f.
     * @param xtm1 a Eigen::Matrix<double,dimx,1> representing the previous time's state.
     * @return a Eigen::Matrix<double,dimx,1> state sample for the current time.
     */
    virtual ssv fSamp (const ssv &xtm1) = 0;
    
    
    /**
     * @brief Evaluates the log of q1.
     * @param x1 a Eigen::Matrix<double,dimx,1> representing time 1's state.
     * @param y1 a Eigen::Matrix<double,dimy,1> representing time 1's data observation.
     * @return a double evaluation.
     */
    virtual double logQ1Ev (const ssv &x1, const osv &y1) = 0;
    
    
    /**
     * @brief Evaluates the log of g.
     * @param yt a Eigen::Matrix<double,dimy,1> representing time t's data observation.
     * @param xt a Eigen::Matrix<double,dimx,1> representing time t's state.
     * @return a double evaluation.
     */
    virtual double logGEv (const osv &yt, const ssv &xt) = 0;


protected:
    /** @brief particle samples */
    std::array<ssv,nparts>  m_particles;
    
    /** @brief particle unnormalized weights */
    std::array<double,nparts> m_logUnNormWeights;
    
    /** @brief curren time */
    unsigned int m_now; 
    
    /** @brief log p(y_t|y_{1:t-1}) or log p(y1) */
    double m_logLastCondLike; 
    
    /** @brief the resampling schedule */
    unsigned int m_rs;
    
    /** @brief resampler object (default ctor'd)*/
    resampT m_resampler;
    
    /** @brief k generator object (default ctor'd)*/
    rvsamp::k_gen<nparts> m_kGen;
    
    /** @brief expectations E[h(x_t) | y_{1:t}] for user defined "h"s */
    std::vector<Mat> m_expectations;
    
};



template<size_t nparts, size_t dimx, size_t dimy, typename resampT>
APF<nparts, dimx, dimy, resampT>::APF(const unsigned int &rs) 
    : m_now(0)
    , m_logLastCondLike(0.0)
    , m_rs(rs)
{
    std::fill(m_logUnNormWeights.begin(), m_logUnNormWeights.end(), 0.0);
}


template<size_t nparts, size_t dimx, size_t dimy, typename resampT>
APF<nparts, dimx, dimy, resampT>::~APF() { }


template<size_t nparts, size_t dimx, size_t dimy, typename resampT>
void APF<nparts, dimx, dimy, resampT>::filter(const osv &data, const std::vector<std::function<const Mat(const ssv&)> >& fs)
{
    
    if(m_now == 0) 
    {
        double max(-1.0/0.0);
        for(size_t ii = 0; ii < nparts; ++ii)
        {
            // sample particles
            m_particles[ii]  = q1Samp(data);
            m_logUnNormWeights[ii]  = logMuEv(m_particles[ii]);
            m_logUnNormWeights[ii] += logGEv(data, m_particles[ii]);
            m_logUnNormWeights[ii] -= logQ1Ev(m_particles[ii], data);
            
            // update maximum
            if( m_logUnNormWeights[ii] > max)
                max = m_logUnNormWeights[ii];
        }
        
        // calculate log-likelihood with log-exp-sum trick
        double sumExp(0.0);
        for( size_t i = 0; i < nparts; ++i){
            sumExp += std::exp( m_logUnNormWeights[i] - max );
        }
        m_logLastCondLike = - std::log( static_cast<double>(nparts) ) + max + std::log(sumExp);
        
        // calculate expectations before you resample
        m_expectations.resize(fs.size());
        unsigned int fId(0);
        for(auto & h : fs){
            
            Mat testOutput = h(m_particles[0]);
            unsigned int rows = testOutput.rows();
            unsigned int cols = testOutput.cols();
            Mat numer = Mat::Zero(rows,cols);
            double denom(0.0);
            for(size_t prtcl = 0; prtcl < nparts; ++prtcl){ // iterate over all particles
                numer += h(m_particles[prtcl]) * std::exp(m_logUnNormWeights[prtcl] - max);
                denom += std::exp(m_logUnNormWeights[prtcl] - max);
            }
            m_expectations[fId] = numer/denom;
            fId++;
        }
        
        // resample if you should (automatically normalizes)
        if( (m_now+1) % m_rs == 0)
            m_resampler.resampLogWts(m_particles, m_logUnNormWeights);

        // advance time step
        m_now += 1;    
    }
    else{ //m_now > 0
        
        // set up "first stage weights" to make k index sampler 
        arrayDouble logFirstStageUnNormWeights = m_logUnNormWeights;
        arrayVec oldPartics = m_particles;
        double m3(-1.0/0.0);
        double m2(-1.0/0.0);
        for(size_t ii = 0; ii < nparts; ++ii)  
        {
            // update m3
            if(m_logUnNormWeights[ii] > m3)
                m3 = m_logUnNormWeights[ii];
            
            // sample
            ssv xtm1                        = oldPartics[ii];
            logFirstStageUnNormWeights[ii] += logGEv(data, propMu(xtm1)); // build up first stage weights
            
            // accumulate things
            if(logFirstStageUnNormWeights[ii] > m2)
                m2 = logFirstStageUnNormWeights[ii];

        }
               
        // draw ks (indexes) (handles underflow issues)
        arrayUInt myKs = m_kGen.sample(logFirstStageUnNormWeights); 
                
        // now draw xts
        double m1(-1.0/0.0);
        double first_cll_sum(0.0);
        double second_cll_sum(0.0);
        double third_cll_sum(0.0);
        ssv xtm1k;
        ssv muT;
        for(size_t ii = 0; ii < nparts; ++ii)   
        {
            // calclations for log p(y_t|y_{1:t-1}) (using log-sum-exp trick)
            second_cll_sum += std::exp( logFirstStageUnNormWeights[ii] - m2 );
            third_cll_sum  += std::exp( m_logUnNormWeights[ii] - m3 );            
            
            // sampling and unnormalized weight update
            xtm1k                   = oldPartics[myKs[ii]];
            m_particles[ii]         = fSamp(xtm1k); 
            muT                     = propMu(xtm1k); 
            m_logUnNormWeights[ii] += logGEv(data, m_particles[ii]) - logGEv(data, muT);
            
            // update m1
            if(m_logUnNormWeights[ii] > m1)
                m1 = m_logUnNormWeights[ii];
        }

        // calculate estimate for log of last conditonal likelihood
        for(size_t p = 0; p < nparts; ++p)
             first_cll_sum += std::exp( m_logUnNormWeights[p] - m1 );
        m_logLastCondLike = m1 + std::log(first_cll_sum) + m2 + std::log(second_cll_sum) - 2*m3 - 2*std::log(third_cll_sum);

        // calculate expectations before you resample
        //std::fill(m_expectations.begin(), m_expectations.end(), ssv::Zero()); 
        unsigned int fId(0);
        for(auto & h : fs){
    
            Mat testOutput = h(m_particles[0]);
            unsigned int rows = testOutput.rows();
            unsigned int cols = testOutput.cols();
            Mat numer = Mat::Zero(rows,cols);
            double denom(0.0);
            
            for(size_t prtcl = 0; prtcl < nparts; ++prtcl){ // iterate over all particles
                numer += h(m_particles[prtcl]) * std::exp(m_logUnNormWeights[prtcl] - m1);
                denom += std::exp(m_logUnNormWeights[prtcl] - m1);
            }
            m_expectations[fId] = numer/denom;
            fId++;
        }

        // if you have to resample
        if( (m_now+1)%m_rs == 0)
            m_resampler.resampLogWts(m_particles, m_logUnNormWeights);
            
        // advance time
        m_now += 1; 
    }
}


template<size_t nparts, size_t dimx, size_t dimy, typename resampT>
double APF<nparts, dimx, dimy, resampT>::getLogCondLike() const
{
    return m_logLastCondLike;
}


template<size_t nparts, size_t dimx, size_t dimy, typename resampT>
auto APF<nparts, dimx, dimy, resampT>::getExpectations() const -> std::vector<Mat>
{
    return m_expectations;
}





#endif //APF_H