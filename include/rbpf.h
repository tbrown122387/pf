#ifndef RBPF_HMM_H
#define RBPF_HMM_H

#include <functional>
#include <vector>
#include <array>
#include <Eigen/Dense>

#include "cf_filters.h" // for closed form filter objects


template<size_t nparts, size_t dimnss, size_t dimsss, size_t dimy, typename resampT>
class rbpf_hmm{
public:

    /** "sampled state size vector" */
    using sssv = Eigen::Matrix<double,dimsss,1>;
    /** "not sampled state size vector" */
    using nsssv = Eigen::Matrix<double,dimsss,1>;
    /** "observation size vector" */
    using osv = Eigen::Matrix<double,dimy,1>;
    /** "sampled state size matrix" */
    using sssMat = Eigen::Matrix<double,dimx,dimx>;
    /** "not sampled state size matrix" */
    using nsssMat = Eigen::Matrix<double,dimx,dimx>;
    /** array of model objects */
    using arrayMod = std::array<hmm,nparts>;
    /** array of samples */
    using arrayVec = std::array<sssv,nparts>;
    /** array of weights */
    using arrayDouble = std::array<double,nparts>;


    //! The constructor.
    /**
     * @brief constructor.
     * @param resamp_sched how often to resample (e.g. once every resamp_sched time periods)
     */
    rbpf_hmm(const unsigned int &resamp_sched);
    

    //! Filter.
    /**
     * @brief filters everything based on a new data point.
     * @param data the most recent time series observation.
     * @param fs a vector of functions computing logE[h(x_1t, x_2t^i)| x_2t^i,y_1:t]. will access the probability vector of x_1t
     */
    void filter(const osv &data,
                const std::vector<std::function<const sssMat(const nsssv &x1tProbs, const sssv &x2t)> >& fs 
                    = std::vector<std::function<const sssMat(const nsssv&, const sssv&)> >());//, const std::vector<std::function<const Mat(const Vec&)> >& fs);


    //! Get the latest conditional likelihood.
    /**
     * @brief Get the latest conditional likelihood.
     * @return the latest conditional likelihood.
     */
    double getLogCondLike() const;
    
    //!
    /**
     * @brief Get vector of expectations.
     * @return vector of expectations
     */
    std::vector<sssMat> getExpectations() const;

    //! Evaluates the first time state density.
    /**
     * @brief evaluates mu.
     * @param x21 component two at time 1
     * @return a double evaluation
     */
    virtual double logMuEv(const sssv &x21) = 0;
    
    
    //! Sample from the first sampler.
    /**
     * @brief samples the second component of the state at time 1.
     * @param y1 most recent datum.
     * @return a Vec sample for x21.
     */
    virtual sssv q1Samp(const osv &y1) = 0;
    
    
    //! Provides the initial mean vector for each HMM filter object.
    /**
     * @brief provides the initial probability vector for each HMM filter object.
     * @param x21 the second state componenent at time 1.
     * @return a Vec representing the probability of each state element.
     */
    virtual nssv initHMMProbVec(const sssv &x21) = 0;
    
    
    //! Provides the transition matrix for each HMM filter object.
    /**
     * @brief provides the transition matrix for each HMM filter object.
     * @param x21 the second state component at time 1. 
     * @return a transition matrix where element (ij) is the probability of transitioning from state i to state j.
     */
    virtual nsssMat initHMMTransMat(const sssv &x21) = 0;

    //! Samples the time t second component. 
    /**
     * @brief Samples the time t second component.
     * @param x2tm1 the previous time's second state component.
     * @param yt the current observation.
     * @return a Vec sample of the second state component at the current time.
     */
    virtual sssv qSamp(const sssv &x2tm1, const osv &yt) = 0;
    
    
    //! Evaluates the proposal density of the second state component at time 1.
    /**
     * @brief Evaluates the proposal density of the second state component at time 1.
     * @param x21 the second state component at time 1 you sampled. 
     * @param y1 time 1 observation.
     * @return a double evaluation of the density.
     */
    virtual double logQ1Ev(const sssv &x21, const osv &y1) = 0;
    
    
    //! Evaluates the state transition density for the second state component.
    /**
     * @brief Evaluates the state transition density for the second state component.
     * @param x2t the current second state component.
     * @param x2tm1 the previous second state component.
     * @return a double evaluation.
     */
    virtual double logFEv(const sssv &x2t, const sssv &x2tm1) = 0;
    
    
    //! Evaluates the proposal density at time t > 1.
    /**
     * @brief Evaluates the proposal density at time t > 1. 
     * @param x2t the current second state component.
     * @param x2tm1 the previous second state component.
     * @param yt the current time series observation.
     * @return a double evaluation.
     */
    virtual double logQEv(const sssv &x2t, const sssv &x2tm1, const osv &yt ) = 0;
    
    
    //! How to update your inner HMM filter object at each time.
    /**
     * @brief How to update your inner HMM filter object at each time.
     * @param aModel a HMM filter object describing the conditional closed-form model.
     * @param yt the current time series observation.
     * @param x2t the current second state component.
     */
    virtual void updateFSHMM(hmm &aModel, const osv &yt, const sssv &x2t) = 0;

private:
    
    /** the current time period */
    unsigned int m_now;
    /** last conditional likelihood */
    double m_lastLogCondLike;
    /** resampling schedue */
    unsigned int m_rs;
    /** the array of inner closed-form models */ 
    arrayMod m_p_innerMods;
    /** the array of samples for the second state portion */
    arrayVec m_p_samps;
    /** the array of unnormalized log-weights */
    arrayDouble m_logUnNormWeights;
    /** the resampler object */
    resampT m_resampler;
    /** the vector of expectations */
    std::vector<sssMat> m_expectations;


};


template<size_t nparts, size_t dimnss, size_t dimsss, size_t dimy, typename resampT>
rbpf_hmm<nparts,dimnss,dimsss,dimy,resampT>::rbpf_hmm(const unsigned int &resamp_sched)
    : m_now(0)
    , m_lastLogCondLike(0.0)
    , m_rs(resamp_sched)
{
    std::fill(m_logUnNormWeights.begin(), m_logUnNormWeights.end());
}


template<size_t nparts, size_t dimnss, size_t dimsss, size_t dimy, typename resampT>
void rbpf_hmm<nparts,dimnss,dimsss,dimy,resampT>::filter(const osv &data, const std::vector<std::function<const sssMat(const nsssv &x1tProbs, const sssv &x2t)> >& fs)
{

    if( m_now == 0){ // first data point coming
    
        // initialize and update the closed-form mods        
        nsssv tmpProbs;
        nsssMat tmpTransMat;
        double logWeightAdj;
        double tmpForFirstLike(0.0);
        for(size_t ii = 0; ii < nparts; ++ii){
            
            m_p_samps[ii] = q1Samp(data); 
            tmpProbs = initHMMProbVec(m_p_samps[ii]);
            tmpTransMat = initHMMTransMat(m_p_samps[ii]);
            //m_p_innerMods.emplace_back(tmpProbs, tmpTransMat); 
            m_p_innerMods[ii] = hmm(tmpProbs, tmpTransMat);
            updateFSHMM(m_p_innerMods[ii], data, m_p_samps[ii]);
            logWeightAdj = m_p_innerMods[ii].getLogCondLike() + logMuEv(m_p_samps[ii]) - logQ1Ev(m_p_samps[ii], data); 

            m_logUnNormWeights[ii] += logWeightAdj;
            tmpForFirstLike += std::exp(logWeightAdj);
        }
        m_lastLogCondLike = std::log(tmpForFirstLike) - std::log(m_numParts); // store likelihood

        // calculate expectations before you resample
        m_expectations.resize(fs.size());
        std::fill(m_expectations.begin(), m_expectations.end(), sssMat::Zero()); 
        int fId(0);
        double m = *std::max_element(m_logUnNormWeights.begin(), m_logUnNormWeights.end());
        for(auto & h : fs){
            sssMat numer = sssMat::Zero();
            sssMat ones = sssMat::Ones();
            double denom(0.0);
            nsssMat tmp;
            for(size_t prtcl = 0; prtcl < nparts; ++prtcl){ 
                tmp = h(m_p_innerMods[prtcl].getFilterVec(), m_p_samps[prtcl]);
                tmp = tmp.array().log().matrix() + (m_logUnNormWeights[prtcl] - m)*ones;
                numer = numer + tmp.array().exp().matrix();
                denom += std::exp( m_logUnNormWeights[prtcl] - m );
            }
            m_expectations[fId] = numer/denom;
            fId++;
        }

        
        // resample (unnormalized weights ok)
        if( (m_now+1) % m_rs == 0)
            m_resampler.resampLogWts(m_p_innerMods, m_p_samps, m_logUnNormWeights);

        // advance time step
        m_now ++;
    }
    else { //m_now > 0
        
        // update
        sssv newX2Samp;
        double logUnNormWeightUpdate;
        double tmpLikeNumer(0.0);
        double tmpLikeDenom(0.0);
        for(unsigned ii = 0; ii < nparts; ++ii){
            
            newX2Samp = qSamp(m_p_samps[ii], data);
            updateFSHMM(m_p_innerMods[ii], data, newX2Samp);
            logUnNormWeightUpdate = m_p_innerMods[ii].getLogCondLike()
                                    + logFEv(newX2Samp, m_p_samps[ii]) 
                                    - logQEv(newX2Samp, m_p_samps[ii], data);
            
            tmpLikeDenom += std::exp(m_logUnNormWeights[ii]);
            m_logUnNormWeights[ii] += logUnNormWeightUpdate;
            tmpLikeNumer += std::exp(m_logUnNormWeights[ii]); 
            m_p_samps[ii] = newX2Samp;
        }
        m_lastLogCondLike = std::log(tmpLikeNumer) - std::log( tmpLikeDenom );
        
        // calculate expectations before you resample
        std::fill(m_expectations.begin(), m_expectations.end(), sssMat::Zero()); 
        unsigned int fId(0);
        double m = *std::max_element(m_logUnNormWeights.begin(), m_logUnNormWeights.end());
        for(auto & h : fs){
            sssMat numer = nsssMat::Zero();
            sssMat ones = nsssMat::Ones();
            sssMat tmp;
            double denom(0.0);
            for(size_t prtcl = 0; prtcl < nparts; ++prtcl){ 
                tmp = h(m_p_innerMods[prtcl].getFilterVec(), m_p_samps[prtcl]);
                tmp = tmp.array().log().matrix() + (m_logUnNormWeights[prtcl] - m)*ones;
                numer = numer + tmp.array().exp().matrix();
                denom += std::exp( m_logUnNormWeights[prtcl] - m );
            }
            m_expectations[fId] = numer/denom;
            fId++;
        }

        // resample (unnormalized weights ok)
        if( (m_now+1) % m_rs == 0)
            m_resampler.resampLogWts(m_p_innerMods, m_p_samps, m_logUnNormWeights);
        
        // update time step
        m_now ++;
    }
    
}


template<size_t nparts, size_t dimnss, size_t dimsss, size_t dimy, typename resampT>
double rbpf_hmm<nparts,dimnss,dimsss,dimy,resampT>::getLogCondLike() const
{
    return m_lastLogCondLike;
}


template<size_t nparts, size_t dimnss, size_t dimsss, size_t dimy, typename resampT>
auto rbpf_hmm<nparts,dimnss,dimsss,dimy,resampT>::getExpectations() const -> std::vector<sssMat>
{
    return m_expectations;
}

#endif //RBPF_HMM_H