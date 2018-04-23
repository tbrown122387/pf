#ifndef RESAMPLERS_H
#define RESAMPLERS_H

#include <chrono>
#include <array>
#include <random>
#include <Eigen/Dense>

    
//! Base class for all resampler types.
/**
 * @class rbase
 * @author taylor
 * @date 15/04/18
 * @file resamplers.h
 * @brief all resamplers must inherit from this. 
 * This will enforce certain structure that are assumed by 
 * all particle filters.
 * @tparam nparts the number of particles.
 * @tparam dimx the dimension of each state sample.
 */
template<size_t nparts, size_t dimx>
class rbase
{
public:

    /** type alias for linear algebra stuff */
    using ssv = Eigen::Matrix<double,dimx,1>;
    /** type alias for array of Eigen Matrices */
    using arrayVec = std::array<ssv, nparts>;
    /** type alias for array of doubles */
    using arrayDouble = std::array<double,nparts>;


    /**
     * @brief The default constructor. This is the only option available. Sets the seed with the clock. 
     */
    rbase();
    
    /**
     * @brief Function to resample from log unnormalized weights
     * @param oldParts
     * @param oldLogUnNormWts
     */
    virtual void resampLogWts(arrayVec &oldParts, arrayDouble &oldLogUnNormWts) = 0;

private:

    /** @brief prng */
    std::mt19937 m_gen;

};


template<size_t nparts, size_t dimx>
rbase<nparts, dimx>::rbase() 
        : m_gen{static_cast<std::uint32_t>(
                    std::chrono::high_resolution_clock::now().time_since_epoch().count()
                                           )}
{
}


//! Performs multinomial resampling.
/**
 * @class mn_resampler
 * @author taylor
 * @date 15/04/18
 * @file resamplers.h
 * @brief Class that performs multinomial resampling.
 * @tparam nparts the number of particles.
 * @tparam dimx the dimension of each state sample.
 */
template<size_t nparts, size_t dimx>
class mn_resampler : public rbase<nparts, dimx>
{
public:
    /** type alias for linear algebra stuff */
    using ssv = Eigen::Matrix<double,dimx,1>;
    /** type alias for linear algebra stuff */
    using arrayVec = std::array<ssv, nparts>;
    /** type alias for array of doubles */
    using arrayDouble = std::array<double,nparts>;
    /** type alias for array of integers */
    using arrayInt = std::array<unsigned int,nparts>;

    /**
     * @brief Default constructor. Only option available.
     */
    mn_resampler();
    
    
    /**
     * @brief resamples particles.
     * @param oldParts the old particles
     * @param oldLogUnNormWts the old log unnormalized weights
     */
    void resampLogWts(arrayVec &oldParts, arrayDouble &oldLogUnNormWts);
    
private:

    /** @brief prng */
    std::mt19937 m_gen;

};




template<size_t nparts, size_t dimx>
mn_resampler<nparts, dimx>::mn_resampler() : rbase<nparts, dimx>()
{
}


template<size_t nparts, size_t dimx>
void mn_resampler<nparts, dimx>::resampLogWts(arrayVec &oldParts, arrayDouble &oldLogUnNormWts)
{
    // these log weights may be very negative. If that's the case, exponentiating them may cause underflow
    // so we use the "log-exp-sum" trick
    // actually not quite...we just shift the log-weights because after they're exponentiated
    // they have the same normalized probabilities
       
    // Create the distribution with exponentiated log-weights
    arrayDouble w;
    double m = *std::max_element(oldLogUnNormWts.begin(), oldLogUnNormWts.end());
    std::transform(oldLogUnNormWts.begin(), oldLogUnNormWts.end(), w.begin(), 
                    [&m](double& d) -> double { return std::exp( d - m ); } );
    std::discrete_distribution<> idxSampler(w.begin(), w.end());
    
    // create temporary particle vector and weight vector
    arrayVec tmpPartics = oldParts; 
    
    // sample from the original parts and store in tmpParts
    unsigned int whichPart;
    for(size_t part = 0; part < nparts; ++part)
    {
        whichPart = idxSampler(m_gen);
        tmpPartics[part] = oldParts[whichPart];
    }
        
    //overwrite olds with news
    oldParts = std::move(tmpPartics);
    std::fill(oldLogUnNormWts.begin(), oldLogUnNormWts.end(), 0.0); // change back    
}



//! Performs multinomial resampling for a Rao-Blackwellized pf.
/**
 * @class mn_resampler_rbpf
 * @author taylor
 * @file resamplers.h
 * @brief Class that performs multinomial resampling for RBPFs.
 * @tparam nparts the number of particles.
 * @tparam dimsampledx the dimension of each state sample.
 */
template<size_t nparts, size_t dimsampledx, typename cfModT>
class mn_resampler_rbpf
{
public:
    /** type alias for linear algebra stuff */
    using ssv = Eigen::Matrix<double,dimsampledx,1>;
    /** type alias for linear algebra stuff */
    using arrayVec = std::array<ssv, nparts>;
    /** type alias for array of doubles */
    using arrayDouble = std::array<double,nparts>;
    /** type alias for array of closed-form models */
    using arrayMod = std::array<cfModT,nparts>;

    /**
     * @brief Default constructor. Only option available.
     */
    mn_resampler_rbpf();
    
    
    /**
     * @brief resamples particles.
     * @param oldMods the old closed-form models
     * @param oldParts the old particles
     * @param oldLogUnNormWts the old log unnormalized weights
     */
    void resampLogWts(arrayMod &oldMods, arrayVec &oldParts, arrayDouble &oldLogUnNormWts);
    
private:

    /** @brief prng */
    std::mt19937 m_gen;

};


template<size_t nparts, size_t dimsampledx, typename cfModT>
mn_resampler_rbpf<nparts, dimsampledx, cfModT>::mn_resampler_rbpf() 
    : m_gen{static_cast<std::uint32_t>(
                    std::chrono::high_resolution_clock::now().time_since_epoch().count()
                                           )}
{
}


template<size_t nparts, size_t dimsampledx, typename cfModT>
mn_resampler_rbpf<nparts, dimsampledx, cfModT>::resampLogWts(arrayMod &oldMods, arrayVec &oldParts, arrayDouble &oldLogUnNormWts) 
{
    // Create the distribution with exponentiated log-weights
    arrayDouble w;
    double m = *std::max_element(oldLogUnNormWts.begin(), oldLogUnNormWts.end());
    std::transform(oldLogUnNormWts.begin(), oldLogUnNormWts.end(), w.begin(), 
                    [&m](double& d) -> double { return std::exp( d - m ); } );
    std::discrete_distribution<> idxSampler(w.begin(), w.end());
    
    // create temporary vectors for samps and mods
    arrayVec tmpSamps;
    arrayMod tmpMods;
    
    // sample from the original parts and store in temporary
    unsigned int whichPart;
    for(size_t part = 0; part < nparts; ++part)
    {
        whichPart = idxSampler(m_gen);
        tmpSamps[part] = oldSamps[whichPart];
        tmpMods[part] = oldMods[whichPart];
    }
    
    //overwrite olds with news
    oldMods = std::move(tmpMods);
    oldSamps = std::move(tmpSamps);
    std::fill (oldLogWts.begin(), oldLogWts.end(), 0.0);

}


#endif // RESAMPLERS_H