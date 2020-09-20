#ifndef RESAMPLERS_H
#define RESAMPLERS_H

#include <chrono>
#include <array>
#include <random>
#include <numeric> // accumulate, partial_sum
#include <cmath> //floor
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
template<size_t nparts, size_t dimx, typename float_t >
class rbase
{
public:

    /** type alias for linear algebra stuff */
    using ssv = Eigen::Matrix<float_t,dimx,1>;
    /** type alias for array of Eigen Matrices */
    using arrayVec = std::array<ssv, nparts>;
    /** type alias for array of float_ts */
    using arrayFloat = std::array<float_t,nparts>;


    /**
     * @brief The default constructor gets called by default, and it sets the seed with the clock. 
     */
    rbase();


    /**
     * @brief The constructor that sets the seed deterministically. 
     * @param seed the seed 
     */
    rbase(unsigned long seed);


    /**
     * @brief Function to resample from log unnormalized weights
     * @param oldParts
     * @param oldLogUnNormWts
     */
    virtual void resampLogWts(arrayVec &oldParts, arrayFloat &oldLogUnNormWts) = 0;

protected:

    /** @brief prng */
    std::mt19937 m_gen;

};


template<size_t nparts, size_t dimx, typename float_t>
rbase<nparts, dimx, float_t>::rbase() 
        : m_gen{static_cast<std::uint32_t>(
                    std::chrono::high_resolution_clock::now().time_since_epoch().count()
                                           )}
{
}


template<size_t nparts, size_t dimx, typename float_t>
rbase<nparts, dimx, float_t>::rbase(unsigned long seed) 
        : m_gen{static_cast<std::uint32_t>(seed)}
{
}



/**
 * @class mn_resampler
 * @author taylor
 * @date 15/04/18
 * @file resamplers.h
 * @brief Class that performs multinomial resampling for "standard" models.
 * @tparam nparts the number of particles.
 * @tparam dimx the dimension of each state sample.
 */
template<size_t nparts, size_t dimx, typename float_t>
class mn_resampler : private rbase<nparts, dimx, float_t>
{
public:

    /** type alias for linear algebra stuff */
    using ssv = Eigen::Matrix<float_t,dimx,1>;
    /** type alias for array of Eigen Matrices */
    using arrayVec = std::array<ssv, nparts>;
    /** type alias for array of float_ts */
    using arrayFloat = std::array<float_t,nparts>;
    /** type alias for array of integers */
    using arrayInt = std::array<unsigned int,nparts>;

    /**
     * @brief Default constructor. 
     */
    mn_resampler() = default;


    /**
     * @brief Constructor that sets the seed.
     * @param seed
     */
    mn_resampler(unsigned long seed);
    
    
    /**
     * @brief resamples particles.
     * @param oldParts the old particles
     * @param oldLogUnNormWts the old log unnormalized weights
     */
    void resampLogWts(arrayVec &oldParts, arrayFloat &oldLogUnNormWts);
    
};


template<size_t nparts, size_t dimx, typename float_t>
mn_resampler<nparts, dimx, float_t>::mn_resampler(unsigned long seed)
    : rbase<nparts, dimx, float_t>(seed)
{
}


template<size_t nparts, size_t dimx, typename float_t>
void mn_resampler<nparts, dimx, float_t>::resampLogWts(arrayVec &oldParts, arrayFloat &oldLogUnNormWts)
{
    // these log weights may be very negative. If that's the case, exponentiating them may cause underflow
    // so we use the "log-exp-sum" trick
    // actually not quite...we just shift the log-weights because after they're exponentiated
    // they have the same normalized probabilities
       
    // Create the distribution with exponentiated log-weights
    arrayFloat w;
    float_t m = *std::max_element(oldLogUnNormWts.begin(), oldLogUnNormWts.end());
    std::transform(oldLogUnNormWts.begin(), oldLogUnNormWts.end(), w.begin(), 
                    [&m](float_t& d) -> float_t { return std::exp( d - m ); } );
    std::discrete_distribution<> idxSampler(w.begin(), w.end());
    
    // create temporary particle vector and weight vector
    arrayVec tmpPartics = oldParts; 
    
    // sample from the original parts and store in tmpParts
    unsigned int whichPart;
    for(size_t part = 0; part < nparts; ++part)
    {
        whichPart = idxSampler(this->m_gen);
        tmpPartics[part] = oldParts[whichPart];
    }
        
    //overwrite olds with news
    oldParts = std::move(tmpPartics);
    std::fill(oldLogUnNormWts.begin(), oldLogUnNormWts.end(), 0.0); // change back    
}



/**
 * @class mn_resampler_rbpf
 * @author taylor
 * @file resamplers.h
 * @brief Class that performs multinomial resampling for RBPFs.
 * @tparam nparts the number of particles.
 * @tparam dimsampledx the dimension of each state sample.
 * @tparam cfModT the type of closed form model
 * @tparam float_t the type of floating point number
 */
template<size_t nparts, size_t dimsampledx, typename cfModT, typename float_t>
class mn_resampler_rbpf 
{
public:

    /** type alias for linear algebra stuff */
    using ssv = Eigen::Matrix<float_t,dimsampledx,1>;
    /** type alias for linear algebra stuff */
    using arrayVec = std::array<ssv, nparts>;
    /** type alias for array of float_ts */
    using arrayFloat = std::array<float_t,nparts>;
    /** type alias for array of closed-form models */
    using arrayMod = std::array<cfModT,nparts>;

    /**
     * @brief Default constructor. 
     */
    mn_resampler_rbpf();
    
 
    /**
     * @brief Default constructor. 
     */
    mn_resampler_rbpf(unsigned long seed);


    /**
     * @brief resamples particles.
     * @param oldMods the old closed-form models
     * @param oldParts the old particles
     * @param oldLogUnNormWts the old log unnormalized weights
     */
    void resampLogWts(arrayMod &oldMods, arrayVec &oldParts, arrayFloat &oldLogUnNormWts);


private:

    /** @brief prng */
    std::mt19937 m_gen;
};


template<size_t nparts, size_t dimsampledx, typename cfModT, typename float_t>
mn_resampler_rbpf<nparts, dimsampledx, cfModT,float_t>::mn_resampler_rbpf() 
    : m_gen{static_cast<std::uint32_t>(
                    std::chrono::high_resolution_clock::now().time_since_epoch().count()
                                           )}
{
}


template<size_t nparts, size_t dimsampledx, typename cfModT, typename float_t>
mn_resampler_rbpf<nparts, dimsampledx, cfModT,float_t>::mn_resampler_rbpf(unsigned long seed)
    : m_gen{ static_cast<std::uint32_t>(seed) }
{
}


template<size_t nparts, size_t dimsampledx, typename cfModT, typename float_t>
void mn_resampler_rbpf<nparts, dimsampledx, cfModT,float_t>::resampLogWts(arrayMod &oldMods, arrayVec &oldSamps, arrayFloat &oldLogUnNormWts) 
{
    // Create the distribution with exponentiated log-weights
    arrayFloat w;
    float_t m = *std::max_element(oldLogUnNormWts.begin(), oldLogUnNormWts.end());
    std::transform(oldLogUnNormWts.begin(), oldLogUnNormWts.end(), w.begin(), 
                    [&m](float_t& d) -> float_t { return std::exp( d - m ); } );
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
    std::fill(oldLogUnNormWts.begin(), oldLogUnNormWts.end(), 0.0);

}


/**
 * @class resid_resampler
 * @author taylor
 * @date 10/25/19
 * @file resamplers.h
 * @brief Class that performs residual resampling on "standard" models.
 * @tparam nparts the number of particles.
 * @tparam dimx the dimension of each state sample.
 * @tparam float_t the floating point for samples
 */
template<size_t nparts, size_t dimx, typename float_t>
class resid_resampler : private rbase<nparts, dimx, float_t>
{
public:

    /** type alias for linear algebra stuff */
    using ssv = Eigen::Matrix<float_t,dimx,1>;
    /** type alias for array of Eigen Matrices */
    using arrayVec = std::array<ssv, nparts>;
    /** type alias for array of float_ts */
    using arrayFloat = std::array<float_t,nparts>;
    /** type alias for array of integers */
    using arrayInt = std::array<unsigned int, nparts>;


    /**
     * @brief Default constructor.
     */
    resid_resampler() = default;
    
 
    /**
     * @brief Constructor that sets the seed.
     * @param seed
     */
    resid_resampler(unsigned long seed);
   

    /**
     * @brief resamples particles.
     * @param oldParts the old particles
     * @param oldLogUnNormWts the old log unnormalized weights
     */
    void resampLogWts(arrayVec &oldParts, arrayFloat &oldLogUnNormWts);
    
};

    
template<size_t nparts, size_t dimx, typename float_t>
resid_resampler<nparts, dimx, float_t>::resid_resampler(unsigned long seed)
    : rbase<nparts, dimx, float_t>(seed)
{
}


template<size_t nparts, size_t dimx, typename float_t>
void resid_resampler<nparts, dimx, float_t>::resampLogWts(arrayVec &oldParts, arrayFloat &oldLogUnNormWts)
{

    // calculate normalized weights
    arrayFloat w; 
    float_t m = *std::max_element(oldLogUnNormWts.begin(), oldLogUnNormWts.end());
    std::transform(oldLogUnNormWts.begin(), oldLogUnNormWts.end(), w.begin(), 
                    [&m](const float_t& d) -> float_t { return std::exp( d - m ); } );
    float_t norm_const (0.0);
    norm_const = std::accumulate(w.begin(), w.end(), norm_const);
    for( auto& weight : w)
        weight = weight/norm_const;

    // calc unNormWBars and numRandomSamples (N-R using IIHMM notation)
    size_t i;
    arrayFloat unNormWBar;
    float_t numRandomSamples(0.0);
    for(i = 0; i < nparts; ++i) {
        unNormWBar[i] = w[i]*nparts - std::floor(w[i]*nparts);
        numRandomSamples += unNormWBar[i];
    }

    // make multinomial distribution for residuals
    std::discrete_distribution<> idxSampler(unNormWBar.begin(), unNormWBar.end());

    // start resampling by producing a count vector
    arrayInt sampleCounts;
    for(i = 0; i < nparts; ++i) {
        sampleCounts[i] = static_cast<unsigned int>(std::floor(nparts*w[i])); // initial
    }
    for(i = 0; i < static_cast<unsigned int>(numRandomSamples); ++i) {
        sampleCounts[idxSampler(this->m_gen)]++;
    }
    
    // now resample the particles using the counts 
    arrayVec tmpPartics;
    unsigned int c(0);
    for(i = 0; i < nparts; ++i) { // over count container
        unsigned int num_replicants = sampleCounts[i];
        if( num_replicants > 0) {
            for(size_t j = 0; j < num_replicants; ++j) { // assign the same thing several times
                tmpPartics[c] = oldParts[i];
                c++;
            }
        }
    }

    //overwrite olds with news
    oldParts = std::move(tmpPartics);
    std::fill(oldLogUnNormWts.begin(), oldLogUnNormWts.end(), 0.0); // change back    
}


/**
 * @class stratif_resampler
 * @author taylor
 * @date 10/25/19
 * @file resamplers.h
 * @brief Class that performs stratified resampling on "standard" models.
 * @tparam nparts the number of particles.
 * @tparam dimx the dimension of each state sample.
 * @tparam float_t the floating point for samples
 */
template<size_t nparts, size_t dimx, typename float_t>
class stratif_resampler : private rbase<nparts, dimx, float_t>
{
public:

    /** type alias for linear algebra stuff */
    using ssv = Eigen::Matrix<float_t,dimx,1>;
    /** type alias for array of Eigen Matrices */
    using arrayVec = std::array<ssv, nparts>;
    /** type alias for array of float_ts */
    using arrayFloat = std::array<float_t,nparts>;
    /** type alias for array of integers */
    using arrayInt = std::array<unsigned int, nparts>;


    /**
     * @brief Default constructor. 
     */
    stratif_resampler() = default;
    
 
    /**
     * @brief Constructor that sets the seed
     * @param seed 
     */
    stratif_resampler(unsigned long seed);
  

    /**
     * @brief resamples particles.
     * @param oldParts the old particles
     * @param oldLogUnNormWts the old log unnormalized weights
     */
    void resampLogWts(arrayVec &oldParts, arrayFloat &oldLogUnNormWts);
    
};


template<size_t nparts, size_t dimx, typename float_t>
stratif_resampler<nparts, dimx, float_t>::stratif_resampler(unsigned long seed)
    : rbase<nparts, dimx, float_t>(seed)
{
}


template<size_t nparts, size_t dimx, typename float_t>
void stratif_resampler<nparts, dimx, float_t>::resampLogWts(arrayVec &oldParts, arrayFloat &oldLogUnNormWts)
{

    // calculate normalized weights
    arrayFloat w; 
    float_t m = *std::max_element(oldLogUnNormWts.begin(), oldLogUnNormWts.end());
    std::transform(oldLogUnNormWts.begin(), oldLogUnNormWts.end(), w.begin(), 
                    [&m](const float_t& d) -> float_t { return std::exp( d - m ); } );
    float_t norm_const (0.0);
    norm_const = std::accumulate(w.begin(), w.end(), norm_const);
    for( auto& weight : w)
        weight = weight/norm_const;

    // calculate the cumulative sums of the weights
    // TODO: possible bug
    arrayFloat cumsums;
    std::partial_sum(w.begin(), w.end(), cumsums.begin());

    // samplethe Uis
    std::uniform_real_distribution<float_t> u_sampler(0.0, 1.0/nparts);
    arrayFloat u_samples;
    for(size_t i = 0; i < nparts; ++i) {
        u_samples[i] = i/nparts + u_sampler(this->m_gen);
    }

    // resample
    arrayVec tmpPartics;
    for(size_t i = 0; i < nparts; ++i){ // tmpPartics, Uis

        // find which index
        unsigned int idx;
        for(unsigned int j = 0; j < nparts; ++j){
            
            // get the first time it gets covered by a cumsum
            if(cumsums[j] >= u_samples[i]){ 
                idx = j;
                break;
            }   
        }

        // assign
        tmpPartics[i] = oldParts[idx];
    }

    //overwrite olds with news
    oldParts = std::move(tmpPartics);
    std::fill(oldLogUnNormWts.begin(), oldLogUnNormWts.end(), 0.0); // change back    
}


/**
 * @class systematic_resampler
 * @author taylor
 * @date 10/25/19
 * @file resamplers.h
 * @brief Class that performs systematic resampling on "standard" models.
 * @tparam nparts the number of particles.
 * @tparam dimx the dimension of each state sample.
 * @tparam float_t the floating point for samples
 */
template<size_t nparts, size_t dimx, typename float_t>
class systematic_resampler : private rbase<nparts, dimx, float_t>
{
public:

    /** type alias for linear algebra stuff */
    using ssv = Eigen::Matrix<float_t,dimx,1>;
    /** type alias for array of Eigen Matrices */
    using arrayVec = std::array<ssv, nparts>;
    /** type alias for array of float_ts */
    using arrayFloat = std::array<float_t,nparts>;
    /** type alias for array of integers */
    using arrayInt = std::array<unsigned int, nparts>;


    /**
     * @brief Default constructor. 
     */
    systematic_resampler() = default;
    

    /**
     * @brief Constructor that sets the seed.
     * @param seed.
     */
    systematic_resampler(unsigned long seed);


    /**
     * @brief resamples particles.
     * @param oldParts the old particles
     * @param oldLogUnNormWts the old log unnormalized weights
     */
    void resampLogWts(arrayVec &oldParts, arrayFloat &oldLogUnNormWts);
    
};


template<size_t nparts, size_t dimx, typename float_t>
systematic_resampler<nparts, dimx, float_t>::systematic_resampler(unsigned long seed)
    : rbase<nparts, dimx, float_t>(seed)
{
}


template<size_t nparts, size_t dimx, typename float_t>
void systematic_resampler<nparts, dimx, float_t>::resampLogWts(arrayVec &oldParts, arrayFloat &oldLogUnNormWts)
{

    // calculate normalized weights
    arrayFloat w; 
    float_t m = *std::max_element(oldLogUnNormWts.begin(), oldLogUnNormWts.end());
    std::transform(oldLogUnNormWts.begin(), oldLogUnNormWts.end(), w.begin(), 
                    [&m](const float_t& d) -> float_t { return std::exp( d - m ); } );
    float_t norm_const (0.0);
    norm_const = std::accumulate(w.begin(), w.end(), norm_const);
    for( auto& weight : w)
        weight = weight/norm_const;

    // calculate the cumulative sums of the weights
    arrayFloat cumsums;
    std::partial_sum(w.begin(), w.end(), cumsums.begin());

    // samplethe Uis
    std::uniform_real_distribution<float_t> u_sampler(0.0, 1.0/nparts);
    arrayFloat u_samples;
    u_samples[0] = u_sampler(this->m_gen);
    for(size_t i = 1; i < nparts; ++i) {
        u_samples[i] = u_samples[i-1] + 1.0/nparts;
    }

    // resample (same code from here on as stratified)
    arrayVec tmpPartics;
    for(size_t i = 0; i < nparts; ++i){ // tmpPartics, Uis

        // find which index
        unsigned int idx;
        for(unsigned int j = 0; j < nparts; ++j){
            
            // get the first time it gets covered by a cumsum
            if(cumsums[j] >= u_samples[i]){ 
                idx = j;
                break;
            }   
        }

        // assign
        tmpPartics[i] = oldParts[idx];
    }

    //overwrite olds with news
    oldParts = std::move(tmpPartics);
    std::fill(oldLogUnNormWts.begin(), oldLogUnNormWts.end(), 0.0); // change back    
}


/**
 * @class mn_resamp_fast1
 * @author taylor
 * @file resamplers.h
 * @brief Class that performs multinomial resampling for "standard" models. 
 * For justification, see page 244 of "Inference in Hidden Markov Models"
 * @tparam nparts the number of particles.
 * @tparam dimx the dimension of each state sample.
 */
template<size_t nparts, size_t dimx, typename float_t>
class mn_resamp_fast1 : private rbase<nparts, dimx, float_t>
{
public:

    /** type alias for linear algebra stuff */
    using ssv = Eigen::Matrix<float_t,dimx,1>;
    /** type alias for array of Eigen Matrices */
    using arrayVec = std::array<ssv, nparts>;
    /** type alias for array of float_ts */
    using arrayFloat = std::array<float_t,nparts>;
    /** type alias for array of integers */
    using arrayInt = std::array<unsigned int,nparts>;

    /**
     * @brief Default constructor. 
     */
    mn_resamp_fast1() = default;
    
 
    /**
     * @brief Default constructor. 
     */
    mn_resamp_fast1(unsigned long seed);


    /**
     * @brief resamples particles.
     * @param oldParts the old particles
     * @param oldLogUnNormWts the old log unnormalized weights
     */
    void resampLogWts(arrayVec &oldParts, arrayFloat &oldLogUnNormWts);
    
};


template<size_t nparts, size_t dimx, typename float_t>
mn_resamp_fast1<nparts, dimx, float_t>:: mn_resamp_fast1(unsigned long seed)
    : rbase<nparts, dimx, float_t>(seed)
{
}


template<size_t nparts, size_t dimx, typename float_t>
void mn_resamp_fast1<nparts, dimx, float_t>::resampLogWts(arrayVec &oldParts, arrayFloat &oldLogUnNormWts)
{
    // these log weights may be very negative. If that's the case, exponentiating them may cause underflow
    // so we use the "log-exp-sum" trick
    // actually not quite...we just shift the log-weights because after they're exponentiated
    // they have the same normalized probabilities
       
    // Also, we're using a fancier algorthm detailed on page 244 of IHMM 

    // Create unnormalized weights
    arrayFloat unnorm_weights;
    float_t m = *std::max_element(oldLogUnNormWts.begin(), oldLogUnNormWts.end());
    std::transform(oldLogUnNormWts.begin(), oldLogUnNormWts.end(), unnorm_weights.begin(), 
                    [&m](float_t& d) -> float_t { return std::exp( d - m ); } );
    
    // get a uniform rv sampler
    std::uniform_real_distribution<float_t> u_sampler(0.0, 1.0);
    
    // two things: 
    // 1.) calculate normalizing constant for weights, and 
    // 2.) generate all these exponentials to help with getting order statistics
    // NB: you never need to store E_{N+1}! (this is subtle)
    float_t weight_norm_const(0.0);
    arrayFloat exponentials;
    float_t G(0.0);
    for(size_t i = 0; i < nparts; ++i) {
        weight_norm_const += unnorm_weights[i];
        exponentials[i] = -std::log(u_sampler(this->m_gen));   
        G += exponentials[i];
    }
    G+= std::log(u_sampler(this->m_gen)); // E_{N+1}

    // see Fig 7.15 in IHMM on page 243
    arrayVec tmpPartics = oldParts;                // the new particles 
    float_t uniform_order_stat(0.0);               // U_{(i)} in the notation of IHMM
    float_t running_sum_normalized_weights(unnorm_weights[0]/weight_norm_const); // \sum_{j=1}^I \omega^j in the notation of IHMM
    float_t one_less_summand(0.0);                 // \sum_{j=1}^{I-1} \omega^j 
    unsigned int idx = 0;
    for(size_t i = 0; i < nparts; ++i){
        uniform_order_stat += exponentials[i]/G; // add a spacing E_i/G
        do {
            if( one_less_summand < uniform_order_stat <= running_sum_normalized_weights ) {
                // select index idx
                tmpPartics[i] = oldParts[idx];
                break;
            }else{
                // increment idx because it will never be chosen (all the other order statistics are even higher) 
                idx++;
                running_sum_normalized_weights += unnorm_weights[idx]/weight_norm_const;
                one_less_summand += unnorm_weights[idx-1]/weight_norm_const;
            }
        }while(true);
    }

    //overwrite olds with news
    oldParts = std::move(tmpPartics);
    std::fill(oldLogUnNormWts.begin(), oldLogUnNormWts.end(), 0.0);  
}

#endif // RESAMPLERS_H
