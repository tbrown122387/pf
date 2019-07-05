#ifndef RV_SAMP_H
#define RV_SAMP_H

#include <chrono>
#include <Eigen/Dense> //linear algebra stuff
#include <random>

namespace rvsamp{


//! Base class for all random variable sampler types. Primary benefit is that it sets the seed for you.
/**
 * @class rvsamp_base
 * @author taylor
 * @file rv_samp.h
 * @brief all rv samplers must inherit from this. 
 * @tparam dim the dimension of each random vector sample.
 */
class rvsamp_base
{
public:

    /**
     * @brief The default constructor. This is the only option available. Sets the seed with the clock. 
     */
    inline rvsamp_base() : 
        m_rng{static_cast<std::uint32_t>(std::chrono::high_resolution_clock::now().time_since_epoch().count())} 
    {}

protected:

    /** @brief prng */
    std::mt19937 m_rng;

};



//! A class that performs sampling from a univariate Normal distribution.
/**
* @class UnivNormSampler
* @author taylor
* @file rv_samp.h
* @brief Samples from univariate Normal distribution.
*/
template<typename float_t>
class UnivNormSampler : public rvsamp_base
{
    
public:


    /**
     * @brief Default-constructor sets up for standard Normal random variate generation.
     */
    UnivNormSampler();


     /**
      * @brief The user must supply both mean and std. dev.
      * @param mu a float_t for the mean of the sampling distribution.
      * @param sigma a float_t (> 0) representing the standard deviation of the samples.
      */
    UnivNormSampler(const float_t &mu, const float_t &sigma);


    /**
     * @brief sets the standard deviation of the sampler.
     * @param sigma the desired standard deviation.
     */
    void setStdDev(const float_t &sigma);
    
    
    /**
     * @brief sets the mean of the sampler.
     * @param mu the desired mean.
     */
    void setMean(const float_t &mu);
    
        
     /**
      * @brief Draws a random number.
      * @return a random sample of type float_t.
      */
    float_t sample();    
    

private:

    /** @brief makes normal random variates */
    std::normal_distribution<float_t> m_z_gen;
    
    /** @brief the mean */
    float_t m_mu;
    
    /** @brief the standard deviation */
    float_t m_sigma;

};


template<typename float_t>
UnivNormSampler<float_t>::UnivNormSampler()
    : rvsamp_base()
    , m_z_gen(0.0, 1.0)
{
    setMean(0.0);
    setStdDev(1.0);
}


template<typename float_t>
UnivNormSampler<float_t>::UnivNormSampler(const float_t &mu, const float_t &sigma)
    : rvsamp_base()
    , m_z_gen(0.0, 1.0)
{
    setMean(mu); 
    setStdDev(sigma);
}


template<typename float_t>
void UnivNormSampler<float_t>::setMean(const float_t &mu)
{
    m_mu = mu;
}


template<typename float_t>
void UnivNormSampler<float_t>::setStdDev(const float_t &sigma)
{
    m_sigma = sigma;
}


template<typename float_t>
float_t UnivNormSampler<float_t>::sample()
{
    return m_mu + m_sigma * m_z_gen(m_rng);
}



//! A class that performs sampling from a univariate Bernoulli distribution.
/**
* @class BernSampler
* @author taylor
* @file rv_samp.h
* @brief Samples from univariate Bernoulli distribution.
*/
template<typename float_t, typename int_t>
class BernSampler : public rvsamp_base
{
    
public:


    /**
     * @brief Default-constructor sets up for Bernoulli random variate generation with p = .5.
     */
    BernSampler();


     /**
      * @brief Constructs Bernoulli sampler with user-specified p.
      * @param p a float_t for the probability that the rv equals 1.
      */
    BernSampler(const float_t &p);


    /**
     * @brief sets the parameter p.
     * @param p the p(X=1) = 1-p(X=0).
     */
    void setP(const float_t &p);
    

    /** 
      * @brief Draws a random number.
      * @return a random sample of type float_t.
      */
    int_t sample();    
    

private:

    /** @brief makes normal random variates */
    std::bernoulli_distribution m_B_gen;
    
    /** @brief the mean */
    float_t m_p;
};


template<typename float_t, typename int_t>
BernSampler<float_t, int_t>::BernSampler() 
    : rvsamp_base(), m_B_gen(.5)
{
}


template<typename float_t, typename int_t>
BernSampler<float_t, int_t>::BernSampler(const float_t& p) 
    : rvsamp_base(), m_B_gen(p)
{
}


template<typename float_t, typename int_t>
void BernSampler<float_t, int_t>::setP(const float_t& p)
{
    m_p = p;
}


template<typename float_t, typename int_t>
int_t BernSampler<float_t, int_t>::sample()
{
    return (m_B_gen(m_rng)) ? 1 : 0;
}



//! A class that performs sampling from a multivariate normal distribution.
/**
* @class MVNSampler
* @author taylor
* @file rv_samp.h
* @brief Can sample from a distribution with fixed mean and covariance, fixed mean only, fixed covariance only, or nothing fixed.
*/
template<size_t dim, typename float_t>
class MVNSampler : public rvsamp_base
{
public:

    /** type alias for linear algebra stuff */
    using Vec = Eigen::Matrix<float_t,dim,1>;
    /** type alias for linear algebra stuff */
    using Mat = Eigen::Matrix<float_t,dim,dim>;
    
    /**
     * @todo: implement move semantics 
     */

    /**
     * @brief Default-constructor sets up for multivariate standard Normal random variate generation.
     */
     MVNSampler();


     /**
      * @brief The user must supply both mean and covariance matrix.
      * @param meanVec a Vec for the mean vector of the sampling distribution.
      * @param covMat a Mat representing the covariance matrix of the samples.
      */
    MVNSampler(const Vec &meanVec, const Mat &covMat);


    /**
     * @brief sets the covariance matrix of the sampler.
     * @param covMat the desired covariance matrix.
     */
    void setCovar(const Mat &covMat);
    
    
    /**
     * @brief sets the mean vector of the sampler.
     * @param meanVec the desired mean vector.
     */
    void setMean(const Vec &meanVec);
    
        
     /**
      * @brief Draws a random vector.
      * @return a Vec random sample.
      */
    auto sample() -> Vec;    
    
private:

    /** @brief makes normal random variates */
    std::normal_distribution<float_t> m_z_gen;
    
    /** @brief covariance matrix */
    Mat m_scale_mat;
    
    /** @brief mean vector */
    Vec m_mean;
    
};


template<size_t dim, typename float_t>
MVNSampler<dim, float_t>::MVNSampler()
        : rvsamp_base()
        , m_z_gen(0.0, 1.0)
{
    setMean(Vec::Zero());
    setCovar(Mat::Identity());  
}


template<size_t dim, typename float_t>
MVNSampler<dim, float_t>::MVNSampler(const Vec &meanVec, const Mat &covMat)
    : rvsamp_base()
    , m_z_gen(0.0, 1.0)
{
    setCovar(covMat);
    setMean(meanVec);
}


template<size_t dim, typename float_t>
void MVNSampler<dim, float_t>::setCovar(const Mat &covMat)
{
    Eigen::SelfAdjointEigenSolver<Mat> eigenSolver(covMat);
    m_scale_mat = eigenSolver.eigenvectors() * eigenSolver.eigenvalues().cwiseMax(0).cwiseSqrt().asDiagonal();
}


template<size_t dim, typename float_t>
void MVNSampler<dim, float_t>::setMean(const Vec &meanVec)
{
    m_mean = meanVec;
}


template<size_t dim, typename float_t>
auto MVNSampler<dim, float_t>::sample() -> Vec
{
    Vec Z;
    for (size_t i=0; i< dim; ++i) 
    {
        Z(i) = m_z_gen(this->m_rng);
    }
    return m_mean + m_scale_mat * Z;
}



//! A class that performs sampling from a continuous uniform distribution.
/**
* @class UniformSampler
* @author taylor
* @file rv_samp.h
* @brief 
*/
template<typename float_t>
class UniformSampler : public rvsamp_base
{
public:

     /**
      * @brief The default constructor. Gives a lower bound of 0 and upper bound of 1.
      */
    UniformSampler();
    
    
     /**
      * @brief The constructor
      * @param lower the lower bound of the PRNG.
      * @param upper the upper bound of the PRNG.
      */
    UniformSampler(const float_t &lower, const float_t &upper);
    
    
     /**
      * @brief Draws a sample.
      * @return a sample of type float_t.
      */
    float_t sample();

private:

    /** @brief makes uniform random variates */
    std::uniform_real_distribution<float_t> m_unif_gen;

};


template<typename float_t>
UniformSampler<float_t>::UniformSampler() 
        : rvsamp_base()
        , m_unif_gen(0.0, 1.0)
{
}


template<typename float_t>
UniformSampler<float_t>::UniformSampler(const float_t &lower, const float_t &upper) 
        : rvsamp_base()
        , m_unif_gen(lower, upper)
{    
}


template<typename float_t>
float_t UniformSampler<float_t>::sample()
{
    return m_unif_gen(m_rng);
}



//! A class that performs sampling with replacement (useful for the index sampler in an APF)
/**
 * @class k_gen
 * @author taylor
 * @file rv_samp.h
 * @brief Basically a wrapper for std::discrete_distribution<>
 * outputs are in the rage (0,1,...N-1)
 */
template<size_t N, typename float_t>
class k_gen : public rvsamp_base
{
public:
    /**
     * @brief default constructor. only one available.
     */
    k_gen();

    
    /**
     * @brief sample N times from (0,1,...N-1) 
     * @param logWts possibly unnormalized type std::array<float_t, N>
     * @return the integers in a std::array<unsigned int, N>
     */
    std::array<unsigned int, N> sample(const std::array<float_t, N> &logWts);     
};


template<size_t N, typename float_t>
k_gen<N, float_t>::k_gen() : rvsamp_base() {}


template<size_t N, typename float_t>
std::array<unsigned int, N> k_gen<N, float_t>::sample(const std::array<float_t, N> &logWts)
{
    // these log weights may be very negative. If that's the case, exponentiating them may cause underflow
    // so we use the "log-exp-sum" trick
    // actually not quite...we just shift the log-weights because after they're exponentiated
    // they have the same normalized probabilities
    
   // Create the distribution with exponentiated log-weights
   // subtract the max first to prevent underflow
   // normalization is taken care of by std::discrete_distribution
    std::array<float_t, N> w;
    float_t m = *std::max_element(logWts.begin(), logWts.end());
    std::transform(logWts.begin(), logWts.end(), w.begin(), 
                   [&m](const float_t& d) -> float_t { return std::exp(d-m); } );
    std::discrete_distribution<> kGen(w.begin(), w.end());
    
    // sample and return ks
    std::array<unsigned int, N> ks;
    for(size_t i = 0; i < N; ++i){
        ks[i] = kGen(this->m_rng);
    }
    return ks;
}



} // namespace pf{
    
    
    
#endif // RV_SAMP_H
