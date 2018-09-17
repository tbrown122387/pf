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
template<size_t dim = 1>
class rvsamp_base
{
public:

    /** "state size vector" type alias for linear algebra stuff */
    using ssv = Eigen::Matrix<double,dim,1>;

    /**
     * @brief The default constructor. This is the only option available. Sets the seed with the clock. 
     */
    rvsamp_base();

protected:

    /** @brief prng */
    std::mt19937 m_rng;

};


template<size_t dim>
rvsamp_base<dim>::rvsamp_base() 
    : m_rng{static_cast<std::uint32_t>(std::chrono::high_resolution_clock::now().time_since_epoch().count())} 
{
}


//! A class that performs sampling from a univariate Normal distribution.
/**
* @class UnivNormSampler
* @author taylor
* @file rv_samp.h
* @brief Samples from univariate Normal distribution.
*/
class UnivNormSampler : public rvsamp_base<1>
{
    
public:


    /**
     * @brief Default-constructor sets up for standard Normal random variate generation.
     */
    UnivNormSampler();


     /**
      * @brief The user must supply both mean and std. dev.
      * @param mu a double for the mean of the sampling distribution.
      * @param sigma a double (> 0) representing the standard deviation of the samples.
      */
    UnivNormSampler(const double &mu, const double &sigma);


    /**
     * @brief sets the standard deviation of the sampler.
     * @param sigma the desired standard deviation.
     */
    void setStdDev(const double &sigma);
    
    
    /**
     * @brief sets the mean of the sampler.
     * @param mu the desired mean.
     */
    void setMean(const double &mu);
    
        
     /**
      * @brief Draws a random number.
      * @return a random sample of type double.
      */
    double sample();    
    

private:

    /** @brief makes normal random variates */
    std::normal_distribution<> m_z_gen;
    
    /** @brief the mean */
    double m_mu;
    
    /** @brief the standard deviation */
    double m_sigma;

};


//! A class that performs sampling from a univariate Bernoulli distribution.
/**
* @class BernSampler
* @author taylor
* @file rv_samp.h
* @brief Samples from univariate Bernoulli distribution.
*/
class BernSampler : public rvsamp_base<1>
{
    
public:


    /**
     * @brief Default-constructor sets up for Bernoulli random variate generation with p = .5.
     */
    BernSampler();


     /**
      * @brief Constructs Bernoulli sampler with user-specified p.
      * @param p a double for the probability that the rv equals 1.
      */
    BernSampler(const double &p);


    /**
     * @brief sets the parameter p.
     * @param p the p(X=1) = 1-p(X=0).
     */
    void setP(const double &p);
    
    /** 
      * @brief Draws a random number.
      * @return a random sample of type double.
      */
    int sample();    
    

private:

    /** @brief makes normal random variates */
    std::bernoulli_distribution m_B_gen;
    
    /** @brief the mean */
    double m_p;
};




//! A class that performs sampling from a multivariate normal distribution.
/**
* @class MVNSampler
* @author taylor
* @file rv_samp.h
* @brief Can sample from a distribution with fixed mean and covariance, fixed mean only, fixed covariance only, or nothing fixed.
*/
template<size_t dim>
class MVNSampler : public rvsamp_base<dim>
{
public:

    /** type alias for linear algebra stuff */
    using Vec = Eigen::Matrix<double,dim,1>;
    /** type alias for linear algebra stuff */
    using Mat = Eigen::Matrix<double,dim,dim>;
    
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
    std::normal_distribution<> m_z_gen;
    
    /** @brief covariance matrix */
    Mat m_scale_mat;
    
    /** @brief mean vector */
    Vec m_mean;
    
};


template<size_t dim>
MVNSampler<dim>::MVNSampler()
        : rvsamp_base<dim>()
        , m_z_gen(0.0, 1.0)
{
    setMean(Vec::Zero());
    setCovar(Mat::Identity());  
}


template<size_t dim>
MVNSampler<dim>::MVNSampler(const Vec &meanVec, const Mat &covMat)
    : rvsamp_base<dim>()
    , m_z_gen(0.0, 1.0)
{
    setCovar(covMat);
    setMean(meanVec);
}


template<size_t dim>
void MVNSampler<dim>::setCovar(const Mat &covMat)
{
    Eigen::SelfAdjointEigenSolver<Mat> eigenSolver(covMat);
    m_scale_mat = eigenSolver.eigenvectors() * eigenSolver.eigenvalues().cwiseMax(0).cwiseSqrt().asDiagonal();
}


template<size_t dim>
void MVNSampler<dim>::setMean(const Vec &meanVec)
{
    m_mean = meanVec;
}


template<size_t dim>
auto MVNSampler<dim>::sample() -> Vec
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
class UniformSampler : public rvsamp_base<1>
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
    UniformSampler(const double &lower, const double &upper);
    
    
     /**
      * @brief Draws a sample.
      * @return a sample of type double.
      */
    double sample();

private:

    /** @brief makes uniform random variates */
    std::uniform_real_distribution<> m_unif_gen;

};



//! A class that performs sampling with replacement (useful for the index sampler in an APF)
/**
 * @class k_gen
 * @author taylor
 * @file rv_samp.h
 * @brief Basically a wrapper for std::discrete_distribution<>
 * outputs are in the rage (0,1,...N-1)
 */
template<size_t N>
class k_gen : public rvsamp_base<N>
{
public:
    /**
     * @brief default constructor. only one available.
     */
    k_gen();

    
    /**
     * @brief sample N times from (0,1,...N-1) 
     * @param logWts possibly unnormalized type std::array<double, N>
     * @return the integers in a std::array<unsigned int, N>
     */
    std::array<unsigned int, N> sample(const std::array<double, N> &logWts);     
};


template<size_t N>
k_gen<N>::k_gen() : rvsamp_base<N>() {}


template<size_t N>
std::array<unsigned int, N> k_gen<N>::sample(const std::array<double, N> &logWts)
{
    // these log weights may be very negative. If that's the case, exponentiating them may cause underflow
    // so we use the "log-exp-sum" trick
    // actually not quite...we just shift the log-weights because after they're exponentiated
    // they have the same normalized probabilities
    
   // Create the distribution with exponentiated log-weights
   // subtract the max first to prevent underflow
   // normalization is taken care of by std::discrete_distribution
    std::array<double, N> w;
    double m = *std::max_element(logWts.begin(), logWts.end());
    std::transform(logWts.begin(), logWts.end(), w.begin(), 
                   [&m](const double& d) -> double { return std::exp(d-m); } );
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
