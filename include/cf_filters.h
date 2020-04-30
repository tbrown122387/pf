#ifndef CF_FILTERS_H
#define CF_FILTERS_H

#include <Eigen/Dense> //linear algebra stuff
#include <math.h>       /* log */

#include "rv_eval.h"


//! Abstract Base Class for all closed-form filters.
/**
 * @class cf_filter
 * @author taylor
 * @file cf_filters.h
 * @brief forces structure on the closed-form filters.
 */
template<size_t dimstate, size_t dimobs, typename float_t>
class cf_filter{

public:
    
    /** "state size vector" type alias for linear algebra stuff */
    using ssv = Eigen::Matrix<float_t,dimstate,1>;
    /** "observation size vector" type alias for linear algebra stuff */
    using osv = Eigen::Matrix<float_t,dimstate,1>;
    
    /**
     * @brief The (virtual) destructor.
     */
    virtual ~cf_filter();
    
    
    /**
     * @brief returns the log of the most recent conditional likelihood
     * @return log p(y_t | y_{1:t-1}) or log p(y_1)
     */
    virtual float_t getLogCondLike() const = 0;
};


template<size_t dimstate, size_t dimobs, typename float_t>
cf_filter<dimstate,dimobs,float_t>::~cf_filter() {}


//! A class template for Kalman filtering.
/**
 * @class kalman
 * @author taylor
 * @file cf_filters.h
 * @brief Inherit from this for a model that admits Kalman filtering.
 */
template<size_t dimstate, size_t dimobs, size_t diminput, typename float_t>
class kalman : public cf_filter<dimstate, dimobs, float_t> {

public:    
    
    /** "state size vector" type alias for linear algebra stuff */
    using ssv = Eigen::Matrix<float_t,dimstate,1>;
    
    /** "observation size vector" type alias for linear algebra stuff */
    using osv = Eigen::Matrix<float_t,dimobs,1>;
    
    /** "input size vector" type alias for linear algebra stuff */
    using isv = Eigen::Matrix<float_t,diminput,1>;
    
    /** "state size matrix" type alias for linear algebra stuff */
    using ssMat = Eigen::Matrix<float_t,dimstate,dimstate>;
    
    /** "observation size matrix" type alias for linear algebra stuff */
    using osMat = Eigen::Matrix<float_t,dimobs,dimobs>;
    
    /** "state dim by input dimension matrix" */
    using siMat = Eigen::Matrix<float_t,dimstate,diminput>;
        
    /** "observation dimension by input dim matrix" */
    using oiMat = Eigen::Matrix<float_t,dimobs,diminput>;

    /** "observation dimension by state dimension -sized matrix" */
    using obsStateSizeMat = Eigen::Matrix<float_t,dimobs,dimstate>;    

    /** "state dimension by observation dimension matrix */
    using stateObsSizeMat = Eigen::Matrix<float_t,dimstate,dimobs>;


    //! Default constructor. 
    /**
     * @brief Need ths fir constructing default std::array<>s. Fills all vectors and matrices with zeros.
     */
    kalman();


    //! Non-default constructor.
    /**
     * @brief Non-default constructor.
     */
    kalman(const ssv &initStateMean, const ssMat &initStateVar);
    
    
    /**
     * @brief The (virtual) destructor
     */
    virtual ~kalman();
    

    /**
     * @brief returns the log of the latest conditional likelihood.
     * @return log p(y_t | y_{1:t-1}) or log p(y_1)
     */
    float_t getLogCondLike() const;
    
    
    /**
     * @brief Get the current filter mean.
     * @return E[x_t | y_{1:t}]
     */
    ssv getFiltMean() const;
    
    
    /**
     * @brief Get the current filter variance-covariance matrix.
     * @return V[x_t | y_{1:t}]
     */
    ssMat getFiltVar() const;
    

    /**
     * @brief get the one-step-ahead point forecast for y
     * @return E[y_{t+1} | y_{1:t}, params]
     */
    osv getPredYMean(const ssMat &stateTrans,
                     const obsStateSizeMat &obsMat, 
                     const siMat &stateInptAffector,
                     const oiMat &obsInptAffector, 
                     const isv &inputData) const;


    /**
     * @brief get the one-step-ahead forecast variance
     * @return V[y_{t+1} | y_{1:t}, params]
     */
    osMat getPredYVar(const ssMat &stateTrans,
                      const ssMat &cholStateVar,
                      const obsStateSizeMat &obsMat,
                      const osMat &cholObsVar) const;


    //! Perform a Kalman filter predict-and-update.
    /**
     * @param yt the new data point.
     * @param stateTrans the transition matrix of the state
     * @param cholStateVar the Cholesky Decomposition of the state noise covariance matrix.
     * @param stateInptAffector the matrix affecting how input data affects state transition.
     * @param inputData exogenous input data
     * @param obsMat the observation/emission matrix of the observation's conditional (on the state) distn.
     * @param obsInptAffector the matrix affecting how input data affects the observational distribution.
     * @param cholObsVar the Cholesky Decomposition of the observatio noise covariance matrix.
     */      
    void update(const osv &yt, 
                const ssMat &stateTrans, 
                const ssMat &cholStateVar, 
                const siMat &stateInptAffector, 
                const isv &inputData,
                const obsStateSizeMat &obsMat,
                const oiMat &obsInptAffector, 
                const osMat &cholObsVar);
                
private: 

    /** @brief predictive state mean */
    ssv m_predMean;
    
    /** @brief filter mean */
    ssv m_filtMean;
    
    /** @brief predictive var matrix */
    ssMat m_predVar;
    
    /** @brief filter var matrix */
    ssMat m_filtVar;
    
    /** @brief latest log conditional likelihood */
    float_t m_lastLogCondLike; 
    
    /** @brief has data been observed? */
    bool m_fresh;
    
    /** @brief pi */
    const float_t m_pi;
    
    /**
     * @todo handle diagonal variance matrices, and ensure symmetricness in other ways
     */

    /**
     * @brief Predicts the next state.
     * @param stateTransMat
     * @param cholStateVar
     * @param stateInptAffector
     * @param inputData
     */
    void updatePrior(const ssMat &stateTransMat, 
                     const ssMat &cholStateVar, 
                     const siMat &stateInptAffector, 
                     const isv &inputData);
                     
    
    /**
     * @brief Turns prediction into new filtering distribution.
     * @param yt
     * @param obsMat
     * @param obsInptAffector
     * @param inputData
     * @param cholObsVar
     */
    void updatePosterior(const osv &yt, 
                         const obsStateSizeMat &obsMat, 
                         const oiMat &obsInptAffector, 
                         const isv &inputData, 
                         const osMat &cholObsVar);
};


template<size_t dimstate, size_t dimobs, size_t diminput, typename float_t>  
kalman<dimstate,dimobs,diminput,float_t>::kalman() 
        : cf_filter<dimstate,dimobs,float_t>()
        , m_predMean(ssv::Zero())
        , m_predVar(ssMat::Zero()) 
        , m_fresh(true)
        , m_pi(3.14159265358979)
{
}
    

template<size_t dimstate, size_t dimobs, size_t diminput, typename float_t>  
kalman<dimstate,dimobs,diminput,float_t>::kalman(const ssv &initStateMean, const ssMat &initStateVar) 
        : cf_filter<dimstate,dimobs,float_t>()
        , m_predMean(initStateMean)
        , m_predVar(initStateVar) 
        , m_fresh(true)
        , m_pi(3.14159265358979)
{
}


template<size_t dimstate, size_t dimobs, size_t diminput, typename float_t>
kalman<dimstate,dimobs,diminput,float_t>::~kalman() {}


template<size_t dimstate, size_t dimobs, size_t diminput, typename float_t>
void kalman<dimstate,dimobs,diminput,float_t>::updatePrior(const ssMat &stateTransMat, 
                        const ssMat &cholStateVar, 
                        const siMat &stateInptAffector, 
                        const isv &inputData)
{
    ssMat Q = cholStateVar.transpose() * cholStateVar;
    m_predMean = stateTransMat * m_filtMean + stateInptAffector * inputData;
    m_predVar  = stateTransMat * m_filtVar * stateTransMat.transpose() + Q;
}


template<size_t dimstate, size_t dimobs, size_t diminput, typename float_t>
void kalman<dimstate,dimobs,diminput,float_t>::updatePosterior(const osv &yt, 
                             const obsStateSizeMat &obsMat, 
                             const oiMat &obsInptAffector, 
                             const isv &inputData, 
                             const osMat &cholObsVar)
{
    osMat R = cholObsVar.transpose() * cholObsVar; //obs
    osMat sigma = obsMat * m_predVar * obsMat.transpose() + R; // pred or APA' + R 
    osMat symSigma = (sigma.transpose() + sigma )/2.0; // ensure symmetric
    osMat siginv = symSigma.inverse();
    stateObsSizeMat K = m_predVar * obsMat.transpose() * siginv;
    osv obsPred = obsMat * m_predMean + obsInptAffector * inputData;
    osv innov = yt - obsPred;
    m_filtMean = m_predMean + K*innov;
    m_filtVar  = m_predVar - K * obsMat * m_predVar;

    // conditional likelihood stuff
    osMat quadForm = innov.transpose() * siginv * innov;
    osMat cholSig ( sigma.llt().matrixL() );
    float_t logDet = 2.0*cholSig.diagonal().array().log().sum();
    m_lastLogCondLike = -.5*innov.rows()*log(2*m_pi) - .5*logDet - .5*quadForm(0,0);
}


template<size_t dimstate, size_t dimobs, size_t diminput, typename float_t>
float_t kalman<dimstate,dimobs,diminput,float_t>::getLogCondLike() const
{
    return m_lastLogCondLike;
}


template<size_t dimstate, size_t dimobs, size_t diminput, typename float_t>
auto kalman<dimstate,dimobs,diminput,float_t>::getFiltMean() const -> ssv
{
    return m_filtMean;
}


template<size_t dimstate, size_t dimobs, size_t diminput, typename float_t>
auto kalman<dimstate,dimobs,diminput,float_t>::getFiltVar() const -> ssMat
{
    return m_filtVar;
}


template<size_t dimstate, size_t dimobs, size_t diminput, typename float_t>
void kalman<dimstate,dimobs,diminput,float_t>::update(const osv &yt, 
                                              const ssMat &stateTrans, 
                                              const ssMat &cholStateVar, 
                                              const siMat &stateInptAffector, 
                                              const isv &inData,
                                              const obsStateSizeMat &obsMat,
                                              const oiMat &obsInptAffector, 
                                              const osMat &cholObsVar)
{
    // this assumes that we have latent states x_{1:...} and y_{1:...} (NOT x_{0:...})
    // for that reason, we don't have to run updatePrior() on the first iteration
    if (m_fresh == true)
    {
        this->updatePosterior(yt, obsMat, obsInptAffector, inData, cholObsVar);
        m_fresh = false;
    }else 
    {
        this->updatePrior(stateTrans, cholStateVar, stateInptAffector, inData);
        this->updatePosterior(yt, obsMat, obsInptAffector, inData, cholObsVar);
    }
}
    
 
template<size_t dimstate, size_t dimobs, size_t diminput, typename float_t>
auto kalman<dimstate,dimobs,diminput,float_t>::getPredYMean(
        const ssMat &stateTrans,
        const obsStateSizeMat &obsMat, 
        const siMat &stateInptAffector,
        const oiMat &obsInptAffector, 
        const isv &futureInputData) const -> osv
{
    return obsMat * (stateTrans * m_filtMean + stateInptAffector * futureInputData) + obsInptAffector * futureInputData;
}


template<size_t dimstate, size_t dimobs, size_t diminput, typename float_t>
auto kalman<dimstate,dimobs,diminput,float_t>::getPredYVar(
        const ssMat &stateTrans,
        const ssMat &cholStateVar,
        const obsStateSizeMat &obsMat,
        const osMat &cholObsVar) const -> osMat
{
    return obsMat * (stateTrans * m_filtVar * stateTrans.transpose() + cholStateVar.transpose()*cholStateVar) * obsMat.transpose() + cholObsVar.transpose() * cholObsVar;
}


//! A class template for HMM filtering.
/**
 * @class hmm
 * @author taylor
 * @file cf_filters.h
 * @brief Inherit from this for a model that admits HMM filtering.
 */
template<size_t dimstate, size_t dimobs, typename float_t>
class hmm : public cf_filter<dimstate,dimobs,float_t>
{

public:

    /** @brief "state size vector" */
    using ssv = Eigen::Matrix<float_t,dimstate,1>;
    
    /** @brief "observation size vector" */
    using osv = Eigen::Matrix<float_t,dimobs,1>;
    
    /** @brief "state size matrix" */
    using ssMat = Eigen::Matrix<float_t,dimstate,dimstate>;


    //! Default constructor. 
    /**
     * @brief Need ths fir constructing default std::array<>s. Fills all vectors and matrices with zeros.
     */
    hmm();


    //! Constructor
    /**
     * @brief allows specification of initstate distn and transition matrix.
     * @param initStateDistr first time state prior distribution.
     * @param transMat time homogeneous transition matrix.
    */
    hmm(const ssv &initStateDistr, const ssMat &transMat);
    
    
    /**
     * @brief The (virtual) desuctor.
     */
    virtual ~hmm();
    

    //! Get the latest conditional likelihood.
    /**
     * @return the latest conditional likelihood.
     */  
    float_t getLogCondLike() const;
    
    
    //! Get the current filter vector.
    /**
     * @brief get the current filter vector.
     * @return a probability vector p(x_t | y_{1:t})
     */
    ssv getFilterVec() const;
    
        
    //! Perform a HMM filter update.
    /**
     * @brief Perform a HMM filter update.
     * @param condDensVec the vector (in x_t) of p(y_t|x_t)
     */
    void update(const ssv &condDensVec);


private:

    /** @brief filter vector */
    ssv m_filtVec;
    
    /** @brief transition matrix */
    ssMat m_transMatTranspose;
    
    /** @brief last conditional likelihood */
    float_t m_lastCondLike; 
    
    /** @brief has data been observed? */
    bool m_fresh;

};


template<size_t dimstate, size_t dimobs, typename float_t>
hmm<dimstate,dimobs,float_t>::hmm() 
    : cf_filter<dimstate,dimobs,float_t>::cf_filter()
    , m_filtVec(ssv::Zero())
    , m_transMatTranspose(ssMat::Zero())
    , m_lastCondLike(0.0)
    , m_fresh(true)
{
}
    

template<size_t dimstate, size_t dimobs, typename float_t>
hmm<dimstate,dimobs,float_t>::hmm(const ssv &initStateDistr, const ssMat &transMat) 
    : cf_filter<dimstate,dimobs,float_t>()
    , m_filtVec(initStateDistr)
    , m_transMatTranspose(transMat.transpose())
    , m_lastCondLike(0.0)
    , m_fresh(true)
{
}


template<size_t dimstate, size_t dimobs, typename float_t>
hmm<dimstate,dimobs,float_t>::~hmm() {}


template<size_t dimstate, size_t dimobs, typename float_t>
auto hmm<dimstate,dimobs,float_t>::getLogCondLike() const -> float_t
{
    return std::log(m_lastCondLike);
}


template<size_t dimstate, size_t dimobs, typename float_t>
auto hmm<dimstate,dimobs,float_t>::getFilterVec() const -> ssv
{
    return m_filtVec;
}


template<size_t dimstate, size_t dimobs, typename float_t>
void hmm<dimstate,dimobs,float_t>::update(const ssv &condDensVec)
{
    if (m_fresh)  // hasn't seen data before and so filtVec is just time 1 state prior
    {
        m_filtVec = m_filtVec.cwiseProduct( condDensVec ); // now it's p(x_1, y_1)
        m_lastCondLike = m_filtVec.sum();
        m_filtVec /= m_lastCondLike;
        m_fresh = false;
        
    } else { // has seen data before
        m_filtVec = m_transMatTranspose * m_filtVec; // now p(x_t |y_{1:t-1})
        m_filtVec = m_filtVec.cwiseProduct( condDensVec ); // now p(y_t,x_t|y_{1:t-1})
        m_lastCondLike = m_filtVec.sum();
        m_filtVec /= m_lastCondLike; // now p(x_t|y_{1:t})
    }
}
    







//! A class template for Gamma filtering.
/**
 * @class gamFilter
 * @author taylor
 * @file cf_filters.h
 * @brief Inherit from this for a model that admits Gamma filtering.
 */
template<size_t dim_pred, typename float_t>
class gamFilter : public cf_filter<1,1,float_t>
{

public:

    /** @brief "predictor size vector" */
    using psv = Eigen::Matrix<float_t,dim_pred,1>;
    
    /** @brief "two by 1 vector" */
    using tsv = Eigen::Matrix<float_t,2,1>;


    //! Default constructor. 
    /**
     * @brief Need ths fir constructing default std::array<>s. Fills all vectors and matrices with zeros.
     */
    //gamFilter();


    //! Constructor
    /**
     * @brief 
     * @param nOneTilde degrees of freedom for time 1 prior.
     * @param dOneTilde rate parameter for time 1 prior.
    */
    gamFilter(const float_t &nOneTilde, const float_t &dOneTilde);
    
    
    /**
     * @brief The (virtual) desuctor.
     */
    virtual ~gamFilter();
    

    //! Get the latest conditional likelihood.
    /**
     * @return the latest conditional likelihood.
     */  
    float_t getLogCondLike() const;
    
    
    //! Get the current filter vector.
    /**
     * @brief get the current filtering distribution. First element is the shape, second is the rate.
     * @returns a vector of the shape and rate parameters of f(p_t | y_{1:t})
     */
    tsv getFilterVec() const;
    
        
    //! Perform a filtering update.
    /**
     * @brief Perform a Gamma filter update.
     * @param yt the most recent dependent random variable
     * @param xt the most recent predictor vector
     * @param beta the beta vector
     * @param sigmaSquared the observation variance scale parameter.
     * @param delta between 0 and 1 the discount parameter
     */
    void update(const float_t& yt, const psv &xt, const psv& beta, const float_t& sigmaSquared, const float_t& delta);


private:

    /** @brief filter vector (shape and rate) */
    tsv m_filtVec;
    
    /** @brief last log of the conditional likelihood */
    float_t m_lastLogCondLike; 
    
    /** @brief has data been observed? */
    bool m_fresh;

};


//template<typename dim_pred, typename float_t>
//gamFilter<dim_pred,float_t>::gamFilter() 
//    : cf_filter<1,1,float_t>::cf_filter()
//    , m_filtVec(tsv::Zero())
//    , m_lastCondLike(0.0)
//    , m_fresh(true)
//{
//}
    

template<size_t dim_pred, typename float_t>
gamFilter<dim_pred,float_t>::gamFilter(const float_t &nOneTilde, const float_t &dOneTilde)
    : cf_filter<1,1,float_t>()
    , m_lastLogCondLike(0.0)
    , m_fresh(true)
{
    m_filtVec(0) = nOneTilde;
    m_filtVec(1) = dOneTilde;
}


template<size_t dim_pred, typename float_t>
gamFilter<dim_pred,float_t>::~gamFilter() {}


template<size_t dim_pred, typename float_t>
auto gamFilter<dim_pred,float_t>::getLogCondLike() const -> float_t
{
    return m_lastLogCondLike;
}


template<size_t dim_pred, typename float_t>
auto gamFilter<dim_pred,float_t>::getFilterVec() const -> tsv
{
    return m_filtVec;
}



template<size_t dim_pred, typename float_t>
void gamFilter<dim_pred,float_t>::update(const float_t& yt, const psv &xt, const psv& beta, const float_t& sigmaSquared, const float_t& delta)
{

    if(sigmaSquared <= 0 || delta <= 0)
        throw std::invalid_argument("ME: both sigma squared and delta have to be positive.\n");

    if (m_fresh)  // hasn't seen data before and so filtVec is just time 1 state prior
    {
        float_t tmpScale = std::sqrt(sigmaSquared*m_filtVec(1)/m_filtVec(0));
        m_lastLogCondLike = rveval::evalScaledT<float_t>(yt, xt.dot(beta), tmpScale, m_filtVec(0), true);
        m_filtVec(0) += 1;
        m_filtVec(1) += (yt - xt.dot(beta))*(yt - xt.dot(beta))/sigmaSquared;
        m_fresh = false;
        
    } else { // has seen data before
        
        m_filtVec(0) *= delta;
        m_filtVec(1) *= delta;
        float_t tmpScale = std::sqrt(sigmaSquared*m_filtVec(1)/m_filtVec(0));
        m_lastLogCondLike = rveval::evalScaledT<float_t>(yt, xt.dot(beta), tmpScale, m_filtVec(0), true);
        m_filtVec(0) += 1;
        m_filtVec(1) += (yt - xt.dot(beta))*(yt - xt.dot(beta))/sigmaSquared;
    }
}
 

//! Another class template for Gamma filtering, but this time
// it's for a multivariate response.
/**
 * @class multivGamFilter
 * @author taylor
 * @file cf_filters.h
 * @brief Inherit from this for a model that admits Gamma filtering.
 */
template<size_t dim_obs, size_t dim_pred, typename float_t>
class multivGamFilter : public cf_filter<1,dim_obs,float_t>
{

public:

    /** @brief "predictor size vector" */
    using psv = Eigen::Matrix<float_t,dim_pred,1>;
    
    /** @brief "beta size matrix" */
    using bsm = Eigen::Matrix<float_t,dim_obs, dim_pred>;

    /** @brief "two by 1 vector" to store size and shapes of gamma distributions */
    using tsv = Eigen::Matrix<float_t,2,1>;

    /** @brief "observation size vector"  */
    using osv = Eigen::Matrix<float_t, dim_obs, 1>;

    /** @brief "observation size matrix" */
    using osm = Eigen::Matrix<float_t, dim_obs, dim_obs>;


    //! Constructor
    /**
     * @brief 
     * @param nOneTilde degrees of freedom for time 1 prior.
     * @param dOneTilde rate parameter for time 1 prior.
    */
    multivGamFilter(const float_t &nOneTilde, const float_t &dOneTilde);
    
    
    /**
     * @brief The (virtual) desuctor.
     */
    virtual ~multivGamFilter();
    

    //! Get the latest conditional likelihood.
    /**
     * @return the latest conditional likelihood.
     */  
    float_t getLogCondLike() const;
    
    
    //! Get the current filter vector.
    /**
     * @brief get the current filtering distribution. First element is the shape, second is the rate.
     * @returns a vector of the shape and rate parameters of f(p_t | y_{1:t})
     */
    tsv getFilterVec() const;
    
        
    //! Perform a filtering update.
    /**
     * @brief Perform a Gamma filter update.
     * @param yt the most recent dependent random variable
     * @param xt the most recent predictor vector
     * @param B the loadings matrix
     * @param Sigma the observation "shape" matrix.
     * @param delta between 0 and 1 the discount parameter
     */
    void update(const osv& yt, const psv &xt, const bsm& B, const osm& Sigma, const float_t& delta);


    //! Get the forecast mean (assuming filtering has been performed already)
    /**
     * @brief gets the forecast mean!
     * @param xtp1 the next time period's predictor vector
     * @param B the loadings matrix
     * @param Sigma the observation "shape" matrix 
     * @param delta between 0 and 1 the discount parameter
     * @return a mean vector
     */
    osv getFcastMean(const psv &xtp1, const bsm& B, const osm& Sigma, const float_t& delta);


    //! Get the forecast covariance matrix (assuming filtering has been performed already)
    /**
     * @brief gets the forecast covariance matrix!
     * @param xtp1 the next time period's predictor vector
     * @param B the loadings matrix
     * @param Sigma the observation "shape" matrix 
     * @param delta between 0 and 1 the discount parameter
     * @return a forecast covariance matrix
     */
    osm getFcastCov(const psv &xtp1, const bsm& B, const osm& Sigma, const float_t& delta);


private:

    /** @brief filter vector (shape and rate) */
    tsv m_filtVec;
    
    /** @brief last log of the conditional likelihood */
    float_t m_lastLogCondLike; 
    
    /** @brief has data been observed? */
    bool m_fresh;

};


template<size_t dim_obs, size_t dim_pred, typename float_t>
multivGamFilter<dim_obs,dim_pred,float_t>::multivGamFilter(const float_t &nOneTilde, const float_t &dOneTilde)
    : cf_filter<1,dim_obs,float_t>()
    , m_lastLogCondLike(0.0)
    , m_fresh(true)
{
    m_filtVec(0) = nOneTilde;
    m_filtVec(1) = dOneTilde;
}


template<size_t dim_obs, size_t dim_pred, typename float_t>
multivGamFilter<dim_obs,dim_pred,float_t>::~multivGamFilter() {}


template<size_t dim_obs, size_t dim_pred, typename float_t>
auto multivGamFilter<dim_obs,dim_pred,float_t>::getLogCondLike() const -> float_t
{
    return m_lastLogCondLike;
}


template<size_t dim_obs, size_t dim_pred, typename float_t>
auto multivGamFilter<dim_obs,dim_pred,float_t>::getFilterVec() const -> tsv
{
    return m_filtVec;
}



template<size_t dim_obs, size_t dim_pred, typename float_t>
void multivGamFilter<dim_obs,dim_pred,float_t>::update(const osv& yt, const psv &xt, const bsm& B, const osm& Sigma, const float_t& delta)
{

    // TODO: doesn't check that Sigma is positive definite or symmetric!
    if(delta <= 0)
        throw std::invalid_argument("ME: delta has to be positive (you're not even checking Sigma).\n");

    if (m_fresh)  // hasn't seen data before and so filtVec is just time 1 state prior
    {
        osm scaleMat = Sigma * m_filtVec(1)/m_filtVec(0);
        osv modeVec = B*xt;
        m_lastLogCondLike = rveval::evalMultivT<dim_obs,float_t>(yt, modeVec, scaleMat, m_filtVec(0), true);
        m_filtVec(0) += 1; 
        m_filtVec(1) += (Sigma.ldlt().solve(yt - modeVec)).squaredNorm();
        m_fresh = false;
        
    } else { // has seen data before
        
        m_filtVec(0) *= delta;
        m_filtVec(1) *= delta;
        
        osm scaleMat = Sigma * m_filtVec(1)/m_filtVec(0);
        osv modeVec = B*xt;
        m_lastLogCondLike = rveval::evalMultivT<dim_obs,float_t>(yt, modeVec, scaleMat, m_filtVec(0), true);
        
        m_filtVec(0) += 1;
        m_filtVec(1) += (Sigma.ldlt().solve(yt - modeVec)).squaredNorm();
    }
}
 

template<size_t dim_obs, size_t dim_pred, typename float_t>
auto multivGamFilter<dim_obs,dim_pred,float_t>::getFcastMean(const psv &xtp1, const bsm& B, const osm& Sigma, const float_t& delta) -> osv
{
    if(delta*m_filtVec(0) > 1.0)
        return B*xtp1;
}


template<size_t dim_obs, size_t dim_pred, typename float_t>
auto multivGamFilter<dim_obs,dim_pred,float_t>::getFcastCov(const psv &xtp1, const bsm& B, const osm& Sigma, const float_t& delta) -> osm
{
    if(delta * m_filtVec(0) > 2.0)
        return Sigma * delta * m_filtVec(1) / (delta * m_filtVec(0) - 2.0);
}




#endif //CF_FILTERS_H
