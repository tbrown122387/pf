#ifndef CF_FILTERS_H
#define CF_FILTERS_H

#include <Eigen/Dense> //linear algebra stuff
#include <math.h>       /* log */


//! Abstract Base Class for Kalman filter and HMM filter.
/**
 * @class cf_filter
 * @author taylor
 * @file cf_filters.h
 * @brief forces structure on the closed-form filters.
 */
template<size_t dimstate, size_t dimobs>
class cf_filter{

public:
    
    /** "state size vector" type alias for linear algebra stuff */
    using ssv = Eigen::Matrix<double,dimstate,1>;
    /** "observation size vector" type alias for linear algebra stuff */
    using osv = Eigen::Matrix<double,dimstate,1>;
    
    /**
     * @brief The (virtual) destructor.
     */
    virtual ~cf_filter();
    
    
    /**
     * @brief returns the log of the most recent conditional likelihood
     * @return log p(y_t | y_{1:t-1}) or log p(y_1)
     */
    virtual double getLogCondLike() const = 0;
};


template<size_t dimstate, size_t dimobs>
cf_filter<dimstate,dimobs>::~cf_filter() {}


//! A class template for Kalman filtering.
/**
 * @class kalman
 * @author taylor
 * @file cf_filters.h
 * @brief Inherit from this for a model that admits Kalman filtering.
 */
template<size_t dimstate, size_t dimobs, size_t diminput>
class kalman{
    
    /** "state size vector" type alias for linear algebra stuff */
    using ssv = Eigen::Matrix<double,dimstate,1>;
    
    /** "observation size vector" type alias for linear algebra stuff */
    using osv = Eigen::Matrix<double,dimobs,1>;
    
    /** "input size vector" type alias for linear algebra stuff */
    using isv = Eigen::Matrix<double,diminput,1>;
    
    /** "state size matrix" type alias for linear algebra stuff */
    using ssMat = Eigen::Matrix<double,dimstate,dimstate>;
    
    /** "observation size matrix" type alias for linear algebra stuff */
    using osMat = Eigen::Matrix<double,dimobs,dimobs>;
    
    /** "state dim by input dimension matrix" */
    using siMat = Eigen::Matrix<double,dimstate,diminput>;
        
    /** "observation dimension by input dim matrix" */
    using oiMat = Eigen::Matrix<double,dimobs,diminput>;
    

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
    double getLogCondLike() const;
    
    
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
                const osMat &obsMat,
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
    double m_lastLogCondLike; 
    
    /** @brief has data been observed? */
    bool m_fresh;
    
    /** @brief pi */
    const double m_pi;
    
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
                         const osMat &obsMat, 
                         const oiMat &obsInptAffector, 
                         const isv &inputData, 
                         const osMat &cholObsVar);
};


template<size_t dimstate, size_t dimobs, size_t diminput>  
kalman<dimstate,dimobs,diminput>::kalman() 
        : cf_filter<dimstate,dimobs>()
        , m_fresh(true)
        , m_predMean(ssv::Zero())
        , m_predVar(ssMat::Zero()) 
        , m_pi(3.14159265358979)
{
}
    

template<size_t dimstate, size_t dimobs, size_t diminput>  
kalman<dimstate,dimobs,diminput>::kalman(const ssv &initStateMean, const ssMat &initStateVar) 
        : cf_filter<dimstate,dimobs>()
        , m_fresh(true)
        , m_predMean(initStateMean)
        , m_predVar(initStateVar) 
        , m_pi(3.14159265358979)
{
}


template<size_t dimstate, size_t dimobs, size_t diminput>
kalman<dimstate,dimobs,diminput>::~kalman() {}


template<size_t dimstate, size_t dimobs, size_t diminput>
void kalman<dimstate,dimobs,diminput>::updatePrior(const ssMat &stateTransMat, 
                        const ssMat &cholStateVar, 
                        const siMat &stateInptAffector, 
                        const isv &inputData)
{
    ssMat Q = cholStateVar.transpose() * cholStateVar;
    m_predMean = stateTransMat * m_filtMean + stateInptAffector * inputData;
    m_predVar  = stateTransMat * m_filtVar * stateTransMat.transpose() + Q;
}


template<size_t dimstate, size_t dimobs, size_t diminput>
void kalman<dimstate,dimobs,diminput>::updatePosterior(const osv &yt, 
                             const osMat &obsMat, 
                             const oiMat &obsInptAffector, 
                             const isv &inputData, 
                             const osMat &cholObsVar)
{
    osMat R = cholObsVar.transpose() * cholObsVar; //obs
    osMat sigma = obsMat * m_predVar * obsMat.transpose() + R; // pred or APA' + R 
    osMat symSigma = (sigma.transpose() + sigma )/2.0; // ensure symmetric
    osMat siginv = symSigma.inverse();
    ssMat K = m_predVar * obsMat.transpose() * siginv;
    osv obsPred = obsMat * m_predMean + obsInptAffector * inputData;
    osv innov = yt - obsPred;
    m_filtMean = m_predMean + K*innov;
    m_filtVar  = m_predVar - K * obsMat * m_predVar;

    // conditional likelihood stuff
    osMat quadForm = innov.transpose() * siginv * innov;
    osMat cholSig ( sigma.llt().matrixL() );
    double logDet = 2.0*cholSig.diagonal().array().log().sum();
    m_lastLogCondLike = -.5*innov.rows()*log(2*m_pi) - .5*logDet - .5*quadForm(0,0);
}


template<size_t dimstate, size_t dimobs, size_t diminput>
double kalman<dimstate,dimobs,diminput>::getLogCondLike() const
{
    return m_lastLogCondLike;
}


template<size_t dimstate, size_t dimobs, size_t diminput>
auto kalman<dimstate,dimobs,diminput>::getFiltMean() const -> ssv
{
    return m_filtMean;
}


template<size_t dimstate, size_t dimobs, size_t diminput>
auto kalman<dimstate,dimobs,diminput>::getFiltVar() const -> ssMat
{
    return m_filtVar;
}


template<size_t dimstate, size_t dimobs, size_t diminput>
void kalman<dimstate,dimobs,diminput>::update(const osv &yt, 
                                              const ssMat &stateTrans, 
                                              const ssMat &cholStateVar, 
                                              const siMat &stateInptAffector, 
                                              const isv &inData,
                                              const osMat &obsMat,
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
    
    
//! A class template for HMM filtering.
/**
 * @class hmm
 * @author taylor
 * @file cf_filters.h
 * @brief Inherit from this for a model that admits HMM filtering.
 */
template<size_t dimstate, size_t dimobs>
class hmm : public cf_filter<dimstate,dimobs>
{

public:

    /** @brief "state size vector" */
    using ssv = Eigen::Matrix<double,dimstate,1>;
    
    /** @brief "observation size vector" */
    using osv = Eigen::Matrix<double,dimobs,1>;
    
    /** @brief "state size matrix" */
    using ssMat = Eigen::Matrix<double,dimstate,dimstate>;


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
    double getLogCondLike() const;
    
    
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
    double m_lastCondLike; 
    
    /** @brief has data been observed? */
    bool m_fresh;

};


template<size_t dimstate, size_t dimobs>
hmm<dimstate,dimobs>::hmm() 
    : cf_filter<dimstate,dimobs>()
    , m_filtVec(ssv::Zero())
    , m_transMatTranspose(ssMat::Zero())
    , m_lastCondLike(0.0)
    , m_fresh(false)
{
}
    

template<size_t dimstate, size_t dimobs>
hmm<dimstate,dimobs>::hmm(const ssv &initStateDistr, const ssMat &transMat) 
    : cf_filter<dimstate,dimobs>()
    , m_filtVec(initStateDistr)
    , m_transMatTranspose(transMat.transpose())
    , m_lastCondLike(0.0)
    , m_fresh(false)
{
}


template<size_t dimstate, size_t dimobs>
hmm<dimstate,dimobs>::~hmm() {}


template<size_t dimstate, size_t dimobs>
double hmm<dimstate,dimobs>::getLogCondLike() const
{
    return std::log(m_lastCondLike);
}


template<size_t dimstate, size_t dimobs>
auto hmm<dimstate,dimobs>::getFilterVec() const -> ssv
{
    return m_filtVec;
}


template<size_t dimstate, size_t dimobs>
void hmm<dimstate,dimobs>::update(const ssv &condDensVec)
{
    if (!m_fresh)  // hasn't seen data before and so filtVec is just time 1 state prior
    {
        m_filtVec = m_filtVec.cwiseProduct( condDensVec ); // now it's p(x_1, y_1)
        m_lastCondLike = m_filtVec.sum();
        m_filtVec /= m_lastCondLike;
        m_fresh = true;
        
    } else { // has seen data before
        m_filtVec = m_transMatTranspose * m_filtVec; // now p(x_t |y_{1:t-1})
        m_filtVec = m_filtVec.cwiseProduct( condDensVec ); // now p(y_t,x_t|y_{1:t-1})
        m_lastCondLike = m_filtVec.sum();
        m_filtVec /= m_lastCondLike; // now p(x_t|y_{1:t})
    }
}
    

#endif //CF_FILTERS_H