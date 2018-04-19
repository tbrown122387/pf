#ifndef RV_EVAL_H
#define RV_EVAL_H

#include <cstddef> // std::size_t
#include <Eigen/Dense>


namespace pf{

namespace rveval{
    
////////////////////////////////////////////////
/////////         Constants            /////////
////////////////////////////////////////////////

/** (2 pi)^(-1/2) */
const double inv_sqrt_2pi(0.3989422804014327);

/** (2/pi)^(1/2) */
const double sqrt_two_over_pi(0.797884560802865);

/** log(2pi) */
const double log_two_pi (1.83787706640935);

/** log(2/pi) */
const double log_two_over_pi (-0.451582705289455);


////////////////////////////////////////////////
/////////      Transformations         /////////
////////////////////////////////////////////////

/**
 * @brief Maps (-1, 1) to the reals.
 * @param phi
 * @return psi
 */
double twiceFisher(const double &phi);


/**
 * @brief Maps a real number to the itnerval (-1,1).
 * @param psi
 * @return phi
 */
double invTwiceFisher(const double &psi);
    
    
/**
 * @brief Maps (0,1) to the reals.
 * @param p
 * @return logit(p)
 */
double logit(const double &p);
    
    
/**
 * @brief Maps the reals to (0,1)
 * @param r
 * @return p = invlogit(p)
 */
double inv_logit(const double &r);


////////////////////////////////////////////////
/////////       double evals           /////////
////////////////////////////////////////////////
    
/**
 * @brief Evaluates the univariate Normal density.
 * @param x the point at which you're evaluating.
 * @param mu the mean.
 * @param sigma the standard deviation.
 * @param log true if you want the log-density. False otherwise.
 * @return a double evaluation.
 */
double evalUnivNorm(const double &x, const double &mu, const double &sigma, bool log = false);


/**
 * @brief Evaluates the standard Normal CDF.
 * @param x the quantile.
 * @return the probability Z < x
 */
double evalUnivStdNormCDF(const double &x); 


/**
 * @brief Evaluates the univariate Beta density
 * @param x the point
 * @param alpha parameter 1 
 * @param beta parameter 2
 * @param log true if you want log density
 * @return double evaluation.
*/       
double evalUnivBeta(const double &x, const double &alpha, const double &beta, bool log = false);


/**
 * @brief Evaluates the univariate Inverse Gamma density
 * @param x the point
 * @param alpha shape parameter  
 * @param beta rate parameter 
 * @param log true if you want log density.
 * @return double evaluation.
*/       
double evalUnivInvGamma(const double &x, const double &alpha, const double &beta, bool log = false);


/**
 * @brief Evaluates the half-normal density
 * @param x the point you're evaluating at
 * @param sigmaSqd the scale parameter
 * @param log true if you want log density.
 * @return double evaluation.
 */
double evalUnivHalfNorm(const double &x, const double &sigmaSqd, bool log = false);


/**
 * @brief Evaluates a truncated Normal density.
 * @param x the quantile
 * @param mu the mode
 * @param sigma the scale parameter.
 * @param lower the lower truncation point (may be negative infinity)
 * @param upper the upper truncation point (may be positive infinity).
 * @param log true if you want the log density.
 * @return 
 */
double evalUnivTruncNorm(const double &x, const double &mu, const double &sigma, const double &lower, const double &upper, bool log = false);


/**
 * @brief Evaluates the logit-Normal distribution (see Wiki for more info)
 * @param x in [0,1] the point you're evaluating at
 * @param mu location parameter that can take any real number
 * @param sigma scale parameter that needs to be positive
 * @param log true if you want to evalute the log-density. False otherwise.
 * @return a double evaluation
 */
double evalLogitNormal(const double &x, const double &mu, const double &sigma, bool log = false);


/**
 * @brief Evaluates what I call the "twice-fisher-Normal" distribution
 * https://stats.stackexchange.com/questions/321905/what-is-the-name-of-this-random-variable/321907#321907
 * @param x in [-1,1] the point you are evaluating at
 * @param mu the location parameter (all real numbers)
 * @param sigma the scale parameter (positive)
 * @param log true if you want to evaluate the log-density. False otherwise.
 * @return a double evaluation
 */
double evalTwiceFisherNormal(const double &x, const double &mu, const double &sigma, bool log = false);


/**
 * @brief Evaluates the lognormal density
 * @param x in (0,infty) the point you are evaluating at
 * @param mu the location parameter
 * @param sigma in (0, infty) the scale parameter
 * @param log true if you want to evaluate the log-density. False otherwise.
 * @return a double evaluation
 */
double evalLogNormal(const double &x, const double &mu, const double &sigma, bool log = false);


/**
 * @brief Evaluates the uniform density.
 * @param x in (lower, upper] the point you are evaluating at.
 * @param lower the lower bound of the support for x.
 * @param upper the upper bound for the support of x.
 * @param log true if you want to evaluate the log-density. False otherwise.
 * @return a double evaluation.
 */
double evalUniform(const double &x, const double &lower, const double &upper, bool log = false);


/**
 * @brief Evaluates discrete uniform pmf
 * @param x the hypothetical value of a rv 
 * @param k the size of the support i.e. (1,2,...k)
 * @param log true if you want log pmf
 * @return P(X=x) probability that X equals x
 */
double evalDiscreteUnif(const int &x, const int &k, bool log = false);



////////////////////////////////////////////////
/////////      Eigen Evals             /////////
////////////////////////////////////////////////


/**
 * @brief Evaluates the multivariate Normal density
 * @tparam dim the size of the vectors 
 * @param x the point you're evaluating at.
 * @param meanVec the mean vector.
 * @param covMat the positive definite, symmetric covariance matrix.
 * @param log true if you want to return the log density. False otherwise.
 * @return a double evaluation.
 */
template<std::size_t dim>
double evalMultivNorm(const Eigen::Matrix<double,dim,1> &x, 
                      const Eigen::Matrix<double,dim,1> &meanVec, 
                      const Eigen::Matrix<double,dim,dim> &covMat, 
                      bool log = false)
{
    using Mat = Eigen::Matrix<double,dim,dim>;
    
    // from Eigen: Remember that Cholesky decompositions are not rank-revealing. 
    /// This LLT decomposition is only stable on positive definite matrices, 
    // use LDLT instead for the semidefinite case. Also, do not use a Cholesky 
    // decomposition to determine whether a system of equations has a solution.
    Eigen::LLT<Mat> lltM(covMat);
    double quadform = (lltM.matrixL().solve(x-meanVec)).squaredNorm();
    if (log){

        // calculate log-determinant using cholesky decomposition too
        double ld (0.0);
        Mat L = lltM.matrixL(); // the lower diagonal L such that M = LL^T

        // add up log of diagnols of Cholesky L
        for(size_t i = 0; i < dim; ++i){
            ld += std::log(L(i,i));
        }
        ld *= 2; // covMat = LL^T

        return -.5*log_two_pi * dim - .5*ld - .5*quadform;


    }else{  // not the log density
        double normConst = std::pow(inv_sqrt_2pi, dim) / lltM.matrixL().determinant();
        return normConst * std::exp(-.5* quadform);
    }
}


/**
 * @brief Evaluates the multivariate Normal density using the Woodbury Matrix Identity to speed up inversion. 
 * Sigma = A + UCU'. This function assumes A is diagonal and C is symmetric.
 * @param x the point you're evaluating at.
 * @param meanVec the mean vector.
 * @param A  of A + UCU' in vector form because we explicitly make it diagonal.
 * @param U of A + UCU'
 * @param C of A + UCU'
 * @param log true if you want to return the log density. False otherwise.
 * @return a double evaluation.
 */
template<std::size_t bigd, std::size_t smalld>
double evalMultivNormWBDA(const Eigen::Matrix<double,bigd,1> &x, 
                          const Eigen::Matrix<double,bigd,1> &meanVec, 
                          const Eigen::Matrix<double,bigd,1> &A, 
                          const Eigen::Matrix<double,bigd,smalld> &U, 
                          const Eigen::Matrix<double,smalld,smalld> &C, 
                          bool log = false)
{
    
    using bigmat = Eigen::Matrix<double,bigd,bigd>;
    using smallmat = Eigen::Matrix<double,smalld,smalld>;

    bigmat Ainv = A.asDiagonal().inverse();
    smallmat Cinv = C.inverse();
    smallmat I =  Cinv + U.transpose()*Ainv*U;
    bigmat SigInv = Ainv - Ainv * U * I.ldlt().solve(U.transpose() * Ainv);
    Eigen::LLT<bigmat> lltSigInv(SigInv);
    bigmat L = lltSigInv.matrixL(); // LL' = Sig^{-1}
    double quadform = (L * (x-meanVec)).squaredNorm();    
    if (log){

        // calculate log-determinant using cholesky decomposition (assumes symmetric and positive definite)
        double halfld (0.0);
        // add up log of diagnols of Cholesky L
        for(size_t i = 0; i < bigd; ++i){
            halfld += std::log(L(i,i));
        }

        return -.5*log_two_pi * bigd + halfld - .5*quadform;


    }else{  // not the log density
        double normConst = std::pow(inv_sqrt_2pi, bigd) * L.determinant();
        return normConst * std::exp(-.5* quadform);
    }
                              
}


} //namespace rveval


} // namespace pf{

#endif //RV_EVAL_H