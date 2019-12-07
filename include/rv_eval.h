#ifndef RV_EVAL_H
#define RV_EVAL_H

#include <cstddef> // std::size_t
#include <Eigen/Dense>
#include <iostream> // cerr
#include "boost/math/special_functions.hpp"


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

/** log(pi) */
const double log_pi (1.1447298858494);


////////////////////////////////////////////////
/////////      Transformations         /////////
////////////////////////////////////////////////

/**
 * @brief Maps (-1, 1) to the reals.
 * @param phi
 * @return psi
 */
template<typename float_t>
float_t twiceFisher(const float_t &phi)
{
    if ( (phi <= -1.0) || (phi >= 1.0) )
        throw std::invalid_argument( "error: phi was not between -1 and 1" );
    else
        return std::log(1.0 + phi) - std::log(1.0 - phi);
}



/**
 * @brief Maps a real number to the itnerval (-1,1).
 * @param psi
 * @return phi
 */
template<typename float_t>
float_t invTwiceFisher(const float_t &psi)
{
    float_t ans = (1.0 - std::exp(psi)) / ( -1.0 - std::exp(psi) );
    
    if ( (ans <= -1.0) || (ans >= 1.0) )
        std::cerr << "error: there was probably overflow for exp(psi) \n";
    
    return ans;    
}
    
    
/**
 * @brief Maps (0,1) to the reals.
 * @param p
 * @return logit(p)
 */
template<typename float_t>
float_t logit(const float_t &p)
{
    if ( (p <= 0.0) || (p >= 1.0))
        std::cerr << "error: p was not between 0 and 1 \n";
    
    return std::log(p) - std::log(1.0 - p);
}
    
    
/**
 * @brief Maps the reals to (0,1)
 * @param r
 * @return p = invlogit(p)
 */
template<typename float_t>
float_t inv_logit(const float_t &r)
{
    float_t ans = 1.0/( 1.0 + std::exp(-r) );
    
    if ( (ans <= 0.0) || (ans >= 1.0))
        std::cerr << "error: there was probably underflow for exp(-r) \n";
    
    return ans;
}


/**
 * @brief Maps the reals to the reals
 * @param r
 * @return log(invlogit(p))
 */
template<typename float_t>
float_t log_inv_logit(const float_t& r)
{
    if(r < -750.00 || r > 750.00) std::cerr << "warning: log_inv_logit might be under/over-flowing\n";
    return -std::log(1.0 + std::exp(-r));
}


////////////////////////////////////////////////
/////////       float_t evals           /////////
////////////////////////////////////////////////
    
/**
 * @brief Evaluates the univariate Normal density.
 * @param x the point at which you're evaluating.
 * @param mu the mean.
 * @param sigma the standard deviation.
 * @param log true if you want the log-density. False otherwise.
 * @return a float_t evaluation.
 */
template<typename float_t>
float_t evalUnivNorm(const float_t &x, const float_t &mu, const float_t &sigma, bool log)
{
    float_t exponent = -.5*(x - mu)*(x-mu)/(sigma*sigma);
    if( sigma > 0.0){
        if(log){
            return -std::log(sigma) - .5*log_two_pi + exponent;
        }else{
            return inv_sqrt_2pi * std::exp(exponent) / sigma;
        }
    }else{
        if(log){
            return -std::numeric_limits<float_t>::infinity();
        }else{
            return 0.0;
        }
    }
}


/**
 * @brief Evaluates the standard Normal CDF.
 * @param x the quantile.
 * @return the probability Z < x
 */
template<typename float_t>
float_t evalUnivStdNormCDF(const float_t &x) // john cook code
{
    // constants
    float_t a1 =  0.254829592;
    float_t a2 = -0.284496736;
    float_t a3 =  1.421413741;
    float_t a4 = -1.453152027;
    float_t a5 =  1.061405429;
    float_t p  =  0.3275911;

    // Save the sign of x
    int sign = 1;
    if (x < 0)
        sign = -1;
    float_t xt = std::fabs(x)/std::sqrt(2.0);

    // A&S formula 7.1.26
    float_t t = 1.0/(1.0 + p*xt);
    float_t y = 1.0 - (((((a5*t + a4)*t) + a3)*t + a2)*t + a1)*t*std::exp(-xt*xt);

    return 0.5*(1.0 + sign*y);
} 


/**
 * @brief Evaluates the univariate Beta density
 * @param x the point
 * @param alpha parameter 1 
 * @param beta parameter 2
 * @param log true if you want log density
 * @return float_t evaluation.
*/  
template<typename float_t>
float_t evalUnivBeta(const float_t &x, const float_t &alpha, const float_t &beta, bool log)
{
    if( (x > 0.0) && (x < 1.0) && (alpha > 0.0) && (beta > 0.0) ){ // x in support and parameters acceptable
        if(log){
            return std::lgamma(alpha + beta) - std::lgamma(alpha) - std::lgamma(beta) + (alpha - 1.0)*std::log(x) + (beta - 1.0) * std::log(1.0 - x);
        }else{
            return pow(x, alpha-1.0) * pow(1.0-x, beta-1.0) * std::tgamma(alpha + beta) / ( std::tgamma(alpha) * std::tgamma(beta) );
        }

    }else{ //not ( x in support and parameters acceptable )
        if(log){
            return -std::numeric_limits<float_t>::infinity();
        }else{
            return 0.0;
        }
    }
}


/**
 * @brief Evaluates the univariate Inverse Gamma density
 * @param x the point
 * @param alpha shape parameter  
 * @param beta rate parameter 
 * @param log true if you want log density.
 * @return float_t evaluation.
*/    
template<typename float_t>
float_t evalUnivInvGamma(const float_t &x, const float_t &alpha, const float_t &beta, bool log)
{
    if ( (x > 0.0) && (alpha > 0.0) && (beta > 0.0) ){ // x in support and acceptable parameters
        if (log){
            return alpha * std::log(beta) - std::lgamma(alpha) - (alpha + 1.0)*std::log(x) - beta/x;
        }else{
            return pow(x, -alpha-1.0) * exp(-beta/x) * pow(beta, alpha) / std::tgamma(alpha);
        }
    }else{ // not ( x in support and acceptable parameters )
        if (log){
            return -std::numeric_limits<float_t>::infinity();
        }else{
            return 0.0;
        }
    }
}


/**
 * @brief Evaluates the half-normal density
 * @param x the point you're evaluating at
 * @param sigmaSqd the scale parameter
 * @param log true if you want log density.
 * @return float_t evaluation.
 */
template<typename float_t>
float_t evalUnivHalfNorm(const float_t &x, const float_t &sigmaSqd, bool log)
{
    if( (x >= 0.0) && (sigmaSqd > 0.0)){
        if (log){
            return .5*log_two_over_pi - .5*std::log(sigmaSqd) - .5*x*x / sigmaSqd;
        }else{
            return std::exp(-.5*x*x/sigmaSqd) * sqrt_two_over_pi / std::sqrt(sigmaSqd);
        }
    }else{
        if (log){
            return -std::numeric_limits<float_t>::infinity();
        }else{
            return 0.0;
        }
    }
}

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
template<typename float_t>
float_t evalUnivTruncNorm(const float_t &x, const float_t &mu, const float_t &sigma, const float_t &lower, const float_t &upper, bool log)
{
    if( (sigma > 0.0) && (lower <= x) & (x <= upper) ){
        if(log){
            return evalUnivNorm(x, mu, sigma, true) 
                - std::log( evalUnivStdNormCDF((upper-mu)/sigma) - evalUnivStdNormCDF((lower-mu)/sigma));
        }else{
            return evalUnivNorm(x,mu,sigma,false)
                / ( evalUnivStdNormCDF((upper-mu)/sigma) - evalUnivStdNormCDF((lower-mu)/sigma) );
        }
        
    }else{
        if (log){
            return -std::numeric_limits<float_t>::infinity();
        }else{
            return 0.0;
        }
    }
}



/**
 * @brief Evaluates the logit-Normal distribution (see Wiki for more info)
 * @param x in [0,1] the point you're evaluating at
 * @param mu location parameter that can take any real number
 * @param sigma scale parameter that needs to be positive
 * @param log true if you want to evalute the log-density. False otherwise.
 * @return a float_t evaluation
 */
template<typename float_t>
float_t evalLogitNormal(const float_t &x, const float_t &mu, const float_t &sigma, bool log)
{
    if( (x >= 0.0) && (x <= 1.0) && (sigma > 0.0)){
        
        float_t exponent = -.5*(logit(x) - mu)*(logit(x) - mu) / (sigma*sigma);
        if(log){
            return -std::log(sigma) - .5*log_two_pi - std::log(x) - std::log(1.0-x) + exponent;
        }else{
            return inv_sqrt_2pi * std::exp(exponent) / (x * (1.0-x) * sigma);   
        }
    }else{
        if(log){
            return -std::numeric_limits<float_t>::infinity();
        }else{
            return 0.0;
        }
    }
}


/**
 * @brief Evaluates what I call the "twice-fisher-Normal" distribution
 * https://stats.stackexchange.com/questions/321905/what-is-the-name-of-this-random-variable/321907#321907
 * @param x in [-1,1] the point you are evaluating at
 * @param mu the location parameter (all real numbers)
 * @param sigma the scale parameter (positive)
 * @param log true if you want to evaluate the log-density. False otherwise.
 * @return a float_t evaluation
 */
template<typename float_t>
float_t evalTwiceFisherNormal(const float_t &x, const float_t &mu, const float_t &sigma, bool log)
{
    if( (x >= -1.0) && (x <= 1.0) && (sigma > 0.0)){
        
        float_t exponent = -.5*(std::log((1.0+x)/(1.0-x)) - mu)*(std::log((1.0+x)/(1.0-x)) - mu)/(sigma* sigma);
        if(log){
            return -std::log(sigma) - .5*log_two_pi + std::log(2.0) - std::log(1.0+x) - std::log(1.0-x) + exponent;
        }else{
            return inv_sqrt_2pi * 2.0 * std::exp(exponent)/( (1.0-x)*(1.0+x)*sigma );
        }
    }else{
        if(log){
            return -std::numeric_limits<float_t>::infinity();
        }else{
            return 0.0;
        }
    }
}


/**
 * @brief Evaluates the lognormal density
 * @param x in (0,infty) the point you are evaluating at
 * @param mu the location parameter
 * @param sigma in (0, infty) the scale parameter
 * @param log true if you want to evaluate the log-density. False otherwise.
 * @return a float_t evaluation
 */
template<typename float_t>
float_t evalLogNormal(const float_t &x, const float_t &mu, const float_t &sigma, bool log)
{
    if( (x > 0.0) && (sigma > 0.0)){
        
        float_t exponent = -.5*(std::log(x)-mu)*(std::log(x)-mu)/(sigma*sigma);
        if(log){
            return -std::log(x) - std::log(sigma) - .5*log_two_pi + exponent;
        }else{
            return inv_sqrt_2pi*std::exp(exponent)/(sigma*x);
        }
    }else{
        if(log){
            return -std::numeric_limits<float_t>::infinity();
        }else{
            return 0.0;
        }
    }
}


/**
 * @brief Evaluates the uniform density.
 * @param x in (lower, upper] the point you are evaluating at.
 * @param lower the lower bound of the support for x.
 * @param upper the upper bound for the support of x.
 * @param log true if you want to evaluate the log-density. False otherwise.
 * @return a float_t evaluation.
 */
template<typename float_t>
float_t evalUniform(const float_t &x, const float_t &lower, const float_t &upper, bool log)
{

    if( (x > lower) && (x <= upper)){
        
        float_t width = upper-lower;
        if(log){
            return -std::log(width);
        }else{
            return 1.0/width;
        }
    }else{
        if(log){
            return -std::numeric_limits<float_t>::infinity();
        }else{
            return 0.0;
        }
    }
    
}


template<typename float_t>
float_t evalScaledT(const float_t& x, const float_t& mu, const float_t& sigma, const float_t& dof, bool log)
{

    if( (sigma > 0.0) && (dof > 0.0) ){

        float_t logDens = std::lgamma(.5*(dof+1.0)) - std::log(sigma) - .5*std::log(dof) - .5*log_pi - std::lgamma(.5*dof) - .5*(dof+1.0)*std::log( 1 + ((x - mu)/sigma)*((x-mu)/sigma)/dof );
        if(log)
            return logDens;
        else
            return std::exp(logDens);    
    } else{
        if(log)
            return -std::numeric_limits<float_t>::infinity();
        else
            return 0.0;
    }
} 



/**
 * @brief Evaluates discrete uniform pmf
 * @param x the hypothetical value of a rv 
 * @param k the size of the support i.e. (1,2,...k)
 * @param log true if you want log pmf
 * @return P(X=x) probability that X equals x
 */
template<typename int_t, typename float_t>
float_t evalDiscreteUnif(const int_t &x, const int &k, bool log)
{
    if( (1 <= x) && (x <= k) ){
        if(log){
            return -std::log(static_cast<float_t>(k));
        }else{
            return 1.0 / static_cast<float_t>(k);
        }
    }else{ // x not in support
        if(log){
            return -std::numeric_limits<float_t>::infinity();
        }else{
            return 0.0;
        }
    }
}


/**
 * @brief Evaluates the Bernoulli pmf
 * @param x the hypothetical value of a rv
 * @param p the probability that the rv equals 1
 */
template<typename int_t, typename float_t>
float_t evalBernoulli(const int_t& x, const float_t& p, bool log)
{
    if( ((x == 0) || (x == 1)) && ( (0.0 <= p) && (p <= 1.0)  ) ){ // if valid x and valid p
        if(log){
            return (x==1) ? std::log(p) : std::log(1.0-p);
        }else{
            return (x==1) ? p : (1.0-p);
        }    
    }else{ // either invalid x or invalid p
        if(log){
            return -std::numeric_limits<float_t>::infinity();
        }else{
            return 0.0;
        }
    }
}


/**
 * @brief Evaluates the Skellam pmf.
 * @param x the point at which you're evaluating.
 * @param mu1.
 * @param mu2.
 * @param log true if you want the log-mass. False otherwise.
 * @return a float_t evaluation.
 */
template<typename int_t, typename float_t>
float_t evalSkellam(const int_t &x, const float_t &mu1, const float_t &mu2, bool log)
{
    if( (mu1 > 0) && (mu2 > 0) ){

        float_t log_mass = -mu1 - mu2 + .5*x*(std::log(mu1) - std::log(mu2));
        log_mass += std::log(boost::math::cyl_bessel_i<int_t, float_t>(x,2.0*std::sqrt(mu1*mu2)));

        if(log) {
            return log_mass;
        }else {
            return std::exp(log_mass);
        }
    }else {
        if(log) {
            return -std::numeric_limits<float_t>::infinity();
        }else {
            return 0.0;
        }
    }        
}


////////////////////////////////////////////////
/////////      Eigen Evals             /////////
////////////////////////////////////////////////


/**
 * @brief Evaluates the multivariate Normal density. 
 * If covariance matrix isn't pd, then returns 0 
 * (or negative infinity if log is true)
 * @tparam dim the size of the vectors 
 * @tparam float_t the floating point type
 * @param x the point you're evaluating at.
 * @param meanVec the mean vector.
 * @param covMat the positive definite, symmetric covariance matrix.
 * @param log true if you want to return the log density. False otherwise.
 * @return a float_t evaluation.
 */
template<std::size_t dim, typename float_t>
float_t evalMultivNorm(const Eigen::Matrix<float_t,dim,1> &x, 
                      const Eigen::Matrix<float_t,dim,1> &meanVec, 
                      const Eigen::Matrix<float_t,dim,dim> &covMat, 
                      bool log = false)
{
    using Mat = Eigen::Matrix<float_t,dim,dim>;
    Eigen::LLT<Mat> lltM(covMat);
    if(lltM.info() == Eigen::NumericalIssue) return log ? -std::numeric_limits<float_t>::infinity() : 0.0; // if not pd return 0 dens
    Mat L = lltM.matrixL(); // the lower diagonal L such that M = LL^T
    float_t quadform = (lltM.solve(x-meanVec)).squaredNorm();
    float_t ld (0.0);  // calculate log-determinant using cholesky decomposition too
    // add up log of diagnols of Cholesky L
    for(size_t i = 0; i < dim; ++i){
        ld += std::log(L(i,i));
    }
    ld *= 2; // covMat = LL^T

    float_t logDens = -.5*log_two_pi * dim - .5*ld - .5*quadform;

    if(log){
        return logDens;
    }else{
        return std::exp(logDens);
    }
}


/**
 * @brief Evaluates the multivariate T density. 
 * If covariance matrix isn't pd, then returns 0 
 * (or negative infinity if log is true)
 * @tparam dim the size of the vectors 
 * @tparam float_t the floating point type
 * @param x the point you're evaluating at.
 * @param locVec the location vector.
 * @param shapeMat the positive definite, symmetric shape matrix.
 * @param log true if you want to return the log density. False otherwise.
 * @return a float_t evaluation.
 */
template<std::size_t dim, typename float_t>
float_t evalMultivT(const Eigen::Matrix<float_t,dim,1> &x, 
                    const Eigen::Matrix<float_t,dim,1> &locVec, 
                    const Eigen::Matrix<float_t,dim,dim> &shapeMat,
                    const float_t& dof, 
                    bool log = false)
{
    if(dof <= 0.0) return log ? -std::numeric_limits<float_t>::infinity() : 0.0; // degrees of freedom must be positive 
    using Mat = Eigen::Matrix<float_t,dim,dim>;
    Eigen::LLT<Mat> lltM(shapeMat);
    if(lltM.info() == Eigen::NumericalIssue) return log ? -std::numeric_limits<float_t>::infinity() : 0.0; // if not pd return 0 dens
    Mat L = lltM.matrixL(); // the lower diagonal L such that M = LL^T
    float_t quadform = (lltM.solve(x-locVec)).squaredNorm();
    float_t ld (0.0);  // calculate log-determinant using cholesky decomposition too
    // add up log of diagnols of Cholesky L
    for(size_t i = 0; i < dim; ++i){
        ld += std::log(L(i,i));
    }
    ld *= 2; // shapeMat = LL^T

    float_t logDens = std::lgamma(.5*(dof+dim)) - .5*dim*std::log(dof) - .5*dim*log_pi 
        - std::lgamma(.5*dof) -.5*ld - .5*(dof+dim)*std::log( 1.0 + quadform/dof );

    if(log){
        return logDens;
    }else{
        return std::exp(logDens);
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
 * @return a float_t evaluation.
 */
template<std::size_t bigd, std::size_t smalld, typename float_t>
float_t evalMultivNormWBDA(const Eigen::Matrix<float_t,bigd,1> &x, 
                          const Eigen::Matrix<float_t,bigd,1> &meanVec, 
                          const Eigen::Matrix<float_t,bigd,1> &A, 
                          const Eigen::Matrix<float_t,bigd,smalld> &U, 
                          const Eigen::Matrix<float_t,smalld,smalld> &C, 
                          bool log = false)
{
    
    using bigmat = Eigen::Matrix<float_t,bigd,bigd>;
    using smallmat = Eigen::Matrix<float_t,smalld,smalld>;

    bigmat Ainv = A.asDiagonal().inverse();
    smallmat Cinv = C.inverse();
    smallmat I =  Cinv + U.transpose()*Ainv*U;
    bigmat SigInv = Ainv - Ainv * U * I.ldlt().solve(U.transpose() * Ainv);
    Eigen::LLT<bigmat> lltSigInv(SigInv);
    bigmat L = lltSigInv.matrixL(); // LL' = Sig^{-1}
    float_t quadform = (L * (x-meanVec)).squaredNorm();    
    if (log){

        // calculate log-determinant using cholesky decomposition (assumes symmetric and positive definite)
        float_t halfld (0.0);
        // add up log of diagnols of Cholesky L
        for(size_t i = 0; i < bigd; ++i){
            halfld += std::log(L(i,i));
        }

        return -.5*log_two_pi * bigd + halfld - .5*quadform;


    }else{  // not the log density
        float_t normConst = std::pow(inv_sqrt_2pi, bigd) * L.determinant();
        return normConst * std::exp(-.5* quadform);
    }
                              
}


/**
 * @brief Evaluates the Wishart density
 * returns 0 if either matrix is not sym pd (
 * or - infinity if log is true)
 * @tparam dim the number of rows of the square matrix
 * @tparam float_t the type of floating point number
 * @param X the matrix you're evaluating at.
 * @param Vinv the INVERSE of the scale matrix.
 * @param n the degrees of freedom.
 * @param log true if you want to return the log density. False otherwise.
 * @return a float_t evaluation.
 */
template<std::size_t dim, typename float_t>
float_t evalWishart(const Eigen::Matrix<float_t,dim,dim> &X, 
                    const Eigen::Matrix<float_t,dim,dim> &Vinv, 
                    const unsigned int &n, 
                    bool log = false)
{
    using Mat = Eigen::Matrix<float_t,dim,dim>;
    Eigen::LLT<Mat> lltX(X);
    Eigen::LLT<Mat> lltVinv(Vinv);
    if((n < dim) | (lltX.info() == Eigen::NumericalIssue) | (lltVinv.info() == Eigen::NumericalIssue)) return log ? -std::numeric_limits<float_t>::infinity() : 0.0; 
    // https://stackoverflow.com/questions/35227131/eigen-check-if-matrix-is-positive-semi-definite 

    float_t ldx (0.0); // log determinant of X
    float_t ldvinv (0.0);
    Mat Lx = lltX.matrixL(); // the lower diagonal L such that X = LL^T
    Mat Lvi = lltVinv.matrixL();
    float_t logGammaNOver2 = .25*dim*(dim-1)*log_pi; // existence guaranteed when n > dim-1

    // add up log of diagonals of each Cholesky L
    for(size_t i = 0; i < dim; ++i){
        ldx += std::log(Lx(i,i));
        ldvinv += std::log(Lvi(i,i));
        logGammaNOver2 += std::lgamma(.5*(n-i)); // recall j = i+1
    }
    ldx *= 2.0; // X = LL^T
    ldvinv *= 2.0;

    float_t logDens = .5*(n - dim -1)*ldx - .5*(Vinv*X).trace() - .5*n*dim*std::log(2.0) + .5*n*ldvinv - logGammaNOver2;

    if(log){
        return logDens;
    }else{
        return std::exp(logDens);
    }
}


/**
 * @brief Evaluates the Inverse Wishart density
 * returns 0 if either matrix is not sym pd (
 * or - infinity if log is true)
 * @tparam dim the number of rows of the square matrix
 * @tparam float_t the type of floating point number
 * @param X the matrix you're evaluating at.
 * @param Psi the scale matrix.
 * @param nu the degrees of freedom.
 * @param log true if you want to return the log density. False otherwise.
 * @return a float_t evaluation.
 */
template<std::size_t dim, typename float_t>
float_t evalInvWishart(const Eigen::Matrix<float_t,dim,dim> &X, 
                       const Eigen::Matrix<float_t,dim,dim> &Psi, 
                       const unsigned int &nu, 
                       bool log = false)
{
    using Mat = Eigen::Matrix<float_t,dim,dim>;
    Eigen::LLT<Mat> lltX(X);
    Eigen::LLT<Mat> lltPsi(Psi);
    if((nu < dim) | (lltX.info() == Eigen::NumericalIssue) | (lltPsi.info() == Eigen::NumericalIssue)) return log ? -std::numeric_limits<float_t>::infinity() : 0.0; 
    // https://stackoverflow.com/questions/35227131/eigen-check-if-matrix-is-positive-semi-definite 

    float_t ldx (0.0); // log determinant of X
    float_t ldPsi (0.0);
    Mat Lx = lltX.matrixL(); // the lower diagonal L such that X = LL^T
    Mat Lpsi = lltPsi.matrixL();
    float_t logGammaNuOver2 = .25*dim*(dim-1)*log_pi; // existence guaranteed when n > dim-1

    // add up log of diagonals of each Cholesky L
    for(size_t i = 0; i < dim; ++i){
        ldx += std::log(Lx(i,i));
        ldPsi += std::log(Lpsi(i,i));
        logGammaNuOver2 += std::lgamma(.5*(nu-i)); // recall j = i+1
    }
    ldx *= 2.0; // X = LL^T
    ldPsi *= 2.0;

    // TODO: this will probably be faster if you find an analogue...
    //float_t logDens = .5*nu*ldPsi - .5*(nu + dim + 1.0)*ldx - .5*(X.solve(Psi)).trace() - .5*nu*dim*std::log(2.0) - logGammaNuOver2;
    float_t logDens = .5*nu*ldPsi - .5*(nu + dim + 1.0)*ldx - .5*(Psi*X.inverse() ).trace() - .5*nu*dim*std::log(2.0) - logGammaNuOver2;


    if(log){
        return logDens;
    }else{
        return std::exp(logDens);
    }
}


} //namespace rveval


#endif //RV_EVAL_H
