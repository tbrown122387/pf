#include "rv_eval.h"

#include <cmath> // tgamma, pow, exp, otherstuff
#include <stdexcept> // std::invalid_argument
#include <iostream> // std::cerr


double rveval::twiceFisher(const double &phi)
{
    if ( (phi <= -1.0) || (phi >= 1.0) )
        throw std::invalid_argument( "error: phi was not between -1 and 1" );
    else
        return std::log(1.0 + phi) - std::log(1.0 - phi);
}


double rveval::invTwiceFisher(const double &psi)
{
    double ans = (1.0 - std::exp(psi)) / ( -1.0 - std::exp(psi) );
    
    if ( (ans <= -1.0) || (ans >= 1.0) )
        std::cerr << "error: there was probably overflow for exp(psi) \n";
    
    return ans;    
}


double rveval::logit(const double &p)
{
    if ( (p <= 0.0) || (p >= 1.0))
        std::cerr << "error: p was not between 0 and 1 \n";
    
    return std::log(p) - std::log(1.0 - p);
}


double rveval::inv_logit(const double &r)
{
    double ans = 1.0/( 1.0 + std::exp(-r) );
    
    if ( (ans <= 0.0) || (ans >= 1.0))
        std::cerr << "error: there was probably underflow for exp(-r) \n";
    
    return ans;
}


double rveval::log_inv_logit(const double& r)
{
    if(r < -750.00 || r > 750.00) std::cerr << "warning: log_inv_logit might be under/over-flowing\n";
    return -std::log(1.0 + std::exp(-r));
}


double rveval::evalUnivNorm(const double &x, const double &mu, const double &sigma, bool log)
{
    double exponent = -.5*(x - mu)*(x-mu)/(sigma*sigma);
    if( sigma > 0.0){
        if(log){
            return -std::log(sigma) - .5*log_two_pi + exponent;
        }else{
            return inv_sqrt_2pi * std::exp(exponent) / sigma;
        }
    }else{
        if(log){
            return -1.0/0.0;
        }else{
            return 0.0;
        }
    }
}


double rveval::evalUnivStdNormCDF(const double &x) // john cook code
{
    // constants
    double a1 =  0.254829592;
    double a2 = -0.284496736;
    double a3 =  1.421413741;
    double a4 = -1.453152027;
    double a5 =  1.061405429;
    double p  =  0.3275911;

    // Save the sign of x
    int sign = 1;
    if (x < 0)
        sign = -1;
    double xt = std::fabs(x)/std::sqrt(2.0);

    // A&S formula 7.1.26
    double t = 1.0/(1.0 + p*xt);
    double y = 1.0 - (((((a5*t + a4)*t) + a3)*t + a2)*t + a1)*t*std::exp(-xt*xt);

    return 0.5*(1.0 + sign*y);
}


double rveval::evalUnivBeta(const double &x, const double &alpha, const double &beta, bool log)
{
    if( (x > 0.0) && (x < 1.0) && (alpha > 0.0) && (beta > 0.0) ){ // x in support and parameters acceptable
        if(log){
            return std::lgamma(alpha + beta) - std::lgamma(alpha) - std::lgamma(beta) + (alpha - 1.0)*std::log(x) + (beta - 1.0) * std::log(1.0 - x);
        }else{
            return pow(x, alpha-1.0) * pow(1.0-x, beta-1.0) * std::tgamma(alpha + beta) / ( std::tgamma(alpha) * std::tgamma(beta) );
        }

    }else{ //not ( x in support and parameters acceptable )
        if(log){
            return -1.0/0.0;
        }else{
            return 0.0;
        }
    }
}


double rveval::evalUnivInvGamma(const double &x, const double &alpha, const double &beta, bool log)
{
    if ( (x > 0.0) && (alpha > 0.0) && (beta > 0.0) ){ // x in support and acceptable parameters
        if (log){
            return alpha * std::log(beta) - std::lgamma(alpha) - (alpha + 1.0)*std::log(x) - beta/x;
        }else{
            return pow(x, -alpha-1.0) * exp(-beta/x) * pow(beta, alpha) / std::tgamma(alpha);
        }
    }else{ // not ( x in support and acceptable parameters )
        if (log){
            return -1.0/0.0;
        }else{
            return 0.0;
        }
    }
}


double rveval::evalUnivHalfNorm(const double &x, const double &sigmaSqd, bool log)
{
    if( (x >= 0.0) && (sigmaSqd > 0.0)){
        if (log){
            return .5*log_two_over_pi - .5*std::log(sigmaSqd) - .5*x*x / sigmaSqd;
        }else{
            return std::exp(-.5*x*x/sigmaSqd) * sqrt_two_over_pi / std::sqrt(sigmaSqd);
        }
    }else{
        if (log){
            return -1.0/0.0;
        }else{
            return 0.0;
        }
    }
}


double rveval::evalUnivTruncNorm(const double &x, const double &mu, const double &sigma, const double &lower, const double &upper, bool log)
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
            return -1.0/0.0;
        }else{
            return 0.0;
        }
    }
}


double rveval::evalLogitNormal(const double &x, const double &mu, const double &sigma, bool log)
{
    if( (x >= 0.0) && (x <= 1.0) && (sigma > 0.0)){
        
        double exponent = -.5*(logit(x) - mu)*(logit(x) - mu) / (sigma*sigma);
        if(log){
            return -std::log(sigma) - .5*log_two_pi - std::log(x) - std::log(1.0-x) + exponent;
        }else{
            return inv_sqrt_2pi * std::exp(exponent) / (x * (1.0-x) * sigma);   
        }
    }else{
        if(log){
            return -1.0/0.0;
        }else{
            return 0.0;
        }
    }
}
 

double rveval::evalTwiceFisherNormal(const double &x, const double &mu, const double &sigma, bool log)
{
    if( (x >= -1.0) && (x <= 1.0) && (sigma > 0.0)){
        
        double exponent = -.5*(std::log((1.0+x)/(1.0-x)) - mu)*(std::log((1.0+x)/(1.0-x)) - mu)/(sigma* sigma);
        if(log){
            return -std::log(sigma) - .5*log_two_pi + std::log(2.0) - std::log(1.0+x) - std::log(1.0-x) + exponent;
        }else{
            return inv_sqrt_2pi * 2.0 * std::exp(exponent)/( (1.0-x)*(1.0+x)*sigma );
        }
    }else{
        if(log){
            return -1.0/0.0;
        }else{
            return 0.0;
        }
    }
}


double rveval::evalLogNormal(const double &x, const double &mu, const double &sigma, bool log)
{
    if( (x > 0.0) && (sigma > 0.0)){
        
        double exponent = -.5*(std::log(x)-mu)*(std::log(x)-mu)/(sigma*sigma);
        if(log){
            return -std::log(x) - std::log(sigma) - .5*log_two_pi + exponent;
        }else{
            return inv_sqrt_2pi*std::exp(exponent)/(sigma*x);
        }
    }else{
        if(log){
            return -1.0/0.0;
        }else{
            return 0.0;
        }
    }
}


double rveval::evalUniform(const double &x, const double &lower, const double &upper, bool log)
{

    if( (x > lower) && (x <= upper)){
        
        double width = upper-lower;
        if(log){
            return -std::log(width);
        }else{
            return 1.0/width;
        }
    }else{
        if(log){
            return -1.0/0.0;
        }else{
            return 0.0;
        }
    }
    
}


double rveval::evalDiscreteUnif(const int &x, const int &k, bool log)
{
    if( (1 <= x) && (x <= k) ){
        if(log){
            return -std::log(static_cast<double>(k));
        }else{
            return 1.0 / static_cast<double>(k);
        }
    }else{ // x not in support
        if(log){
            return -1.0/0.0;
        }else{
            return 0.0;
        }
    }
}


double rveval::evalBernoulli(const int& x, const double& p, bool log)
{
    if( ((x == 0) || (x == 1)) && ( (0.0 <= p) && (p <= 1.0)  ) ){ // if valid x and valid p
        if(log){
            return (x==1) ? std::log(p) : std::log(1.0-p);
        }else{
            return (x==1) ? p : (1.0-p);
        }    
    }else{ // either invalid x or invalid p
        if(log){
            return -1.0/0.0;
        }else{
            return 0.0;
        }
    }
}
