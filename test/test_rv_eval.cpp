//#include "UnitTest++.h"
#include <UnitTest++/UnitTest++.h>
#include "rv_eval.h"

#define PREC .0001 // define the precision for floating point tests

#define bigdim 2
#define smalldim 1

class DensFixture
{
public:

    using bigVec   = Eigen::Matrix<double,bigdim,1>;
    using bigMat   = Eigen::Matrix<double,bigdim,bigdim>;
    using smallVec = Eigen::Matrix<double,smalldim,1>;
    using smallMat = Eigen::Matrix<double,smalldim,smalldim>;

    // for multivariate Gaussian
    bigVec x;
    bigVec mu;
    bigMat covMat;
    bigVec A;
    smallMat C;
    bigVec U;
    double beta1p;
    double beta2p;
    double sigmaSquaredHN;
    double invgamma1p;
    double invgamma2p;
    double pi;
    double lower;
    double upper;
    double lnMu;
    double lnSigma;
    

    DensFixture() 
    {
        // MVN NORMAL
        x(0) = .02;
        x(1) = -.01;
        mu(0) = 0.0;
        mu(1) = 0.0;
        covMat(0,0) = 3.0;
        covMat(0,1) = 1.0;
        covMat(1,0) = 1.0;
        covMat(1,1) = 3.0;
        
        // MVN NORM woodbury        
        A(0,0) = A(1,0) = 2.0;
        U(0,0) = U(1,0) = 1.0;
        C(0,0) = 1.0;
        
        // beta
        beta1p = .2;
        beta2p = .3;
        
        // inverse gamma 
        invgamma1p = .2;
        invgamma2p = 5.2;
        
        // half normal
        sigmaSquaredHN = 1.5;
        pi = 3.141592653589793;
        
        // cts uniform
        lower = 1.3;
        upper = 5.2;
        
        // lognormal
        lnMu = .5;
        lnSigma = 5.3;

    }
    
};


TEST_FIXTURE(DensFixture, univNormalTest)
{
    // via R dnorm(.5, 2, 1.5, T)
    CHECK_CLOSE(rveval::evalUnivNorm<double>(.5, 2.0, 1.5, true), -1.824404, PREC);
    CHECK_CLOSE(rveval::evalUnivNorm<double>(.5, 2.0, 1.5, false), 0.1613138, PREC);
}


TEST_FIXTURE(DensFixture, univNormCDFTest)
{
    // via R pnorm(.1)
    CHECK_CLOSE(rveval::evalUnivStdNormCDF<double>(.1), 0.5398278, PREC);
    CHECK_CLOSE(rveval::evalUnivStdNormCDF<double>(0.0), .5, PREC);
    CHECK_CLOSE(rveval::evalUnivStdNormCDF<double>(1.0/0.0), 1.0, PREC);
    CHECK_CLOSE(rveval::evalUnivStdNormCDF<double>(-1.0/0.0), 0.0, PREC);
}


TEST_FIXTURE(DensFixture, truncNormTest)
{
    // check bounds can be infinite
    CHECK_CLOSE(rveval::evalUnivTruncNorm<double>(0.0, 0.0, 1.0, -1.0/0.0, 1.0/0.0, true),
                rveval::evalUnivNorm<double>(0.0, 0.0, 1.0, true), PREC);
    CHECK_CLOSE(rveval::evalUnivTruncNorm<double>(0.0, 0.0, 1.0, -1.0/0.0, 1.0/0.0, false),
                rveval::evalUnivNorm<double>(0.0, 0.0, 1.0, false), PREC);
    // check support is good
    CHECK_CLOSE(rveval::evalUnivTruncNorm<double>(0.0, 0.0, 1.0, .1, 20.0, false), 0.0, PREC);
    CHECK_CLOSE(rveval::evalUnivTruncNorm<double>(0.0, 0.0, 1.0, .1, 20.0, true), -1.0/0.0, PREC);
    CHECK_CLOSE(rveval::evalUnivTruncNorm<double>(0.0, 0.0, 1.0, -20.0, -.1, false), 0.0, PREC);
    CHECK_CLOSE(rveval::evalUnivTruncNorm<double>(0.0, 0.0, 1.0, -20.0, -.1, true), -1.0/0.0, PREC);
    // check a real evaluation comparing it to R's truncnorm::dtruncnorm(0, -5, 5, 0, 2)
    CHECK_CLOSE(rveval::evalUnivTruncNorm<double>(0.0, 0.0, 2.0, -5.0, 5.0, false), 0.2019796, PREC);
    CHECK_CLOSE(rveval::evalUnivTruncNorm<double>(0.0, 0.0, 2.0, -5.0, 5.0, true), -1.599589, PREC);
}


TEST_FIXTURE(DensFixture, multivariateGaussianTest)
{
    // via R dmvnorm(c(.02, -.01),sigma=matrix(c(3,1,1,3),nrow=2))
    double num = rveval::evalMultivNorm<bigdim,double>(x, mu, covMat, true);    
    CHECK_CLOSE(num,
                -2.877717,
                PREC);

    double num2 = rveval::evalMultivNorm<bigdim,double>(x, mu, covMat, false);
    CHECK_CLOSE(num2, 
                0.05626309,
                PREC);
}


TEST_FIXTURE(DensFixture, multivNormWoodburyTest)
{
    double normeval = rveval::evalMultivNorm<bigdim,double>(x, mu, covMat, true);
    double wbdanormeval = rveval::evalMultivNormWBDA<bigdim,smalldim,double>(x, mu, A, U, C, true);
    CHECK_CLOSE(normeval, wbdanormeval, PREC);
                
    double normeval2 = rveval::evalMultivNorm<bigdim,double>(x, mu, covMat, false);
    double wbdanormeval2 = rveval::evalMultivNormWBDA<bigdim,smalldim,double>(x, mu, A, U, C, false);
    CHECK_CLOSE(normeval2, wbdanormeval2, PREC);
}


TEST_FIXTURE(DensFixture, univBeta)
{
    // via R dbeta(.5, .2, .3, F)
    CHECK_CLOSE(rveval::evalUnivBeta<double>(.5, beta1p, beta2p, true),
                -1.007776,
                PREC);

    CHECK_CLOSE(rveval::evalUnivBeta<double>(.5, beta1p, beta2p, false),
                0.3650299,
                PREC);

    CHECK_EQUAL(rveval::evalUnivBeta<double>(-.5, beta1p, beta2p, true), -1.0/0.0);

    CHECK_EQUAL(rveval::evalUnivBeta<double>(-.5, beta1p, beta2p, false), 0.0);
}


TEST_FIXTURE(DensFixture, invGammaTest)
{
    CHECK_CLOSE(rveval::evalUnivInvGamma<double>(3.2, invgamma1p, invgamma2p, true),
                -4.215113,
                PREC);
                
    CHECK_CLOSE(rveval::evalUnivInvGamma<double>(3.2, invgamma1p, invgamma2p, false),
                0.01477065,
                PREC);
                
    CHECK_EQUAL(rveval::evalUnivInvGamma<double>(-3.2, invgamma1p, invgamma2p, true), -1.0/0.0); 
   
    CHECK_EQUAL(rveval::evalUnivInvGamma<double>(-3.2, invgamma1p, invgamma2p, false), 0.0);
}


TEST_FIXTURE(DensFixture, halfNormalTest)
{
    // fdrtool::dhalfnorm(.2, sqrt(pi/(2*1.5)))
    CHECK_CLOSE(rveval::evalUnivHalfNorm<double>(.2, sigmaSquaredHN, true), -0.4418572400321429, PREC);
    CHECK_CLOSE(rveval::evalUnivHalfNorm<double>(.2, sigmaSquaredHN, false), 0.6428414009228908, PREC);
    CHECK_EQUAL(rveval::evalUnivHalfNorm<double>(-.2, sigmaSquaredHN, false), 0.0);
    CHECK_EQUAL(rveval::evalUnivHalfNorm<double>(-.2, sigmaSquaredHN, true), -1.0/0.0);
}


TEST_FIXTURE(DensFixture, ctsUniformTest)
{
    CHECK_CLOSE(rveval::evalUniform<double>((lower+upper)/2.0, lower, upper, false), 1.0/(upper - lower), PREC);
    CHECK_CLOSE(rveval::evalUniform<double>((lower+upper)/2.0, lower, upper, true), -std::log(upper-lower), PREC);
    CHECK_EQUAL(rveval::evalUniform<double>(lower-.01, lower, upper, false), 0.0);
    CHECK_EQUAL(rveval::evalUniform<double>(lower-.01, lower, upper, true), -1.0/0.0);
}


TEST_FIXTURE(DensFixture, evalLogNormalTest)
{
    // dlnorm(.2, .5, 5.3, T)
    CHECK_CLOSE(rveval::evalLogNormal<double>(.2, lnMu, lnSigma, true), 
                -1.056412288363436,
                PREC);
    CHECK_CLOSE(rveval::evalLogNormal<double>(.2, lnMu, lnSigma, false), 
                0.3477010262745334,
                PREC);
    CHECK_EQUAL(rveval::evalLogNormal<double>(-2, lnMu, lnSigma, true),
                -1.0/0.0);
    CHECK_EQUAL(rveval::evalLogNormal<double>(-2, lnMu, lnSigma, false), 0.0);
                
}


//TEST_FIXTURE(DensFixture, evalLogitNormalTest)
//{
//    
//}
//
//
//TEST_FIXTURE(DensFixture, evalTwiceFisherNormalTest)
//{
//    
//}


TEST_FIXTURE(DensFixture, evalBernoulliTest)
{
    /// dbinom(1, 1, .001, T) 
    CHECK_CLOSE(-6.907755, rveval::evalBernoulli(1, .001, true), PREC);
    CHECK_CLOSE(0.001, rveval::evalBernoulli(1, .001, false), PREC);
    
    CHECK_CLOSE(-1.0/0.0, rveval::evalBernoulli(-1, .5, true), PREC);
    CHECK_CLOSE(0.0, rveval::evalBernoulli(1, 1.1, false), PREC);
}



