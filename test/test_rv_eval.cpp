#include "UnitTest++.h"
#include "rv_eval.h"

using namespace rveval;

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
    CHECK_CLOSE(evalUnivNorm(.5, 2.0, 1.5, true), -1.824404, 0.00001);
    CHECK_CLOSE(evalUnivNorm(.5, 2.0, 1.5, false), 0.1613138, 0.00001);
}


TEST_FIXTURE(DensFixture, univNormCDFTest)
{
    // via R pnorm(.1)
    CHECK_CLOSE(evalUnivStdNormCDF(.1), 0.5398278, 0.00001);
    CHECK_CLOSE(evalUnivStdNormCDF(0.0), .5, .000001);
    CHECK_CLOSE(evalUnivStdNormCDF(1.0/0.0), 1.0, .000001);
    CHECK_CLOSE(evalUnivStdNormCDF(-1.0/0.0), 0.0, .000001);
}


TEST_FIXTURE(DensFixture, truncNormTest)
{
    // check bounds can be infinite
    CHECK_CLOSE(evalUnivTruncNorm(0.0, 0.0, 1.0, -1.0/0.0, 1.0/0.0, true),
                evalUnivNorm(0.0, 0.0, 1.0, true), .000001);
    CHECK_CLOSE(evalUnivTruncNorm(0.0, 0.0, 1.0, -1.0/0.0, 1.0/0.0, false),
                evalUnivNorm(0.0, 0.0, 1.0, false), .000001);
    // check support is good
    CHECK_CLOSE(evalUnivTruncNorm(0.0, 0.0, 1.0, .1, 20.0, false), 0.0, .000001);
    CHECK_CLOSE(evalUnivTruncNorm(0.0, 0.0, 1.0, .1, 20.0, true), -1.0/0.0, .000001);
    CHECK_CLOSE(evalUnivTruncNorm(0.0, 0.0, 1.0, -20.0, -.1, false), 0.0, .000001);
    CHECK_CLOSE(evalUnivTruncNorm(0.0, 0.0, 1.0, -20.0, -.1, true), -1.0/0.0, .000001);
    // check a real evaluation comparing it to R's truncnorm::dtruncnorm(0, -5, 5, 0, 2)
    CHECK_CLOSE(evalUnivTruncNorm(0.0, 0.0, 2.0, -5.0, 5.0, false), 0.2019796, .000001);
    CHECK_CLOSE(evalUnivTruncNorm(0.0, 0.0, 2.0, -5.0, 5.0, true), -1.599589, .000001);


}


TEST_FIXTURE(DensFixture, multivariateGaussianTest)
{
    // via R dmvnorm(c(.02, -.01),sigma=matrix(c(3,1,1,3),nrow=2))
    CHECK_CLOSE(evalMultivNorm<bigdim>(x, mu, covMat, true),
                -2.877717,
                0.00001);
    CHECK_CLOSE(evalMultivNorm<bigdim>(x, mu, covMat, false), 
                0.05626309,
                0.00001);
}


TEST_FIXTURE(DensFixture, multivNormWoodburyTest)
{
    double normeval = evalMultivNorm<bigdim>(x, mu, covMat, true);
    double wbdanormeval = evalMultivNormWBDA<bigdim,smalldim>(x, mu, A, U, C, true);
    CHECK_CLOSE(normeval, wbdanormeval, 0.00001);
                
    double normeval2 = evalMultivNorm<bigdim>(x, mu, covMat, false);
    double wbdanormeval2 = evalMultivNormWBDA<bigdim,smalldim>(x, mu, A, U, C, false);
    CHECK_CLOSE(normeval2, wbdanormeval2, 0.00001);
}


TEST_FIXTURE(DensFixture, univBeta)
{
    // via R dbeta(.5, .2, .3, F)
    CHECK_CLOSE(evalUnivBeta(.5, beta1p, beta2p, true),
                -1.007776,
                0.00001);

    CHECK_CLOSE(evalUnivBeta(.5, beta1p, beta2p, false),
                0.3650299,
                0.00001);

    CHECK_EQUAL(evalUnivBeta(-.5, beta1p, beta2p, true), -1.0/0.0);

    CHECK_EQUAL(evalUnivBeta(-.5, beta1p, beta2p, false), 0.0);
}


TEST_FIXTURE(DensFixture, invGammaTest)
{
    CHECK_CLOSE(evalUnivInvGamma(3.2, invgamma1p, invgamma2p, true),
                -4.215113,
                0.00001);
                
    CHECK_CLOSE(evalUnivInvGamma(3.2, invgamma1p, invgamma2p, false),
                0.01477065,
                0.00001);
                
    CHECK_EQUAL(evalUnivInvGamma(-3.2, invgamma1p, invgamma2p, true), -1.0/0.0); 
   
    CHECK_EQUAL(evalUnivInvGamma(-3.2, invgamma1p, invgamma2p, false), 0.0);
}


TEST_FIXTURE(DensFixture, halfNormalTest)
{
    // fdrtool::dhalfnorm(.2, sqrt(pi/(2*1.5)))
    CHECK_CLOSE(evalUnivHalfNorm(.2, sigmaSquaredHN, true), -0.4418572400321429, 0.00001);
    CHECK_CLOSE(evalUnivHalfNorm(.2, sigmaSquaredHN, false), 0.6428414009228908, 0.00001);
    CHECK_EQUAL(evalUnivHalfNorm(-.2, sigmaSquaredHN, false), 0.0);
    CHECK_EQUAL(evalUnivHalfNorm(-.2, sigmaSquaredHN, true), -1.0/0.0);
}


TEST_FIXTURE(DensFixture, ctsUniformTest)
{
    CHECK_CLOSE(evalUniform((lower+upper)/2.0, lower, upper, false), 1.0/(upper - lower), 0.00001);
    CHECK_CLOSE(evalUniform((lower+upper)/2.0, lower, upper, true), -std::log(upper-lower), 0.00001);
    CHECK_EQUAL(evalUniform(lower-.01, lower, upper, false), 0.0);
    CHECK_EQUAL(evalUniform(lower-.01, lower, upper, true), -1.0/0.0);
}


TEST_FIXTURE(DensFixture, evalLogNormalTest)
{
    // dlnorm(.2, .5, 5.3, T)
    CHECK_CLOSE(evalLogNormal(.2, lnMu, lnSigma, true), 
                -1.056412288363436,
                0.00001);
    CHECK_CLOSE(evalLogNormal(.2, lnMu, lnSigma, false), 
                0.3477010262745334,
                0.00001);
    CHECK_EQUAL(evalLogNormal(-2, lnMu, lnSigma, true),
                -1.0/0.0);
    CHECK_EQUAL(evalLogNormal(-2, lnMu, lnSigma, false), 0.0);
                
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


