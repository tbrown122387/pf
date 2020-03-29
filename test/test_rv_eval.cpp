//#include "UnitTest++.h"
#include <UnitTest++/UnitTest++.h>
#include "rv_eval.h"

#define PREC .001 // define the precision for floating point tests

#define bigdim 2
#define smalldim 1

class DensFixture
{
public:

    using bigVec   = Eigen::Matrix<double,bigdim,1>;
    using bigMat   = Eigen::Matrix<double,bigdim,bigdim>;
    using smallVec = Eigen::Matrix<double,smalldim,1>;
    using smallMat = Eigen::Matrix<double,smalldim,smalldim>;

    // all arguments
    bigVec x;
    bigVec mu;
    bigMat covMat;
    bigMat badCovMat;
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
    Eigen::Matrix<double,2,2> Omega;
    Eigen::Matrix<double,2,2> S;
    Eigen::Matrix<double,2,2> Sinv;
    double scaledTMu, scaledTdof, scaledTSigma;


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
        badCovMat(0,0) = badCovMat(0,1) = badCovMat(1,0) = badCovMat(1,1) = 1.0; 

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

        // wishart
        Omega(0,0) = 2.0;
        Omega(0,1) = -.3;
        Omega(1,0) = -.3;
        Omega(1,1) = 4.0;
        Sinv(0,0) = 1.010101010101010;
        Sinv(1,1) = 1.010101010101010;
        Sinv(0,1) = -0.101010101010101;
        Sinv(1,0) = -0.101010101010101;
        S(0,0) = S(1,1) = 1.0;
        S(0,1) = S(1,0) = .1;

        // scaled t
        scaledTMu = 23.2;
        scaledTdof = 3.6;
        scaledTSigma = 1.7;
    }
    
};


TEST_FIXTURE(DensFixture, univNormalTest)
{
    // via R dnorm(.5, 2, 1.5, T)
    CHECK_CLOSE(rveval::evalUnivNorm<double>(.5, 2.0, 1.5, true), -1.824404, PREC);
    CHECK_CLOSE(rveval::evalUnivNorm<double>(.5, 2.0, 1.5, false), 0.1613138, PREC);
}


TEST_FIXTURE(DensFixture, univScaledT)
{
    // via dt.scaled(1.23, 3.6, 23.2, 1.7, log =T)
    CHECK_CLOSE(rveval::evalScaledT<double>(1.23, scaledTMu, scaledTSigma, scaledTdof, true), -10.39272, PREC);
    CHECK_CLOSE(rveval::evalScaledT<double>(1.23, scaledTMu, scaledTSigma, scaledTdof, false), 3.065496e-05, PREC);

    // test broken ones
    CHECK_CLOSE(rveval::evalScaledT<double>(1.23, scaledTMu, -1.0*scaledTSigma, scaledTdof, true), -1.0/0.0, PREC);
    CHECK_CLOSE(rveval::evalScaledT<double>(1.23, scaledTMu, -1.0*scaledTSigma, scaledTdof, false), 0.0, PREC);
    CHECK_CLOSE(rveval::evalScaledT<double>(1.23, scaledTMu, scaledTSigma, -scaledTdof, true), -1.0/0.0, PREC);
    CHECK_CLOSE(rveval::evalScaledT<double>(1.23, scaledTMu, scaledTSigma, -scaledTdof, false), 0.0, PREC);
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

    double badNormLogDens = rveval::evalMultivNorm<bigdim,double>(x,mu,badCovMat,true);
    double badNormDens = rveval::evalMultivNorm<bigdim,double>(x,mu,badCovMat,false);
    CHECK_CLOSE(-1.0/0.0, badNormLogDens, PREC);
    CHECK_CLOSE(0.0, badNormDens, PREC);
}


TEST_FIXTURE(DensFixture, multivariateTTest)
{
    // via R dmvt(c(.02, -.01), c(0,0), sigma=matrix(c(3,1,1,3),nrow=2), 10)
    // reusing some of the multivariate normal variables, but the names are
    // a bit off..so I apologize
    double num = rveval::evalMultivT<bigdim,double>(x, mu, covMat, 3, true);    
    CHECK_CLOSE(num,
                -2.877796,
                PREC);

    double num2 = rveval::evalMultivT<bigdim,double>(x, mu, covMat, 3, false);
    CHECK_CLOSE(num2, 
                0.05625863,
                PREC);

    double badNormLogDens = rveval::evalMultivT<bigdim,double>(x,mu,badCovMat,3,true);
    double badNormDens = rveval::evalMultivT<bigdim,double>(x,mu,badCovMat,3,false);
    CHECK_CLOSE(-1.0/0.0, badNormLogDens, PREC);
    CHECK_CLOSE(0.0, badNormDens, PREC);
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


TEST_FIXTURE(DensFixture, evalSkellamTest)
{
    /////////////////////////
    // when first arg is 0 //
    /////////////////////////
    // z < 7.75
    // dskellam(0, 1.0, .025, log = T)
    CHECK_CLOSE(-1.000155, rveval::evalSkellam(0, 1.0, .025, true), PREC);
    // dskellam(0, 1.0, .025, log = F)
    CHECK_CLOSE(0.3678226, rveval::evalSkellam(0, 1.0, .025, false), PREC);
    // z < 500
    // dskellam(0, 115.2, 114.3, log = T)
    CHECK_CLOSE(-3.638105, rveval::evalSkellam(0, 115.2, 114.3, true), PREC);
    // dskellam(0, 115.2, 114.3, log = F)
    CHECK_CLOSE(0.02630214, rveval::evalSkellam(0, 115.2, 114.3, false), PREC);
    // otherwise... 
    // dskellam(0, 400.0, 10.0, log = T)
    CHECK_CLOSE(-286.8469, rveval::evalSkellam(0, 400.0, 10.0, true), PREC);
    // dskellam(0, 400.0, 10.0, log = F)
    CHECK_CLOSE(2.654379e-125, rveval::evalSkellam(0, 400.0, 10.0, false), PREC);
 
    ///////////////////////////
    // when first arg is +-1 //
    ///////////////////////////
    // z < 7.75
    // dskellam(1, 1.0, .025, log = T)
    CHECK_CLOSE(-1.012526, rveval::evalSkellam(1.0, 1.0, .025, true), PREC);
    // dskellam(1, 1.0, .025, log = F)
    CHECK_CLOSE(0.363300132, rveval::evalSkellam(1.0, 1.0, .025, false), PREC);
    // dskellam(-1, 1.0, .025, log = T)
    CHECK_CLOSE(-4.701405, rveval::evalSkellam(-1, 1.0, .025, true), PREC);
    // dskellam(-1, 1.0, .025, log = F)
    CHECK_CLOSE(0.009082504, rveval::evalSkellam(-1, 1.0, .025, false), PREC);
    // z < 500
    // dskellam(1, 115.2, 114.3, log = T)
    CHECK_CLOSE(-3.636367, rveval::evalSkellam(1, 115.2, 114.3, true), PREC);
    // dskellam(1, 115.2, 114.3, log = F)
    CHECK_CLOSE(0.02634789, rveval::evalSkellam(1, 115.2, 114.3, false), PREC);
    // otherwise... 
    // dskellam(1, 400.0, 10.0, log = T)
    CHECK_CLOSE(-285.0065, rveval::evalSkellam(1, 400.0, 10.0, true), PREC);
    // dskellam(1, 400.0, 10.0, log = F)
    CHECK_CLOSE(1.672127e-124, rveval::evalSkellam(1, 400.0, 10.0, false), PREC);

    ///////////////////
    // miscellaneous //
    ///////////////////
 
    // dskellam(-3, .2, .3, log = F)
    CHECK_CLOSE(0.002770575, rveval::evalSkellam(-3, .2, .3, false), PREC);

    // dskellam(-3, .2, .3, log = T)
    CHECK_CLOSE(-5.8887, rveval::evalSkellam(-3, .2, .3, true), PREC);
    
    // dskellam(3, .2, .3, log = F)
    CHECK_CLOSE(0.0008209112, rveval::evalSkellam(3, .2, .3, false), PREC);

    // dskellam(3, .2, .3, log = T)
    CHECK_CLOSE(-7.105096, rveval::evalSkellam(3, .2, .3, true), PREC);

    // out of bounds parameters    
    CHECK_CLOSE(-1.0/0.0, rveval::evalSkellam(-1, .5, -.5, true), PREC);
    CHECK_CLOSE(-1.0/0.0, rveval::evalSkellam(-1, -.5, .5, true), PREC);
    CHECK_CLOSE(0.0, rveval::evalSkellam(-1, .5, -.5, false), PREC);
    CHECK_CLOSE(0.0, rveval::evalSkellam(-1, -.5, .5, false), PREC);

}


TEST_FIXTURE(DensFixture, evalWishartTest)
{
    // library(LaplacesDemon)
    // dwishart(matrix(c(2,-.3,-.3,4),2,2), 3, matrix(c(1,.1,.1,1),2,2), log=T)
    // -5.5765548037951
    // dwishart(matrix(c(2,-.3,-.3,4),2,2), 3, matrix(c(1,.1,.1,1),2,2))
    // 0.00378558516193494

    double goodLogDens =  rveval::evalWishart<2,double>(Omega, Sinv, 3, true);   
    CHECK_CLOSE(-5.57655, goodLogDens, PREC);
    double goodDens = rveval::evalWishart<2,double>(Omega, Sinv, 3, false);
    CHECK_CLOSE(0.003785, goodDens, PREC);
    double badLogDens = rveval::evalWishart<2,double>(Omega, Sinv, 1, true);
    CHECK_CLOSE(-1.0/0.0, badLogDens, PREC);
    double badDens = rveval::evalWishart<2,double>(Omega,Sinv, 1, false);
    CHECK_CLOSE(0.0, badDens, PREC);
    double badLogDens2 = rveval::evalWishart<2,double>(Omega, badCovMat, 3, true);
    CHECK_CLOSE(-1.0/0.0, badLogDens2, PREC);
    double badDens2 = rveval::evalWishart<2,double>(Omega, badCovMat, 3, false);
    CHECK_CLOSE(0.0, badDens2, PREC);
    double badLogDens3 = rveval::evalWishart<2,double>(badCovMat, Sinv, 3, true);
    double badDens3 = rveval::evalWishart<2,double>(badCovMat,Sinv,3,false);
    CHECK_CLOSE(-1.0/0.0, badLogDens3, PREC);
    CHECK_CLOSE(0.0, badDens3, PREC);
        
}


TEST_FIXTURE(DensFixture, evalInvWishart)
{
    //library(LaplacesDemon)
    //dinvwishart(matrix(c(2,-.3,-.3,4),2,2), 3, matrix(c(1,.1,.1,1),2,2))
    //0.0001079824
    //dinvwishart(matrix(c(2,-.3,-.3,4),2,2), 3, matrix(c(1,.1,.1,1),2,2), log = T)
    // -9.133543
    
    double goodLogDens = rveval::evalInvWishart<2,double>(Omega, S, 3, true);
    double goodDens = rveval::evalInvWishart<2,double>(Omega,S,3,false);
    double badLogDens = rveval::evalInvWishart<2,double>(Omega,badCovMat,3,true);
    double badDens = rveval::evalInvWishart<2,double>(Omega,badCovMat,3,false);
    double badLogDens2 = rveval::evalInvWishart<2,double>(badCovMat,S,3,true);
    double badDens2 = rveval::evalInvWishart<2,double>(badCovMat,S,3,false);
    double badLogDens3 = rveval::evalInvWishart<2,double>(Omega,S,1,true);
    double badDens3 = rveval::evalInvWishart<2,double>(Omega,S,1,false);
    
    CHECK_CLOSE(-1.0/0.0, badLogDens, PREC);
    CHECK_CLOSE(-1.0/0.0, badLogDens2, PREC);
    CHECK_CLOSE(-1.0/0.0, badLogDens3, PREC);
    CHECK_CLOSE(0.0, badDens, PREC);
    CHECK_CLOSE(0.0, badDens2, PREC);
    CHECK_CLOSE(0.0, badDens3, PREC);

    CHECK_CLOSE(-9.133543, goodLogDens, PREC);
    CHECK_CLOSE(0.0001079824, goodDens, PREC);
        
        
}


