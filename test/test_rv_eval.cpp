#include <catch2/catch.hpp>
#include <pf/rv_eval.h>

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


TEST_CASE_METHOD(DensFixture, "univariate normal test", "[dentiies]") {
    // via R dnorm(.5, 2, 1.5, T)
    REQUIRE(rveval::evalUnivNorm<double>(.5, 2.0, 1.5, true) == Approx( -1.824404) );
    REQUIRE(rveval::evalUnivNorm<double>(.5, 2.0, 1.5, false) == Approx( 0.1613138));
}


TEST_CASE_METHOD(DensFixture, "univScaledT", "[densities]")
{
    // via dt.scaled(1.23, 3.6, 23.2, 1.7, log =T)
    REQUIRE(rveval::evalScaledT<double>(1.23, scaledTMu, scaledTSigma, scaledTdof, true) == Approx( -10.39272) );
    REQUIRE(rveval::evalScaledT<double>(1.23, scaledTMu, scaledTSigma, scaledTdof, false) == Approx( 3.065496e-05) );

    // test broken ones
    REQUIRE(rveval::evalScaledT<double>(1.23, scaledTMu, -1.0*scaledTSigma, scaledTdof, true) == -std::numeric_limits<double>::infinity()  );
    REQUIRE(rveval::evalScaledT<double>(1.23, scaledTMu, -1.0*scaledTSigma, scaledTdof, false) == 0.0 );
    REQUIRE(rveval::evalScaledT<double>(1.23, scaledTMu, scaledTSigma, -scaledTdof, true) == -std::numeric_limits<double>::infinity() );
    REQUIRE(rveval::evalScaledT<double>(1.23, scaledTMu, scaledTSigma, -scaledTdof, false) == 0.0 );
}


TEST_CASE_METHOD(DensFixture, "univNormCDFTest", "[densities]")
{
    // via R pnorm(.1)
    REQUIRE(rveval::evalUnivStdNormCDF<double>(.1) == Approx( 0.5398278) );
    REQUIRE(rveval::evalUnivStdNormCDF<double>(0.0) == Approx(.5) );
    REQUIRE(rveval::evalUnivStdNormCDF<double>(1.0/0.0) == Approx(1.0));
    REQUIRE(rveval::evalUnivStdNormCDF<double>(-std::numeric_limits<float_t>::infinity()) == Approx( 0.0) );
}


TEST_CASE_METHOD(DensFixture, "truncNormTest", "[densities]")
{
    // check bounds can be infinite
    REQUIRE(rveval::evalUnivTruncNorm<double>(0.0, 0.0, 1.0, -std::numeric_limits<float_t>::infinity(), std::numeric_limits<float_t>::infinity(), true) == Approx(
                rveval::evalUnivNorm<double>(0.0, 0.0, 1.0, true) ) );
    REQUIRE(rveval::evalUnivTruncNorm<double>(0.0, 0.0, 1.0, -std::numeric_limits<float_t>::infinity(), std::numeric_limits<float_t>::infinity(), false) == Approx(
                rveval::evalUnivNorm<double>(0.0, 0.0, 1.0, false)));
    // check support is good
    REQUIRE(rveval::evalUnivTruncNorm<double>(0.0, 0.0, 1.0, .1, 20.0, false) == 0.0);
    REQUIRE(rveval::evalUnivTruncNorm<double>(0.0, 0.0, 1.0, .1, 20.0, true) == -std::numeric_limits<double>::infinity() );
    REQUIRE(rveval::evalUnivTruncNorm<double>(0.0, 0.0, 1.0, -20.0, -.1, false) == 0.0);
    REQUIRE(rveval::evalUnivTruncNorm<double>(0.0, 0.0, 1.0, -20.0, -.1, true) == -std::numeric_limits<double>::infinity() );
    // check a real evaluation comparing it to R's truncnorm::dtruncnorm(0, -5, 5, 0, 2)
    REQUIRE(rveval::evalUnivTruncNorm<double>(0.0, 0.0, 2.0, -5.0, 5.0, false) == Approx( 0.2019796) );
    REQUIRE(rveval::evalUnivTruncNorm<double>(0.0, 0.0, 2.0, -5.0, 5.0, true) == Approx(-1.599589) );
}


TEST_CASE_METHOD(DensFixture, "multivariateGaussianTest", "[densities]")
{
    // via R mvtnorm::dmvnorm(c(.02, -.01),sigma=matrix(c(3,1,1,3),nrow=2))
    double num = rveval::evalMultivNorm<bigdim,double>(x, mu, covMat, true);    
    REQUIRE( std::abs(num - (-2.877716587249263)) < .0001 );

    double num2 = rveval::evalMultivNorm<bigdim,double>(x, mu, covMat, false);
    REQUIRE( std::abs(num2 - 0.05626309) < .0001 );

    double badNormLogDens = rveval::evalMultivNorm<bigdim,double>(x,mu,badCovMat,true);
    double badNormDens = rveval::evalMultivNorm<bigdim,double>(x,mu,badCovMat,false);
    REQUIRE(- std::numeric_limits<double>::infinity() ==  badNormLogDens);
    REQUIRE(badNormDens == 0.0);
}


TEST_CASE_METHOD(DensFixture, "multivariateTTest", "[densities]")
{
    // via R dmvt(c(.02, -.01), c(0,0), sigma=matrix(c(3,1,1,3),nrow=2), 10)
    // reusing some of the multivariate normal variables, but the names are
    // a bit off..so I apologize
    double num = rveval::evalMultivT<bigdim,double>(x, mu, covMat, 3, true);    
    REQUIRE( std::abs(num - ( -2.877796 )) < .001);

    double num2 = rveval::evalMultivT<bigdim,double>(x, mu, covMat, 3, false);
    REQUIRE( std::abs(num2 -0.05625863) < .0001);

    double badNormLogDens = rveval::evalMultivT<bigdim,double>(x,mu,badCovMat,3,true);
    double badNormDens = rveval::evalMultivT<bigdim,double>(x,mu,badCovMat,3,false);
    REQUIRE(badNormLogDens == - std::numeric_limits<double>::infinity());
    REQUIRE(badNormDens == 0.0);
}


TEST_CASE_METHOD(DensFixture, "multivNormWoodburyTest", "[densities]")
{
    double normeval = rveval::evalMultivNorm<bigdim,double>(x, mu, covMat, true);
    double wbdanormeval = rveval::evalMultivNormWBDA<bigdim,smalldim,double>(x, mu, A, U, C, true);
    REQUIRE( std::abs(normeval - wbdanormeval) < .0001 );
                
    double normeval2 = rveval::evalMultivNorm<bigdim,double>(x, mu, covMat, false);
    double wbdanormeval2 = rveval::evalMultivNormWBDA<bigdim,smalldim,double>(x, mu, A, U, C, false);
    REQUIRE( std::abs(normeval2 - wbdanormeval2) < .0001 );
}


TEST_CASE_METHOD(DensFixture, "univBeta", "[densities]")
{
    // via R dbeta(.5, .2, .3, F)
    REQUIRE(rveval::evalUnivBeta<double>(.5, beta1p, beta2p, true) == Approx( -1.007776) );

    REQUIRE(rveval::evalUnivBeta<double>(.5, beta1p, beta2p, false) == Approx( 0.3650299 ));

    REQUIRE(rveval::evalUnivBeta<double>(-.5, beta1p, beta2p, true) == -std::numeric_limits<double>::infinity());

    REQUIRE(rveval::evalUnivBeta<double>(-.5, beta1p, beta2p, false) == 0.0);
}


TEST_CASE_METHOD(DensFixture, "invGammaTest", "[densities]")
{
    REQUIRE(rveval::evalUnivInvGamma<double>(3.2, invgamma1p, invgamma2p, true) == Approx( -4.215113 ));
    REQUIRE(rveval::evalUnivInvGamma<double>(3.2, invgamma1p, invgamma2p, false) == Approx( 0.01477065) );
                
    REQUIRE(rveval::evalUnivInvGamma<double>(-3.2, invgamma1p, invgamma2p, true) == -std::numeric_limits<double>::infinity() ); 
    REQUIRE(rveval::evalUnivInvGamma<double>(-3.2, invgamma1p, invgamma2p, false) == 0.0);
}


TEST_CASE_METHOD(DensFixture, "halfNormalTest", "[densities]")
{
    // fdrtool::dhalfnorm(.2, sqrt(pi/(2*1.5)))
    REQUIRE(rveval::evalUnivHalfNorm<double>(.2, sigmaSquaredHN, true) == Approx( -0.4418572400321429));
    REQUIRE(rveval::evalUnivHalfNorm<double>(.2, sigmaSquaredHN, false) == Approx(0.6428414009228908));
    REQUIRE(rveval::evalUnivHalfNorm<double>(-.2, sigmaSquaredHN, false) == 0.0);
    REQUIRE(rveval::evalUnivHalfNorm<double>(-.2, sigmaSquaredHN, true) ==  -std::numeric_limits<double>::infinity());
}


TEST_CASE_METHOD(DensFixture, "ctsUniformTest", "[densities]")
{
    REQUIRE(rveval::evalUniform<double>((lower+upper)/2.0, lower, upper, false) == Approx( 1.0/(upper - lower)));
    REQUIRE(rveval::evalUniform<double>((lower+upper)/2.0, lower, upper, true) == Approx(-std::log(upper-lower)) );
    REQUIRE(rveval::evalUniform<double>(lower-.01, lower, upper, false) ==  0.0);
    REQUIRE(rveval::evalUniform<double>(lower-.01, lower, upper, true) == -std::numeric_limits<double>::infinity());
}


TEST_CASE_METHOD(DensFixture, "evalLogNormalTest", "[densities]")
{
    // dlnorm(.2, .5, 5.3, T)
    REQUIRE(rveval::evalLogNormal<double>(.2, lnMu, lnSigma, true) == Approx( -1.056412288363436));
    REQUIRE(rveval::evalLogNormal<double>(.2, lnMu, lnSigma, false) == Approx(0.3477010262745334));
    REQUIRE(rveval::evalLogNormal<double>(-2, lnMu, lnSigma, true) == -std::numeric_limits<double>::infinity());
    REQUIRE(rveval::evalLogNormal<double>(-2, lnMu, lnSigma, false) == 0.0);
}


//TEST_CASE_METHOD(DensFixture, evalLogitNormalTest)
//{
//    
//}
//
//
//TEST_CASE_METHOD(DensFixture, evalTwiceFisherNormalTest)
//{
//    
//}


TEST_CASE_METHOD(DensFixture, "evalBernoulliTest", "[densities]")
{
    /// dbinom(1, 1, .001, T) 
    REQUIRE( Approx(-6.907755) ==  rveval::evalBernoulli(1, .001, true));
    REQUIRE( Approx(0.001) ==  rveval::evalBernoulli(1, .001, false));
    
    REQUIRE( -std::numeric_limits<double>::infinity() == rveval::evalBernoulli(-1, .5, true));
    REQUIRE(0.0 == rveval::evalBernoulli(1, 1.1, false));
}


TEST_CASE_METHOD(DensFixture, "evalSkellamTest", "[densities]")
{
    /////////////////////////
    // when first arg is 0 //
    /////////////////////////
    // z < 7.75
    // dskellam(0, 1.0, .025, log = T)
    REQUIRE(Approx(-1.000155) == rveval::evalSkellam(0, 1.0, .025, true));
    // dskellam(0, 1.0, .025, log = F)
    REQUIRE( Approx(0.3678226) == rveval::evalSkellam(0, 1.0, .025, false));
    // z < 500
    // dskellam(0, 115.2, 114.3, log = T)
    REQUIRE( Approx(-3.638105) == rveval::evalSkellam(0, 115.2, 114.3, true));
    // dskellam(0, 115.2, 114.3, log = F)
    REQUIRE( Approx(0.02630214) == rveval::evalSkellam(0, 115.2, 114.3, false));
    // otherwise... 
    // dskellam(0, 400.0, 10.0, log = T)
    REQUIRE( Approx(-286.8469) ==  rveval::evalSkellam(0, 400.0, 10.0, true));
    // dskellam(0, 400.0, 10.0, log = F)
    REQUIRE( Approx(2.654379e-125) == rveval::evalSkellam(0, 400.0, 10.0, false));
 
    ///////////////////////////
    // when first arg is +-1 //
    ///////////////////////////
    // z < 7.75
    // dskellam(1, 1.0, .025, log = T)
    REQUIRE( Approx(-1.012526) == rveval::evalSkellam(1.0, 1.0, .025, true));
    // dskellam(1, 1.0, .025, log = F)
    REQUIRE( Approx(0.363300132) == rveval::evalSkellam(1.0, 1.0, .025, false));
    // dskellam(-1, 1.0, .025, log = T)
    REQUIRE( Approx(-4.701405) == rveval::evalSkellam(-1, 1.0, .025, true));
    // dskellam(-1, 1.0, .025, log = F)
    REQUIRE( Approx(0.009082504) == rveval::evalSkellam(-1, 1.0, .025, false));
    // z < 500
    // dskellam(1, 115.2, 114.3, log = T)
    REQUIRE( Approx(-3.636367) == rveval::evalSkellam(1, 115.2, 114.3, true));
    // dskellam(1, 115.2, 114.3, log = F)
    REQUIRE( Approx(0.02634789) ==  rveval::evalSkellam(1, 115.2, 114.3, false));
    // otherwise... 
    // dskellam(1, 400.0, 10.0, log = T)
    REQUIRE( Approx(-285.0065) == rveval::evalSkellam(1, 400.0, 10.0, true));
    // dskellam(1, 400.0, 10.0, log = F)
    REQUIRE( Approx(1.672127e-124) == rveval::evalSkellam(1, 400.0, 10.0, false));

    /////////////////////////////////////
    // two above are false and z > 100 //
    /////////////////////////////////////
    // dskellam(2, 100.0, 1.3, log = T)
    REQUIRE( Approx(-76.72014) == rveval::evalSkellam(2, 100.0, 1.3, true));
    // dskellam(2, 100.0, 1.3, log = F)
    REQUIRE( Approx(4.795877e-34) == rveval::evalSkellam(2, 100.0, 1.3, false));
   

    ///////////////////
    // miscellaneous //
    ///////////////////
 
    // dskellam(-3, .2, .3, log = F)
    REQUIRE( Approx(0.002770575) == rveval::evalSkellam(-3, .2, .3, false));

    // dskellam(-3, .2, .3, log = T)
    REQUIRE( Approx(-5.8887) == rveval::evalSkellam(-3, .2, .3, true));
    
    // dskellam(3, .2, .3, log = F)
    REQUIRE( Approx(0.0008209112) == rveval::evalSkellam(3, .2, .3, false));

    // dskellam(3, .2, .3, log = T)
    REQUIRE( Approx(-7.105096) == rveval::evalSkellam(3, .2, .3, true));

    // out of bounds parameters    
    REQUIRE(-std::numeric_limits<double>::infinity() == rveval::evalSkellam(-1, .5, -.5, true));
    REQUIRE(-std::numeric_limits<double>::infinity() ==  rveval::evalSkellam(-1, -.5, .5, true));
    REQUIRE(0.0 == rveval::evalSkellam(-1, .5, -.5, false));
    REQUIRE(0.0 == rveval::evalSkellam(-1, -.5, .5, false));
}


TEST_CASE_METHOD(DensFixture, "evalWishartTest", "[densities]")
{
    // library(LaplacesDemon)
    // dwishart(matrix(c(2,-.3,-.3,4),2,2), 3, matrix(c(1,.1,.1,1),2,2), log=T)
    // -5.5765548037951
    // dwishart(matrix(c(2,-.3,-.3,4),2,2), 3, matrix(c(1,.1,.1,1),2,2))
    // 0.00378558516193494

    double goodLogDens =  rveval::evalWishart<2,double>(Omega, Sinv, 3, true);   
    REQUIRE( Approx(-5.57655) == goodLogDens);
    double goodDens = rveval::evalWishart<2,double>(Omega, Sinv, 3, false);
    REQUIRE( std::abs(0.003785 - goodDens) < .0001 );
    double badLogDens = rveval::evalWishart<2,double>(Omega, Sinv, 1, true);
    REQUIRE(-std::numeric_limits<double>::infinity() == badLogDens);
    double badDens = rveval::evalWishart<2,double>(Omega,Sinv, 1, false);
    REQUIRE(0.0 == badDens);
    double badLogDens2 = rveval::evalWishart<2,double>(Omega, badCovMat, 3, true);
    REQUIRE(-std::numeric_limits<double>::infinity() == badLogDens2);
    double badDens2 = rveval::evalWishart<2,double>(Omega, badCovMat, 3, false);
    REQUIRE(0.0 == badDens2);
    double badLogDens3 = rveval::evalWishart<2,double>(badCovMat, Sinv, 3, true);
    double badDens3 = rveval::evalWishart<2,double>(badCovMat,Sinv,3,false);
    REQUIRE(-std::numeric_limits<double>::infinity() ==  badLogDens3);
    REQUIRE(0.0 == badDens3);
}


TEST_CASE_METHOD(DensFixture, "evalInvWishart", "[densities]")
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
    
    REQUIRE(-std::numeric_limits<double>::infinity() ==  badLogDens);
    REQUIRE(-std::numeric_limits<double>::infinity() ==  badLogDens2);
    REQUIRE(-std::numeric_limits<double>::infinity() ==  badLogDens3);
    REQUIRE(0.0 == badDens);
    REQUIRE(0.0 == badDens2);
    REQUIRE(0.0 == badDens3);

    REQUIRE( Approx(-9.133543) == goodLogDens);
    REQUIRE( Approx(0.0001079824) == goodDens);
}
