#include <catch2/catch_all.hpp>

#include <pf/cf_filters.h>

#define bigdim 2
#define smalldim 1


using namespace pf;
using Catch::Approx;


class HmmFixture
{
public:

	using Vec = Eigen::Matrix<double,bigdim,1>;
	using Mat = Eigen::Matrix<double,bigdim,bigdim>;

	// data members    
    pf::filters::hmm<bigdim,smalldim,double> mod;
	
	HmmFixture() 
	{
		Vec initial_log_probs;
		initial_log_probs << std::log(.7), std::log(.3) ;
		Mat log_trans_mat;
		log_trans_mat << std::log(.9) , std::log(.1), std::log(.2), std::log(.8);
		mod = pf::filters::hmm<bigdim,smalldim,double>(initial_log_probs, log_trans_mat);
	}


};

TEST_CASE_METHOD(HmmFixture, "test log product", "[hmm]")
{

	Mat A;
	A << std::log(.9), std::log(.1), std::log(.2), std::log(.8);
	Vec x;
	x << std::log(.3) , std::log(.7);
	// should be log (.34,.62) = -1.0788097, -0.4780358
	Vec result = mod.log_product(A, x);
	REQUIRE( std::abs( result(0) - std::log(.34)) < .0001 );
	REQUIRE( std::abs( result(1) == std::log(.62)) < .0001 );
}

TEST_CASE_METHOD(HmmFixture, "test correct update 1", "[hmm]")
{
    // all probability goes on first element
    Vec logCondDensVec;
    logCondDensVec << std::log(1.0), std::log(0.0);
    mod.update(logCondDensVec);

    REQUIRE(mod.getFilterVecLogProbs()(0) == log(1.0));
    REQUIRE(mod.getFilterVecLogProbs()(1) == log(0.0));
    REQUIRE(mod.getLogCondLike() == std::log(.7));


	// one more update
	logCondDensVec << std::log(.5), std::log(.5);
	mod.update(logCondDensVec);
	
	REQUIRE( std::abs( mod.getFilterVecLogProbs()(0) - std::log(.9)) < .0001 );
	REQUIRE( std::abs( mod.getFilterVecLogProbs()(1) - std::log(.1)) < .0001 );
	REQUIRE( std::abs( mod.getLogCondLike() - std::log(.5)) < .00001);

}

TEST_CASE_METHOD(HmmFixture, "test correct update 2", "[hmm]")
{
	// all probability goes on first element
    Vec logCondDensVec;
	logCondDensVec << std::log(0.0), std::log(1.0);
    mod.update(logCondDensVec);

    REQUIRE(mod.getFilterVecLogProbs()(0) == log(0.0));
    REQUIRE(mod.getFilterVecLogProbs()(1) == log(1.0));
    REQUIRE(mod.getLogCondLike() == std::log(.3));

	// one more update
	logCondDensVec << std::log(.5), std::log(.5);
	mod.update(logCondDensVec);
	
	REQUIRE( std::abs( mod.getFilterVecLogProbs()(0) - std::log(.2)) < .0001 );
	REQUIRE( std::abs( mod.getFilterVecLogProbs()(1) - std::log(.8)) < .0001 );
	REQUIRE( std::abs( mod.getLogCondLike() - std::log(.5)) < .00001);

}

