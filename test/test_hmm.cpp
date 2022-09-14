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
 
  
  
   HmmFixture() 
    {
    }


};


TEST_CASE_METHOD(HmmFixture, "test correct init", "[hmm]")
{
    Vec initial_log_probs;
    initial_log_probs << std::log(.5), std::log(.5) ;
    Mat log_trans_mat;
    log_trans_mat << std::log(.5) , std::log(.5), std::log(.5), std::log(.5);
    pf::filters::hmm<bigdim,smalldim,double> mod(initial_log_probs, log_trans_mat);
}

TEST_CASE_METHOD(HmmFixture, "test correct update", "[hmm]")
{
    // same as above
    Vec initial_log_probs;
    initial_log_probs << std::log(.5), std::log(.5) ;
    Mat log_trans_mat;
    log_trans_mat << std::log(.5) , std::log(.5), std::log(.5), std::log(.5);
    pf::filters::hmm<bigdim,smalldim,double> mod(initial_log_probs, log_trans_mat);


    Vec logCondDensVec;
    // all probability goes on first element
    logCondDensVec << std::log(1.0), std::log(0.0);
    mod.update(logCondDensVec);

    REQUIRE(mod.getFilterVecLogProbs()(0) == log(1.0));
    REQUIRE(mod.getFilterVecLogProbs()(1) == log(0.0));
    REQUIRE(mod.getLogCondLike() == std::log(.5));
}

