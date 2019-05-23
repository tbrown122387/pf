#include <Eigen/Dense>
#include <UnitTest++/UnitTest++.h>
#include "utils.h"


#define PREC .0001

TEST(data_reader_test){
    // cannot have header!
    // need to know num cols
    std::vector<Eigen::Matrix<double,2,1>> data = utils::readInData<2,double>("test_data.csv");
    CHECK(data.size() == 1);
    CHECK_CLOSE(1.23, data[0](0), PREC);
    CHECK_CLOSE(4.56, data[0](1), PREC);
}

