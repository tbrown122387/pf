#include <fstream> // for ifstream

#include <pf/resamplers.h> 

#include "data_reader.h"
#include "acv_bs.h" // bs filter for svol

// some template parameters
#define dimstate 4 
#define dimobs   2
#define numparts 1000
#define FLOATTYPE float // choose float (faster) or double (slower)


int main(int argc, char** argv)
{

    // choose your resampler
    //using multinomialR = mn_resampler<numparts,dimstate,FLOATTYPE>;
    using fastMultinomR= mn_resamp_fast1<numparts,dimstate,FLOATTYPE>;

    // model parameters 
    FLOATTYPE var_s0 = 4.0;
    FLOATTYPE var_u0 = 1.0;
    FLOATTYPE var_s = .02;
    FLOATTYPE var_u = .001;
    FLOATTYPE scale_y = .1;
    FLOATTYPE nu_y = 10.0;
    FLOATTYPE Delta = .1;

    // instantiate a boostrap filter almost constant variance model object
    acv_bs<numparts, dimstate, dimobs, fastMultinomR,FLOATTYPE> 
        mod(var_s0, var_u0, var_s, var_u, scale_y, nu_y, Delta);

    // read in some data
    using osv = Eigen::Matrix<FLOATTYPE,dimobs,1>;   
    std::vector<osv> data = readInData<FLOATTYPE,dimobs>("data.csv", ' ');

    // optional lambda for filtering expectation approximations 
    using ssv = Eigen::Matrix<FLOATTYPE,dimstate,1>;
    using Mat = Eigen::Matrix<FLOATTYPE,Eigen::Dynamic,Eigen::Dynamic>;    // at each time point, which approximates the 
    auto idtyLambda = [](const ssv& xt) -> const Mat  
    {
        return xt;
    };
    std::vector<std::function<const Mat(const ssv&)>> v;
    v.push_back(idtyLambda);

    // iterate over the data (finally)
    // printing stuff is obviously optional
    for(size_t row = 0; row < data.size(); ++row){
        mod.filter(data[row], v); 
        std::cout << mod.getExpectations()[0] << "\n";
    }
   
   return 1; 
}
