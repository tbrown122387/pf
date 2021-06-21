#include <fstream> // for ifstream

#include <pf/resamplers.h> // for mn_resampler

#include "svol_comparison.h" // function header
#include "data_reader.h"
#include "svol_bs.h" // bs filter for svol
#include "svol_apf.h" // apf filter for svol
#include "svol_sisr.h" // sisr filter for svol
#include "svol_bs_withcovs.h"


// some template parameters
#define dimstate 1
#define dimobs   1
#define numparts 100
#define FLOATTYPE float // choose float (faster) or double (slower)

using namespace pf::resamplers;

void run_svol_comparison(const std::string &csv)
{
    // "state size vector"
    using ssv = Eigen::Matrix<FLOATTYPE,dimstate,1>;
    // "observation sized vector"
    using osv = Eigen::Matrix<FLOATTYPE,dimobs,1>;    
    // "covariate-sized vector"
    using cvsv = Eigen::Matrix<FLOATTYPE,1,1>;
    // dynamically-sized square matrix
    using Mat = Eigen::Matrix<FLOATTYPE,Eigen::Dynamic,Eigen::Dynamic>;
    
    // model parameters that are assumed known
    FLOATTYPE phi = .91;
    FLOATTYPE beta = .5;
    FLOATTYPE sigma = 1.0;

    // the same model in three different particle filters
    svol_bs<numparts, dimstate, dimobs,mn_resampler<numparts,dimstate,FLOATTYPE>,FLOATTYPE> bssvol(phi, beta, sigma);
    svol_apf<numparts,dimstate,dimobs,mn_resampler<numparts,dimstate,FLOATTYPE>,FLOATTYPE> apfsvol(phi,beta,sigma);
    svol_sisr<numparts,dimstate,dimobs,mn_resampler<numparts,dimstate,FLOATTYPE>,FLOATTYPE> sisrsvol(phi,beta,sigma);
    svol_bs_wc<numparts,dimstate,dimobs,mn_resampler<numparts,dimstate,FLOATTYPE>,FLOATTYPE> bssvolwc(phi,beta,sigma);

    // read in some data
    std::vector<osv> data = readInData<FLOATTYPE,dimobs>(csv);

    // optional lambda
    // filter() will use this to approx.
    // an expectation now
    // in this case, the sample mean is calculated
    // at each time point, which approximates the 
    // the sequence of filtering means
    auto idtyLambda = [](const ssv& xt) -> const Mat  
    {
        return xt;
    };
    std::vector<std::function<const Mat(const ssv&)>> v;
    v.push_back(idtyLambda);

    // another identity lambda because the signature for "with covariate" models is different
    auto idtyLambdaWC = [](const ssv& xt, const cvsv& zt) -> const Mat
    {
        return xt;
    };
    std::vector<std::function<const Mat(const ssv&, const cvsv&)>> v2;
    v2.push_back(idtyLambdaWC);

    // iterate over the data (finally)
    // printing stuff is obviously optional
    for(size_t row = 0; row < data.size(); ++row){
        bssvol.filter(data[row], v);
        apfsvol.filter(data[row], v);
        sisrsvol.filter(data[row], v);
        bssvolwc.filter(data[row], cvsv::Zero(), v2);
         
        std::cout << bssvol.getExpectations()[0] << ", " 
                  << apfsvol.getExpectations()[0] << ", "
                  << sisrsvol.getExpectations()[0] << ", "
                  << bssvolwc.getExpectations()[0] << ", "
                  << bssvol.getLogCondLike() << ", " 
                  << apfsvol.getLogCondLike() << ", "
                  << sisrsvol.getLogCondLike() << ","
                  << bssvolwc.getLogCondLike() << "\n";
    }
    
}
