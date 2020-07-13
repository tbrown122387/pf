#include <fstream> // for ifstream

#include <pf/resamplers.h> 

#include "resamp_comparison.h" // function header
#include "data_reader.h"
#include "svol_bs.h" // bs filter for svol

// some template parameters
#define dimstate 1
#define dimobs   1
#define numparts 5000
#define FLOATTYPE float // choose float (faster) or double (slower)


void run_resamp_comparison(const std::string &csv)
{
    // "state size vector"
    using ssv = Eigen::Matrix<FLOATTYPE,dimstate,1>;
    // "observation sized vector"
    using osv = Eigen::Matrix<FLOATTYPE,dimobs,1>;    
    // dynamically-sized square matrix
    using Mat = Eigen::Matrix<FLOATTYPE,Eigen::Dynamic,Eigen::Dynamic>;

    using multinomialR = mn_resampler        <numparts,dimstate,FLOATTYPE>;
    using residR       = resid_resampler     <numparts,dimstate,FLOATTYPE>;
    using stratifR     = stratif_resampler   <numparts,dimstate,FLOATTYPE>;
    using systematicR  = systematic_resampler<numparts,dimstate,FLOATTYPE>;
    using fastMultinomR= mn_resamp_fast1     <numparts,dimstate,FLOATTYPE>;

    // model parameters that are assumed known
    FLOATTYPE phi = .91;
    FLOATTYPE beta = .5;
    FLOATTYPE sigma = 1.0;

    // the same model in four different particle filters
    svol_bs<numparts, dimstate, dimobs, multinomialR, FLOATTYPE> bssvol1(phi, beta, sigma);
    svol_bs<numparts, dimstate, dimobs, residR      , FLOATTYPE> bssvol2(phi, beta, sigma);
    svol_bs<numparts, dimstate, dimobs, stratifR    , FLOATTYPE> bssvol3(phi, beta, sigma);
    svol_bs<numparts, dimstate, dimobs, systematicR , FLOATTYPE> bssvol4(phi, beta, sigma);
    svol_bs<numparts, dimstate, dimobs, fastMultinomR,FLOATTYPE> bssvol5(phi, beta, sigma);

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

    // iterate over the data (finally)
    // printing stuff is obviously optional
    for(size_t row = 0; row < data.size(); ++row){
        bssvol1.filter(data[row], v);
        bssvol2.filter(data[row], v);
        bssvol3.filter(data[row], v);
        bssvol4.filter(data[row], v);
        bssvol5.filter(data[row], v);

        std::cout << bssvol1.getExpectations()[0] << ", ";
        std::cout << bssvol2.getExpectations()[0] << ", ";
        std::cout << bssvol3.getExpectations()[0] << ", ";
        std::cout << bssvol4.getExpectations()[0] << ", ";
        std::cout << bssvol5.getExpectations()[0] << "\n";
    }
    
}
