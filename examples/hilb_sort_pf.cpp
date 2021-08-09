#include <fstream> // for ifstream

#include <pf/resamplers.h> // for sys_hilb_resampler and mn_resampler

#include "hilb_sort_pf.h" // function header
#include "data_reader.h"
#include "svol_sisr_hilb.h" // sisr filter for svol with hilbert sort resampling
#include "svol_bs.h" // for bootstrap comparison filter


// some template parameters
#define numbits  5
#define numparts 100
#define FLOATTYPE float // choose float (faster) or double (slower)


using namespace pf::resamplers;

void run_hilb_pf_example(const std::string &csv)
{
    // "state size vector"
    using ssv = Eigen::Matrix<FLOATTYPE,1,1>;
    // "observation sized vector"
    using osv = Eigen::Matrix<FLOATTYPE,1,1>;    
    // "U sized vector"
    using usv = Eigen::Matrix<FLOATTYPE,1,1>;
    // "U sized vector for resampling"
    using usvr = Eigen::Matrix<FLOATTYPE,1,1>;
    // array of all Us needed to filter for one time point
    using arrayUs = std::array<usv,numparts>;    
 
    // model parameters that are assumed known
    FLOATTYPE phi = .91;
    FLOATTYPE beta = .5;
    FLOATTYPE sigma = 1.0;

    // a hilbert pf compared with a standard boostrap filter
    svol_sisr_hilb<numparts,numbits,sys_hilb_resampler<numparts,1,numbits,FLOATTYPE>,FLOATTYPE> sisrsvol_hilb(phi,beta,sigma);
    svol_bs<numparts, 1, 1, mn_resampler<numparts,1,FLOATTYPE>,FLOATTYPE> bssvol(phi, beta, sigma);
    
    // read in some data
    std::vector<osv> data = readInData<FLOATTYPE,1>(csv);

    // iterate over the data (finally)
    // printing stuff is obviously optional
    arrayUs this_time_Us;
    pf::rvsamp::UnivNormSampler<FLOATTYPE> std_norm_sampler;
    for(size_t row = 0; row < data.size(); ++row){
         
        // refresh normal random variates
        for(size_t p = 0; p < numparts; ++p){
            this_time_Us[p] = usv(std_norm_sampler.sample());
        }
        sisrsvol_hilb.filter(data[row], this_time_Us, usvr(std_norm_sampler.sample()));
        bssvol.filter(data[row]);
         
        std::cout << sisrsvol_hilb.getLogCondLike() << ", " << bssvol.getLogCondLike() << "\n";
    }
    
}
