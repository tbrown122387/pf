#include <fstream> // for ifstream

#include "svol_comparison.h" // function header

#include "svol_bs.h" // bs filter for svol
#include "svol_apf.h" // apf filter for svol
#include "svol_sisr.h" // sisr filter for svol
#include "resamplers.h" // for mn_resampler

// some template parameters
#define dimstate 1
#define dimobs   1
#define numparts 5000
#define FLOATTYPE float // choose float (faster) or double (slower)

// a function to read in data from a csv file
// this csv file must not have a header
// probably a lot of better functions out there, 
// but I didn't want to add a dependency for
// this one example. sorry...
std::vector<Eigen::Matrix<FLOATTYPE,dimobs,1> > readInData(const std::string &fileLoc)
{
   
    // build this up and return it
    std::vector<Eigen::Matrix<FLOATTYPE,dimobs,1> > data;
    
    // start reading
    std::string line;
    std::ifstream ifs(fileLoc);
    std::string one_number;
    unsigned int num_col;
    if(!ifs.is_open()){
        std::cerr << "readInData() failed to read data from: " << fileLoc << "\n";
    }

    // didn't fail...keep going
    while(std::getline(ifs, line)){
        
        std::vector<FLOATTYPE> data_row;
        try{
            
            // get a single element in a single row
            std::istringstream buff(line);

            // use commas to split up the line
            num_col = 0;
            while(std::getline(buff, one_number, ',')){
                data_row.push_back(std::stod(one_number));
                num_col ++;
            }

        } catch(const std::invalid_argument& ia){
            std::cerr << "Invalid Argument: " << ia.what() << "\n";
            continue;
        }

        // now append this vector to your collection
        Eigen::Map<Eigen::Matrix<FLOATTYPE,dimobs,1>> drw(&data_row[0], num_col);
        data.push_back(drw);
    }

    return data;
}


void run_svol_comparison(const std::string &csv)
{
    // "state size vector"
    using ssv = Eigen::Matrix<FLOATTYPE,dimstate,1>;
    // "observation sized vector"
    using osv = Eigen::Matrix<FLOATTYPE,dimobs,1>;    
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

    // read in some data
    std::vector<osv> data = readInData(csv);

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
        bssvol.filter(data[row], v);
        apfsvol.filter(data[row], v);
        sisrsvol.filter(data[row], v);
        
        std::cout << bssvol.getExpectations()[0] << ", " << apfsvol.getExpectations()[0] << ", " << sisrsvol.getExpectations()[0] << "\n";
        //std::cout << bssvol.getLogCondLike() << ", " << apfsvol.getLogCondLike() <<", " << sisrsvol.getLogCondLike() << "\n";
    }
    
}
