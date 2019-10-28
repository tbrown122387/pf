#ifndef DATA_READER_H
#define DATA_READER_H

#include <vector>
#include <Eigen/Dense>
#include <string>
#include <fstream>
#include <iostream>


// a function to read in data from a csv file
// this csv file must not have a header
// probably a lot of better functions out there,
// but I didn't want to add a dependency for
// this one example. sorry...
template<typename float_t, size_t dimobs>
std::vector<Eigen::Matrix<float_t,dimobs,1> > readInData(const std::string &fileLoc)
{
   
    // build this up and return it
    std::vector<Eigen::Matrix<float_t,dimobs,1> > data;
    
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
        
        std::vector<float_t> data_row;
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
        Eigen::Map<Eigen::Matrix<float_t,dimobs,1>> drw(&data_row[0], num_col);
        data.push_back(drw);
    }

    return data;
}


#endif // DATA_READER_H
