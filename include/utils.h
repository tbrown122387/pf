#ifndef UTILIS_H
#define UTILIS_H

#include <Eigen/Dense>
#include <fstream>
#include <array>
#include <iostream> // std::cerr
#include <fstream>  // std::ofstream
#include <string>   // string
#include <vector>   // vector


namespace utils{

    // return string that has str plus the current date/time, format is YYYY-MM-DD.HH:mm:ss
    std::string genStringWithTime(const std::string& str); 

    
    /**
     * @todo fully understand the MatrixBase thing. More info
     * here: https://eigen.tuxfamily.org/dox/TopicFunctionTakingEigenTypes.html
     */
     
    /**
     * @brief Writes to a row of file. Only appends. Doesn't close file stream.
     * @tparam dim the size of the Eigen::Vector.
     * @param vec the Eigen::Vector.
     * @param ofs the target ofstream.
     */
    template<size_t dim, typename float_t>
    void logParams(const Eigen::Matrix<float_t,dim,1> &vec, std::ofstream &ofs);
     
    
    /**
     * @brief Writes to a row of file. Only appends. Opens and closes file stream.
     * @tparam size the size of the Eigen::Vector.
     * @param vec the Eigen::Vector.
     * @param outfile the target file path.
     */
    template<size_t dim, typename float_t>
    void logParams(const Eigen::Matrix<float_t,dim,1> &vec, const std::string &outfile);//std::ofstream &ofs);


    /**
     * @brief Writes to a row of an ofstream. Only appends.
     * @tparam size the number of doubles in the array.
     * @param arr the array to be written.
     * @param outfile the target file path.
     */
    template<size_t size, typename float_t>
    void logParams(const std::array<float_t, size> &arr, const std::string &outfile);//std::ofstream &ofs);
    
    
    /**
     * @brief reads in data in a csv file with no header and separated by commas.
     * @tparam nc the number of columns.
     * @param fileLoc the string filepath of the file.
     * @return a std::vector of your data. Each elemenet is a row in Eigen::Vector form.
     */
    template<size_t nc, typename float_t>
    std::vector<Eigen::Matrix<float_t,nc,1> > readInData(const std::string& fileLoc);


template<size_t dim, typename float_t>
void logParams(const Eigen::Matrix<float_t,dim,1> &vec, std::ofstream &ofs)
{
    // make sure open and doesn't close
    if(ofs.is_open()){
        
        // write stuff
        for(size_t i = 0; i < dim; ++i){
            if( i == 0){
                ofs << vec(i,0);
            } else {
                ofs << "," << vec(i,0);                                        
            }
         }
        ofs << "\n";
        
        
    }else{
        std::cerr << "tried to write to a closed ofstream! " << "\n";
    }
}


template<size_t dim, typename float_t>
void logParams(const Eigen::Matrix<float_t,dim,1> &vec, const std::string &outfile)
{

    // open the file in append mode
    std::ofstream f(outfile, std::ios::app);
    
    // make sure open
    if(f.is_open()){
        
        // write stuff
        for(size_t i = 0; i < dim; ++i){
            if( i == 0){
                f << vec(i,0);
            } else {
                f << "," << vec(i,0);                                        
            }
         }
        f << "\n";
        
        f.close();
        
    }else{
        std::cerr << "logParams() failed to open file at: " << outfile << "\n";
    }
}


template<size_t size, typename float_t>
void logParams(const std::array<float_t, size> &arr, const std::string &outfile)
{
    /**
     * \todo test to make sure write worked (e.g. if( !f << "derp" ) ) 
     */

    // open the file in append mode
    std::ofstream f(outfile, std::ios::app);

    // make sure file open
    if(f.is_open()){
        
        // write
        for(size_t i = 0; i < size; ++i){
            if( i == 0){
                f << arr[i];
            } else {
                f << "," << arr[i];                                        
            }
        }
        f << "\n";   
    }else{
        std::cerr << "logParams() failed to open file at: " << outfile << "\n";
    }
}


/**
 * @brief reads in comma separated data
 * NB: you need to know the number of columns beforehand, and pass this number in as a template parameter
 * NB: the data cannot have a header
 */
template<size_t nc, typename float_t>
std::vector<Eigen::Matrix<float_t,nc,1> > readInData(const std::string& fileLoc)
{
    // returning this. gotta build it up
    std::vector<Eigen::Matrix<float_t,nc,1> > data;
    
    // start reading
    std::string line;
    std::ifstream ifs(fileLoc);
    std::string one_number;    
    if(!ifs.is_open()){     // check if we can open inFile
        std::cerr << "readInData() failed to read data from: " << fileLoc << "\n";
    }
    
    // didn't fail...keep going
    while ( std::getline(ifs, line) ){     // get a whole row as a string
    
        std::vector<float_t> data_row;
        try{
            // get a single element on a row
            std::istringstream buff(line);
            
            // make one number between commas
            while(std::getline(buff, one_number, ',')){ 
                data_row.push_back(std::stod(one_number));
            }
            
        } catch (const std::invalid_argument& ia){
            std::cerr << "Invalid Argument: " << ia.what() << "\n";
            continue;
        }   
        
        // now append this Vec to your collection
        Eigen::Map<Eigen::Matrix<float_t,nc,1> > drw(&data_row[0], nc);
        data.push_back(drw);
    }
    
    return data;
    
} 

} // namespace utils


#endif // UTILIS_H
