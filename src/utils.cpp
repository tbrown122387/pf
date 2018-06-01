#include "utils.h"

#include <ctime>

std::string utils::genStringWithTime(const std::string& str) {
    time_t     now = time(0);
    struct tm  tstruct;
    char       buf[80];
    tstruct = *localtime(&now);
    // Visit http://en.cppreference.com/w/cpp/chrono/c/strftime
    // for more information about date/time format
//    strftime(buf, sizeof(buf), "%Y-%m-%d.%X", &tstruct);
    strftime(buf, sizeof(buf), "%Y-%m-%d.%H-%M-%S", &tstruct);
    
    return str + "_" + buf;
}
