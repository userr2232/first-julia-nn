#include "boost/date_time/posix_time/posix_time.hpp"
#include <iostream>

int main() {
    using namespace boost::gregorian;
    using namespace boost::posix_time;

    ptime start(date(2002, Jan, 1)), end(date(2002, Dec, 31));
    time_iterator titr(start, hours(24));
    while(titr <= end) {
        std::cout << to_simple_string(*titr) << std::endl;
        ++titr;
    }
    std::cout << "Now backward" << std::endl;
    while(titr >= start) {
        std::cout << to_simple_string(*titr) << std::endl;
        --titr;
    }
}