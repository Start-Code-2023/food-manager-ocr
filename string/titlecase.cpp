// titlecase.cpp

#include "titlecase.h"
#include <cctype>

std::string upperCase(const std::string &input) {
    std::string result = input;

    for (char &c: result) {
        c = char(std::toupper(c));
    }

    return result;
}
