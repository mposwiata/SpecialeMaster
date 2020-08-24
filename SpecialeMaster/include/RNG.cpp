//  RNG.cpp
//  speciale_cpp
//
//  Created by Martin Poswiata on 24/08/2020.
//  Copyright © 2020 Martin Poswiata. All rights reserved.
//
#include "../include/RNG.hpp"

// Default constructor
RNG::RNG(){}

RNG::RNG(int length){
    std::default_random_engine generator;
    std::normal_distribution<double> distribution(0.0, 1.0);
    for(int i = 0; i < length; ++i){
        randomVector.push_back(distribution(generator));
    }
}

// Destructor
RNG::~RNG () {}

//getRandom
std::vector<double> RNG::getValue(){
    return randomVector;
}
