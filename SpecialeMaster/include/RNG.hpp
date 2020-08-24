//
//  RNG.hpp
//  SpecialeMaster
//
//  Created by Martin Poswiata on 24/08/2020.
//  Copyright © 2020 Martin Poswiata. All rights reserved.
//

#ifndef RNG_H
#define RNG_H
#include <vector>
#include <random>
#include <stdio.h>

class RNG {
    private:
        std::vector<double> randomVector;
    public:
        RNG(); //constructor
        RNG(int length);
        ~RNG(); //destructor
        std::vector<double> getValue();
};

#endif
