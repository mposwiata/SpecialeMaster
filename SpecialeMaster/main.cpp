//
//  main.cpp
//  speciale_cpp
//
//  Created by Martin Poswiata on 24/08/2020.
//  Copyright © 2020 Martin Poswiata. All rights reserved.
//

#include <iostream>
#include "include/RNG.hpp"

int main(int argc, const char * argv[]) {
    // insert code here...
    
    RNG test = RNG(10);
    
    for(int i = 0; i < 10; ++i){
        std::cout << test.getValue()[i] << "\n";
    }
        
    return 0;
}
