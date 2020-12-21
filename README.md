This entire GitHub project consists of all the code which has been used to create the results for my thesis. There is a lot of unnecessary code, but this will not be cleaned up.

The most important scripts and the application of these are discussed below:

Thesis/Heston/ - This folder includes all the scripts used for generating artificial neural networks, plots, etc. 

Thesis/Heston/HestonModel.py - includes the Class object for the heston model

Thesis/Heston/AndersenLake.py - This script includes methods for calculating option prices with the Andersen & Lake method

Thesis/Heston/DataGeneration.py - includes methods and code for generate uniform and grid sequence data

Thesis/Heston/ModelGenerator.py - includes code for generation of all the artificial neural networks created

Thesis/Heston/NNModelGenerator.py - methods for creating, compiling, fitting of artificial neural networks.

Thesis/Heston/MonteCarlo.py - methods for the Monte Carlo simulations under Heston

Thesis/Heston/SimpleExample.py - a simple example using the Andersen & Lake method

The remaining scripts in the Heston folder are meassy, and primarily just a bunch of function callings.


Thesis/BlackScholes/BlackScholes.py - the BlackScholes class object

Thesis/BlackScholes/MonteCarlo.py - MonteCarlo for Black Scholes


Thesis/misc/VanillaOptions.py - the class object for vanillaoptions (puts and calls)