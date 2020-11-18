import numpy as np
import itertools
from multiprocessing import Pool, cpu_count
from Thesis.misc import VanillaOptions as vo
from Thesis.Heston import MC_al as mc

if __name__ == "__main__":
    ### Generating input data
    spot = np.linspace(start = 50, stop = 150, num = 1000)
    vol1 = 0.04
    vol2 = 0.01
    kappa1 = 2
    kappa2 = 0.1
    theta1 = 0.04
    theta2 = 0.01
    epsilon1 = 0.5
    epsilon2 = 2
    epsilon = np.linspace(start = 0.5, stop = 2, num = 10)
    rho1 = -0.7
    rho2 = 0.8
    rate = 0.05

    input_array = np.array(list(itertools.product(spot, epsilon)))

    paral_list = []
    for some_row in input_array:
        paral_list.append([some_row[0], some_row[1]])

    tau = 1.005 #set to match option data
    strike = 100

    some_option = vo.EUCall(tau, strike)

    spot = np.reshape(spot, (-1, 1))
    """
    print("Calculating single data")
    ### For single input
    al_output1 = np.zeros(len(spot))
    mc_output1 = np.zeros(len(spot))
    al_output2 = np.zeros(len(spot))
    mc_output2 = np.zeros(len(spot))
    for i in range(len(spot)):
        some_model = hm.HestonClass(spot[i], vol1, kappa1, theta1, epsilon1, rho1, rate) # case 1
        some_model2 = hm.HestonClass(spot[i], vol2, kappa2, theta2, epsilon2, rho2, rate)
        al_output1[i] = al.Andersen_Lake(some_model, some_option)
        mc_output1[i] = mc.Heston_monte_carlo(some_model, some_option, 10000)
        al_output2[i] = al.Andersen_Lake(some_model2, some_option)
        mc_output2[i] = mc.Heston_monte_carlo(some_model2, some_option, 10000)

    np.savetxt("Data/al_output1.csv", al_output1, delimiter=",")
    np.savetxt("Data/mc_output1.csv", mc_output1, delimiter=",")
    np.savetxt("Data/al_output2.csv", al_output2, delimiter=",")
    np.savetxt("Data/mc_output2.csv", mc_output2, delimiter=",")
"""
    print("Calculating multi data")
    ### For multiple input
    # going parallel
    cpu_cores = cpu_count()

    # parallel
    pool = Pool(cpu_cores)
    res = pool.starmap(mc.calc_prices, paral_list)
    pool.close()
    al_output_multiple_1 = np.array(res)[:,0]
    mc_output_multiple_1 = np.array(res)[:,1]
    al_output_multiple_2 = np.array(res)[:,2]
    mc_output_multiple_2 = np.array(res)[:,3]

    np.savetxt("Data/al_output_multiple_1.csv", al_output_multiple_1, delimiter=",")
    np.savetxt("Data/mc_output_multiple_1.csv", mc_output_multiple_1, delimiter=",")
    np.savetxt("Data/al_output_multiple_2.csv", al_output_multiple_2, delimiter=",")
    np.savetxt("Data/mc_output_multiple_2.csv", mc_output_multiple_2, delimiter=",")
