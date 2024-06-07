import numpy as np
import pickle
import sys
import time
import matplotlib.pyplot as plt

sys.path.append("..")
from Defaults import defaultSimulate as default
from Helper import ClusterModelNEST
import psutil

if __name__ == '__main__':
    Savepath = "Data"

    CPUcount=psutil.cpu_count(logical = False)
    if CPUcount>8:
        CPUcount-=2

    startTime = time.time()


    params = {'n_jobs': CPUcount, 'N_E': 4000, 'N_I': 1000
        , 'dt': 0.1, 'neuron_type': 'iaf_psc_exp', 'simtime': 2000, 'delta_I_xE': 0.,
              'delta_I_xI': 0., 'record_voltage': False, 'record_from': 1, 'warmup': 1000,
              'Q': 10, 'tau_E': 10., 'tau_I': 5.,
              'clustering': 'weight'
                            #'probabilities'
              }

    Rj = 0.75  # 0.75 default value  #works with 0.95 and gif wo adaptation
    #jep = 10.0  # clustering strength
    #jip = 1. + (jep - 1) * jip_ratio
    #params['jplus'] = np.array([[jep, jip], [jip, jip]])
    params['ps'] = np.ones((2,2))*0.2#np.array([[0.25,0.25],[0.25,0.25]])
    #pep = 3.0  # clustering strength
    pep = 3.0
    pip = 1. + (pep - 1) * Rj

    params['pplus'] = np.array([[pep, pip],[pip, pip]])
    params['jplus'] = params['pplus']


    I_ths = [1.13,
             0.75]  # 3,5,Hz        #background stimulation of E/I neurons -> sets firing rates and changes behavior
    # to some degree # I_ths = [5.34,2.61] # 10,15,Hzh

    params['I_th_E'] = I_ths[0]
    params['I_th_I'] = I_ths[1]
    timeout = 18000  # 5h

    params['matrixType'] = "PROCEDURAL_GLOBALG"  # not needed only added for easier plotting scripts

    EI_Network = ClusterModelNEST.ClusteredNetworkNEST_Timing(default, params)
    # Creates object which creates the EI clustered network in NEST
    Result = EI_Network.get_simulation(timeout=timeout)
    plt.figure()
    plt.plot(Result['spiketimes'][0], Result['spiketimes'][1], 'k.', markersize=0.5)
    plt.show()

    if params['clustering']=='probabilities':
        path=Savepath+"_prob.pkl"
    else:
        path=Savepath+"_weight.pkl"
    with open(path, 'wb') as f:
        pickle.dump(Result, f, pickle.HIGHEST_PROTOCOL)
