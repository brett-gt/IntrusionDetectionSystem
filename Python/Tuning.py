""" Tuning contains functions used to perform searches across network hyperparmaters. 
"""

import IDS
import pandas as pd
import numpy as np
import Hyper as HyperP

from tensorflow import random
from numpy.random import seed

#----------------------------------------------------------------------------------
def grid_sparse(data_encoding, train_data, test_data, results_file_name = "grid_sparse_results.csv"):
    """Does a grid search on a sparse network. 
    
       Trains sparse networks, finds best thresholds, and records the best value using that thresh in a csv
    """

    # Look at this option for using tensorflow tuning
    # https://medium.com/ml-book/neural-networks-hyperparameter-tuning-in-tensorflow-2-0-a7b4e2b574a1

    #TODO: These shouldn't be needed
    seed(1234)
    random.set_seed(2345)

    column = ['Input Layer', 'Bottleneck Layer', 'Input Sparse', 'Hidden Sparse', 'Thresh', 'Score']
    results = pd.DataFrame(columns=column)

    max_layer_outer = 80
    min_layer_outer = 60
    inc_layer_outer = 5

    min_layer_inner = 50
    inc_layer_inner = 4

    max_sparse_outer = 0.3
    min_sparse_outer = 0.1
    inc_sparse_outer = 0.1

    max_sparse_inner = 0.5
    min_sparse_inner = 0.1
    inc_sparse_inner = 0.1
  
    for i in range(min_layer_outer, max_layer_outer, inc_layer_outer):
        for j in range(min_layer_inner, i, inc_layer_inner):
            for k in np.arange(min_sparse_outer, max_sparse_outer, inc_sparse_outer):
                for l in np.arange(min_sparse_inner, max_sparse_inner, inc_sparse_inner):

                    param_string = str(i) + "_" + str(j) + "_" + str(k) + "_" + str(l);
                    print("Running " + param_string)
                    file_name = "result_" + param_string + ".txt"
                    model_name = param_string + ".h5"

                    params = HyperP.cAutoHyper(model_name,[i, j],[0.005, 0], [k, l, 0])

                    model = IDS.train(train_data, data_encoding, params)
                    thresh, score = IDS.find_best_thresh(model, test_data)

                    results = results.append({'Input Layer':i, 'Bottleneck Layer':j, 'Input Sparse': k, 'Hidden Sparse': l,
                                         'Thresh': thresh, 'Score': score}, ignore_index=True)
                    results.to_csv(results_file_name)

    print(results)

