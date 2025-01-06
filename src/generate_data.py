## Python 3.11.7

from os import name
from src.SlovotvirModel import run_model_parallel
from src.helpers import (binning,
                         plot_posterior,
                         train_and_amortize, 
                         preprocess_input)
import pickle
import numpy as np
import time

# set up the size of the data
train_size = 100000
test_size = 10000
valid_size = 10000

# priors
training_params = (np.random.uniform(-5, 1, train_size), 
                np.random.uniform(-2, 2, train_size), 
                np.random.lognormal(0, 0.5, train_size))
testing_params = (np.random.uniform(-5, 1, test_size), 
                np.random.uniform(-2, 2, test_size), 
                np.random.lognormal(0, 0.5,  test_size))
validation_params =(np.random.uniform(-5, 1, valid_size), 
                    np.random.uniform(-2, 2, valid_size), 
                    np.random.lognormal(0, 0.5,  valid_size))


if name == '__main__':
    print('Generating data...')
    # generating
    training_data = run_model_parallel(training_params[0], 
                                    training_params[1], 
                                    training_params[2], 
                                    train_size)
    testing_data = run_model_parallel(testing_params[0], 
                                    testing_params[1], 
                                    testing_params[2], 
                                    test_size)
    validation_data = run_model_parallel(validation_params[0], 
                                        validation_params[1], 
                                        validation_params[2], 
                                        valid_size)

    # refactoring priors
    training_params = np.vstack(training_params)
    testing_params = np.vstack(testing_params)
    validation_params = np.vstack(validation_params)

    # BayesFlow format
    ## train data
    train_data = {}
    train_data["prior_non_batchable_context"] = None
    train_data["prior_batchable_context"] = None
    train_data["prior_draws"] = training_params.reshape(3, train_size).T
    train_data["sim_non_batchable_context"] = None
    train_data["sim_batchable_context"] = None
    train_data['sim_data'] = preprocess_input(training_data)
    ## test data
    test_data = {}
    test_data["prior_non_batchable_context"] = None
    test_data["prior_batchable_context"] = None
    test_data["prior_draws"] = testing_params.reshape(3, test_size).T
    test_data["sim_non_batchable_context"] = None
    test_data["sim_batchable_context"] = None
    test_data['sim_data'] = preprocess_input(testing_data)
    ## validation data
    valid_data = {}
    valid_data["prior_non_batchable_context"] = None
    valid_data["prior_batchable_context"] = None
    valid_data["prior_draws"] = validation_params.reshape(3, valid_size).T
    valid_data["sim_non_batchable_context"] = None
    valid_data["sim_batchable_context"] = None
    valid_data['sim_data'] = preprocess_input(validation_data)

    print('Saving data...')

    # get today's date
    pickle.dump(train_data, 
                open(f"../data/train_data_{time.strftime("%Y%m%d")}.pkl", "wb"))
    pickle.dump(test_data, 
                open(f"../data/test_data_{time.strftime("%Y%m%d")}.pkl", "wb"))
    pickle.dump(valid_data, 
                open(f"../data/valid_data_{time.strftime("%Y%m%d")}.pkl", "wb"))
