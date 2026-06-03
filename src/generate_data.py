## Python 3.11.7

import argparse
import os
from pathlib import Path
from src.SlovotvirModel import run_model_parallel
from src.helpers import preprocess_input
import pickle
import numpy as np
import time

DATA_DIR = Path(__file__).resolve().parent.parent / 'data'


def parse_args():
    parser = argparse.ArgumentParser(description='Generate Slovotvir model data.')
    parser.add_argument('--quick', action='store_true',
                        help='Run a small smoke-test generation.')
    parser.add_argument('--max-procs', type=int, default=None,
                        help='Maximum worker processes to use.')
    parser.add_argument('--batch-size', type=int, default=1000,
                        help='Number of simulations per multiprocessing batch.')
    return parser.parse_args()


def make_params(size):
    return (np.random.uniform(-5, 1, size),
            np.random.uniform(-2, 2, size),
            np.random.lognormal(0, 0.5, size))


if __name__ == '__main__':
    args = parse_args()
    quick = args.quick or os.getenv('QUICK') == '1'

    # set up the size of the data
    train_size = 10 if quick else 100000
    test_size = 4 if quick else 10000
    valid_size = 4 if quick else 10000
    batch_size = min(args.batch_size, train_size) if quick else args.batch_size
    max_procs = args.max_procs if args.max_procs is not None else (1 if quick else None)

    # priors
    training_params = make_params(train_size)
    testing_params = make_params(test_size)
    validation_params = make_params(valid_size)

    print('Generating data...')
    # generating
    training_data = run_model_parallel(training_params[0], 
                                    training_params[1], 
                                    training_params[2], 
                                    train_size,
                                    batch_size=batch_size,
                                    max_procs=max_procs)
    testing_data = run_model_parallel(testing_params[0], 
                                    testing_params[1], 
                                    testing_params[2], 
                                    test_size,
                                    batch_size=batch_size,
                                    max_procs=max_procs)
    validation_data = run_model_parallel(validation_params[0], 
                                        validation_params[1], 
                                        validation_params[2], 
                                        valid_size,
                                        batch_size=batch_size,
                                        max_procs=max_procs)

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
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    today = time.strftime('%Y%m%d')
    with open(DATA_DIR / f'train_data_{today}.pkl', 'wb') as f:
        pickle.dump(train_data, f)
    with open(DATA_DIR / f'test_data_{today}.pkl', 'wb') as f:
        pickle.dump(test_data, f)
    with open(DATA_DIR / f'valid_data_{today}.pkl', 'wb') as f:
        pickle.dump(valid_data, f)
