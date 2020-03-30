import argparse

from model import *
from sacred import Experiment
from sacred.observers import MongoObserver

import config

ex = Experiment("Hyperparameter_Search_CV")
ex.observers.append(MongoObserver.create(url='mongodb://%s:%s@127.0.0.1:27017' % ('mongo_user', 'mongo_password'),
                                          db_name="sacred"))



def get_default_params():

    parser = argparse.ArgumentParser()

    parser.add_argument("--learning_rate", type=float, default=0.01)
    parser.add_argument('--l2_leaf_reg', type=float, default=0.01)
    parser.add_argument('--random_strength', type=int, default=2020)
    parser.add_argument("--max_depth", type=int, default=10)
    parser.add_argument("--feature_fraction", type=float, default=0.5)
    parser.add_argument("--bagging_fraction", type=float, default=0.5)
    parser.add_argument("--bagging_seed", type=int, default=2020)
    parser.add_argument("--bagging_freq", type=int, default=5)
    parser.add_argument("--min_data_in_leaf", type=int, default=500)
    parser.add_argument("--seed", type=int, default=2020)
    parser.add_argument("--feature_fraction_seed", type=int, default=2020)
    parser.add_argument("--extra_seed", type=int, default=2020)
    parser.add_argument("--data_random_seed", type=int, default=2020)
    parser.add_argument("--objective_seed", type=int, default=2020)

    args = parser.parse_args()

    return args


@ex.config
def hyperparam():
    args = get_default_params()

    args.model = 'catboost'
    args.inference = True # Set whether automl searching or inference

    # for Catboost
    """@nni.variable(nni.loguniform(0.00001, 0.1), name=args.learning_rate)"""
    args.learning_rate = args.learning_rate
    """@nni.variable(nni.loguniform(0.00001, 1), name=args.l2_leaf_reg)"""
    args.l2_leaf_reg = args.l2_leaf_reg
    """@nni.variable(nni.randint(0, 100000), name=args.random_strength)"""
    args.random_strength = args.random_strength

    # for LightGBM
    """@nni.variable(nni.randint(1, 15), name=args.max_depth)"""
    args.max_depth = args.max_depth
    """@nni.variable(nni.uniform(0.5, 1.0), name=args.feature_fraction)"""
    args.feature_fraction = args.feature_fraction
    """@nni.variable(nni.uniform(0.5, 1.0), name=args.bagging_fraction)"""
    args.bagging_fraction = args.bagging_fraction
    """@nni.variable(nni.randint(0, 100000), name=args.bagging_seed)"""
    args.bagging_seed = args.bagging_seed
    """@nni.variable(nni.randint(1, 10), name=args.bagging_freq)"""
    args.bagging_freq = args.bagging_freq
    """@nni.variable(nni.randint(100, 2000), name=args.min_data_in_leaf)"""
    args.min_data_in_leaf = args.min_data_in_leaf
    """@nni.variable(nni.randint(0, 100000), name=args.seed)"""
    args.seed = args.seed
    """@nni.variable(nni.randint(0, 100000), name=args.feature_fraction_seed)"""
    args.feature_fraction_seed = args.feature_fraction_seed
    """@nni.variable(nni.randint(0, 100000), name=args.extra_seed)"""
    args.extra_seed = args.extra_seed
    """@nni.variable(nni.randint(0, 100000), name=args.data_random_seed)"""
    args.data_random_seed = args.data_random_seed
    """@nni.variable(nni.randint(0, 100000), name=args.objective_seed)"""
    args.objective_seed = args.objective_seed
    
    args.train_set = None
    print("hyperparam - ", args)


@ex.automain
def run(args):
    test_loss = train_model(args)

    ex.log_scalar("loss", test_loss)

    return test_loss


