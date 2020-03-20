import logging

import numpy as np
import pandas as pd

import lightgbm as lgb
from catboost import CatBoostClassifier

from sklearn.metrics import accuracy_score, log_loss
from sklearn.metrics import precision_recall_curve, f1_score
from sklearn.model_selection import RepeatedStratifiedKFold, StratifiedKFold


def lgb_f1_score(y_hat, data):
    y_true = data.get_label()
    y_hat = np.round(y_hat)  # scikits f1 doesn't like probabilities
    return 'f1', f1_score(y_true, y_hat), True


def train_model(args):
    logging.info("[Train_start]")

    train_path = 'data/train.csv'
    test_path  = 'data/predict_input.csv'

    train = pd.read_csv(train_path)
    test  = pd.read_csv(test_path)

    train_smiles = train['SMILES']
    test_smiles = test['SMILES']

    del train['SMILES']
    del test['SMILES']

    train_input = train.iloc[:, :-1].values
    label = train['label'].values.astype(int)
    test_input  = test.values

    y_train_pred = np.zeros((train_input.shape[0],), dtype=np.float32)
    y_test_pred = np.zeros((test_input.shape[0], ), dtype=np.float32)

    val_f1 = list()
    val_acc = list()
    val_loss = list()

    num_folds = 1
    n_splits = 5
    num_iters = 100
    for rs in range(2020, 2020 + num_folds):
        kfold = StratifiedKFold(n_splits=n_splits,
                                random_state=rs,
                                shuffle=True)
        for i, (train_idx, valid_idx) in enumerate(kfold.split(train_input, label)):
            x_train, x_valid = np.copy(train_input[train_idx, :]), np.copy(train_input[valid_idx, :])
            y_train, y_valid = label[train_idx], label[valid_idx]

            means = np.mean(x_train[:, -4:], axis=0)
            stds = np.std(x_train[:, -4:], axis=0)

            x_train[:, -4:] -= means
            x_train[:, -4:] /= (stds + 1e-5)

            x_valid[:, -4:] -= means
            x_valid[:, -4:] /= (stds + 1e-5)

            x_test = np.copy(test_input)
            x_test[:, -4:] -= means
            x_test[:, -4:] /= (stds + 1e-5)

            if args.model == 'lgbm':
                params = {'learning_rate': args.learning_rate,
                          'max_depth': args.max_depth,
                          'boosting': 'gbdt',
                          'objective': 'binary',
                          'metric': 'binary_logloss',
                          'is_training_metric': True,
                          'feature_fraction': args.feature_fraction,
                          'feature_fraction_seed': args.feature_fraction_seed,
                          'bagging_fraction': args.bagging_fraction,
                          'bagging_freq': args.bagging_freq,
                          'bagging_seed': args.bagging_seed,
                          "min_data_in_leaf": args.min_data_in_leaf,
                          'seed': args.seed,
                          'num_threads': -1,  # cpu 코어 수
                          'num_class': 1,
                          'tree_learner': 'data',
                          'extra_seed': args.extra_seed,
                          'data_random_seed': args.data_random_seed,
                          'objective_seed': args.objective_seed,
                          # 'device':'gpu', # gpu 사용하는 경우 주석 해제
                          'verbosity': -1}

                # LightGBM Model
                d_train = lgb.Dataset(x_train, y_train)
                d_valid = lgb.Dataset(x_valid, y_valid)

                model = lgb.train(params=params,
                                  train_set=d_train,
                                  num_boost_round=num_iters,
                                  valid_sets=[d_train, d_valid],
                                  verbose_eval=1000,
                                  early_stopping_rounds=500)

                y_valid_pred = model.predict(x_valid, num_iteration=model.best_iteration)
                y_valid_label = np.round(y_valid_pred).astype(int)

                y_train_pred[valid_idx] += model.predict(x_valid) / num_folds
                y_test_pred += model.predict(x_test) / (n_splits * num_folds)
            else:
                # CatBoost Model
                model = CatBoostClassifier(iterations=num_iters,
                                           learning_rate=args.learning_rate,
                                           l2_leaf_reg=args.l2_leaf_reg,
                                           # task_type="GPU", # gpu 사용하는 경우 주석 해제
                                           #  devices='0-3',
                                           random_strength=args.random_strength,
                                           use_best_model=True,
                                           eval_metric='F1',
                                           custom_loss='F1',
                                           thread_count=-1)

                model.fit(x_train, y_train,
                          early_stopping_rounds=1000,
                          eval_set=(x_valid, y_valid),
                          metric_period=100)

                y_valid_pred = model.predict_proba(x_valid)
                y_valid_label = np.argmax(y_valid_pred, axis=1)

                y_train_pred[valid_idx] += model.predict_proba(x_valid)[:, -1] / num_folds
                y_test_pred += model.predict_proba(x_test)[:, -1] / (n_splits * num_folds)

            bce_loss = log_loss(y_valid, y_valid_pred)
            f1 = f1_score(y_valid, y_valid_label)
            acc = accuracy_score(y_valid, y_valid_label)

            val_f1.append(f1)
            val_acc.append(acc)
            val_loss.append(bce_loss)

    f1 = np.mean(val_f1)
    eval_loss = 1 - f1

    if args.inference:
        columns = ['type', 'SMILES', 'predict_proba', 'label']

        types = np.concatenate([np.array(['train' for _ in range(train_input.shape[0])]).reshape(-1, 1),
                                np.array(['test' for _ in range(test_input.shape[0])]).reshape(-1, 1)
                                ], axis=0)

        smiles = np.concatenate([train_smiles.values.reshape(-1, 1),
                                 test_smiles.values.reshape(-1, 1)],
                                 axis=0)

        predict_proba = np.concatenate([y_train_pred.reshape(-1, 1),
                                        y_test_pred.reshape(-1, 1)],
                                        axis=0)

        labels = np.concatenate([label.reshape(-1, 1),
                                 np.array(['' for _ in range(test_input.shape[0])]).reshape(-1, 1)],
                                 axis=0)

        values = np.concatenate([types,
                                 smiles,
                                 predict_proba,
                                 labels], axis=1)

        result_df = pd.DataFrame(values, columns=columns)

        acc = np.mean(val_acc)
        bce_loss = np.mean(val_loss)

        make_submission(result_df, acc, bce_loss)
        return

    logging.info('F1 Loss: %.4f' % eval_loss)

    """@nni.report_final_result(eval_loss)"""
    logging.debug('Final result is %g', eval_loss)
    logging.debug('Send final result done.')

    return eval_loss


def make_submission(result_df, val_acc, val_loss):
    def scoring(y_true, y_proba, verbose=True):
        def threshold_search(y_true, y_proba):
            precision, recall, thresholds = precision_recall_curve(y_true, y_proba)
            thresholds = np.append(thresholds, 1.001)
            F = 2 / (1 / (precision + 1e-5) + 1 / (recall + 1e-5))
            best_score = np.max(F)
            best_th = thresholds[np.argmax(F)]
            return best_th

        rkf = RepeatedStratifiedKFold(n_splits=5, n_repeats=10)

        scores = []
        ths = []
        for train_index, test_index in rkf.split(y_true, y_true):
            y_prob_train, y_prob_test = y_proba[train_index], y_proba[test_index]
            y_true_train, y_true_test = y_true[train_index], y_true[test_index]

            # determine best threshold on 'train' part
            best_threshold = threshold_search(y_true_train, y_prob_train)

            # use this threshold on 'test' part for score
            sc = f1_score(y_true_test, (y_prob_test >= best_threshold).astype(int))
            ths.append(best_threshold)

        best_th = np.mean(ths)
        score = np.mean(scores)

        if verbose: print(f'Best threshold: {np.round(best_th, 4)}, Score: {np.round(score, 5)}')

        return best_th, score

    train_idx = result_df['type'] == 'train'
    test_idx = result_df['type'] == 'test'

    train_y_true = result_df[train_idx]['label'].values.reshape(-1, 1)
    train_y_pred = result_df[train_idx]['predict_proba'].values.reshape(-1, 1)

    best_threshold, score = scoring(train_y_true.astype(int), train_y_pred)

    result = pd.DataFrame(np.concatenate([result_df[test_idx]['SMILES'].values.reshape(-1, 1),
                                          result_df[test_idx]['predict_proba'].map(
                                              lambda x: 1 if x > best_threshold else 0).values.reshape(-1, 1)],
                                         axis=1),
                          columns=['SMILES', 'label'])

    result.to_csv('results/model_acc_{:.4f}_f1_{:.4f}_loss_{:.4f}.csv'.format(val_acc, score, val_loss), index=False)

    print('F1_Score Threshold Tune Results, F1_loss:{:.4f}, Accuracy:{:.4f}, BCE Loss:{:.4f}'.format(score,
                                                                                                     val_acc,
                                                                                                     val_loss))