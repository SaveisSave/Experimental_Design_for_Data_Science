from sklearn.ensemble import VotingClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import f1_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.linear_model import LogisticRegression
import numpy as np
import pandas as pd


def stacking_classifier_performance_cv(estimators, metadata_selected, vis_selected, text_selected, audio_selected, y):
    voting_cf_results_cv = []
    label_stacking_results_cv = []
    label_and_feature_stacking_results_cv = []

    for i in range(10):
        counter = 0
        split_percentage_1 = (9. - i) / 10.
        split_percentage_2 = split_percentage_1 + 0.1
        index_1 = np.floor(split_percentage_1 * y.shape[0]).astype(int)
        index_2 = np.floor(split_percentage_2 * y.shape[0]).astype(int)
        Y_train = y.drop(y.index[range(index_1, index_2)])
        Y_test = y.iloc[index_1:index_2]
        estimator_predictions_train = np.zeros((Y_train.shape[0], len(estimators)))
        estimator_predictions_test = np.zeros((Y_test.shape[0], len(estimators)))
        for j in estimators:
            if 'meta' in j[0]:
                X_train = metadata_selected.drop(metadata_selected.index[range(index_1, index_2)])
                X_test = metadata_selected[index_1:index_2]
            elif 'vis' in j[0]:
                X_train = vis_selected.drop(vis_selected.index[range(index_1, index_2)])
                X_test = vis_selected[index_1:index_2]
            elif 'text' in j[0]:
                X_train = text_selected.drop(text_selected.index[range(index_1, index_2)])
                X_test = text_selected[index_1:index_2]
            elif 'audio' in j[0]:
                X_train = audio_selected.drop(audio_selected.index[range(index_1, index_2)])
                X_test = audio_selected[index_1:index_2]
            j[1].fit(X_train, Y_train)
            estimator_predictions_train[:, counter] = j[1].predict(X_train)
            estimator_predictions_test[:, counter] = j[1].predict(X_test)
            counter += 1

        def vote(x):
            return 1 if sum(x) > len(estimators) / 2. else 0

        Y_pred_voting = np.apply_along_axis(vote, axis=1, arr=estimator_predictions_test)

        print(Y_pred_voting)

        voting_results = (
            f1_score(Y_test, Y_pred_voting), precision_score(Y_test, Y_pred_voting),
            recall_score(Y_test, Y_pred_voting))
        voting_cf_results_cv.append(voting_results)

        label_stacking_cf = LogisticRegression(penalty='l2', dual=False, tol=0.0001, C=1.0, fit_intercept=True,
                                               intercept_scaling=1,
                                               class_weight=None, random_state=0, solver='liblinear', max_iter=100,
                                               multi_class='ovr',
                                               verbose=0, warm_start=False, n_jobs=-1)
        label_stacking_cf.fit(estimator_predictions_train, Y_train)
        Y_pred_label_stacking = label_stacking_cf.predict(estimator_predictions_test)
        label_stacking_results = (
            f1_score(Y_test, Y_pred_label_stacking), precision_score(Y_test, Y_pred_label_stacking),
            recall_score(Y_test, Y_pred_label_stacking))
        label_stacking_results_cv.append(label_stacking_results)

        complete_data = pd.concat([metadata_selected, vis_selected, text_selected, audio_selected], axis=1, join='inner', sort=False)
        X_train_features = pd.DataFrame(complete_data.drop(complete_data.index[range(index_1, index_2)])).reset_index(drop=True)
        X_train_pred_and_features = X_train_features.join(
            pd.DataFrame(estimator_predictions_train), lsuffix='df_1',rsuffix='df_2')
        X_test_features = pd.DataFrame(complete_data[index_1:index_2]).reset_index(drop=True)
        X_test_pred_and_features = X_test_features.join(pd.DataFrame(estimator_predictions_test), lsuffix='df_3',rsuffix='df_4')

        label_and_feature_stacking_cf = LogisticRegression(penalty='l2', dual=False, tol=0.0001, C=1.0,
                                                           fit_intercept=True, intercept_scaling=1,
                                                           class_weight=None, random_state=0, solver='liblinear',
                                                           max_iter=100, multi_class='ovr',
                                                           verbose=0, warm_start=False, n_jobs=-1)
        label_and_feature_stacking_cf.fit(X_train_pred_and_features, Y_train)
        Y_pred_label_and_feature_stacking = label_and_feature_stacking_cf.predict(X_test_pred_and_features)
        label_and_feature_stacking_results = (
        f1_score(Y_test, Y_pred_label_and_feature_stacking), precision_score(Y_test, Y_pred_label_and_feature_stacking),
        recall_score(Y_test, Y_pred_label_and_feature_stacking))
        label_and_feature_stacking_results_cv.append(label_and_feature_stacking_results)

    return np.mean(voting_cf_results_cv, axis=0), np.mean(label_stacking_results_cv, axis=0), np.mean(label_and_feature_stacking_results_cv, axis=0)


def stacking_classifier_performance_on_test_set(estimators, metadata_selected, vis_selected, text_selected,
                                                audio_selected, y, split_index):
    meta_train = metadata_selected[0:split_index]
    vis_train = vis_selected[0:split_index]
    text_train = text_selected[0:split_index]
    audio_train = audio_selected[0:split_index]
    meta_test = metadata_selected[split_index:]
    vis_test = vis_selected[split_index:]
    text_test = text_selected[split_index:]
    audio_test = audio_selected[split_index:]
    Y_train = y[0:split_index]
    Y_test = y[split_index:]

    estimator_predictions_train = np.zeros((Y_train.shape[0], len(estimators)))
    estimator_predictions_test = np.zeros((Y_test.shape[0], len(estimators)))

    counter = 0
    for j in estimators:
        X_train = None
        X_test = None
        if 'meta' in j[0]:
            X_train = meta_train
            X_test = meta_test
        elif 'vis' in j[0]:
            X_train = vis_train
            X_test = vis_test
        elif 'text' in j[0]:
            X_train = text_train
            X_test = text_test
        elif 'audio' in j[0]:
            X_train = audio_train
            X_test = audio_test

        j[1].fit(X_train, Y_train)
        estimator_predictions_train[:, counter] = j[1].predict(X_train)
        estimator_predictions_test[:, counter] = j[1].predict(X_test)
        counter += 1

    def vote(x):
        return 1 if sum(x) > len(estimators) / 2. else 0

    Y_pred_voting = np.apply_along_axis(vote, axis=1, arr=estimator_predictions_test)

    voting_results = (
        f1_score(Y_test, Y_pred_voting), precision_score(Y_test, Y_pred_voting), recall_score(Y_test, Y_pred_voting))

    label_stacking_cf = LogisticRegression(penalty='l2', dual=False, tol=0.0001, C=1.0, fit_intercept=True,
                                           intercept_scaling=1,
                                           class_weight=None, random_state=0, solver='liblinear', max_iter=100,
                                           multi_class='ovr',
                                           verbose=0, warm_start=False, n_jobs=-1)
    label_stacking_cf.fit(estimator_predictions_train, Y_train)
    Y_pred_label_stacking = label_stacking_cf.predict(estimator_predictions_test)
    label_stacking_results = (
        f1_score(Y_test, Y_pred_label_stacking), precision_score(Y_test, Y_pred_label_stacking),
        recall_score(Y_test, Y_pred_label_stacking))

    complete_data = pd.concat([metadata_selected, vis_selected, text_selected, audio_selected], axis=1, join='inner',
                              sort=False)
    X_train_features = pd.DataFrame(complete_data[0:split_index])
    X_train_pred_and_features = X_train_features.join(
        pd.DataFrame(estimator_predictions_train), lsuffix='df_1', rsuffix='df_2')
    X_test_features = pd.DataFrame(complete_data[split_index:]).reset_index(drop=True)
    X_test_pred_and_features = X_test_features.join(pd.DataFrame(estimator_predictions_test), lsuffix='df_3',
                                                    rsuffix='df_4')

    label_and_feature_stacking_cf = LogisticRegression(penalty='l2', dual=False, tol=0.0001, C=1.0,
                                                       fit_intercept=True, intercept_scaling=1,
                                                       class_weight=None, random_state=0, solver='liblinear',
                                                       max_iter=100, multi_class='ovr',
                                                       verbose=0, warm_start=False, n_jobs=-1)
    label_and_feature_stacking_cf.fit(X_train_pred_and_features, Y_train)
    Y_pred_label_and_feature_stacking = label_and_feature_stacking_cf.predict(X_test_pred_and_features)
    label_and_feature_stacking_results = (
        f1_score(Y_test, Y_pred_label_and_feature_stacking), precision_score(Y_test, Y_pred_label_and_feature_stacking),
        recall_score(Y_test, Y_pred_label_and_feature_stacking))

    return voting_results, label_stacking_results, label_and_feature_stacking_results
