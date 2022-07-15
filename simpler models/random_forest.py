from matplotlib import pyplot

import db
import numpy as np
import joblib
import pickle
import matplotlib.pyplot as plt
import random

from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, RepeatedStratifiedKFold, cross_val_score, RepeatedKFold
from sklearn.metrics import confusion_matrix, accuracy_score

devId = 3036
authKey = 'E00ABB875BD945E1ACC8119F12BE27AD'
dbConn = db.DataBase()


def get_win_good_with(gods, matches):
    # synergy win rate
    gods = list(map(lambda x: x[0], gods))
    good_with = np.zeros((len(gods), len(gods))) * 0.5
    good_with_num_matches = np.zeros((len(gods), len(gods)))
    for match in matches:
        for god_1_id in match[:5]:
            i = gods.index(god_1_id)
            for god_2_id in match[:5]:
                if god_1_id != god_2_id:
                    j = gods.index(god_2_id)
                    good_with[i, j] += 1 if match[len(match) - 1] == 1 else 0
                    good_with_num_matches[i, j] += 1

        for god_1_id in match[5:10]:
            i = gods.index(god_1_id)
            for god_2_id in match[5:10]:
                if god_1_id != god_2_id:
                    j = gods.index(god_2_id)
                    good_with[i, j] += 1 if match[len(match) - 1] == 2 else 0
                    good_with_num_matches[i, j] += 1

    win_good_with = np.divide(good_with, good_with_num_matches, out=np.zeros_like(good_with), where=good_with_num_matches != 0)
    pickle.dump(win_good_with, open("../pickles/win_good_with.p", "wb"))

    return win_good_with


def get_win_good_against(gods, matches):
    # countering win rate
    gods = list(map(lambda x: x[0], gods))
    good_against = np.zeros((len(gods), len(gods))) * 0.5
    good_against_num_matches = np.zeros((len(gods), len(gods)))
    for match in matches:
        for god_1_id in match[:5]:
            i = gods.index(god_1_id)
            for god_2_id in match[5:10]:
                j = gods.index(god_2_id)

                if match[len(match) - 1] == 1:
                    good_against[i, j] += 1
                else:
                    good_against[j, i] += 1

                good_against_num_matches[i, j] += 1
                good_against_num_matches[j, i] += 1

    win_good_against = np.divide(good_against, good_against_num_matches, out=np.zeros_like(good_against), where=good_against_num_matches != 0)
    pickle.dump(win_good_against, open("../pickles/win_good_against.p", "wb"))

    return win_good_against


def get_data(gods):
    matches = dbConn.select("SELECT "
                            "t1_god1, t1_god2, t1_god3, t1_god4, t1_god5, "
                            "t2_god1, t2_god2, t2_god3, t2_god4, t2_god5, "
                            "ban1, ban2, ban3, ban4, ban5, ban6, ban7, ban8, ban9, ban10, "
                            "win FROM Match", ())

    win_good_with = get_win_good_with(gods, matches)
    win_good_against = get_win_good_against(gods, matches)

    gods = list(map(lambda x: x[0], gods))
    X = []
    y = []
    for match in matches:
        # features = list(map(lambda god: 1 if god in match[:5] else 2 if god in match[5:10] else 0, gods))
        features_team_1 = list(map(lambda god: 1 if god in match[:5] else 0, gods))
        features_team_2 = list(map(lambda god: 1 if god in match[5:10] else 0, gods))
        features = features_team_1 + features_team_2

        s_team_1 = 0
        s_team_2 = 0
        n_synergy_team_1 = 0
        n_synergy_team_2 = 0
        counter = 0
        n_counter = 0
        for god_1_id in match[:5]:
            i = gods.index(god_1_id)
            for god_2_id in match[:5]:
                if god_1_id != god_2_id:
                    j = gods.index(god_2_id)
                    s_team_1 += win_good_with[i, j]
                    n_synergy_team_1 += 1

        for god_1_id in match[5:10]:
            i = gods.index(god_1_id)
            for god_2_id in match[5:10]:
                if god_1_id != god_2_id:
                    j = gods.index(god_2_id)
                    s_team_2 += win_good_with[i, j]
                    n_synergy_team_2 += 1

            for god_2_id in match[:5]:
                if god_1_id != god_2_id:
                    j = gods.index(god_2_id)
                    counter += win_good_against[i, j]
                    n_counter += 1

        s_team_1 = s_team_1 / n_synergy_team_1
        s_team_2 = s_team_2 / n_synergy_team_2
        features.append(s_team_1 / (s_team_1 + s_team_2))
        features.append(counter / n_counter)

        X.append(features)
        y.append(match[len(match) - 1] - 1)

    # randomly shuffle cases to avoid overfitting
    c = list(zip(X, y))
    random.shuffle(c)
    X, y = zip(*c)
    return X, y, win_good_with, win_good_against


def build_model():
    gods = dbConn.select("SELECT * FROM God ORDER BY name", ())
    X, y, win_good_with, win_good_against = get_data(gods)

    print(len(X), "samples")
    print(len(X[0]), "features")

    # Split the dataset into training and test dataset
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=0)

    # model = RandomForestClassifier()
    model = RandomForestRegressor()

    model.fit(X_train, y_train)

    # Use the forest's predict method on the test data
    predictions = model.predict(X_test)
    # Calculate the absolute errors
    errors = abs(predictions - y_test)
    # Print out the mean absolute error (mae)
    print('Mean Absolute Error:', round(np.mean(errors), 2), 'degrees.')

    # Calculate mean absolute percentage error (MAPE)
    correct = 0
    for i in range(len(predictions)):
        if predictions[i] > 0.5 and y_test[i] == 1:
            correct = correct + 1
        elif predictions[i] < 0.5 and y_test[i] == 0:
            correct = correct + 1

    # Calculate and display accuracy
    accuracy = 100 * (correct / len(y_test))
    print('Accuracy:', accuracy, '%.')

    # Use the forest's predict method on the test data
    # predictions = model.predict(X_test)
    # Calculate the absolute errors
    # errors = abs(predictions - y_test)
    ## Print out the mean absolute error (mae)
    # print('Mean Absolute Error:', round(np.mean(errors), 2), 'degrees.')
    #
    ## Calculate mean absolute percentage error (MAPE)
    # mape = 100 * (errors / y_test)
    ## Calculate and display accuracy
    # accuracy = 100 - np.mean(mape)
    # print('Accuracy:', round(accuracy, 2), '%.')

    joblib.dump(model, "../models/rf.joblib")


def predict():
    model = joblib.load("../models/rf.joblib")

    gods = dbConn.select("SELECT * FROM God ORDER BY name", ())
    matches = dbConn.select("SELECT "
                            "t1_god1, t1_god2, t1_god3, t1_god4, t1_god5, "
                            "t2_god1, t2_god2, t2_god3, t2_god4, t2_god5, "
                            "ban1, ban2, ban3, ban4, ban5, ban6, ban7, ban8, ban9, ban10, "
                            "win FROM Match", ())

    win_good_with = get_win_good_with(gods, matches)
    win_good_against = get_win_good_against(gods, matches)
    # win_good_with = pickle.load(open("pickles/win_good_with.p", "rb"))
    # win_good_against = pickle.load(open("pickles/win_good_against.p", "rb"))

    # heatmap = plt.pcolor(win_good_against)
    # plt.colorbar(heatmap)
    # plt.show()

    team_1 = ['Achilles', 'Ares', 'Hera', 'Poseidon', 'Ratatoskr']
    team_2 = ['Athena', 'Kuzenbo', 'Nu Wa', 'Sylvanus', 'Terra']
    features_team_1 = list(map(lambda god: 1 if god[1] in team_1 else 0, gods))
    features_team_2 = list(map(lambda god: 1 if god[1] in team_2 else 0, gods))
    features = features_team_1 + features_team_2

    gods = list(map(lambda x: x[1], gods))

    X = []
    s_team_1 = 0
    s_team_2 = 0
    n_synergy_team_1 = 0
    n_synergy_team_2 = 0
    counter = 0
    n_counter = 0
    for god_1_name in team_1:
        i = gods.index(god_1_name)
        for god_2_name in team_1:
            if god_1_name != god_2_name:
                j = gods.index(god_2_name)
                s_team_1 += win_good_with[i, j] if win_good_with[i, j] > 0 else 0.5
                n_synergy_team_1 += 1

    for god_1_name in team_2:
        i = gods.index(god_1_name)
        for god_2_name in team_2:
            if god_1_name != god_2_name:
                j = gods.index(god_2_name)
                s_team_2 += win_good_with[i, j] if win_good_with[i, j] > 0 else 0.5
                n_synergy_team_2 += 1

        for god_2_name in team_1:
            if god_1_name != god_2_name:
                j = gods.index(god_2_name)
                counter += win_good_against[i, j]
                n_counter += 1

    s_team_1 = s_team_1 / n_synergy_team_1
    s_team_2 = s_team_2 / n_synergy_team_2
    synergy = s_team_1 / (s_team_1 + s_team_2)
    features.append(synergy)
    features.append(counter / n_counter)

    print("synergy", synergy)
    print("counter", counter / n_counter)

    X.append(features)
    y_pred = model.predict(X)[0]  # 0 - team 1 wins, 1 - team 2 wins
    print(y_pred)


# build_model()
predict()
