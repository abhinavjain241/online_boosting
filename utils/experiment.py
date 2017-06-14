from random import shuffle
from collections import defaultdict
import numpy as np
from sklearn.metrics import f1_score, accuracy_score


def test(Booster, Learner, data, m, trials=1, should_shuffle=True):
    results = []
    for t in range(trials):
        if should_shuffle:
            shuffle(data)
        results.append(run_test(Booster, Learner, data, m))
    results = zip(*results)
    def avg(x):
        return sum(x) / len(x)
    #return map(avg, zip(*results[0])), map(avg, zip(*results[1])), results[2], results[3]
    return results[0], results[1], results[2], results[3]#, results[4]

def run_test(Booster, Learner, data, m):
    classes = np.unique(np.array([y for (x, y) in data]))
    baseline = Learner(classes)
    predictor = Booster(Learner, classes=classes, M=m)
    correct_booster = 0.0
    correct_baseline = 0.0
    t = 0
    booster_scores = []
    baseline_scores = []
    performance_booster = []
    performance_baseline = []
    # predict_booster = [] # Predicted labels by booster
    # predict_baseline = [] # Predicted lables by baseline
    # booster_accuracy_score = [] # Booster Accuracy Score - Same as (TP + TN) / (Total) - essentially what cmarsh has done
    true_labels = [] # True labels as taken from the data
    for (features, label) in data:
        # boost_pred = predictor.predict(features)
        # if boost_pred == label:
        if predictor.predict(features) == label:
            correct_booster += 1
        booster_scores.append(predictor.get_score())
        # predict_booster.append(boost_pred)
        predictor.update(features, label)
        # base_pred = baseline.predict(features)
        # if base_pred == label:
        if baseline.predict(features) == label:
            correct_baseline += 1
        baseline_scores.append(baseline.get_score())
        # predict_baseline.append(base_pred)    
        baseline.partial_fit(features, label)
        t += 1
        true_labels.append(label)
        performance_booster.append(correct_booster / t)
        performance_baseline.append(correct_baseline / t)
        # boosterscore = f1_score(true_labels, predict_booster)
        # baselinescore = f1_score(true_labels, predict_baseline)
        # booster_accuracy_score.append(accuracy_score(true_labels, predict_booster))
        # booster_scores.append(boosterscore)
        # baseline_scores.append(baselinescore)


    return performance_booster, performance_baseline, booster_scores, baseline_scores#, booster_accuracy_score


def testNumLearners(Booster, Learner, data, start, end, inc, trials=1):
    results = defaultdict(int)
    for t in range(trials):
        shuffle(data)
        for m in range(start, end + 1, inc):
            accuracy = test(Booster, Learner, data, m)[0][-1]
            print m, accuracy
            results[m] += accuracy
    for m in results:
        results[m] /= trials
    return results
