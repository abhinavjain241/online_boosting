import warnings

from sklearn.datasets import load_svmlight_file
warnings.filterwarnings("ignore", module="sklearn")

from ensemblers.adaboost import AdaBooster
from ensemblers.ogboost import OGBooster
from ensemblers.ocpboost import OCPBooster
from ensemblers.expboost import EXPBooster
from ensemblers.osboost import OSBooster
from ensemblers.smoothboost import SmoothBooster

from learners.naive_bayes_gaussian import NaiveBayes as GaussianNB
from learners.naive_bayes_binary import NaiveBayes as BinaryNB
from learners.perceptron import Perceptron
from learners.random_stump import RandomStump
#from learners.decision_stump import DecisionStump
from learners.decision_tree import DecisionTree
from learners.knn import kNN
from learners.histogram import RNB
from learners.winnow import Winnow
from learners.mlp import MLP
from learners.mlp2 import MLP as MLP2


def load_data(filename):
    X, y = load_svmlight_file(filename)

    data = zip(X, y)
    return data


ensemblers = {
    "AdaBooster": AdaBooster,
    "OCPBooster": OCPBooster,
    "EXPBooster": EXPBooster,
    "OGBooster": OGBooster,
    "OSBooster": OSBooster,
    "SmoothBooster": SmoothBooster
}


def get_ensembler(ensembler_name):
    return ensemblers[ensembler_name]


weak_learners = {
    "GaussianNB": GaussianNB,
    "BinaryNB": BinaryNB,
    "kNN": kNN,
    #"MLP": MLP,
    #"MLP2": MLP2,
    #"DecisionStump": DecisionStump,
    "DecisionTree": DecisionTree,
    "Perceptron": Perceptron,
    "RandomStump": RandomStump,
    #"Winnow": Winnow,
    #"Histogram": RNB
}


def get_weak_learner(weak_learner_name):
    return weak_learners[weak_learner_name]
