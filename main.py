import argparse
from random import seed
from yaml import dump
from utils.experiment import test
from utils.utils import *


if __name__ == "__main__":
    seed(0)

    parser = argparse.ArgumentParser(
        description='Test error for a combination of ensembler and weak learner.')
    parser.add_argument('dataset', help='dataset filename')
    parser.add_argument('ensembler', help='chosen ensembler')
    parser.add_argument('weak_learner', help='chosen weak learner')
    parser.add_argument('M', metavar='# weak_learners',
                        help='number of weak learners', type=int)
    parser.add_argument(
        'trials', help='number of trials (each with different shuffling of the data); defaults to 1', type=int, default=1, nargs='?')
    parser.add_argument('--record', action='store_const', const=True,
                        default=False, help='export the results in YAML format')
    args = parser.parse_args()

    ensembler = get_ensembler(args.ensembler)
    weak_learner = get_weak_learner(args.weak_learner)
    data = load_data("data/" + args.dataset)

    accuracy, baseline, boost_score, base_score, accuracy_scores = test(
        ensembler, weak_learner, data, args.M, trials=args.trials)

    print "Accuracy:"
    print accuracy
    print "Baseline:"
    print baseline[-1]
    print "Booster Scores"
    print boost_score
    print "Baseline Scores:"
    print base_score
    print "Sklearn accuracy_score"
    print accuracy_scores

    if args.record:
        results = {
            'm': args.M,
            'accuracy': accuracy,
            'baseline': baseline[-1],
            'booster': args.ensembler,
            'weak_learner': args.weak_learner,
            'trials': args.trials,
            'seed': 0
        }
        filename = args.ensembler + "_" + \
            args.weak_learner + "_" + str(args.M) + ".yml"
        f = open(filename, 'w+')
        f.write(dump(results))
