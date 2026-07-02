
import itertools, operator

import matplotlib.pyplot as plt

def pseudo_roc(labels = [True, False, False, True, False, True], ranks = [1,2,3,1,2,3], add_baseline=False, *args, **kwargs):
    # pseudo-ROC plots as in: https://www.sciencedirect.com/science/article/pii/S1535947620316522#fig5
    labels_scores_sorted = [(label, score) for label, score in sorted(zip(labels, ranks), key=operator.itemgetter(1))]
    x = [0]
    y = [0]
    for label, rank in itertools.groupby(labels_scores_sorted, key=operator.itemgetter(1)):
        x_next = x[-1] if len(x) > 0 else 0
        y_next = y[-1] if len(x) > 0 else 0
        for (label_, rank_) in rank:
            if label_:
                y_next += 1
            else:
                x_next += 1
        x.append(x_next)
        y.append(y_next)
    #plt.step(x, y, where='pre', *args, **kwargs) # Does not work in "tie cases" where equal scores have both labels
    plt.plot(x, y, *args, **kwargs) # Plots "ties" as diagnoal
    if add_baseline:
        plt.plot([0, len(ranks)], [0, sum(labels)], color='k', linestyle='dashed', linewidth=0.5)
