import pickle
import matplotlib.pyplot as plt
from models.core.train_eval import utils
import json


def get_evalMetrics(exp_ids):
    eval_metrics = {}

    for exp_id in exp_ids:
        dirName = './models/experiments/'+exp_id+'/eval_metrics'

        with open(dirName, "rb") as f:
            eval_metrics[exp_id] = pickle.load(f)

    return eval_metrics

explogs = utils.loadExplogs()
completed_exp = utils.get_completedExpIDs(explogs)
completed_exp
# %%
exp_ids = completed_exp
eval_metrics = get_evalMetrics(exp_ids)

def vis(exp_ids, eval_metrics):
    for exp_id in exp_ids:
        em_id = eval_metrics[exp_id]
        plt.figure()
        plt.scatter(em_id['x_test'], em_id['y_test'])
        # plt.scatter(em_id['x_test'], em_id['prediction_mean'])
        plt.scatter(em_id['x_test'], em_id['prediction_samples'])

        plt.legend(['True', 'Predicted mean'])
        plt.title(exp_id)



vis(exp_ids, eval_metrics)
# %%
