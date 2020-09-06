import pickle
import matplotlib.pyplot as plt



def get_evalMetrics(exp_ids):
    eval_metrics = {}

    for exp_id in exp_ids:
        dirName = './models/experiments/'+exp_id+'/eval_metrics'

        with open(dirName, "rb") as f:
            eval_metrics[exp_id] = pickle.load(f)

    return eval_metrics

exp_ids = ['exp001']
eval_metrics = get_evalMetrics(exp_ids)

eval_metrics['exp001'].keys()
# %%
def vis(exp_ids, eval_metrics):

    for exp_id in exp_ids:
        em_id = eval_metrics[exp_id]
        plt.figure()
        plt.scatter(em_id['x_test'], em_id['y_test'])
        plt.scatter(em_id['x_test'], em_id['prediction_mean'])
        plt.legend(['True', 'Predicted mean'])
        plt.title(exp_id)



vis(exp_ids, eval_metrics)
# %%
