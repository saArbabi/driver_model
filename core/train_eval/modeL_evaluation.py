import models.core.tf_models.abstract_model as am
from models.core.tf_models.utils import nll_loss, get_predictionMean, get_predictionSamples
import pickle
from tensorflow import keras


def dumpEvalmetrics(exp_dir, eval_metrics):
    with open(exp_dir+'/eval_metrics', "wb") as f:
        pickle.dump(eval_metrics, f)

def modelEvaluate(model, validation_data, config):
    """
    Function for evaluating the model.
    Performance metrics are:
        - nll loss, training and validation
        - RWSE
        -
    """
    x_test, y_test = validation_data

    predictions = model.predict(y_test)
    eval_metrics = {}
    eval_metrics['prediction_mean'] = get_predictionMean(predictions, config).numpy()
    eval_metrics['prediction_samples'] = get_predictionSamples(1, predictions, config).numpy()

    eval_metrics['x_test'] = x_test
    eval_metrics['y_test'] = y_test
    dumpEvalmetrics(model.exp_dir, eval_metrics)
