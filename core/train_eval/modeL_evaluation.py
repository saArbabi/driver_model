import models.core.tf_models.abstract_model as am
from models.core.tf_models.utils import nll_loss, get_predictionMean
import pickle

def dumpEvalmetrics(exp_dir, eval_metrics):
    with open(exp_dir+'/eval_metrics', "wb") as f:
        pickle.dump(eval_metrics, f)

def modelEvaluate(x_test, y_test, config):
    """
    Function for evaluating the model.
    Performance metrics are:
        - nll loss, training and validation
        - RWSE
        -
    """
    model = keras.models.load_model(model.exp_dir+'/trained_model',
                                            custom_objects={'loss': nll_loss(config)})

    predictions = model.predict(y_test)
    eval_metrics = {}
    eval_metrics['prediction_mean'] = get_predictionMean(predictions, config)
    eval_metrics['x_test'] = x_test
    eval_metrics['y_test'] = y_test
    dumpEvalmetrics(model.exp_dir, eval_metrics)
