import os
import json
import pandas as pd
import pickle
from models.core.train_eval import utils

def get_expDir(config):
    exp_dir = './models/'
    if config['exp_type']['model'] == 'controller':
        exp_dir += 'controller/'

    elif config['exp_type']['model'] == 'driver_model':
        exp_dir += 'driver_model/'
    else:
        raise Exception("Unknown experiment type")

    return exp_dir + config['exp_id']



# %%
def genConfig(config):
    dirName = './models/experiments/'+config['exp_id']
    if not os.path.exists(dirName):
        os.mkdir(dirName)
        if not 'config.json' in os.listdir(dirName):
            with open(dirName+'/config.json','w') as f:
                json.dump(config, f, sort_keys=True,
                                indent=4, separators=(',', ': '))

def genExpID(last_exp_id):
    id = "{0:0=3d}".format(int(last_exp_id[3:])+1)
    return '{}{}'.format('exp',str(id))

def get_lastExpID(explogs):
    if not explogs:
        return 'exp000'
    else:
        return list(explogs.keys())[-1]

def genExpSeires(test_variables=None):
    """
    Function for generating series of folders for storing experiment reasults.
    :input: config_series defines the experiment series
    """
    if os.path.getsize(explogs_path) == 0:
        explogs = {}
    else:
        explogs = utils.loadExplogs(explogs_path)

    if test_variables:
        for param in test_variables['param_values']:
            last_exp_id = get_lastExpID(explogs)
            config_i = config
            config_i['model_config'][test_variables['param_name']] = param
            exp_id = genExpID(last_exp_id)
            config_i['exp_id'] = exp_id
            genConfig(config_i)
            explogs[exp_id] = explog
    else:
        last_exp_id = get_lastExpID(explogs)
        exp_id = genExpID(last_exp_id)
        config['exp_id'] = exp_id
        genConfig(config)
        explogs[exp_id] = explog

    utils.dumpExplogs(explogs_path, explogs)


# %%
# config = {
#  "model_config": {
#     "learning_rate": 1e-2,
#     "hidden_size": 5,
#     "components_n": 4
# },
# "data_config": {
#     "step_size": 3,
#     "sequence_length": 5,
#     "features": ['vel', 'pc','gap_size', 'dx', 'act_long_p', 'act_lat_p','lc_type'],
#     "history_drop": {"percentage":0, "vehicle":'mveh'},
#     "scaler":{"StandardScaler":['vel', 'pc','gap_size', 'dx',
#                                 'act_long_p', 'act_lat_p', 'act_long', 'act_lat']},
#     "scaler_path": '/models/experiments/scaler001'
# },
# "exp_id": 'NA',
# "exp_type": {"target_name":'yveh', "model":"controller"},
# "Note": 'NA'
# }
config = {
 "model_config": {
    "learning_rate": 1e-2,
    "hidden_size": 5,
    "components_n": 4
},
"data_config": {},
"exp_id": 'NA',
"exp_type": {"target_name":'yveh', "model":"controller"},
"Note": 'NA'
}

test_variables = {'param_name':'hidden_size', 'param_values': [1,2,3]} # variables being tested
explog = {'exp_state':'NA', 'target_name':'NA', 'model':'NA',
            'loss':'NA'}

explogs_path = './models/experiments/exp_logs.json'
from importlib import reload
reload(utils)
genExpSeires()
