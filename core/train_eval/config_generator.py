import os
import json
from models.core.train_eval import utils

explogs_path = './models/experiments/exp_logs.json'
explog = {'exp_state':'NA',
        'model':'NA',
        'train_loss':'NA',
        'val_loss':'NA'}

def genConfig(config):
    dirName = './models/experiments/'+config['exp_id']
    if not os.path.exists(dirName):
        os.mkdir(dirName)
    if not 'config.json' in os.listdir(dirName):
        with open(dirName+'/config.json','w') as f:
            json.dump(config, f,
                            indent=4, separators=(',', ': '))

def genExpID(series_id, last_exp_id):
    if last_exp_id[0:9] == series_id:
        id = "{0:0=3d}".format(int(last_exp_id[12:])+1)
    else:
        id = "{0:0=3d}".format(0)
    return  series_id + '{}{}'.format('exp',id)

def get_lastExpID(explogs):
    if not explogs:
        return 'series000exp000'
    else:
        return list(explogs.keys())[-1]

def genExpSeires(series_id, test_variables, config):
    """
    Function for generating series of folders for storing experiment reasults.
    :input: config_series defines the experiment series
    """
    # config = utils.loadConfigBase(config_base)

    if os.path.getsize(explogs_path) == 0:
        explogs = {}
    else:
        explogs = utils.loadExplogs()

    if test_variables:
        for variable in test_variables:
            for param in test_variables[variable]:
                last_exp_id = get_lastExpID(explogs)
                config_i = config
                config_i['model_config'][variable] = param
                exp_id = genExpID(series_id, last_exp_id)
                config_i['exp_id'] = exp_id
                genConfig(config_i)
                explogs[exp_id] = explog
    else:
        last_exp_id = get_lastExpID(explogs)
        exp_id = genExpID(series_id, last_exp_id)
        config['exp_id'] = exp_id
        genConfig(config)
        explogs[exp_id] = explog

    utils.dumpExplogs(explogs_path, explogs)
    print("You are ready to run experiment ", exp_id)
