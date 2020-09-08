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

def genExpSeires(config_base, test_variables=None):
    """
    Function for generating series of folders for storing experiment reasults.
    :input: config_series defines the experiment series
    """
    config = utils.loadConfigBase(config_base)

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
    print("You are ready to run your experiments")

{
 "model_config": {
     "learning_rate": 1e-2,
     "neurons_n": 50,
     "layers_n": 2,
     "epochs_n": 5,
     "batch_n": 128,
     "components_n": 5
},
"data_config": {    "step_size": 3,
    "sequence_n": 1,
    "veh_states":{"mveh":["lc_type", "vel", "pc", "gap_size", "dx", "act_long_p", "act_lat_p"],
                    "yveh":["vel", "act_long_p"]}
},
"exp_id": "NA",
"model_type": "merge_controller",
"Note": "NA"
}
genExpSeires(config_base='baseline_run.json', test_variables=test_variables)
