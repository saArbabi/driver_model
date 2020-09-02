import os
import json

def get_expDir(config):
    exp_dir = './models/'
    if config['exp_type']['model'] == 'controller':
        exp_dir += 'controller/'

    elif config['exp_type']['model'] == 'driver_model':
        exp_dir += 'driver_model/'
    else:
        raise Exception("Unknown experiment type")

    return exp_dir + config['exp_name']



config = {
 "model_config": {
    "learning_rate": 1e-2,
    "hidden_size": 5,
    "components_n": 4
},
"data_config": {
    "step_size": 3,
    "sequence_length": 5,
    "features": ['vel', 'pc','gap_size', 'dx', 'act_long_p', 'act_lat_p','lc_type'],
    "history_drop": {"percentage":0, "vehicle":'mveh'},
    "scaler":{"StandardScaler":['vel', 'pc','gap_size', 'dx',
                                'act_long_p', 'act_lat_p', 'act_long', 'act_lat']},
    "scaler_path": './driver_model/experiments/scaler001'
},
"exp_name": 'exp001',
"exp_type": {"target_name":'yveh', "model":"controller"},
}
# %%

# %%
def generate_config(config):
    dirName = get_expDir(config)
    if not os.path.exists(dirName):
        os.mkdir(dirName)
        if not 'config.json' in os.listdir(dirName):
            with open(dirName+'/config.json','w') as f:
                json.dump(config, f)
