import json
from driver_model.core.model import Model
from driver_model.core.prepare_data import PrepData
import numpy as np
import os
from importlib import reload
# reload(PrepData)
"""
Reads config file for the experiment
Preps data
Trains model
Test models [tensorboard, prediction profiles, key metrics. ]
Saves all files in output
"""


# os.getcwd()
def load_config_file(exp):
    path = '/experiments/' + exp + '/job_config.json'
    with open(path) as f:
        config_file = json.loads(f.read())
        return config_file["model_config"], config_file["data_config"]

_, data_config = load_config_file('exp001')
data_config
data_preprocessor = PrepData(data_config)
data_preprocessor.actions
data_preprocessor.sequence_length
data_preprocessor.__dict__
a[2]

a = np.array(range(100))
a = []
for i in range(100):
    a.append([i , 1, 2, 3, 4])

a
data_preprocessor.sequence(a)


    Model(model_config)


experiment_sets = ['exp001', 'exp002']


model = load_model('exp001')






for exp in experiment_sets:

    model_config, data_config  = load_config_file(exp)
    model = Model(model_config)
    model.train()
