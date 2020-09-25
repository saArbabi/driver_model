import json
import os
explogs_path = './models/experiments/exp_logs.json'

# %%
def loadConfigBase(file_name):
    dirName = './models/experiments/'+file_name
    with open(dirName, 'r') as f:
        return json.load(f)

def loadConfig(exp_id):
    dirName = './models/experiments/'+exp_id+'/config.json'
    with open(dirName, 'r') as f:
        return json.load(f)

def loadExplogs():
    with open(explogs_path, 'r') as f:
        explogs = json.load(f)
        _explogs = {}
        for key in sorted(explogs):
            _explogs[key] = explogs[key]
        return _explogs

def dumpExplogs(explogs_path, explogs):
    with open(explogs_path, 'w') as f:
        json.dump(explogs, f,
                        indent=4, separators=(',', ': '))

def get_undoneExpIDs(explogs):
    undone_exp = []
    for key, value in explogs.items():
        if value['exp_state'] == 'NA':
            undone_exp.append(key)
    return undone_exp

def get_completedExpIDs(explogs):
    undone_exp = []
    for key, value in explogs.items():
        if value['exp_state'] == 'complete':
            undone_exp.append(key)
    return undone_exp

def updateExpstate(model, explogs, exp_id, exp_state):
    for key, value in explogs.items():
        # first remove failed experiments
        if value['exp_state'] != ('complete' or 'NA') and key != exp_id:
            explogs[key]['exp_state'] = 'failed'

    explogs[exp_id]['exp_state'] = exp_state
    explogs[exp_id]['train_loss'] = round(model.train_loss.result().numpy().item(), 2)
    explogs[exp_id]['val_loss'] = round(model.test_loss.result().numpy().item(), 2)
    explogs[exp_id]['epoch'] += 1
    dumpExplogs(explogs_path, explogs)

def delete_experiment(exp_id):
    dirName = './models/experiments/'+exp_id
    explogs = loadExplogs()
    if exp_id in explogs:
        explogs.pop(exp_id)

    dumpExplogs(explogs_path, explogs)
    if exp_id in os.listdir('./models/experiments'):
        if 'config.json' in os.listdir(dirName):
            os.remove(dirName+'/config.json')
