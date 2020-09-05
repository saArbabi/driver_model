import json
explogs_path = './models/experiments/exp_logs.json'

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
        return json.load(f)

def dumpExplogs(explogs_path, explogs):
    with open(explogs_path, 'w') as f:
        json.dump(explogs, f, sort_keys=True,
                        indent=4, separators=(',', ': '))

def get_undoneExpIDs(explogs):
    undone_exp = []
    for key, value in explogs.items():  # for name, age in dictionary.iteritems():  (for Python 2.x)
        if value['exp_state'] == 'NA':
            undone_exp.append(key)
    return undone_exp

def updateExpstate(explogs, exp_id, exp_state):
    explogs[exp_id]['exp_state'] = exp_state
    dumpExplogs(explogs_path, explogs)
    if exp_state == 'complete':
        print('Experiment 0 ', exp_id, ' has been complete')
    elif exp_state == 'in progress':
        print('Experiment 0 ', exp_id, 'is in progress')
