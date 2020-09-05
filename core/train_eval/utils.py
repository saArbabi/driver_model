import json
def loadConfig(exp_id):
    dirName = './models/experiments/'+exp_id+'/config.json'
    with open(dirName, 'r') as f:
        return json.load(f)

def loadExplogs(explogs_path):
    with open(explogs_path, 'r') as f:
        return json.load(f)

def dumpExplogs(explogs_path, explogs):
    with open(explogs_path, 'w') as f:
        json.dump(explogs, f, sort_keys=True,
                        indent=4, separators=(',', ': '))
