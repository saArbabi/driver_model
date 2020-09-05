
def get_expDir(config):
    exp_dir = './models/'
    if config['exp_type']['model'] == 'controller':
        exp_dir += 'controller/'

    elif config['exp_type']['model'] == 'driver_model':
        exp_dir += 'driver_model/'
    else:
        raise Exception("Unknown experiment type")

    return exp_dir + config['exp_id']

    model = keras.models.load_model(model.exp_dir+'/trained_model',
                                        custom_objects={'loss': nll_loss(config)})

    model = keras.models.load_model(model.exp_dir+'/trained_model',
                                            custom_objects={'loss': nll_loss(config)})

# %%
