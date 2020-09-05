
model = keras.models.load_model(model.exp_dir+'/trained_model',
                                        custom_objects={'loss': utils.nll_loss(config)})
