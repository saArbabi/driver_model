
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
    @tf.function
    def tracemodel(self, x):
        """Trace model execution - use for writing model graph
        :param: A sample input
        """
        return self(x)

    def saveGraph(self, x):
        writer = tf.summary.create_file_writer(self.exp_dir+'/graph')
        tf.summary.trace_on(graph=True, profiler=True)
        # Forward pass
        z = self.tracemodel(x.reshape(-1,1))
        with writer.as_default():
            tf.summary.trace_export(name='model_trace',
                                step=0, profiler_outdir=self.exp_dir+'/graph')
        writer.close()
