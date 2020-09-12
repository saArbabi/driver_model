    def trajCompute(self, state_t0, config):
        """
        :Param:
            state0 - current vehicle state
        :Return: [[x_cor, y_cor], ...]
        """
        state_arr = np.repeat([state_t0], self.samples_n, axis=0)
        for i in range(10):
            state_arr = self.step(state_arr, i)
            self.x.append(self.x[-1] + state_arr[:, self.mvehindx['vel']]*0.1)
            self.y.append(self.y[-1] + act_lat*0.1)

        return self.x, self.y

    def step(self, state_arr, i):
        act_long, act_lat = self.get_actions(state_arr, config)

        state_arr[:, self.mvehindx['vel']] += act_long*0.1
        state_arr[:, self.mvehindx['pc']] += act_lat*0.1
        # lc_left = state_arr[:, self.mvehindx['pc']]  > 1.85
        # state_arr[lc_left, self.mvehindx['pc']] = -1.85
        # lc_right = state_arr[:, self.mvehindx['pc']]  < -1.85
        # state_arr[lc_right, self.mvehindx['pc']] = 1.85
        dv = state_arr[:, self.mvehindx['vel']] - state_arr[:, self.yvehindx['vel']]
        state_arr[:, self.mvehindx['dx']] += dv*0.1

        state_arr[:, self.yvehindx['vel']] = self.veh_state[i+1][self.yvehindx['vel']]

        return state_arr
        
def get_expDir(config):
    exp_dir = './models/'
    if config['model_type']['model'] == 'controller':
        exp_dir += 'controller/'

    elif config['model_type']['model'] == 'driver_model':
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

def build_toy_dataset(nsample=10000):
    y_data = np.float32(np.random.uniform(-10.5, 10.5, (1, nsample))).T
    r_data = np.float32(np.random.normal(size=(nsample,1))) # random noise
    x_data = np.float32(np.sin(0.75*y_data)*7.0+y_data*0.5+r_data*1.0)
    return train_test_split(x_data, y_data, random_state=42, train_size=0.8)


###################################


        self.y = [np.zeros(self.samples_n)]
        self.x = [np.zeros(self.samples_n)]
