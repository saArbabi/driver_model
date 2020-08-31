



class Model(object):
    def __init__(self, model_config):
        self.config = model_config


    def train_model():
        pass

    def scale_data(self):
        pass

    def xy_split(self, xy_array):
        pass

    def history_drop(self):
        pass

    def sequence(self, xy_array):

        sequential_data = []
        i_reset = 0
        i = 0
        for chunks in range(self.step_size):
            prev_states = deque(maxlen=self.sequence_length)
            while i < len(xy_array):
                row = xy_array[i]
                prev_states.append([n for n in row[:len(self.actions)]])
                if len(prev_states) == self.sequence_length:
                    sequential_data.append([np.array(prev_states), row[len(self.actions):]])
                i += self.step_size
            i_reset += 1
            i = i_reset

        return sequential_data


    def preprocess(self, data):

        xy_set = []
        for scenario in datasets:
            df_scen = df.loc[df['scenario'] == scenario]
            car_ids = df_scen['id'].unique()
            for id in car_ids:
                df_id =  df_scen.loc[df_scen['id'] == id].reset_index(drop=True)
                indx = df_id['frm'].diff()[df_id['frm'].diff() != 1].index.values
                n_lc = len(indx)  # number of LCs for this car

                if n_lc > 1:
                    init_index = 0
                    for n in range(1,n_lc+1):
                        if n == n_lc:
                            df_id_sec = df_id.iloc[init_index:]
                        else:
                            end_index = indx[n] - 1
                            df_id_sec = df_id.iloc[init_index:end_index]
                        init_index = end_index + 1
                        if len(df_id_sec)>10:
                            sequential_data = sequence(df_id_sec)
                            xy_set.append(sequential_data)

                else:
                    sequential_data = sequence(df_id)
                    xy_set.append(sequential_data)

        return self.xy_split(xy_array)

    def data_prep(self):
        train_set, val_set = self.load_data()

        x_train, y_train = preprocess(train_set)
        x_val, y_val = preprocess(train_set)

        return x_train, y_train, x_val, y_val



        return train_set[self.config['features']], val_set[self.config['features']]
