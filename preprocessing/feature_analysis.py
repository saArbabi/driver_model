import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

mveh_col = ['id','episode_id', 'name', 'frm', 'scenario', 'vel', 'act_long', 'act_lat',
       'gap_size', 'pc', 'dx']

yveh_col = ['id', 'episode_id', 'name','frm', 'scenario', 'vel', 'act_long']

mveh_df = pd.read_csv('./driver_model/datasets/mveh_df.txt', delimiter=' ',
                        header=None, names=mveh_col)
yveh_df = pd.read_csv('./driver_model/datasets/yveh_df.txt', delimiter=' ',
                        header=None, names=yveh_col)

len(mveh_df)
len(yveh_df)
mveh_df['gap_size'].min()
mveh_df['act_long'].max()
mveh_df['act_lat'].min()

mveh_df['vel'].max()
mveh_df['dx'].max()


feature_col = ['vel', 'act_long', 'act_lat', 'gap_size', 'pc', 'dx']


sample_scaler = standard_scaler_sample.fit(train_set[sample_col].values)
sample = sample_scaler.transform(train_set[sample_col].values)
scaled_data = pd.DataFrame(sample, columns=sample_col)
scaled_data = pd.DataFrame(scaled_data, columns=sample_col)

for feature in feature_col:
    plt.figure()
    mveh_df[feature].plot.hist(bins=125)
    plt.title(feature)



len(mveh_df.loc[mveh_df['vel']>5])/len(mveh_df)

mveh_df.loc[mveh_df['vel']>15]['act_long'].plot.hist(bins=125)
['act_long'].max()
mveh_df.loc[(mveh_df['vel']>5) & (mveh_df['act_long']>15)]
['act_long'].max()

test_car = mveh_df.loc[mveh_df['episode_id']=='r2046']
print()
plt.plot(test_car['act_long'])

vel = test_car['vel'].values
N = len(vel)
acc = []
for row_i in range(N):
    if row_i < N-1:
        acc_c = (vel[row_i+1] - vel[row_i])/0.1

    else:
        acc_c = (vel[row_i-1] - vel[row_i])/0.1

    acc.append(acc_c)


plt.plot(acc)
plt.plot(test_car['act_long'])

plt.plot(test_car['vel'])
