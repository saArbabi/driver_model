import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

mveh_col = ['id', 'episode_id','lc_type', 'name', 'frm', 'scenario', 'vel', 'pc',
       'gap_size', 'dx', 'act_long_p', 'act_lat_p', 'act_long', 'act_lat']

yveh_col = ['id', 'episode_id','lc_type', 'name', 'frm', 'scenario', 'vel', 'act_long_p', 'act_long']

mveh_df = pd.read_csv('/datasets/mveh_df.txt', delimiter=' ',
                        header=None, names=mveh_col)
yveh_df = pd.read_csv('/datasets/yveh_df.txt', delimiter=' ',
                        header=None, names=yveh_col)

len(mveh_df)
len(yveh_df)
mveh_df['gap_size'].min()
mveh_df['act_long'].max()
mveh_df['act_lat'].min()

mveh_df['vel'].max()
mveh_df['dx'].max()


feature_col = ['vel', 'act_long', 'act_lat', 'gap_size', 'pc', 'dx']

mveh_df['act_lat'].plot.hist(bins=125)
for feature in feature_col:
    plt.figure()
    mveh_df[feature].plot.hist(bins=125)
    plt.title(feature)



mveh_df.loc[mveh_df['act_long']>8]
['act_long'].max()

test_car = mveh_df.loc[mveh_df['episode_id']==243]

acc =  (test_car['vel'].iloc[1:].values - test_car['vel'].iloc[:-1].values)/0.1
test_car.drop(test_car.index[0], inplace=True)
test_car.loc[1:,['act_long', 'a']] = [2,2]
test_car.loc[1:,['a','v']] = [2, 2]


acc_p =  test_car['vel'].iloc[:-1].values


- test_car['vel'].iloc[:-1].values)/0.1
test_car.loc[['g','w']] = pd.DataFrame(test_car[['vel','act_long']].values)



test_car[['a','b']] = 0
plt.plot(test_car['vel'].values)
plt.plot(test_car['act_long'].values)
plt.grid()

plt.plot(acc)

plt.plot(acc)
plt.plot(test_car['act_long'])

plt.plot(test_car['vel'])
