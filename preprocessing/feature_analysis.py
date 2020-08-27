import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

mveh_col = ['id', 'frm', 'scenario', 'name', 'vel', 'act_long', 'act_lat',
       'gap_size', 'pc', 'dx']

yveh_col = ['scenario', 'frm', 'id', 'name', 'vel', 'act_long']
mveh_df = pd.read_csv('./driver_model/datasets/mveh_df.txt', delimiter=' ',
                        header=None, names=mveh_col)
yveh_df = pd.read_csv('./driver_model/datasets/yveh_df.txt', delimiter=' ',
                        header=None, names=yveh_col)

len(mveh_df)
len(yveh_df)
mveh_df['gap_size'].min()
mveh_df['act_long'].max()
mveh_df['act_lat'].max()
mveh_df['dx'].max()
