import numpy as np
import pandas as pd
import pickle as pk
import glob
import os

county_list = os.listdir('data/set_features')
print(county_list)

to_remove = []

for id, file_name_ in enumerate(county_list[0:1]):
    print(id)

    s = pd.read_parquet('data/set_features/' + file_name_, engine='pyarrow')
    # for i in s.index:
    #     print(s.loc[i].tolist())
    nan_list = s.columns[s.isna().any()].tolist()
    str_list = []
    for c in s.columns:
        # if s[c].dtype == object:
        #     print('damn')
        #     print(c)
        #     print(s[c])
        if isinstance(s.iloc[0][c], str):
            str_list.append(c)

    non_numer_list = []
    for c in s.columns:
        if not pd.to_numeric(s[c], errors='coerce').notnull().all():
            non_numer_list.append(c)

    # print(len(nan_list))
    # print(len(str_list))
    # print(len(non_numer_list))

    to_remove = list(set.union(set(to_remove), set(nan_list), set(str_list), set(non_numer_list)))


print(to_remove)
print(len(to_remove))

for file_name_ in county_list:
    s = pd.read_parquet('data/set_features/' + file_name_, engine='pyarrow')
    s = s.drop(columns=to_remove).to_numpy()
    s = np.float32(s)
    with open('data/cleaned_features/' + file_name_[:-3] + '.pkl', 'wb') as f:
        pk.dump(s, f)
