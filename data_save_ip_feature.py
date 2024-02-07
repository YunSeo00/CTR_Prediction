import time
import os
import sys
import argparse

import torch
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

# load dataset and preprocessing
train_data = pd.read_parquet("../input/train.parquet")
test_data = pd.read_parquet("../input/test.parquet")

## make date feature
def make_date_feature(df):
    df["hour"] = df["date"].dt.hour
    return df

train_data = make_date_feature(train_data)
test_data = make_date_feature(test_data)

def hex_to_A_class(hex_str):
    return int(hex_str[0:2], 16)

def hex_to_demical(hex_str):
    return int(hex_str, 16)

train_data['site_id_A_class'] = train_data['site_id'].apply(hex_to_A_class)
train_data['app_ip_A_class'] = train_data['app_id'].apply(hex_to_A_class)

test_data['site_id_A_class'] = test_data['site_id'].apply(hex_to_A_class)
test_data['app_ip_A_class'] = test_data['app_id'].apply(hex_to_A_class)

## drop unused columns
train_data = train_data.drop(['id', 'site_id', 'site_domain', 'app_id', 'app_domain', 'device_id', 'device_ip', 'device_model', 'f_2', 'f_5', 'f_7', 'f_9'], axis=1)
test_data = test_data.drop(['id', 'site_id', 'site_domain', 'app_id', 'app_domain', 'device_id', 'device_ip', 'device_model', 'f_2', 'f_5', 'f_7', 'f_9'], axis=1)

## categorical feature encoding
cat_val = ['hour', 'banner_pos', 'site_category', 'app_category', 'device_type', 'device_conn_type', 'f_1', 'f_3', 'f_4', 'f_6', 'f_8', 'site_id_A_class', 'app_ip_A_class']
field_dims = [len(set(train_data[col])) for col in cat_val] # using for embedding dimension (model input)

le = LabelEncoder()
for i in range(len(cat_val)):
    train_data[cat_val[i]] = le.fit_transform(train_data[cat_val[i]])
    test_data[cat_val[i]] = le.transform(test_data[cat_val[i]])

## split train, validation set
tmp_stratify = train_data['date'].astype(str) + '-' +train_data['click'].astype(str)
train_data, val_data = train_test_split(train_data, test_size=0.2, random_state=42, stratify=tmp_stratify)

X_train = train_data[cat_val].values
y_train = train_data["click"].values
X_val = val_data[cat_val].values
y_val = val_data["click"].values
X_test = test_data[cat_val].values

offsets = torch.IntTensor((0, *np.cumsum(field_dims)[:-1]))
total_field_dims = sum(field_dims)

## Save numpy vector
np.save("./data/ip_feature/X_train.npy", X_train)
np.save("./data/ip_feature/y_train.npy", y_train)
np.save("./data/ip_feature/X_val.npy", X_val)
np.save("./data/ip_feature/y_val.npy", y_val)
np.save("./data/ip_feature/X_test.npy", X_test)

## save offset, total_field_dims
np.save("./data/ip_feature/offsets.npy", offsets.cpu().numpy())
np.save("./data/ip_feature/total_field_dims.npy", total_field_dims)
