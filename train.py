#!/usr/local/bin/python3

import sys
import os
import json
import csv
import random
import pandas as pd
from mercari import PriceModel

model = PriceModel()
model.load_data("train.10000.tsv")
model.train()

print("train_data: type", type(model.train_data))

with open('item1.json') as f:
    item_json = json.load(f)

item_df = pd.DataFrame().append(item_json, ignore_index=True)
print("item_df:", item_df)


model.predict(item_df)
