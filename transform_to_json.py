import sys
import pandas as pd
import json

#train_id = int(sys.argv[1])

df = pd.read_csv('train.100000.tsv', sep='\t')

print("df.shape:", df.shape)
df_sliced = df.iloc[:10000]

for index, row in df_sliced.iterrows():
    part_df = df_sliced.loc[[index]]
    dict_obj = part_df.to_dict('records')[0]

    # print("dict_obj:", dict_obj)
    train_id = dict_obj.get('train_id', 0)
    category = dict_obj.get("category_name", "")
    brand_name = dict_obj.get("brand_name", "")

    if(category == "Electronics/Cell Phones & Accessories/Cell Phones & Smartphones" and brand_name == "Apple"):
        print("train_id:", train_id)
        with open("items/item_{}.json".format(train_id), 'w') as f:
            json.dump(dict_obj, f, indent=4)
