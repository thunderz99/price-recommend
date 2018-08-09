import gc
import time
import numpy as np
import pandas as pd
from pandas.api.types import CategoricalDtype
# import matplotlib.pyplot as plt
# import seaborn as sns
from scipy.sparse import csr_matrix, hstack
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.preprocessing import LabelBinarizer
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import mean_squared_error, mean_absolute_error
from tabulate import tabulate

import lightgbm as lgb


NUM_BRANDS = 4000
NUM_CATEGORIES = 1000
NAME_MIN_DF = 10
MAX_FEATURES_ITEM_DESCRIPTION = 50000

# Split the dataset in to train and test. We are using training data only for
# EDA.


def handle_missing_inplace(dataset):
    dataset['category_name'].fillna(value='missing', inplace=True)
    dataset['brand_name'].fillna(value='missing', inplace=True)
    dataset['item_description'].replace(
        'No description yet,''missing', inplace=True)
    dataset['item_description'].fillna(value='missing', inplace=True)


def cutting(dataset):
    pop_brand = dataset['brand_name'].value_counts(
    ).loc[lambda x: x.index != 'missing'].index[:NUM_BRANDS]
    dataset.loc[~dataset['brand_name'].isin(
        pop_brand), 'brand_name'] = 'missing'
    pop_category = dataset['category_name'].value_counts(
    ).loc[lambda x: x.index != 'missing'].index[:NUM_CATEGORIES]


class PriceModel:

    def __init__(self):
        pass

    def load_data(self, filename):

        print(filename)
        df = pd.read_csv(filename, sep='\t')
        print("df len:", len(df))

        msk = np.arange(len(df)) < len(df) * 0.8
        train = df[msk]
        test = df[~msk]

        test_new = test.drop('price', axis=1)
        # origin price
        self.y_test_origin = test["price"]
        # price preprocessed
        self.y_test = np.log1p(self.y_test_origin)

        # Drop rows where price = 0
        train = train[train.price != 0].reset_index(drop=True)

        nrow_train = train.shape[0]
        self.y = np.log1p(train["price"])
        merge = pd.concat([train, test_new])

        self.merge = merge

        handle_missing_inplace(self.merge)
        cutting(self.merge)
        self.to_categorical(self.merge)

        print("head of train data:")
        print(self.merge.head())

        # Count vectorize name and category name columns.
        self.name_cv = CountVectorizer(min_df=NAME_MIN_DF)
        X_name = self.name_cv.fit_transform(self.merge['name'])

        self.cat_cv = CountVectorizer()
        X_category = self.cat_cv.fit_transform(self.merge['category_name'])

        # TFIDF Vectorize item_description column.
        self.tv = TfidfVectorizer(max_features=MAX_FEATURES_ITEM_DESCRIPTION,
                                  ngram_range=(1, 3), stop_words='english')
        X_description = self.tv.fit_transform(self.merge['item_description'])

        # Label binarize brand_name column.
        self.lb = LabelBinarizer(sparse_output=True)
        X_brand = self.lb.fit_transform(self.merge['brand_name'])

        print("X_brand.shape", X_brand.shape)
        print("lb.transform Razor:", self.lb.transform(['Razor']))

        # transform item_condtion_id and shipping to vector
        dummies = pd.get_dummies(self.merge[['item_condition_id', 'shipping']],
                                 sparse=False)
        X_dummies = csr_matrix(dummies.values)

        print("X_dummies.shape:", X_dummies.shape)

        # Create sparse merge.
        sparse_merge = hstack(
            (X_dummies, X_description, X_brand, X_category, X_name)).tocsr()

        print("sparse_merge.shape", sparse_merge.shape)

        # Remove features with document frequency <=1.
        self.unuse_feature_mask = np.array(
            np.clip(sparse_merge.getnnz(axis=0) - 1, 0, 1), dtype=bool)
        sparse_merge = sparse_merge[:, self.unuse_feature_mask]

        print("sparse_merge.shape after remove freq<=1:", sparse_merge.shape)

        # Separate train and test data from sparse merge.
        self.X = sparse_merge[:nrow_train]
        self.X_test = sparse_merge[nrow_train:]

        self.train_data = train
        self.test = test

        print("X.shape:", self.X.shape)
        # end of data preparation

    def train(self):

        train_X = lgb.Dataset(self.X, label=self.y)

        params = {
            'learning_rate': 0.75,
            'application': 'regression',
            'max_depth': 3,
            'num_leaves': 100,
            'verbosity': -1,
            'metric': 'RMSE',
        }

        # ### Training
        # Training a model requires a parameter list and data set.
        # And training will take a while.

        self.gbm = lgb.train(params, train_set=train_X,
                             num_boost_round=3200, verbose_eval=100)

        # ### Prediction

        print("X_test.shape", self.X_test.shape)
        y_pred = self.gbm.predict(
            self.X_test, num_iteration=self.gbm.best_iteration)

        y_test = self.y_test
        y_test_origin = self.y_test_origin

        print('The rmse of prediction log1p is:',
              mean_squared_error(y_test, y_pred) ** 0.5)

        print('The rmse of prediction price is:',
              mean_squared_error(y_test_origin, np.expm1(y_pred)) ** 0.5)

        print('The mae of prediction price is:',
              mean_absolute_error(y_test_origin, np.expm1(y_pred)))

        print("type y_test:", type(y_test))
        print("type y_pred:", type(y_pred))

        for i in range(5):
            print("test[{}]".format(i), self.test.iloc[[i]])
            print("log price:{}, predict:{}".format(
                y_test.values[i], y_pred[i]))
            print("$ price:{}, predict:{}".format(
                y_test_origin.values[i], np.expm1(y_pred[i])))

        return "OK"

    def fit(self, filename):

        self.load_data(filename)
        print("fit filename:", filename)
        self.train()

    def predict_file(self, filename):

        item_df = pd.read_csv(filename, sep='\t')
        return predict(self, item_df)

    def predict(self, item_df):
        X_test = self.preprocess(item_df)

        # print("X_test.shape:", X_test.shape)
        # print("X_test:", X_test[0])
        y_pred = self.gbm.predict(
            X_test, num_iteration=self.gbm.best_iteration)

        y = np.expm1(y_pred)

        print("price is: %.1f dollar" % y)
        return y.tolist()

    def preprocess(self, item_df):

        print("item_df, before preprocess:")
        print(tabulate(item_df, headers='keys', tablefmt='psql'))

        if('price' in item_df.columns):
            item_df = item_df.drop('price', axis=1)

        handle_missing_inplace(item_df)
        item_df = self.to_categorical_for_predict(item_df)
        handle_missing_inplace(item_df)

        X_name = self.name_cv.transform(item_df['name'])
        X_category = self.cat_cv.transform(item_df['category_name'])
        # TFIDF Vectorize item_description column.
        X_description = self.tv.transform(item_df['item_description'])
        # Label binarize brand_name column.

        X_brand = self.lb.transform(item_df['brand_name'])

        X_dummies = self.get_dummies_for_predict(item_df)

        print("dummies:", X_dummies)

        sparse_merge = hstack(
            (X_dummies, X_description, X_brand, X_category, X_name)).tocsr()

        print("before remove unuse_feature:", sparse_merge.shape)

        sparse_merge = sparse_merge[:, self.unuse_feature_mask]

        print("after preprocess", sparse_merge.shape)

        return sparse_merge

    def to_categorical(self, dataset):

        # hold category for predict afterwise
        self.cat_type_category = CategoricalDtype(
            categories=pd.unique(dataset['category_name']), ordered=True)
        print("self.cat_type_category:", self.cat_type_category)
        dataset['category_name'] = dataset['category_name'].astype(
            self.cat_type_category)

        self.cat_type_brand = CategoricalDtype(
            categories=pd.unique(dataset['brand_name']), ordered=True)
        dataset['brand_name'] = dataset['brand_name'].astype(
            self.cat_type_brand)

        self.cat_type_condition = CategoricalDtype(
            categories=pd.unique(dataset['item_condition_id']), ordered=True)
        dataset['item_condition_id'] = dataset['item_condition_id'].astype(
            self.cat_type_condition)

    def to_categorical_for_predict(self, item_df):

        print("item_df.shape", item_df.shape)
        print("item_df", item_df)
        print("len(item_df)", len(item_df))

        item_df['category_name'] = item_df['category_name'].astype(
            self.cat_type_category)
        item_df['brand_name'] = item_df['brand_name'].astype(
            self.cat_type_brand)
        item_df['item_condition_id'] = item_df['item_condition_id'].astype(
            self.cat_type_condition)

        return item_df

    def get_dummies_for_predict(self, item_df):

        # TODO, get_dummies for item_df should be ok.
        nrow_origin = self.merge.shape[0]

        dataset = pd.concat([self.merge, item_df])

        X_dummies = csr_matrix(pd.get_dummies(
            dataset[['item_condition_id', 'shipping']], sparse=True).values)

        return X_dummies[nrow_origin:]
