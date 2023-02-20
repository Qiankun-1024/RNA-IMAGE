import os
import pandas as pd
import numpy as np

from sklearn.feature_selection import VarianceThreshold
from sklearn.model_selection import train_test_split

from utils import *

def Variance_feature_selector(x, threshold = 0.0):
    selector = VarianceThreshold(threshold = threshold)
    X = selector.fit_transform(x)
    X = selector.inverse_transform(X)
    x = pd.DataFrame(X, index=x.index, columns=x.columns)
    x = x.loc[:,(x!=0).any(axis=0)]

    return x



def split_data(x_data, category, batch):
    x_train, x_test, c_train, c_test, b_train, b_test = train_test_split(x_data, category, batch, test_size=0.2, random_state=42)
    
    return x_train, x_test, c_train, c_test, b_train, b_test



def label_processing(y_data):
    #y_data.replace('None', np.NaN, inplace=True)
    category = y_data.drop(['batch'], axis=1)
    batch = y_data['batch']

    category_dummies = pd.DataFrame()
    for label in category.columns:
        df = pd.get_dummies(category[label])
        df.columns = map(lambda x: label + '__' + x, df.columns)
        category_dummies = pd.concat([category_dummies, df], axis=1)    

    batch_dummies = pd.get_dummies(batch)

    return category_dummies, batch_dummies


def merge_log_data(datalist):
    log2_x = None
    y = None
    for (x_data, y_data, batch, d_type) in datalist:
        log2_tpm = log2_feature(x_data, d_type)
        y_data = y_data[['cancer_type', 'primary_site']]
        y_data['batch'] = batch

        if log2_x is None:
            log2_x = log2_tpm
            y = y_data
        else:
            log2_x = pd.concat([log2_x, log2_tpm], axis=0)
            y = pd.concat([y, y_data], axis=0)

    log2_x = Variance_feature_selector(log2_x)   

    for label in y.columns:
        counts = y[label].value_counts()
        drop = counts[counts <= 50].index.tolist()
        for i in drop:
            y[y==i] = np.nan
    y.dropna(how='any',inplace=True)

    log2_x = log2_x.loc[y.index]     
    max_matrix = log2_x.max()
    
    category, batch = label_processing(y)

    return log2_x, y, category, batch, max_matrix



if __name__ == '__main__':
    current_dir = os.path.dirname(os.path.abspath(__file__))
    data_path = os.path.join(current_dir, 'data/TCGA_GTEx')
    if not os.path.exists(data_path):
        os.mkdir(data_path)

    tcga_x = pd.read_csv('./data/TCGA_GTEx/tcga_features.tsv', index_col = 0, sep='\t')
    tcga_y = pd.read_csv('./data/TCGA_GTEx/tcga_labels.tsv', index_col = 0, sep='\t')
    gtex_x = pd.read_csv('./data/TCGA_GTEx/gtex_features.tsv',
                            index_col = 0, sep='\t')
    gtex_y = pd.read_csv('./data/TCGA_GTEx/gtex_labels.tsv', index_col = 0, sep='\t')


    data_list = [[tcga_x, tcga_y, 'TCGA', 'FPKM'],
                 [gtex_x, gtex_y, 'GTEx', 'TPM']]
    log2_x_data, y_data, category, batch, max_matrix  = merge_log_data(data_list)
    max_matrix.to_csv(os.path.join(data_path, 'max_norm_tpm.tsv'), sep='\t')
    log2_x_data.to_csv(os.path.join(data_path, 'log2_x_data.tsv'), sep='\t')
    y_data.to_csv(os.path.join(data_path, 'y_data.tsv'), sep='\t')
    category.to_csv(os.path.join(data_path, 'category.tsv'), sep='\t') 
    batch.to_csv(os.path.join(data_path, 'batch.tsv'), sep='\t')

    x_train, x_test, c_train, c_test, b_train, b_test = split_data(log2_x_data, category, batch)
    x_train.to_csv(os.path.join(data_path, 'x_train.tsv'), sep='\t')
    x_test.to_csv(os.path.join(data_path, 'x_test.tsv'), sep='\t')
    c_train.to_csv(os.path.join(data_path, 'c_train.tsv'), sep='\t')
    c_test.to_csv(os.path.join(data_path, 'c_test.tsv'), sep='\t')
    b_train.to_csv(os.path.join(data_path, 'b_train.tsv'), sep='\t')
    b_test.to_csv(os.path.join(data_path, 'b_test.tsv'), sep='\t')
    

    

