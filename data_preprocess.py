import os
import pandas as pd
import numpy as np

from sklearn.feature_selection import VarianceThreshold
from sklearn.model_selection import train_test_split


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



def trans_1d_to_2d(x_data):
    piexl_coords = pd.read_csv("/vol1/cuipeng_group/qiankun/GAN-VAE/data/tsne_sorted_pixels_coords.csv", index_col=0)

    x_len = piexl_coords["x_coord"].to_list()[-1]
    y_len = piexl_coords["y_coord"].to_list()[-1]

    data_df = pd.merge(piexl_coords, x_data.T, how='left', left_index=True, right_index=True, sort=False)
    data_df.drop(["x_coord", "y_coord"], axis=1, inplace=True)
    data = data_df.fillna(0).T.values

    num = data.shape[0]
    data = data.reshape((num, x_len, y_len))
    data = data[:,::-1]

    return data



def trans_2d_to_1d(data):
    piexl_coords = pd.read_csv("/vol1/cuipeng_group/qiankun/GAN-VAE/data/tsne_sorted_pixels_coords.csv", index_col=0)

    data = np.array(data)
    num = data.shape[0]
    data = data[:,::-1].reshape(num,-1)
    data_df = pd.DataFrame(data).T
    data_df.index = piexl_coords.index
    data_df = pd.concat([piexl_coords, data_df], axis=1)
    data_df.drop(["x_coord", "y_coord"], axis=1, inplace=True)
    data_df = data_df[data_df.index.str.startswith('ENSG')]

    return data_df.T


def log2_feature(x_data, data_type='FPKM'):
    if data_type not in ["FPKM", "TPM"]:
        raise ValueError(
            f""" Allowed polarity values: 'positive' or 'negative'
                                    but provided {data_type}"""
        )
    
    if data_type == 'FPKM':
        #fpkm value to tpm value
        fpkm_data = x_data  
        sum = fpkm_data.sum(axis=1)
        tpm_data = fpkm_data.div(sum,axis=0)* 10**6

        log2_tpm = np.log2(tpm_data + 1)

    if data_type == 'TPM':
        tpm_data = x_data
        log2_tpm = np.log2(tpm_data + 1)

    return log2_tpm


def normalize_feature(log2_tpm):
    #normalize the tpm value features on each gene by dividing by its maximum log2tpm value
    max_matirx = pd.read_csv("/vol1/cuipeng_group/qiankun/GAN-VAE/data/max_norm_tpm.tsv", index_col = 0, sep="\t")
    max_matirx.columns = ['max_log2tpm']

    log2_tpm_t = pd.merge(log2_tpm.T, (max_matirx+0.00000001), how='left', left_index=True, right_index=True, sort=False)
    log2_tpm_t = log2_tpm_t.dropna(how='any', axis=0)
    norm_tpm_t = log2_tpm_t.div(log2_tpm_t['max_log2tpm'], axis=0)
    norm_tpm = norm_tpm_t.drop(['max_log2tpm'], axis=1).T

    return norm_tpm


def reverse_log2_feature(norm_tpm):
    #reverse the normalized tpm value features to log2 tpm value on each gene by multiply by its maximum log2tpm value
    max_matirx = pd.read_csv("/vol1/cuipeng_group/qiankun/GAN-VAE/data/max_norm_tpm.tsv", index_col = 0, sep="\t")
    max_matirx.columns = ['max_log2tpm']
    norm_tpm_t = pd.merge(norm_tpm.T, (max_matirx+0.00000001), how='left', left_index=True, right_index=True, sort=False)
    log2_tpm_t = norm_tpm_t.mul(norm_tpm_t['max_log2tpm'], axis=0)
    log2_tpm = log2_tpm_t.drop(['max_log2tpm'], axis=1).T

    return log2_tpm


def reverse_tpm_feature(log2_tpm):
    tpm = 2 ** (log2_tpm) - 1 
    return tpm


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



def load_data():
    current_dir = os.path.dirname(os.path.abspath(__file__))
    data_path = os.path.join(current_dir, "data")
    x_train = pd.read_csv(os.path.join(data_path, "x_train.tsv"),
                          index_col=0, sep="\t")
    x_test = pd.read_csv(os.path.join(data_path, "x_test.tsv"),
                         index_col=0, sep="\t")
    c_train = pd.read_csv(os.path.join(data_path, "c_train.tsv"),
                          index_col=0, sep="\t")
    c_test = pd.read_csv(os.path.join(data_path, "c_test.tsv"),
                         index_col=0, sep="\t")
    b_train = pd.read_csv(os.path.join(data_path, "b_train.tsv"),
                          index_col=0, sep="\t")
    b_test = pd.read_csv(os.path.join(data_path, "b_test.tsv"), index_col=0, sep="\t")

    x_train = normalize_feature(x_train)
    x_test = normalize_feature(x_test)

    x_train = trans_1d_to_2d(x_train)
    x_test = trans_1d_to_2d(x_test)


    return (x_train.astype(np.float32), b_train.to_numpy().astype(np.float32),
            c_train.to_numpy().astype(np.float32),
            x_test.astype(np.float32), b_test.to_numpy().astype(np.float32),
            c_test.to_numpy().astype(np.float32))


def genarate_new_data(x_data, data_type='TPM'):
    if data_type == 'FPKM':
        log2_data = log2_feature(x_data, data_type='FPKM')
    if data_type == 'TPM':
        log2_data = log2_feature(x_data, data_type='TPM')    
    if data_type == 'log2_TPM':
        log2_data = x_data
    
    norm_data = normalize_feature(log2_data)
    x = trans_1d_to_2d(norm_data)
    x = np.expand_dims(x, axis=3)

    return x.astype(np.float32)


if __name__ == '__main__':
    current_dir = os.path.dirname(os.path.abspath(__file__))
    data_path = os.path.join(current_dir, "data")
    if not os.path.exists(data_path):
        os.mkdir(data_path)

    tcga_x = pd.read_csv("/vol1/cuipeng_group/qiankun/GAN-VAE/data/tcga_features.tsv", index_col = 0, sep="\t")
    tcga_y = pd.read_csv("/vol1/cuipeng_group/qiankun/GAN-VAE/data/tcga_labels.tsv", index_col = 0, sep="\t")
    gtex_x = pd.read_csv("/vol1/cuipeng_group/qiankun/GAN-VAE/data/gtex_features.tsv",
                            index_col = 0, sep="\t")
    gtex_y = pd.read_csv("/vol1/cuipeng_group/qiankun/GAN-VAE/data/gtex_labels.tsv", index_col = 0, sep="\t")

    data_list = [[tcga_x, tcga_y, 'TCGA', 'FPKM'],
                 [gtex_x, gtex_y, 'GTEx', 'TPM']]
    log2_x_data, y_data, category, batch, max_matrix  = merge_log_data(data_list)
    max_matrix.to_csv(os.path.join(data_path, "max_norm_tpm.tsv"), sep="\t")
    log2_x_data.to_csv(os.path.join(data_path, "log2_x_data.tsv"), sep="\t")
    y_data.to_csv(os.path.join(data_path, "y_data.tsv"), sep="\t")
    category.to_csv(os.path.join(data_path, "category.tsv"), sep="\t")
    batch.to_csv(os.path.join(data_path, "batch.tsv"), sep="\t")

    x_train, x_test, c_train, c_test, b_train, b_test = split_data(log2_x_data, category, batch)
    x_train.to_csv(os.path.join(data_path, "x_train.tsv"), sep="\t")
    x_test.to_csv(os.path.join(data_path, "x_test.tsv"), sep="\t")
    c_train.to_csv(os.path.join(data_path, "c_train.tsv"), sep="\t")
    c_test.to_csv(os.path.join(data_path, "c_test.tsv"), sep="\t")
    b_train.to_csv(os.path.join(data_path, "b_train.tsv"), sep="\t")
    b_test.to_csv(os.path.join(data_path, "b_test.tsv"), sep="\t")
    
    #use all category labels
    # y = pd.read_csv(os.path.join(data_path, "all_label.tsv"), index_col=0, sep="\t")
    # category, batch= label_processing(y)
    # missing_label_idx = category[category['tumor_stage__None']==1].index
    # category.loc[missing_label_idx,
    #             ['tumor_stage__Stage 0',
    #             'tumor_stage__Stage I',
    #             'tumor_stage__Stage II',
    #             'tumor_stage__Stage III',
    #             'tumor_stage__Stage IV']] = 2 
    # category.drop('tumor_stage__None', axis=1, inplace=True)
    # log2_x_data = pd.read_csv(os.path.join(data_path, "log2_x_data.tsv"), index_col=0, sep="\t")
    # x_train, x_test, c_train, c_test, b_train, b_test = split_data(log2_x_data, category, batch)
    # c_train.to_csv(os.path.join(data_path, "all_c_train.tsv"), sep="\t")
    # c_test.to_csv(os.path.join(data_path, "all_c_test.tsv"), sep="\t")

    

