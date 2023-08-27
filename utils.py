import os
import numpy as np
import pandas as pd
import tensorflow as tf

from model import BF_Generator, Batch_Encoder, Generator


def load_model(ckpt='latest', ckpt_path='training_checkpoints',
               z_dim=100, only_encoder=False):
    bf_generator = BF_Generator((-1, 128, 256, 1), z_dim)
    if os.path.isdir(ckpt_path):
        print("=> loading checkpoint '{}'".format(ckpt_path))
    else:
        print("=> no checkpoint found at '{}'".format(ckpt_path))
        
    checkpoint = tf.train.Checkpoint(bf_generator=bf_generator)
    
    if ckpt == 'latest':
        ckpt_path = tf.train.latest_checkpoint(ckpt_path)
    else:
        ckpt_path = os.path.join(ckpt_path, ckpt)      
          
    checkpoint.restore(ckpt_path).expect_partial()  
    
    if only_encoder:
        return bf_generator.encode
    return bf_generator


def load_batch_model(ckpt='latest', ckpt_path='training_checkpoints', z_dim=100):
    batch_encoder = Batch_Encoder((-1, 128, 256, 1), z_dim)
    if os.path.isdir(ckpt_path):
        print("=> loading checkpoint '{}'".format(ckpt_path))
    else:
        print("=> no checkpoint found at '{}'".format(ckpt_path))
        
    checkpoint = tf.train.Checkpoint(batch_encoder=batch_encoder)
    
    if ckpt == 'latest':
        ckpt_path = tf.train.latest_checkpoint(ckpt_path)
    else:
        ckpt_path = os.path.join(ckpt_path, ckpt)      
          
    checkpoint.restore(ckpt_path).expect_partial()  

    return batch_encoder


def load_generator_model(ckpt='latest', ckpt_path='training_checkpoints', z_dim=100):
    generator = Generator((-1, 128, 256, 1), z_dim)
    if os.path.isdir(ckpt_path):
        print("=> loading checkpoint '{}'".format(ckpt_path))
    else:
        print("=> no checkpoint found at '{}'".format(ckpt_path))
        
    checkpoint = tf.train.Checkpoint(generator=generator)
    
    if ckpt == 'latest':
        ckpt_path = tf.train.latest_checkpoint(ckpt_path)
    else:
        ckpt_path = os.path.join(ckpt_path, ckpt)      
          
    checkpoint.restore(ckpt_path).expect_partial()  

    return generator


def load_data(data_path='data/TCGA_GTEx/', dtype='log2TPM'):
    if dtype not in ['FPKM', 'TPM', 'log2TPM', 'normalizedTPM']:
        raise ValueError(
            f""" Allowed values: 'FPKM' ,'TPM', 'log2TPM' or 'normalizedTPM'
                but provided {dtype}"""
    )

    x_train = pd.read_csv(os.path.join(data_path, 'x_train.tsv'),
                          index_col=0, sep='\t')
    x_test = pd.read_csv(os.path.join(data_path, 'x_test.tsv'),
                         index_col=0, sep='\t')
    c_train = pd.read_csv(os.path.join(data_path, 'c_train.tsv'),
                          index_col=0, sep='\t')
    c_test = pd.read_csv(os.path.join(data_path, 'c_test.tsv'),
                         index_col=0, sep='\t')
    b_train = pd.read_csv(os.path.join(data_path, 'b_train.tsv'),
                          index_col=0, sep='\t')
    b_test = pd.read_csv(os.path.join(data_path, 'b_test.tsv'),
                         index_col=0, sep='\t')

    
    if dtype == 'TPM' or dtype == 'FPKM':
        x_train = log2_feature(x_train, dtype)
        x_test = log2_feature(x_test, dtype)
        
    if dtype != 'nomalizedTPM':
        x_train = normalize_feature(x_train)
        x_test = normalize_feature(x_test)  

    x_train = trans_1d_to_2d(x_train)
    x_test = trans_1d_to_2d(x_test)

    return (x_train.astype(np.float32), b_train.to_numpy().astype(np.float32),
            c_train.to_numpy().astype(np.float32),
            x_test.astype(np.float32), b_test.to_numpy().astype(np.float32),
            c_test.to_numpy().astype(np.float32))



def trans_1d_to_2d(x_data):
    piexl_coords = pd.read_csv('./data/tsne_sorted_pixels_coords.csv', index_col=0)

    x_len = piexl_coords['x_coord'].to_list()[-1]
    y_len = piexl_coords['y_coord'].to_list()[-1]

    data_df = pd.merge(piexl_coords, x_data.T, how='left', left_index=True, right_index=True, sort=False)
    data_df.drop(['x_coord', 'y_coord'], axis=1, inplace=True)
    data = data_df.fillna(0).T.values

    num = data.shape[0]
    data = data.reshape((num, x_len, y_len))
    data = data[:,::-1]

    return data



def trans_2d_to_1d(data):
    piexl_coords = pd.read_csv('./data/tsne_sorted_pixels_coords.csv', index_col=0)

    data = np.array(data)
    num = data.shape[0]
    data = data[:,::-1].reshape(num,-1)
    data_df = pd.DataFrame(data).T
    data_df.index = piexl_coords.index
    data_df = pd.concat([piexl_coords, data_df], axis=1)
    data_df.drop(['x_coord', 'y_coord'], axis=1, inplace=True)
    data_df = data_df[data_df.index.str.startswith('ENSG')]

    return data_df.T


def log2_feature(x_data, dtype='FPKM'):
    if dtype not in ['FPKM', 'TPM']:
        raise ValueError(
            f""" Allowed polarity values: 'positive' or 'negative'
                                    but provided {dtype}"""
        )
    
    if dtype == 'FPKM':
        #fpkm value to tpm value
        fpkm_data = x_data  
        sum = fpkm_data.sum(axis=1)
        tpm_data = fpkm_data.div(sum,axis=0)* 10**6

        log2_tpm = np.log2(tpm_data + 1)

    if dtype == 'TPM':
        tpm_data = x_data
        # tpm_data = tpm_data.mul(list(1000000 / tpm_data.sum(axis=1)), axis=0)
        log2_tpm = np.log2(tpm_data + 1)

    return log2_tpm


def normalize_feature(log2_tpm, max_matrix=None):
    #normalize the tpm value features on each gene by dividing by its maximum log2tpm value
    if max_matrix is None:
        max_matrix = pd.read_csv('./data/TCGA_GTEx/max_norm_tpm.tsv', index_col = 0, sep='\t')
    max_matrix.columns = ['max_log2tpm']

    log2_tpm_t = pd.merge(log2_tpm.T, (max_matrix+0.00000001), how='left', left_index=True, right_index=True, sort=False)
    log2_tpm_t = log2_tpm_t.dropna(how='any', axis=0)
    norm_tpm_t = log2_tpm_t.div(log2_tpm_t['max_log2tpm'], axis=0)
    norm_tpm = norm_tpm_t.drop(['max_log2tpm'], axis=1).T

    return norm_tpm


def reverse_log2_feature(norm_tpm, max_matrix=None):
    #reverse the normalized tpm value features to log2 tpm value on each gene by multiply by its maximum log2tpm value
    if max_matrix is None:
        max_matrix = pd.read_csv('./data/TCGA_GTEx/max_norm_tpm.tsv', index_col = 0, sep='\t')
    max_matrix.columns = ['max_log2tpm']
    norm_tpm_t = pd.merge(norm_tpm.T, (max_matrix+0.00000001), how='left', left_index=True, right_index=True, sort=False)
    log2_tpm_t = norm_tpm_t.mul(norm_tpm_t['max_log2tpm'], axis=0)
    log2_tpm = log2_tpm_t.drop(['max_log2tpm'], axis=1).T

    return log2_tpm


def reverse_tpm_feature(log2_tpm):
    tpm = 2 ** (log2_tpm) - 1 
    return tpm


def gene2img(data, dtype='TPM'):
    if dtype not in ['FPKM', 'TPM', 'log2TPM', 'normalizedTPM']:
        raise ValueError(
            f""" Allowed values: 'FPKM' ,'TPM', 'log2TPM' or 'normalizedTPM'
                but provided {dtype}"""
    )
    
    if dtype == 'FPKM' or dtype == 'TPM':
        data = log2_feature(data, dtype=dtype)
    
    if dtype != 'normalizedTPM':
        data = normalize_feature(data)
        
    data = trans_1d_to_2d(data).astype(np.float32)
    data = np.expand_dims(data, axis=3)  
    return data



def generate_data(data, dtype='TPM', outdtype='TPM', batch_size=256, model=None):
    if dtype not in ['FPKM', 'TPM', 'log2TPM', 'normalizedTPM']:
        raise ValueError(
            f""" Allowed values: 'FPKM' ,'TPM', 'log2TPM' or 'normalizedTPM'
                but provided {dtype}"""
    )
    if outdtype not in ['TPM', 'log2TPM']:
        raise ValueError(
            f""" Allowed values: 'TPM' or 'log2TPM'
                but provided {dtype}"""
    )

    '''load model to generate batch free data'''
    if model is None:
        model = load_model()
    
    img = gene2img(data, dtype=dtype)

    x_dataset = tf.data.Dataset.from_tensor_slices(img).batch(batch_size)
    z, bf_data = None, None
    for x in x_dataset: 
        mean_, logvar_ = model.encode(x)
        z_ = model.reparameterize(mean_, logvar_)
        bf_x = model.decode(z_, apply_sigmoid=True)
        if z is None:
            z = z_
            bf_data = bf_x
        else:
            z = np.concatenate((z,z_),axis=0)
            bf_data = np.concatenate((bf_data,bf_x),axis=0)

    bf_data = trans_2d_to_1d(bf_data)
    bf_data = reverse_log2_feature(bf_data)
    
    if outdtype == 'TPM':
        bf_data = reverse_tpm_feature(bf_data)
    
    z = pd.DataFrame(z, index=data.index)
    bf_data.index = data.index

    return z, bf_data


def generate_distribution(data, dtype='TPM', batch_size=256):
    '''load model to generate batch free data'''
    model = load_model()
    
    img = gene2img(data, dtype=dtype)

    x_dataset = tf.data.Dataset.from_tensor_slices(img).batch(batch_size)
    means, logvars = None, None
    for x in x_dataset: 
        mean_, logvar_ = model.encode(x)
        if means is None:
            means = mean_
            logvars = logvar_
        else:
            means = np.concatenate((means, mean_),axis=0)
            logvars = np.concatenate((logvars, logvar_),axis=0)
        
    means = pd.DataFrame(means, index=data.index)
    logvars = pd.DataFrame(logvars, index=data.index)

    return means, logvars


def generate_batch_data(data, dtype='TPM', batch_size=256):
    '''load model to generate batch free data'''
    model = load_batch_model()
    
    img = gene2img(data, dtype=dtype)

    x_dataset = tf.data.Dataset.from_tensor_slices(img).batch(batch_size)
    z = None
    for x in x_dataset: 
        z_ = model(x)
        if z is None:
            z = z_
        else:
            z = np.concatenate((z,z_),axis=0)
    z = pd.DataFrame(z, index=data.index)
    return z


def Wasserstein_Distance(query_means, query_logvars, target_mean, target_logvar):
    p1 = (tf.reduce_sum(tf.subtract(query_means, target_mean) ** 2, axis=-1)) ** 0.5
    p2 = tf.reduce_sum(tf.subtract(tf.exp(query_logvars) ** 0.5,
                                   tf.exp(target_logvar) ** 0.5)
                       ** 2, axis=-1)
    return (p1+p2).numpy()


def Wasserstein_Distance_parallel(means, logvars):
    means_a2 = tf.reshape(tf.reduce_sum(means ** 2, axis=1), (-1, 1))
    means_b2 = tf.reduce_sum(means ** 2, axis=1)
    means_2ab = tf.matmul(means, tf.transpose(means)) * 2
    p1 = means_a2 + means_b2 - means_2ab
    sqrt_exp_logvars = tf.exp(logvars) ** 0.5
    logvars_a2 = tf.reshape(tf.reduce_sum(sqrt_exp_logvars ** 2, axis=1), (-1, 1))
    logvars_b2 = tf.reduce_sum(sqrt_exp_logvars ** 2, axis=1)
    logvars_2ab = tf.matmul(sqrt_exp_logvars, tf.transpose(sqrt_exp_logvars)) * 2
    p2 = logvars_a2 + logvars_b2 - logvars_2ab
    return (p1+p2).numpy()


def compute_distance(model, data, dtype, batch_size=256, parallel=True):
    index = data.index
    data = gene2img(data, dtype=dtype)
    
    x_dataset = tf.data.Dataset.from_tensor_slices(data).batch(batch_size)
    means, logvars = None, None
    for x in x_dataset: 
        means_, logvars_ = model(x)
        if means is None:
            means = means_
            logvars = logvars_
        else:
            means = np.concatenate((means, means_),axis=0)
            logvars = np.concatenate((logvars, logvars_),axis=0)  
    
       
    if parallel:
        d = Wasserstein_Distance_parallel(means, logvars)
        
        import scipy.sparse as sp
        edge = sp.coo_matrix(d)
        result = pd.DataFrame({'query':edge.row,
                               'target':edge.col,
                               'distance':edge.data})
        result = result[(result['query'] < result['target'])]
        result['query'] = index[result['query']]
        result['target'] = index[result['target']]
    else:
        result = pd.DataFrame()
        for i in range(data.shape[0]-1):
            for j in range(i+1, len(data), batch_size):
                from_ = j
                to = min((from_ + batch_size), len(data))
                d = Wasserstein_Distance(means[from_:to],
                                        logvars[from_:to],
                                        means[i], logvars[i])
                result_ = pd.DataFrame({'query':index[from_:to],
                                        'target':index[i],
                                        'distance':d})
                result = pd.concat([result, result_], axis=0)
    return result

    
def umap_visualization(data, label, plt_classes,
                       z_dim=100, n_neighbors=25,
                       min_dist=0.5):
    import matplotlib.pyplot as plt
    from sklearn.preprocessing import LabelEncoder
    from umap import UMAP
    from sklearn.decomposition import PCA
    from sklearn.preprocessing import StandardScaler
    le = LabelEncoder()
    
    if data.shape[1] > z_dim:
        scaler = StandardScaler()
        data = scaler.fit_transform(data)
        pca = PCA(n_components=z_dim)
        data = pca.fit_transform(data)
    embedding = UMAP(n_neighbors=n_neighbors,
                     min_dist=min_dist).fit_transform(data)

    fig = plt.figure(figsize=(8 * len(plt_classes), 8))
    for i, c in enumerate(plt_classes):
        encoded_label = le.fit_transform(label[c].astype('str'))
        ax = plt.subplot(1, len(plt_classes), i+1)
        scatter = ax.scatter(embedding[:,0], embedding[:,1], alpha=.8, lw=2, c=encoded_label, cmap=plt.cm.Set3)
        plt.xlabel('UMAP-1')
        plt.ylabel('UMAP-2')
        plt.legend(handles = scatter.legend_elements()[0] , loc=0,
                labels=list(np.unique(label[c].astype('str'))), scatterpoints=1)
        
    embedding = pd.DataFrame(embedding)
    embedding.columns = ['UMAP1', 'UMAP2']
    return fig, embedding

