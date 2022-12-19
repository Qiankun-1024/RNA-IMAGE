import numpy as np
import pandas as pd
import os
import tensorflow as tf

import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.preprocessing import LabelEncoder
from sklearn.manifold import TSNE
from umap import UMAP

from data_preprocess import *
from Batch_Free_GAN import BF_Generator, Batch_Encoder

from utils import generate_data, generate_batch_data, umap_visualization

def pca_visualization(data, label, out_name):
    le = LabelEncoder()
    encoded_label = le.fit_transform(label)
    pca = PCA(n_components=2)
    X = pca.fit_transform(data)

    explained_variance_ratio_ = pca.explained_variance_ratio_

    fig = plt.figure(figsize=(15, 8))
    scatter = plt.scatter(X[:,0], X[:,1], alpha=.8, lw=2, c=encoded_label, cmap=plt.cm.Spectral)
    plt.legend(handles = scatter.legend_elements()[0], bbox_to_anchor=(1.05, 1),
                 loc=2, borderaxespad=0, shadow=False, labels=list(np.unique(label)), scatterpoints=1)
    plt.xlabel('PC1({:.2%} explained variance)'.format(explained_variance_ratio_[0]))
    plt.ylabel('PC2({:.2%} explained variance)'.format(explained_variance_ratio_[1]))
    fig.subplots_adjust(right=0.7)

    plt_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'figure')
    if not os.path.exists(plt_dir):
        os.mkdir(plt_dir)
    plt.savefig(os.path.join(plt_dir, (out_name+'.png')))


def tsne_visualization(data, label, out_name):
    le = LabelEncoder()
    encoded_label = le.fit_transform(label)
    X = TSNE(n_components=2, init='pca',
             random_state=0, perplexity=100).fit_transform(data)

    fig = plt.figure(figsize=(15, 8))
    scatter = plt.scatter(X[:,0], X[:,1], s=1, alpha=.8, lw=2, c=encoded_label, cmap=plt.cm.Spectral)
    plt.legend(handles = scatter.legend_elements()[0], bbox_to_anchor=(1.05, 1),
                 loc=2, borderaxespad=0, shadow=False, labels=list(np.unique(label)), scatterpoints=1)
    plt.xlabel('TSNE1')
    plt.ylabel('TSNE2')
    fig.subplots_adjust(right=0.7)

    plt_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'figure', 'TSNE')
    if not os.path.exists(plt_dir):
        os.mkdir(plt_dir)
    plt.savefig(os.path.join(plt_dir, (out_name+'.png')))
    plt.close()
    X_df = pd.DataFrame(X)
    X_df.columns = ['TSNE-1', 'TSNE-2']
    X_df.to_csv(os.path.join(plt_dir, (out_name+'.tsv')), sep='\t')


# def umap_visualization(data, label, out_name):
#     le = LabelEncoder()
#     encoded_label = le.fit_transform(label)
#     X = UMAP(n_neighbors=25,
#              min_dist=0.5).fit_transform(data)

#     fig = plt.figure(figsize=(15, 8))
#     scatter = plt.scatter(X[:,0], X[:,1], s=1, alpha=.8, lw=2, c=encoded_label, cmap=plt.cm.Spectral)
#     plt.legend(handles = scatter.legend_elements()[0], bbox_to_anchor=(1.05, 1),
#                  loc=2, borderaxespad=0, shadow=False, labels=list(np.unique(label)), scatterpoints=1)
#     plt.xlabel('UMAP1')
#     plt.ylabel('UMAP2')
#     fig.subplots_adjust(right=0.7)

#     plt_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'figure', 'TSNE')
#     if not os.path.exists(plt_dir):
#         os.mkdir(plt_dir)
#     plt.savefig(os.path.join(plt_dir, (out_name+'.png')))
#     plt.close()
#     X_df = pd.DataFrame(X)
#     X_df.columns = ['UMAP1', 'UMAP2']
#     X_df.to_csv(os.path.join(plt_dir, (out_name+'.tsv')), sep='\t')


def generate_bf_data(data, label, model):
    '''load model to generate batch free data'''

    x_dataset = tf.data.Dataset.from_tensor_slices(data).batch(128)
    mean, logvar, z, bf_data = None, None, None, None
    for x in x_dataset: 
        mean_, logvar_ = model.encode(x)
        z_ = model.reparameterize(mean_, logvar_)
        bf_x = model.decode(z_, apply_sigmoid=True)
        if z is None:
            z = z_
            mean = mean_
            logvar = logvar_
            bf_data = bf_x
        else:
            z = np.concatenate((z,z_),axis=0)
            mean = np.concatenate((mean,mean_),axis=0)
            logvar = np.concatenate((logvar,logvar_),axis=0)
            bf_data = np.concatenate((bf_data,bf_x),axis=0)

    bf_data = trans_2d_to_1d(bf_data)
    log2_bf_data = reverse_log2_feature(bf_data)
    
    mean = pd.DataFrame(mean, index=label.index)
    logvar = pd.DataFrame(logvar, index=label.index)  
    z = pd.DataFrame(z, index=label.index)
    log2_bf_data.index = label.index

    return mean, logvar, z, log2_bf_data



# def generate_batch_data(data, label, model):
#     '''load model to generate batch free data'''

#     x_dataset = tf.data.Dataset.from_tensor_slices(data).batch(128)
#     mean, logvar, z, bf_data = None, None, None, None
#     for x in x_dataset: 
#         z_ = model(x)
#         if z is None:
#             z = z_
#         else:
#             z = np.concatenate((z,z_),axis=0)

#     z = pd.DataFrame(z, index=label.index)
#     return z
 

def generate_data_pca(model, num, z_dim):
    random_vector_for_generation = tf.random.normal(shape=[num, z_dim])
    generated_x = model.sample(random_vector_for_generation)
    generated_x = trans_2d_to_1d(generated_x)

    pca = PCA(n_components=2)
    pc = pca.fit_transform(generated_x)
    explained_variance_ratio_ = pca.explained_variance_ratio_

    plt.scatter(pc[:,0], pc[:,1], alpha=.8, lw=2)
    plt.xlabel('PC1({:.2%} explained variance)'.format(explained_variance_ratio_[0]))
    plt.ylabel('PC2({:.2%} explained variance)'.format(explained_variance_ratio_[1]))

    plt_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'figure')
    if not os.path.exists(plt_dir):
        os.mkdir(plt_dir)
    plt.savefig(os.path.join(plt_dir, 'generated.png'))



# def generate_data(data, label):
#     data = log2_feature(data, data_type='TPM')
#     data = normalize_feature(data)
#     data = trans_1d_to_2d(data)
#     norm_data = np.expand_dims(data, axis=3)
    
#     ##hyper parameter
#     z_dim = 100
    
#     ##load model
#     checkpoint_dir = '/vol1/cuipeng_group/qiankun/GAN-VAE/training_checkpoints_v3'
#     bf_generator = BF_Generator(norm_data.shape, z_dim)
#     batch_encoder = Batch_Encoder(norm_data.shape, z_dim)
#     checkpoint = tf.train.Checkpoint(bf_generator=bf_generator,
#                                      batch_encoder=batch_encoder)
#     latest = tf.train.latest_checkpoint(checkpoint_dir)
#     # latest = os.path.join(checkpoint_dir, 'ckpt-98')
#     checkpoint.restore(latest).expect_partial()
    
#     mean, logvar, z, bf_x = generate_bf_data(norm_data, label, bf_generator)
#     bf_x = reverse_tpm_feature(bf_x)
#     return z, bf_x

if __name__ == '__main__':
    gpus = tf.config.experimental.list_physical_devices("GPU")
    if gpus:
        gpu1 = gpus[0]
        tf.config.experimental.set_memory_growth(gpu1, True)
        tf.config.experimental.set_visible_devices([gpu1],"GPU")
        print(gpu1)
    ##load data
    # label = pd.read_csv('/vol1/cuipeng_group/qiankun/GAN-VAE/data/y_data.tsv', index_col=0, sep='\t')
    # data = pd.read_csv('/vol1/cuipeng_group/qiankun/GAN-VAE/data/log2_x_data.tsv', index_col=0, sep='\t')
   
    # data = normalize_feature(data)
    # data = trans_1d_to_2d(data)
    # norm_data = np.expand_dims(data, axis=3)
    # ##hyper parameter
    # z_dim = 100
    
    # ##load model
    # checkpoint_dir = '/vol1/cuipeng_group/qiankun/GAN-VAE/training_checkpoints_v3'
    # bf_generator = BF_Generator(norm_data.shape, z_dim)
    # batch_encoder = Batch_Encoder(norm_data.shape, z_dim)
    # checkpoint = tf.train.Checkpoint(bf_generator=bf_generator,
    #                                  batch_encoder=batch_encoder)
    # latest = tf.train.latest_checkpoint(checkpoint_dir)
    # checkpoint.restore(latest).expect_partial()

    
    # mean, logvar, z, bf_x = generate_bf_data(norm_data, label, bf_generator)
    # zb = generate_batch_data(norm_data, label, batch_encoder)
    
    # z_concat = pd.concat([zc, zb], axis=1)
    # pca = PCA(n_components=100)
    # pca_x = pca.fit_transform(x)
    # pca_bfx = pca.fit_transform(bf_x)

    # umap_visualization(pca_x, label['batch'], "x_umap")
    # umap_visualization(z, label['batch'], "z_umap")
    # umap_visualization(pca_bfx, label['batch'], "bf_x_umap")

    
    # bf_x = reverse_tpm_feature(bf_x)
    # x = reverse_tpm_feature(x)
    # x.to_csv('/vol1/cuipeng_group/qiankun/GAN-VAE/data/output/x_tpm.tsv', sep='\t')
    # mean.to_csv('/vol1/cuipeng_group/qiankun/GAN-VAE/data/output/mean.tsv', sep='\t')
    # logvar.to_csv('/vol1/cuipeng_group/qiankun/GAN-VAE/data/output/logvar.tsv', sep='\t')
    # z.to_csv('/vol1/cuipeng_group/qiankun/GAN-VAE/data/output/z.tsv', sep='\t')
    # bf_x.to_csv('/vol1/cuipeng_group/qiankun/GAN-VAE/data/output/bfx_tpm.tsv', sep='\t')
    # zb.to_csv('/vol1/cuipeng_group/qiankun/GAN-VAE/data/output/zb.tsv', sep='\t')
    
    
    #load data
    label = pd.read_csv('/vol1/cuipeng_group/qiankun/GAN-VAE/data/all_label.tsv',
                        index_col=0, sep='\t')
    data = pd.read_csv('/vol1/cuipeng_group/qiankun/GAN-VAE/data/log2_x_data.tsv',
                       index_col=0, sep='\t')
    
    zc, bf_x = generate_data(data, dtype='log2TPM', outdtype='TPM')
    zb = generate_batch_data(data, dtype='log2TPM')
    z_concat = pd.concat([zc, zb], axis=1)
    
    zc.to_csv('./data/output/v5_zc.tsv', sep='\t')
    zb.to_csv('./data/output/zb.tsv', sep='\t')
    z_concat.to_csv('./data/output/z_concat.tsv', sep='\t')
    bf_x.to_csv('./data/output/v5_bfx.tsv', sep='\t')
    
    fig_zb, embed_zb = umap_visualization(zb, label, ['batch'])
    embed_zb.to_csv('./data/figure/TSNE/zb.tsv', sep='\t')
    fig_zc, embed_zc = umap_visualization(zc, label, ['batch'])
    embed_zc.to_csv('./data/figure/TSNE/zc.tsv', sep='\t')
    fig_concat, embed_concat = umap_visualization(z_concat, label, ['batch'])
    embed_concat.to_csv('./data/figure/TSNE/z_concat.tsv', sep='\t')

    


