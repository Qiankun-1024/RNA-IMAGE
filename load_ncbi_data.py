import pandas as pd
import tensorflow as tf
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.preprocessing import LabelEncoder
from Batch_Free_GAN import BF_Generator
from data_preprocess import *


def pca_visualization(data, label, out_name):
    pca = PCA(n_components=2)
    X = pca.fit_transform(data)
    explained_variance_ratio_ = pca.explained_variance_ratio_
    PC = pd.DataFrame(X, columns=['PC1', 'PC2'])
    PC['PMID'] = list(label)

    fig = plt.figure(figsize=(15, 8))
    sns.jointplot(data=PC, x='PC1', y='PC2', hue='PMID')
    fig.subplots_adjust(right=0.7)

    plt_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'figure', 'NCBI')
    if not os.path.exists(plt_dir):
        os.mkdir(plt_dir)
    plt.savefig(os.path.join(plt_dir, (out_name+'.png')))
    
    
data = pd.read_csv('/vol1/cuipeng_group/qiankun/GAN-VAE/data/NCBI/29273624_tpm.tsv',
                   index_col=0, sep='\t')
data = log2_feature(data, data_type='TPM')
norm_data = normalize_feature(data)
norm_data = trans_1d_to_2d(norm_data)
x = np.expand_dims(norm_data, axis=3)

label = pd.DataFrame({'source': 'NCBI'*len(data)}, index=data.index)
# label = pd.read_csv('/vol1/cuipeng_group/qiankun/GAN-VAE/data/NCBI/cancer_sra_list.csv')
# label = label.set_index(['library'])
# label = label.loc[data.index]
# label = label[['PMID']].astype('str')

dataset = pd.read_csv('/vol1/cuipeng_group/qiankun/GAN-VAE/data/log2_x_data.tsv', index_col=0, sep='\t')
dataset_label = pd.read_csv('/vol1/cuipeng_group/qiankun/GAN-VAE/data/all_label.tsv', index_col=0, sep='\t')
dataset_label = dataset_label.drop(['batch'], axis=1)
dataset_z = pd.read_csv('/vol1/cuipeng_group/qiankun/GAN-VAE/data/output/cpu1_ckpt56_z.tsv', index_col=0, sep='\t')
dataset_label = dataset_label[(dataset_label['primary_site']=='Breast')].iloc[0:26]
dataset_z = dataset_z.loc[dataset_label.index]
dataset = dataset.loc[dataset_label.index]

##hyper parameter
z_dim = 25
    
##load model
cpu1_checkpoint_dir = '/vol1/cuipeng_group/qiankun/GAN-VAE/cpu1_training_checkpoints'
bf_generator = BF_Generator(x.shape, z_dim)
checkpoint = tf.train.Checkpoint(bf_generator=bf_generator)
latest = os.path.join(cpu1_checkpoint_dir, 'ckpt-56')
checkpoint.restore(latest).expect_partial()

mean, logvar = bf_generator.encode(x)
z = bf_generator.reparameterize(mean, logvar)
norm_bf_x = bf_generator.decode(z, apply_sigmoid=True)
norm_bf_x = trans_2d_to_1d(norm_bf_x.numpy())
log2_bf_x = reverse_log2_feature(norm_bf_x)
bf_x = reverse_tpm_feature(log2_bf_x)
z = pd.DataFrame(z.numpy(), index=label.index)
z.columns = [i for i in range(25)]
dataset_z.columns = [i for i in range(25)]
z = z.append(dataset_z)
bf_x.index = label.index
dataset_label['source'] = 'TCGA'
label = pd.concat([label, dataset_label[['source']]])
data = pd.concat([data,dataset], join='inner')
# data = data.fillna(0)
pca_visualization(data, label['source'], 'breast_x')
pca_visualization(z, label['source'], 'breast_z')
# pca_visualization(log2_bf_x, label, 'breast_bf_x')
