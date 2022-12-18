import tensorflow as tf
import os
import re
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse
import matplotlib.transforms as transforms
import pandas as pd

from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
from sklearn.preprocessing import LabelEncoder

from data_preprocess import trans_2d_to_1d



def confidence_ellipse(x, y, ax, n_std=3.0, facecolor='none', **kwargs):

    if x.size != y.size:
        raise ValueError("x and y must be the same size")

    cov = np.cov(x, y)
    pearson = cov[0, 1]/np.sqrt(cov[0, 0] * cov[1, 1])
    # Using a special case to obtain the eigenvalues of this
    # two-dimensionl dataset.
    ell_radius_x = np.sqrt(1 + pearson)
    ell_radius_y = np.sqrt(1 - pearson)
    ellipse = Ellipse((0, 0), width=ell_radius_x * 2, height=ell_radius_y * 2,
                      facecolor=facecolor, **kwargs)

    # Calculating the stdandard deviation of x from
    # the squareroot of the variance and multiplying
    # with the given number of standard deviations.
    scale_x = np.sqrt(cov[0, 0]) * n_std
    mean_x = np.mean(x)

    # calculating the stdandard deviation of y ...
    scale_y = np.sqrt(cov[1, 1]) * n_std
    mean_y = np.mean(y)

    transf = transforms.Affine2D() \
        .rotate_deg(45) \
        .scale(scale_x, scale_y) \
        .translate(mean_x, mean_y)

    ellipse.set_transform(transf + ax.transData)
    return ax.add_patch(ellipse)



def comparing_real_and_generated_data(bf_generator, batch_encoder, unet_generator, data, num_selected, epoch):
    x_selected = data[0:num_selected]
    bf_x = bf_generator(x_selected)
    zb = batch_encoder(x_selected)
    x_prob = unet_generator(bf_x, zb, apply_sigmoid=True)

    for i in range(num_selected):
    
        plt.subplot(num_selected, 3, i*3+1)
        plt.imshow(x_selected[i, :, :, 0], cmap='OrRd')
        plt.axis('off')


        plt.subplot(num_selected, 3, i*3+2)
        plt.imshow(x_prob[i, :, :, 0], cmap='OrRd')
        plt.axis('off')

        plt.subplot(num_selected, 3, i*3+3)
        plt.imshow(bf_x[i, :, :, 0], cmap='OrRd')
        plt.axis('off')
    
    plt_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'cpu1_figure', 'comparison')
    if not os.path.exists(plt_dir):
        os.mkdir(plt_dir)
    plt.savefig(os.path.join(plt_dir, 'Comparison_{:04d}.png'.format(epoch)))
    plt.close()



def plot_loss(train_gen_loss, train_disc_loss, test_gen_loss, test_disc_loss):
    plt.title('Training Metrics')

    plt.ylabel("Loss", fontsize=14)
    plt.plot(train_gen_loss)
    plt.plot(train_disc_loss)
    plt.plot(test_gen_loss)
    plt.plot(test_disc_loss)

    plt.legend(['train genarator loss', 'train discriminator loss', 'test genarator loss', 'test discriminator loss',], loc='upper left')
    
    plt_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'cpu1_figure')
    if not os.path.exists(plt_dir):
        os.mkdir(plt_dir)
    plt.savefig(os.path.join(plt_dir, 'loss.png'))



def comparing_tsne_on_test_set(bf_generator, batch_encoder, unet_generator, data, label_type, epoch):
    current_dir = os.path.dirname(os.path.abspath(__file__))
    data_path = os.path.join(current_dir, "data")
    b_test = pd.read_csv(os.path.join(data_path, "b_test.tsv"), index_col=0, sep="\t")
    label = pd.read_csv(os.path.join(data_path, "y_data.tsv"), index_col=0, sep="\t")
    label = label.loc[b_test.index]
    label = label[label_type].to_list()
    le = LabelEncoder()
    encoded_label = le.fit_transform(label)

    bf_x = bf_generator(data)
    zb = batch_encoder(data)
    x_prob = unet_generator(bf_x, zb, apply_sigmoid=True)

    bf_x = trans_2d_to_1d(bf_x)
    x_prob = trans_2d_to_1d(x_prob)

    z_embedded = TSNE(n_components=2, init='pca',
                            random_state=0, perplexity=100).fit_transform(zb)
    pca = PCA(n_components=100)
    bf_pc = pca.fit_transform(bf_x)
    bf_embedded = TSNE(n_components=2, init='pca',
                            random_state=0, perplexity=100).fit_transform(bf_pc)
    pca = PCA(n_components=100)
    x_pc = pca.fit_transform(x_prob)
    x_embedded = TSNE(n_components=2, init='pca',
                            random_state=0, perplexity=100).fit_transform(x_pc)

    fig = plt.figure(1)
    ax1 = plt.subplot(1, 3, 1)
    scatter1 = ax1.scatter(z_embedded[:,0], z_embedded[:,1], alpha=.8, lw=2, c=encoded_label, cmap=plt.cm.Spectral)
    plt.xlabel('tSNE-1')
    plt.ylabel('tSNE-2')
    ax1.set_title('Epoch {}'.format(epoch))
    ax2 = plt.subplot(1, 3, 2)
    scatter2 = ax2.scatter(bf_embedded[:,0], bf_embedded[:,1], alpha=.8, lw=2, c=encoded_label, cmap=plt.cm.Spectral)
    plt.xlabel('tSNE-1')
    plt.ylabel('tSNE-2')
    ax2.set_title('Epoch {}'.format(epoch))
    ax3 = plt.subplot(1, 3, 3)
    scatter3 = ax3.scatter(bf_pc[:,0], bf_pc[:,1], alpha=.8, lw=2, c=encoded_label, cmap=plt.cm.Spectral)
    plt.xlabel('tSNE-1')
    plt.ylabel('tSNE-2')
    ax3.set_title('Epoch {}'.format(epoch))
    handles = scatter3.legend_elements()[0]
    fig.subplots_adjust(top=0.6)
    fig.legend(handles = handles,bbox_to_anchor=(0.50, 0.01),borderaxespad=0,ncol=3,
        loc=8, labels=list(np.unique(label)))
    
    plt_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'cpu1_figure')
    if not os.path.exists(plt_dir):
        os.mkdir(plt_dir)
    plt.savefig(os.path.join(plt_dir, (epoch + '_tsne.png')))
    


def generate_and_save_images(model, epoch, test_input):
    predictions = model.sample(test_input)
    plt.figure(figsize=(4,4))

    for i in range(predictions.shape[0]):
        plt.subplot(4, 4, i+1)
        plt.imshow(predictions[i, :, :, 0], cmap='OrRd')
        plt.axis('off')

    plt_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'cpu1_figure', 'generate')
    if not os.path.exists(plt_dir):
        os.makedirs(plt_dir)
    plt.savefig(os.path.join(plt_dir, 'image_at_epoch_{:04d}.png'.format(epoch)))
    plt.close()    



def comparing_pca_between_x_and_bfx(x, bf_generator, batch_encoder, unet_generator, epoch):
    bf_x = bf_generator(x)
    zb = batch_encoder(x)
    fake_x = unet_generator(bf_x, zb, apply_sigmoid=True)
    data = tf.concat([x, bf_x, fake_x],axis=0)
    data = trans_2d_to_1d(data)

    n = x.shape[0]
    label = ['real data'] * n + ['batch free data'] * n + ['fake data'] * n
    le = LabelEncoder()
    encoded_label = le.fit_transform(label)

    pca = PCA(n_components=2)
    pc = pca.fit_transform(data)
    explained_variance_ratio_ = pca.explained_variance_ratio_

    fig, ax = plt.subplots(figsize=(6, 6))

    confidence_ellipse(pc[0:n,0], pc[0:n,1], ax,
                    alpha=0.5, edgecolor='salmon', facecolor='mistyrose', zorder=0)
    ax.scatter(pc[0:n,0], pc[0:n,1], c='salmon')
    confidence_ellipse(pc[n:2*n:,0], pc[n:2*n,1], ax,
                    alpha=0.5, edgecolor='c', facecolor='lightcyan', zorder=0)
    ax.scatter(pc[n:2*n,0], pc[n:2*n,1], c='c')
    confidence_ellipse(pc[2*n:3*n:,0], pc[2*n:3*n,1], ax,
                    alpha=0.5, edgecolor='orange', facecolor='navajowhite', zorder=0)
    ax.scatter(pc[2*n:3*n,0], pc[2*n:3*n,1], c='orange')
    plt.legend(['real data','batch free data', 'fake data'])
    plt.xlabel('PC1({:.2%} explained variance)'.format(explained_variance_ratio_[0]))
    plt.ylabel('PC2({:.2%} explained variance)'.format(explained_variance_ratio_[1]))
    plt.title('Epoch {}'.format(epoch))

    plt_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'cpu1_figure', 'pca')
    if not os.path.exists(plt_dir):
        os.mkdir(plt_dir)
    plt.savefig(os.path.join(plt_dir, 'image_at_epoch_{:04d}.png'.format(epoch)))
    plt.close()  



def plt_train_log(log):
    log = open(log, 'r')
    txt = log.read()
    zc_batch = re.findall(r'Test zc batch acc: (\d.\d+)', txt)
    zc_category = re.findall(r'Test zc category acc: (\d.\d+)', txt)
    zb_batch = re.findall(r'Test zb batch acc: (\d.\d+)', txt)
    zb_category = re.findall(r'Test zb category acc: (\d.\d+)', txt)
    bf_x_class = re.findall(r'Test bf_x class acc：(\d.\d+)', txt)
    bf_x_batch = re.findall(r'Test bf_x batch acc：(\d.\d+)', txt)
    x_class = re.findall(r'Test x class acc：(\d.\d+)', txt)
    x_batch = re.findall(r'Test x batch acc：(\d.\d+)', txt)
    Epoch = [i+1 for i in range(len(zc_batch))]
    Epoch = Epoch * 8
    Accuracy = zc_batch + zc_category + zb_batch + zb_category + bf_x_class + bf_x_batch + x_class + x_batch
    layer = ['Zc', 'Zb', 'bf_x', 'x']
    layer = [[i]*138 for i in layer]
    layer = [c for r in layer for c in r]
    pred_type = ['batch', 'category', 'batch', 'category', 'class',
                 'batch', 'class', 'batch']
    pred_type = [[i]*69 for i in pred_type]
    pred_type = [c for r in pred_type for c in r]
    data = pd.DataFrame({'Accuracy':Accuracy, 'Epoch':Epoch,
                         'Layer':layer, 'Discriminator':pred_type})
    data['Accuracy'] = data['Accuracy'].map(lambda x: float(x))

    plt_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'cpu1_figure')
    if not os.path.exists(plt_dir):
        os.mkdir(plt_dir)
    sns.lineplot(x="Epoch", y="Accuracy", hue="Layer", style="Discriminator",
                 data=data.iloc[0:200,:])
    plt.savefig(os.path.join(plt_dir, 'z_acc_line.png'))
    plt.close()  
    sns.lineplot(x="Epoch", y="Accuracy", hue="Layer", style="Discriminator",
                 data=data.iloc[200:400,:])
    plt.savefig(os.path.join(plt_dir, 'x_acc_line.png'))
    plt.close()  



if __name__ == "__main__":
    log = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'train.log')
    plt_train_log(log)