import os
import math
import numpy as np
import pandas as pd
import tensorflow as tf
import matplotlib.pyplot as plt

import sys
path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(path)
from data_preprocess import trans_2d_to_1d, trans_1d_to_2d


def interpolate_datas(baseline, data, alphas):
    alphas_x = alphas[:, tf.newaxis, tf.newaxis, tf.newaxis]
    delta = data - baseline
    datas = baseline + alphas_x * delta
    return datas


def integral_approximation(gradients):
    # riemann_trapezoidal
    grads = (gradients[:-1] + gradients[1:]) / tf.constant(2.0)
    integrated_gradients = tf.math.reduce_mean(grads, axis=0)
    return integrated_gradients


def sort_attributions(attrs):
    piexl_coords = pd.read_csv(
        "./data/tsne_sorted_pixels_coords.csv",
        index_col=0)
    attrs = attrs[:,::-1].reshape(1,-1)
    attrs = pd.DataFrame(attrs).T
    attrs.index = piexl_coords.index
    attrs = pd.concat([piexl_coords, attrs], axis=1)
    attrs.drop(["x_coord", "y_coord"], axis=1, inplace=True)
    attrs.columns = ['ig']
    attrs = attrs.sort_values(
        by='ig', axis=0, ascending=False)
    return attrs


  
class ImportantGene():
    def __init__(self, attributions):
        self.attributions = attributions
        
    def filter(self,
               thre_pct=None,
               thre_num=None,
               directions='both'):  
        
        if (thre_pct is not None) & (thre_num is not None):
            raise ValueError(
                f'''Received two parameters :thre_pct and thre_num parameters,
                , two parameters should not exist at the same time.
                '''
                )
            
        if directions not in ['positive', 'negative', 'both']:
            raise ValueError(
                f""" Allowed polarity values: 'positive', 'negative' or 'both'
                                        but provided {directions}"""
            )
        
        if thre_pct:
            threshold_attrs = self.get_pct_thresholds(pct=thre_pct)
        if thre_num:
            threshold_attrs = self.get_num_thresholds(
                num=thre_num, directions=directions
                )
        
        attrs_df = self.cut_off_important_gene(
            threshold_attrs, directions
            ) 
        return attrs_df     


    def get_pct_thresholds(self, pct):   
               
        attrs = np.array(self.attributions)
        indices = np.zeros(attrs.shape[1], dtype=int)
        np.set_printoptions(precision=4)
        # 2. Get the sum of the attributions
        total = np.sum(np.abs(attrs), axis=0)
        # 3. Sort the attributions from largest to smallest.
        sorted_attrs = np.sort(np.abs(attrs), axis=0)[::-1]
        # 4. Calculate the percentage of the total sum that each attribution
        # and the values about it contribute.
        cum_sum = 100.0 * np.cumsum(sorted_attrs, axis=0) / total
        # 5. Threshold the attributions by the percentage
        for i in range(cum_sum.shape[1]):
            if pct > cum_sum[:,i][-1]:
                indice = cum_sum.shape[0] - 1
            else:
                indice = np.where(cum_sum[:,i] >= pct)[0][0]
            indices[i] = indice    
        # 6. Select the desired attributions and return    
        threshold_attrs = sorted_attrs[indices, np.arange(0, attrs.shape[1])]
        return threshold_attrs
        
    def get_num_thresholds(self, num, directions):  
         
        if directions == 'positive':
            total_num = np.sum(np.ravel(self.attributions)>0)
            assert total_num > num
            threshold_attrs = np.sort(self.attributions, axis=0)[::-1][num-1]
            
        if directions == 'negative':
            total_num = np.sum(np.ravel(self.attributions)<0)
            assert total_num > num
            threshold_attrs = np.abs(np.sort(self.attributions, axis=0)[num-1])
        return threshold_attrs
    
    def cut_off_important_gene(self, thresholds, directions):
        
        attrs = self.attributions
        idx = attrs.index
        col = attrs.columns
        if directions == 'positive':
            attrs = attrs * (attrs >= thresholds)
        elif directions == 'negative':
            attrs = attrs * (attrs <= -thresholds)
        else:
            attrs = attrs * (np.abs(attrs) >= thresholds)
        attrs_df = pd.DataFrame(attrs,
                                index=idx,
                                columns=col)     
        return attrs_df
    

def Wasserstein_Distance(query_means, query_logvars, target_mean, target_logvar):
    target_mean = tf.cast(target_mean, tf.float32)
    target_logvar = tf.cast(target_logvar, tf.float32)
    p1 = (tf.reduce_sum(tf.subtract(query_means, target_mean) ** 2, axis=1)) ** 0.5
    p2 = tf.reduce_sum(tf.subtract(tf.exp(query_logvars) ** 0.5,
                                   tf.exp(target_logvar) ** 0.5)
                       ** 2, axis=1)
    return p1+p2


def compute_comparision_gradients(model, datas, mean, logvar):
    with tf.GradientTape() as tape:
        tape.watch(datas)
        query_means, query_log_vars = model(datas)
        distance = Wasserstein_Distance(query_means, query_log_vars, mean, logvar)
    return tape.gradient(distance, datas)



def comparision_integrated_gradients(model,
                                     X, Y,
                                     m_steps=50,
                                     batch_size=32):
    x_mean, x_logvar = model(X)
    y_mean, y_logvar = model(Y)
    # 1. Generate alphas.
    alphas = tf.linspace(start=0.0, stop=1.0, num=m_steps + 1)

    # Initialize TensorArray outside loop to collect gradients.
    # X and Y are each other's baselines to calculate the gradient
    # based on the sample distance.
    x_gradient_batches = tf.TensorArray(tf.float32, size=m_steps + 1)
    y_gradient_batches = tf.TensorArray(tf.float32, size=m_steps + 1)
    
    for alpha in tf.range(0, len(alphas), batch_size):
        from_ = alpha
        to = tf.minimum(from_ + batch_size, len(alphas))
        alpha_batch = alphas[from_:to]
        # 2. Generate interpolated inputs between data X and data Y.
        interpolated_path_input_batch = interpolate_datas(X, Y, alpha_batch)

        # 3. Compute gradients between model outputs and interpolated inputs.
        x_gradient_batch = \
            compute_comparision_gradients(model=model,
                                          datas=interpolated_path_input_batch,
                                          mean=y_mean,
                                          logvar=y_logvar)
        y_gradient_batch = \
            compute_comparision_gradients(model=model,
                                          datas=interpolated_path_input_batch,
                                          mean=x_mean,
                                          logvar=x_logvar)
        # Write batch indices and gradients to extend TensorArray.
        x_gradient_batches = x_gradient_batches.scatter(tf.range(from_, to),
                                                        x_gradient_batch)     
        y_gradient_batches = y_gradient_batches.scatter(tf.range(from_, to),
                                                        y_gradient_batch)     

    # Stack path gradients together row-wise into single tensor.a
    x_total_gradients = x_gradient_batches.stack()
    y_total_gradients = y_gradient_batches.stack()

    # 4. Integral approximation through averaging gradients.
    x_avg_gradients = integral_approximation(gradients=x_total_gradients)
    y_avg_gradients = integral_approximation(gradients=y_total_gradients)

    # 5. Scale integrated gradients with respect to input.
    x_integrated_gradients = (X - Y) * x_avg_gradients
    y_integrated_gradients = (Y - X) * y_avg_gradients
    
    return x_integrated_gradients.numpy(), y_integrated_gradients.numpy()


        
def comparision_attributions(model,
                             X,
                             Y,
                             m_steps=240,
                             batch_size=128,
                             output='1d'):
    
    if output not in ['1d', '2d']:
        raise ValueError(
            f'''Allowed polarity values: '1d', '2d' but provided {output}'''
        )
    
    X = np.expand_dims(X, axis=0)
    Y = np.expand_dims(Y, axis=0)
    x_attr, y_attr = comparision_integrated_gradients(model=model,
                                                      X=X,
                                                      Y=Y,
                                                      m_steps=m_steps,
                                                      batch_size=batch_size)
    mean_attr = np.mean((x_attr, y_attr), axis=0)
    #normalize attributions scores by max value
    mean_attr /=  np.max(np.abs(mean_attr))
    if output == '1d':
        #return data frame
        mean_attr = trans_2d_to_1d(mean_attr)
        mean_attr.index = ['Gene_attrs']
        return mean_attr.T
    if output == '2d':
        return mean_attr



def mask_gene(data, attr, to_, from_=0):
    if data.shape == (128, 256, 1):
        data = np.expand_dims(data, axis=0)
    if data.ndim != 4:
        raise ValueError(
            f'''expected data shape = (1, 128, 256, 1),
            but got {data.shape}'''
            )

    indice = np.unravel_index(np.argsort(attr.ravel())[from_: to_], attr.shape)
    mask = np.ones((attr.shape))
    mask[indice] = 0
    data = data * mask
    return data


def compute_mask_distance(model, X, Y, block_size, order="descending"):
    attr = comparision_attributions(model = model,
                                    X = X,
                                    Y = Y,
                                    m_steps=200,
                                    output='2d')
    
    if order == "descending":
        attr_order = attr[::-1]
    elif order == "ascending":
        attr_order = attr
    num = attr.ravel().shape[0]
    n_block = math.ceil(num/block_size) + 1
    mask_dist = np.zeros(shape = n_block, dtype=np.float32)

    for i in range(n_block):
        from_ = (i-1) * block_size
        from_ = max(from_, 0)
        to_ = i * block_size
        to_ = min(to_, num-1)
        x = mask_gene(X, attr_order, from_=from_, to_=to_)
        y = mask_gene(Y, attr_order, from_=from_, to_=to_)
        x_mean, x_logvar = model(x)
        y_mean, y_logvar = model(y)
        d = Wasserstein_Distance(x_mean, x_logvar, y_mean, y_logvar)
        mask_dist[i] = d
        
    return mask_dist
    

    
def plot_attributions(attributions,
                      baseline,
                      data,
                      cmap=None,
                      overlay_alpha=0.4):

    # Sum of the attributions across color channels for visualization.
    # The attribution mask shape is a grayscale image with height and width
    # equal to the original image.
    attribution_mask = tf.reduce_sum(tf.math.abs(attributions), axis=-1)

    fig, axs = plt.subplots(nrows=2, ncols=2, squeeze=False, figsize=(8, 8))

    axs[0, 0].set_title('Baseline data')
    axs[0, 0].imshow(np.squeeze(baseline, 0))
    axs[0, 0].axis('off')

    axs[0, 1].set_title('Original data')
    axs[0, 1].imshow(np.squeeze(data, 0))
    axs[0, 1].axis('off')

    axs[1, 0].set_title('Attribution mask')
    axs[1, 0].imshow(np.squeeze(attribution_mask, 0), cmap=cmap)
    axs[1, 0].axis('off')

    axs[1, 1].set_title('Overlay')
    axs[1, 1].imshow(np.squeeze(attribution_mask, 0), cmap=cmap)
    axs[1, 1].imshow(np.squeeze(data, 0), alpha=overlay_alpha)
    axs[1, 1].axis('off')

    plt.tight_layout()
    return fig


def plot_similarity_attributions(differential_gene,
                                 common_gene):
    positive_attributions = np.expand_dims(trans_1d_to_2d(differential_gene.T), axis=3)
    negetive_attributions = np.expand_dims(trans_1d_to_2d(common_gene.T), axis=3)
    cliped_attributions = positive_attributions + negetive_attributions
    positive_attributions /= np.max(np.abs(cliped_attributions))
    negetive_attributions /= np.max(np.abs(cliped_attributions))
    cliped_attributions /= np.max(np.abs(cliped_attributions))
    
    fig, axs = plt.subplots(nrows=1, ncols=3, squeeze=False, figsize=(8, 8))

    axs[0,0].set_title('cliped attributions')
    axs[0,0].imshow(np.squeeze(cliped_attributions, 0), cmap='coolwarm')
    axs[0,0].axis('off')

    axs[0,1].set_title('differential gene')
    axs[0,1].imshow(np.squeeze(positive_attributions, 0), cmap='Reds')
    axs[0,1].axis('off')

    axs[0,2].set_title('common gene')
    axs[0,2].imshow(np.squeeze(negetive_attributions, 0), cmap='Blues_r')
    axs[0,2].axis('off')

    plt.tight_layout()
    return fig


    
    
    
    