import tensorflow as tf
import tensorflow.keras.backend as K
import numpy as np
import pandas as pd
from scipy.stats import chisquare
from sklearn.decomposition import PCA
from sklearn.neighbors import NearestNeighbors
from data_preprocess import trans_2d_to_1d, reverse_log2_feature



class accuracy_score(tf.keras.metrics.Metric):
    '''
        ratio of the incorrectly predicted labels over all labels
        '''

    def __init__(self, thresholds=0.5, name="cm", **kwargs):
        super(accuracy_score, self).__init__(name=name, **kwargs)
        self.thresholds = thresholds
        self.true_positives = self.add_weight(
            name="tp", shape=(1,), initializer="zeros")
        self.false_positives = self.add_weight(
            name="fp", shape=(1,), initializer="zeros")
        self.false_negatives = self.add_weight(
            name="fn", shape=(1,), initializer="zeros")
        self.true_negatives = self.add_weight(
            name="tn", shape=(1,), initializer="zeros")

    @tf.function
    def update_state(self, y_true, y_pred):
        y_true = tf.cast(tf.reshape(y_true, (-1,)), tf.float32)
        one = tf.ones_like(y_pred)
        zero = tf.zeros_like(y_pred)
        y_pred = tf.where(y_pred >= self.thresholds, x=one, y=zero)
        y_pred = tf.cast(tf.reshape(y_pred, (-1,)), tf.bool)
        for i in tf.range(tf.shape(y_true)[0]):
            if y_true[i] == 1:
                if y_pred[i]:
                    self.true_positives.assign(self.true_positives + 1.0)
                else:
                    self.false_negatives.assign(self.false_negatives + 1.0)
            if y_true[i] == 0:
                if y_pred[i]:
                    self.false_positives.assign(self.false_positives + 1.0)
                else:
                    self.true_negatives.assign(self.true_negatives + 1.0)
        return (self.true_positives, self.false_positives, self.false_negatives, self.true_negatives)

    def result(self):
        accuracy_score = tf.divide(tf.add(self.true_positives, self.true_negatives),
                                 tf.add_n([self.true_positives, self.false_positives,
                                           self.false_negatives, self.true_negatives]))
        return tf.squeeze(accuracy_score)

    def reset_states(self):
        K.batch_set_value([(v, np.zeros(1)) for v in self.variables])
        
        
        
class label_ranking_average_precision(tf.keras.metrics.Metric):
    '''
    Label ranking average precision (LRAP) is the average over each ground truth label assigned to each sample,
    evaluates the average fraction of labels ranked above ground truth label
    best value is 1
    '''
    def __init__(self, name="label_ranking_average_precision", **kwargs):
        super(label_ranking_average_precision,self).__init__(name=name, **kwargs)
        self.sum_aux = self.add_weight(
            name="sum_aux", shape=(1,), initializer="zeros")
        self.num_samples = self.add_weight(
            name="num_samples", shape=(1,), initializer="zeros")

    @tf.function
    def update_state(self, y_true, y_pred):
        n = tf.cast(tf.shape(y_true)[0], dtype=tf.float32)
        self.num_samples.assign(self.num_samples + n)

        y_true = tf.cast(y_true, tf.float32)
        y_pred = tf.cast(y_pred, tf.float32)

        for i in tf.range(tf.shape(y_true)[0]):
            p_n_label = tf.where(y_true[i, :] >= 0)
            y_true_ = tf.gather_nd(y_true[i, :], p_n_label)
            y_pred_ = tf.gather_nd(y_pred[i, :], p_n_label)

            rank_index = tf.argsort(y_pred_, direction='DESCENDING')
            sort_y_true = tf.gather(y_true_, rank_index)
            sort_y_pred = tf.gather(y_pred_, rank_index)
            truth_index = tf.where(sort_y_true==1)
            pred_rank = tf.argsort(sort_y_pred, direction='DESCENDING') + 1
            truth_rank = tf.gather(pred_rank, truth_index)
            L = tf.argsort(truth_rank, axis=0) + 1
            aux = tf.cond(tf.shape(L)[0] != 0, lambda :tf.cast(tf.reduce_mean(L / truth_rank), tf.float32)
                          , lambda :tf.constant(1.))    #true label rank in all true labels/ true label rank in all labels
            self.sum_aux.assign(self.sum_aux + aux)

        return self.num_samples, self.sum_aux

    def result(self):
        out = tf.divide(self.sum_aux, self.num_samples)
        return tf.squeeze(out)

    def reset_states(self):
        K.batch_set_value([(v, np.zeros(1)) for v in self.variables])



class micro_average(tf.keras.metrics.Metric):

    def __init__(self, thresholds=0.5, name="micro_average", **kwargs):
        super(micro_average, self).__init__(name=name, **kwargs)
        self.thresholds = thresholds
        self.true_positives = self.add_weight(
            name="tp", shape=(1,), initializer="zeros")
        self.false_positives = self.add_weight(
            name="fp", shape=(1,), initializer="zeros")
        self.false_negatives = self.add_weight(
            name="fn", shape=(1,), initializer="zeros")
        self.true_negatives = self.add_weight(
            name="tn", shape=(1,), initializer="zeros")

    @tf.function
    def update_state(self, y_true, y_pred):
        y_true = tf.cast(tf.reshape(y_true, (-1,)), tf.float32)
        one = tf.ones_like(y_pred)
        zero = tf.zeros_like(y_pred)
        y_pred = tf.where(y_pred >= self.thresholds, x=one, y=zero)
        y_pred = tf.cast(tf.reshape(y_pred, (-1,)), tf.bool)
        for i in tf.range(tf.shape(y_true)[0]):
            if y_true[i] == 1:
                if y_pred[i]:
                    self.true_positives.assign(self.true_positives + 1.0)
                else:
                    self.false_negatives.assign(self.false_negatives + 1.0)
            if y_true[i] == 0:
                if y_pred[i]:
                    self.false_positives.assign(self.false_positives + 1.0)
                else:
                    self.true_negatives.assign(self.true_negatives + 1.0)
        return (self.true_positives, self.false_positives, self.false_negatives, self.true_negatives)

    def result(self):
        micro_precision = tf.divide(self.true_positives,
                                    tf.add_n([self.true_positives, self.false_positives, tf.constant([0.000001])]))
        micro_racall = tf.divide(self.true_positives,
                                    tf.add_n([self.true_positives, self.false_negatives, tf.constant([0.000001])]))
        micro_f1 = tf.divide((tf.constant(2.0) * micro_precision * micro_racall),
                             (micro_precision+ micro_racall+ tf.constant([0.000001])))
        micro_score = (micro_precision,micro_racall,micro_f1)
        return tf.squeeze(micro_score)

    def reset_states(self):
        K.batch_set_value([(v, np.zeros(1)) for v in self.variables])



class k_neighbours_batch_effect():
    '''
    Evaluate the batch effect of different batches of samples belonging to the same category
    pi: total proportion of samples in batch i in category c
    k: neighbour size
    fi: local fraction of samples in batch i in k neighbours subset
    '''
    def __init__(self, bf_generator, data, label):   
        y_data = pd.read_csv('/vol1/cuipeng_group/qiankun/GAN-VAE/data/y_data.tsv', index_col=0, sep='\t')
        self.label = y_data.loc[label.index]
        
        self.x = trans_2d_to_1d(data)
        x_dataset = tf.data.Dataset.from_tensor_slices(data).batch(128)
        z = None
        bf_data = None 
        for x in x_dataset: 
            mean, logvar = bf_generator.encode(x)
            z_ = bf_generator.reparameterize(mean, logvar)
            bf_x = bf_generator.decode(z_, apply_sigmoid=True)
            if z is None:
                z = z_
                bf_data = bf_x
            else:
                z = np.concatenate((z,z_),axis=0)
                bf_data = np.concatenate((bf_data,bf_x),axis=0)

        bf_data = trans_2d_to_1d(bf_data)
        log2_bf_data = reverse_log2_feature(bf_data)
        log2_bf_data.index = self.label.index
        z = pd.DataFrame(z, index=self.label.index)
        self.z = z
        self.bf_x = log2_bf_data
                
    def bisection(self, data, selected_data, batch, a, b, pi):
        c = int((a+b)/2)
        fa = self.rejection_rates(data, selected_data, batch, pi, a)
        fb = self.rejection_rates(data, selected_data, batch, pi, b)
        fc = self.rejection_rates(data, selected_data, batch, pi, c)
        max_score = fa
        while (abs(b-a)>10) & (abs(fa-fb)>0.01):
            if fc > fa:
                a = c
            else:
                b = c
            max_score = self.bisection(data, selected_data, batch, a, b, pi)
        return max_score
           
    def rejection_rates(self, data, selected_data, batch, pi, k):
        nbrs = NearestNeighbors(n_neighbors=k, algorithm='ball_tree').fit(data)
        distances, indices = nbrs.kneighbors(selected_data)
        
        rejection_counter = []
        f_df = k * pi
        for i in indices:
            fi = batch.iloc[i].value_counts().astype(float)
            f_df = pd.concat([f_df, fi], join='outer', axis=1)
            f_df = f_df.fillna(0)
            f_exp = f_df.iloc[:,0].values
            fi = f_df.iloc[:,1].values
            chisq, p_value = chisquare(fi, f_exp)
            rejection_counter.append(np.where(p_value >= 0.05, 0, 1))
        rejection_rates = np.mean(rejection_counter)
        return rejection_rates
    
    def result(self):
        z_rejection_rates = []
        bfx_rejection_rates = []
        counts = 0
        for name, group in self.label.groupby(['cancer_type', 'primary_site']):
            if counts > 1:
                break
            if (group['batch'].nunique() > 1) & (len(group) > 20):
                counts += 1
                group_z = self.z.loc[group.index]
                group_bfx = self.bf_x.loc[group.index]
                pca = PCA(n_components=10)
                pc_z = pca.fit_transform(group_z)
                pc_bfx = pca.fit_transform(group_bfx)
                # a = int(len(group) * 0.25)
                # b = int(len(group) * 0.75)
                pi = group['batch'].value_counts()/len(group)
                np.random.seed(0)
                selected_pc_z = pc_z[np.random.choice(len(pc_z), size=10, replace=False)] 
                selected_pc_bfx = pc_bfx[np.random.choice(len(pc_z), size=10, replace=False)]
                # z_score = self.bisection(pc_z, selected_pc_z, group['batch'], a, b, pi)
                # bfx_score = self.bisection(pc_bfx, selected_pc_bfx, group['batch'], a, b, pi)
                z_score = self.rejection_rates(pc_z, selected_pc_z, group['batch'], pi, int(len(group) * 0.5))
                bfx_score = self.rejection_rates(pc_bfx, selected_pc_bfx, group['batch'], pi, int(len(group) * 0.5))
                z_rejection_rates.append(z_score)
                bfx_rejection_rates.append(bfx_score)
        z_mean = np.mean(z_rejection_rates)
        bfx_mean = np.mean(bfx_rejection_rates)
        return z_mean, bfx_mean
