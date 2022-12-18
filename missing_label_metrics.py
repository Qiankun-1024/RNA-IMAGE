import tensorflow as tf
import tensorflow.keras.backend as K
import numpy as np
import pandas as pd
from scipy.stats import chisquare
from sklearn.decomposition import PCA
from sklearn.neighbors import NearestNeighbors



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
            if y_true[i] < 0:
                continue
            if y_true[i] == 1:
                if y_pred[i]:
                    self.true_positives.assign(self.true_positives + 1.0)
                else:
                    self.false_negatives.assign(self.false_negatives + 1.0)
            else:
                if y_pred[i]:
                    self.false_positives.assign(self.false_positives + 1.0)
                else:
                    self.true_negatives.assign(self.true_negatives + 1.0)
        return (self.true_positives, self.false_positives, self.false_negatives, self.true_negatives)
    
    @tf.function
    def result(self):
        accuracy_score = tf.divide(tf.add(self.true_positives, self.true_negatives),
                                 tf.add_n([self.true_positives, self.false_positives,
                                           self.false_negatives, self.true_negatives]))
        return tf.squeeze(accuracy_score)

    def reset_states(self):
        K.batch_set_value([(v, np.zeros(1)) for v in self.variables])