import os
import pandas as pd
import numpy as np
import tensorflow as tf
from sklearn.preprocessing import OneHotEncoder

import sys
import os
path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(path)
from classifier_model import make_classifier
from utils import gene2img


def load_model(pred_target):
    tcga_label = pd.read_csv('./classifier/Breast/data/tcga_label.tsv',
                        index_col=0, sep='\t')
    outsize = len(np.unique(tcga_label[[pred_target]]))
    checkpoint_dir = os.path.join(
        os.path.dirname(os.path.abspath(__file__)),
        'training_checkpoints')
    classifier = make_classifier(outsize)
    checkpoint = tf.train.Checkpoint(classifier=classifier)
    latest = tf.train.latest_checkpoint(os.path.join(checkpoint_dir, 'fine_cluster'))
    checkpoint.restore(latest)
    return classifier


def load_data():
    label = pd.read_csv('./BRCA_classifier/data/other_label.tsv',
                        index_col=0, sep='\t')
    data = pd.read_csv('./BRCA_classifier/data/other_data.tsv',
                       index_col=0, sep='\t')
    data = data.loc[label.index]
    return data, label



class onehot_encoder():
    def __init__(self, columns):
        label = pd.read_csv('./BRCA_classifier/data/tcga_label.tsv',
                        index_col=0, sep='\t')
        label = label[columns].to_numpy().reshape(-1,1)  
        self.enc = OneHotEncoder()
        self.enc.fit(label)
    
    def transform(self, y):
        encoded = self.enc.transform(y).toarray()
        return encoded.astype('float32')
    
    def inverse_transform(self, encoded):
        category = self.enc.inverse_transform(encoded)
        return category
    
    
data, label = load_data()
data = gene2img(data)
col = 'fine_cluster'
enc = onehot_encoder(col)

classifier = load_model(col)
batch_size = 64
dataset = tf.data.Dataset.from_tensor_slices(data).batch(batch_size)
y_pred = None
for x in dataset: 
    y = classifier(x)
    if y_pred is None:
        y_pred = y
    else:
        y_pred = np.concatenate((y_pred,y),axis=0)
y_pred = enc.inverse_transform(y_pred)
y_pred = pd.DataFrame(y_pred)
y_pred.index = label.index
y_pred.columns = ['fine_cluster']
y_pred.to_csv('./BRCA_classifier/data/other_data_class.tsv',
              sep='\t')
