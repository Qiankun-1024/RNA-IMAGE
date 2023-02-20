import pandas as pd
import numpy as np
import tensorflow as tf
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import RandomOverSampler, SMOTE, ADASYN

from classifier_model import make_classifier
import sys
import os
path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(path)
 
from utils import gene2img
from model import focal_loss
from custom_metrics import micro_average



@tf.function
def printbar():
    ts = tf.timestamp()
    today_ts = ts % (24 * 60 * 60)

    hour = tf.cast(today_ts // 3600 + 8, tf.int32) % tf.constant(24)
    minite = tf.cast((today_ts % 3600) // 60, tf.int32)
    second = tf.cast(tf.floor(today_ts % 60), tf.int32)

    def timeformat(m):
        if tf.strings.length(tf.strings.format("{}", m)) == 1:
            return (tf.strings.format("0{}", m))
        else:
            return (tf.strings.format("{}", m))

    timestring = tf.strings.join([timeformat(hour), timeformat(minite),
                                  timeformat(second)], separator=":")
    tf.print("==========" * 10, end="")
    print("=========" * 10, end="")
    tf.print(timestring)
    print(timestring)

    
def load_data():
    label = pd.read_csv('./BRCA_classifier/data/tcga_label.tsv',
                        index_col=0, sep='\t')
    label = label[['coarse_cluster', 'fine_cluster']]
    data = pd.read_csv('./BRCA_classifier/data/tcga_data.tsv',
                       index_col=0, sep='\t')
    return data, label



class onehot_encoder():
    def __init__(self, columns, y):
        label = pd.read_csv('./BRCA_classifier/data/tcga_label.tsv',
                        index_col=0, sep='\t')
        label = label.loc[y.index]
        label = label[columns].to_numpy().reshape(-1,1)  
        self.enc = OneHotEncoder()
        self.enc.fit(label)
    
    def transform(self, y):
        encoded = self.enc.transform(y).toarray()
        return encoded.astype('float32')
    
    def inverse_transform(self, encoded):
        category = self.enc.inverse_transform(encoded)
        return category
        

def over_sampling(x, y, method='random'):
    if method == 'random':
        ros = RandomOverSampler(random_state=42)
    if method == 'SMOTE':
        ros = SMOTE(random_state=42, k_neighbors=4)
    if method == 'ADASYN':
        fine_cluster = np.unique(y['fine_cluster'])
        sample_dict = dict(zip(fine_cluster, len(fine_cluster) * [100]))
        ros = ADASYN(random_state=42,
                    sampling_strategy=sample_dict,
                    n_neighbors=4)
    x_res, y_res = ros.fit_resample(x, y)
    return x_res, y_res
        

    
def train_multi_task_classifier(data, label, batch_size, lr, EPOCHS):
    for col in label.columns:
        y = label[[col]]
        x_res, y_res = over_sampling(data, y, method='ADASYN')
        enc = onehot_encoder(col, y)
        y= enc.transform(y_res)
        x = gene2img(x_res)
        train(x, y, batch_size, lr, EPOCHS, col)


        
def train(x, y, batch_size, lr, EPOCHS, label_type):
    x_train, x_test, y_train, y_test = train_test_split(
        x, y, test_size = 0.1, random_state=0
    )
    train_dataset = tf.data.Dataset.from_tensor_slices(
        (x_train,y_train)).batch(batch_size).shuffle(1000)
    test_dataset = tf.data.Dataset.from_tensor_slices(
        (x_test,y_test)).batch(batch_size).shuffle(1000)
    classifier = make_classifier(y.shape[-1])
    
    lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(
        lr,
        decay_steps=100,
        decay_rate=0.96,
        staircase=True)
    optimizer = tf.keras.optimizers.Adam(learning_rate=lr_schedule )
    
    train_loss = tf.keras.metrics.Mean(name='train_loss')
    train_accuracy = tf.keras.metrics.CategoricalAccuracy(name='train_accuracy')
    train_micro_average = micro_average(name='train_micro_average')
    test_loss = tf.keras.metrics.Mean(name='test_loss')
    test_accuracy = tf.keras.metrics.CategoricalAccuracy(name='test_accuracy')
    test_micro_average = micro_average(name='test_micro_average')
     
    checkpoint_dir = os.path.join(
        os.path.dirname(os.path.abspath(__file__)),
        'training_checkpoints', label_type)
    ckpt = tf.train.Checkpoint(classifier=classifier)
    ckpt_manager = tf.train.CheckpointManager(
        ckpt, checkpoint_dir, max_to_keep=50)
    if not os.path.exists(checkpoint_dir):
        os.makedirs(checkpoint_dir)
    
    patience = 0 
    min_test_loss = 1e8
    for epoch in range(EPOCHS):
        train_loss.reset_states()
        train_accuracy.reset_states()
        train_micro_average.reset_states()
        test_loss.reset_states()
        test_accuracy.reset_states()
        test_micro_average.reset_states()        
        for train_x, train_y in train_dataset:
            train_step(classifier,
                       optimizer,
                       train_x, train_y,
                       train_loss,
                       train_accuracy,
                       train_micro_average)
        for test_x, test_y in test_dataset:
            test_step(classifier,
                      test_x, test_y,
                      test_loss,
                      test_accuracy,
                      test_micro_average)    
        
        printbar()
        print('{} classifier, EPOCH {}'
              .format(label_type, epoch+1))    
        print('Train loss: {},Test loss: {}'
              .format(train_loss.result(), test_loss.result()))
        print('Train accuracy: {},Test accuracy: {}'
              .format(train_accuracy.result(), test_accuracy.result()))
        print('Train precision: {},Test precision: {}'
              .format(train_micro_average.result()[0], test_micro_average.result()[0]))
        print('Train recall: {},Test recall: {}'
              .format(train_micro_average.result()[1], test_micro_average.result()[1]))
        print('Train F1 score: {},Test F1 score: {}'
              .format(train_micro_average.result()[2], test_micro_average.result()[2]))
        
        #early stop
        new_test_loss = test_loss.result()
        if new_test_loss < min_test_loss:
            min_test_loss = new_test_loss
            patience = 0
            ckpt_save_path = ckpt_manager.save()
            print('Saving checkpoint for epoch {} at {}'.format(epoch + 1,
                                                                ckpt_save_path))
        else:
            patience += 1

        if patience == 5:
            print('Test loss has not decreased during the last 5 epochs, early stopping')
            break


from sklearn.model_selection import KFold
def kfold_train(x, y, batch_size, lr, EPOCHS, label_type):
    x, x_val, y, y_val = train_test_split(
        x, y, test_size = 0.1, random_state=0
    )
    val_dataset = tf.data.Dataset.from_tensor_slices(
                (x_val, y_val)).batch(batch_size).shuffle(1000)
    kfold = KFold(n_splits=10, shuffle=True)

    classifier = make_classifier(y.shape[-1])
    lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(
        lr,
        decay_steps=100,
        decay_rate=0.96,
        staircase=True)
    optimizer = tf.keras.optimizers.Adam(learning_rate=lr_schedule)
    
    train_loss = tf.keras.metrics.Mean(name='train_loss')
    train_accuracy = tf.keras.metrics.CategoricalAccuracy(name='train_accuracy')
    train_micro_average = micro_average(name='train_micro_average')
    test_loss = tf.keras.metrics.Mean(name='test_loss')
    test_accuracy = tf.keras.metrics.CategoricalAccuracy(name='test_accuracy')
    test_micro_average = micro_average(name='test_micro_average')
     
    checkpoint_dir = os.path.join(
        os.path.dirname(os.path.abspath(__file__)),
        'training_checkpoints', label_type)
    ckpt = tf.train.Checkpoint(classifier=classifier)
    ckpt_manager = tf.train.CheckpointManager(
        ckpt, checkpoint_dir, max_to_keep=50)
    if not os.path.exists(checkpoint_dir):
        os.makedirs(checkpoint_dir)
    
    patience = 0 
    min_test_loss = 1e8
    for epoch in range(round(EPOCHS/10)):
        for train, test in kfold.split(x, y):
            train_dataset = tf.data.Dataset.from_tensor_slices(
                (x[train], y[train])).batch(batch_size).shuffle(1000)
            test_dataset = tf.data.Dataset.from_tensor_slices(
                (x[test], y[test])).batch(batch_size).shuffle(1000)
            train_loss.reset_states()
            train_accuracy.reset_states()
            train_micro_average.reset_states()
            test_loss.reset_states()
            test_accuracy.reset_states()
            test_micro_average.reset_states()        
            for train_x, train_y in train_dataset:
                train_step(classifier,
                        optimizer,
                        train_x, train_y,
                        train_loss,
                        train_accuracy,
                        train_micro_average)
            for test_x, test_y in val_dataset:
                test_step(classifier,
                        test_x, test_y,
                        test_loss,
                        test_accuracy,
                        test_micro_average)    
            
            printbar()
            print('{} classifier, EPOCH {}'
                .format(label_type, epoch+1))    
            print('Train loss: {},Test loss: {}'
                .format(train_loss.result(), test_loss.result()))
            print('Train accuracy: {},Test accuracy: {}'
                .format(train_accuracy.result(), test_accuracy.result()))
            print('Train precision: {},Test precision: {}'
                .format(train_micro_average.result()[0], test_micro_average.result()[0]))
            print('Train recall: {},Test recall: {}'
                .format(train_micro_average.result()[1], test_micro_average.result()[1]))
            print('Train F1 score: {},Test F1 score: {}'
                .format(train_micro_average.result()[2], test_micro_average.result()[2]))
            
            #early stop
            new_test_loss = test_loss.result()
            if new_test_loss < min_test_loss:
                min_test_loss = new_test_loss
                patience = 0
                ckpt_save_path = ckpt_manager.save()
                print('Saving checkpoint for epoch {} at {}'.format(epoch + 1,
                                                                    ckpt_save_path))
            else:
                patience += 1

            if patience == 5:
                print('Test loss has not decreased during the last 5 epochs, early stopping')
                break

    



def train_step(model, optimizer, features, labels, train_loss, accuracy, micro_average):
    with tf.GradientTape() as tape:
        preds = model(features)    
        loss = focal_loss(labels, preds)
    gradient = tape.gradient(loss, model.trainable_variables)
    optimizer.apply_gradients(zip(gradient, model.trainable_variables))
    train_loss(loss)
    accuracy.update_state(labels, preds)
    micro_average.update_state(labels, preds)
    
    
    
    
def test_step(model, features, labels, test_loss, accuracy, micro_average):
    preds = model(features)    
    loss = focal_loss(labels, preds)
    test_loss(loss)
    accuracy.update_state(labels, preds)
    micro_average.update_state(labels, preds)

        
    
if __name__ == '__main__':
    gpus = tf.config.experimental.list_physical_devices("GPU")
    if gpus:
        gpu = gpus[0]
        tf.config.experimental.set_memory_growth(gpu, True)
        tf.config.experimental.set_visible_devices([gpu],"GPU")  
    
    # batch_size = 64
    # lr = 0.000005
    batch_size = 32
    lr = 0.00001
    EPOCHS = 200
    
    sys.stdout = open('./BRCA_classifier/train.log', mode = 'w') 
    data, label = load_data()
    label = label[['fine_cluster']]
    train_multi_task_classifier(data, label, batch_size, lr, EPOCHS)
