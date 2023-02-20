import tensorflow as tf

import sys
from os.path import dirname, abspath, join
path = dirname(dirname(abspath(__file__)))
sys.path.append(path)
from utils import load_model



def get_encoder_layer(trainable_layer='all'):
    bf_generator = load_model()
    encoder = bf_generator.get_layer(index=0)
    encoder = tf.keras.models.Sequential(encoder.layers[:-7])
    
    if trainable_layer == 'all':
        encoder.trainable = True
    elif trainable_layer == 'none':
        encoder.trainable = False
    elif trainable_layer < len(encoder.layers):
        encoder.trainable = True
        for layer in encoder.layers[:(-1 * trainable_layer)]:
            layer.trainable =  False
    return encoder
    


class Classifier(tf.keras.Model):
    
    def __init__(self, encoder_layer, output_dim):
        super(Classifier, self).__init__()
        self.encoder = encoder_layer
        self.classify = tf.keras.Sequential(
            [   
                tf.keras.layers.Flatten(),
                tf.keras.layers.Dense(100),
                tf.keras.layers.LeakyReLU(),
                tf.keras.layers.Dropout(0.5),
                tf.keras.layers.Dense(output_dim, activation='softmax')
            ]
        )
        
    @tf.function
    def reparameterize(self, mean, logvar):
        eps = tf.random.normal(shape=mean.shape)
        return eps * tf.exp(logvar * .5) + mean

    def call(self, x):
        x = self.encoder(x)
        probs = self.classify(x)
        return probs
    


def make_classifier(output_dim):    
    encoder_layer = get_encoder_layer(trainable_layer='none')
    classifier = Classifier(encoder_layer, output_dim)
    return classifier

