import tensorflow as tf
import tensorflow_addons as tfa
import numpy as np


class BF_Generator(tf.keras.Model):

    def __init__(self, input_shape, z_dim):
        super(BF_Generator, self).__init__()
        self.z_dim = z_dim
        self.encoder = tf.keras.Sequential(
            [
                tf.keras.layers.InputLayer(input_shape=(input_shape[1], input_shape[2], 1)),
                tf.keras.layers.Conv2D(filters=64, kernel_size=5, strides=2, padding='same'),
                tf.keras.layers.LeakyReLU(),
                tf.keras.layers.Conv2D(filters=64, kernel_size=5, strides=2, padding='same'),
                tfa.layers.GroupNormalization(),
                tf.keras.layers.LeakyReLU(),
                tf.keras.layers.Conv2D(filters=128, kernel_size=3, strides=2, padding='same'),
                tfa.layers.GroupNormalization(),
                tf.keras.layers.LeakyReLU(),
                tf.keras.layers.Flatten(),
                tf.keras.layers.Dense(1000),
                tf.keras.layers.LeakyReLU(),
                # No activation
                tf.keras.layers.Dense((self.z_dim + self.z_dim)),
                # tf.keras.layers.Dense(self.z_dim),
            ]
        )

        self.decoder = tf.keras.Sequential(
            [
            tf.keras.layers.InputLayer(input_shape=(self.z_dim,)),
            tf.keras.layers.Dense(units=16*32*64),
            tf.keras.layers.LeakyReLU(),
            tf.keras.layers.Reshape(target_shape=(16, 32, 64)),
            tf.keras.layers.Conv2DTranspose(filters=128, kernel_size=5, strides=2, padding='same'),
            tfa.layers.GroupNormalization(),
            tf.keras.layers.LeakyReLU(),
            tf.keras.layers.Conv2DTranspose(filters=64, kernel_size=5, strides=2, padding='same'),
            tfa.layers.GroupNormalization(),
            tf.keras.layers.LeakyReLU(),
            tf.keras.layers.Conv2DTranspose(filters=64, kernel_size=5, strides=2, padding='same'),
            tfa.layers.GroupNormalization(),
            tf.keras.layers.LeakyReLU(),
            tf.keras.layers.Conv2DTranspose(filters=1, kernel_size=5, strides=1, padding='same'),
            ]
        )

    @tf.function
    def sample(self, eps=None):
        if eps is None:
            eps = tf.random.normal(shape=(100, self.z_dim))
        return self.decode(eps, apply_sigmoid=True)

    def encode(self, x):
        mean, logvar = tf.split(self.encoder(x), num_or_size_splits=2, axis=1)
        return mean, logvar

    def reparameterize(self, mean, logvar):
        eps = tf.random.normal(shape=mean.shape)
        return eps * tf.exp(logvar * .5) + mean

    def decode(self, z, apply_sigmoid=True):
        logits = self.decoder(z)
        if apply_sigmoid:
            probs = tf.sigmoid(logits)
            return probs
        return logits

    def call(self, x, apply_sigmoid=True):
        mean, logvar = tf.split(self.encoder(x), num_or_size_splits=2, axis=1)
        z = self.reparameterize(mean, logvar)
        logits = self.decoder(z)
        if apply_sigmoid:
            probs = tf.sigmoid(logits)
            return probs
        return logits



class Batch_Encoder(tf.keras.Model):
    
    def __init__(self, input_shape, z_dim):
        super(Batch_Encoder, self).__init__()
        self.z_dim = z_dim
        self.encoder = tf.keras.Sequential(
            [
                tf.keras.layers.InputLayer(input_shape=(input_shape[1], input_shape[2], 1)),
                tf.keras.layers.Conv2D(filters=64, kernel_size=5, strides=2),
                tf.keras.layers.LeakyReLU(),
                tf.keras.layers.Conv2D(filters=64, kernel_size=5, strides=2),
                tfa.layers.GroupNormalization(),
                tf.keras.layers.LeakyReLU(),
                tf.keras.layers.Conv2D(filters=128, kernel_size=5, strides=2),
                tfa.layers.GroupNormalization(),
                tf.keras.layers.LeakyReLU(),
                tf.keras.layers.Flatten(),
                tf.keras.layers.Dense(1000),
                tf.keras.layers.LeakyReLU(),
                # No activation
                # tf.keras.layers.Dense(z_dim)
                tf.keras.layers.Dense(z_dim + z_dim)
            ]
        )
    
    @tf.function
    def encode(self, x):
        mean, logvar = tf.split(self.encoder(x), num_or_size_splits=2, axis=1)
        return mean, logvar

    def reparameterize(self, mean, logvar):
        eps = tf.random.normal(shape=mean.shape)
        return eps * tf.exp(logvar * .5) + mean

    def call(self, x):
        mean, logvar = tf.split(self.encoder(x), num_or_size_splits=2, axis=1)
        z = self.reparameterize(mean, logvar)
        # z = self.encoder(x)
        return z



class GroupNormalization(tf.keras.layers.Layer):

    def __init__(self, N=16, group=2, epsilon=1e-5):
        super(GroupNormalization, self).__init__()
        self.group = group
        self.epsilon = epsilon
        self.N = N

    def build(self, input_shape):
        self.gamma = self.add_weight(
            name='gamma',
            shape=input_shape[-1:],
            initializer='ones',
            trainable=True)

        self.beta = self.add_weight(
            name='beta',
            shape=input_shape[-1:],
            initializer='zeros',
            trainable=True)
    
    def call(self, x):
        _, C, H, W = x.shape
        tf.print(x.shape)
        x = tf.reshape(x, [self.N, self.group, C // self.group, H, W])
        mean, var = tf.nn.moments(x, [2, 3, 4], keepdims=True) 
        x = (x - mean) / tf.sqrt(var + self.epsilon)
        x = tf.reshape(x, [self.N, C, H, W]) 
        return self.gamma * x + self.beta



def downsample(filters, kernel_size, norm_type='batchnorm', apply_norm=True):
    result = tf.keras.Sequential()
    result.add(
        tf.keras.layers.Conv2D(filters, kernel_size, strides=2, padding='same')
        )

    if apply_norm:
        if norm_type.lower() == 'batchnorm':
            result.add(tf.keras.layers.BatchNormalization())
        elif norm_type.lower() == 'instancenorm':
            result.add(tfa.layers.InstanceNormalization())
        elif norm_type.lower() == 'groupnorm':    
            result.add(tfa.layers.GroupNormalization())

    result.add(tf.keras.layers.LeakyReLU())

    return result


def upsample(filters, kernel_size, norm_type='batchnorm', apply_dropout=False):
    result = tf.keras.Sequential()
    result.add(
        tf.keras.layers.Conv2DTranspose(filters, kernel_size, strides=2, padding='same')
        )

    if norm_type.lower() == 'batchnorm':
        result.add(tf.keras.layers.BatchNormalization())
    elif norm_type.lower() == 'instancenorm':
        result.add(tfa.layers.InstanceNormalization())
    elif norm_type.lower() == 'groupnorm':    
        result.add(tfa.layers.GroupNormalization())

    if apply_dropout:
        result.add(tf.keras.layers.Dropout(0.5))

    result.add(tf.keras.layers.LeakyReLU())

    return result

    

# class Unet_Generator(tf.keras.Model):
#     def __init__(self, z_dim, norm_type):
#         super(Unet_Generator, self).__init__()
#         self.z_dim = z_dim
#         self.norm_type = norm_type

#     def build(self, input_shape):
#         self.down_stack = [
#             downsample(64, 5, self.norm_type, apply_norm=True),  # (bs, 64, 128, 64)
#             downsample(64, 5, self.norm_type),  # (bs, 32, 64, 64)
#             downsample(128, 5, self.norm_type),  # (bs, 16, 32, 128)
#             downsample(256, 5, self.norm_type),  # (bs, 8, 16, 256)
#             # downsample(512, 5, self.norm_type),  # (bs, 8, 8, 512)
#             # downsample(512, 5, self.norm_type),  # (bs, 4, 4, 512)
#             # downsample(512, 5, self.norm_type),  # (bs, 2, 2, 512)
#             # downsample(512, 5, self.norm_type),  # (bs, 1, 1, 512)
#         ]
#         self.up_stack = [
#             # upsample(512, 5, self.norm_type, apply_dropout=True),  # (bs, 2, 2, 1024)
#             # upsample(512, 5, self.norm_type, apply_dropout=True),  # (bs, 4, 4, 1024)
#             # upsample(512, 5, self.norm_type, apply_dropout=True),  # (bs, 8, 8, 1024)
#             # upsample(512, 5, self.norm_type),  # (bs, 16, 16, 1024)
#             upsample(128, 5, self.norm_type, apply_dropout=True),  # (bs, 16, 32, 256)
#             upsample(64, 5, self.norm_type, apply_dropout=True),  # (bs, 32, 64, 128)
#             upsample(64, 5, self.norm_type),  # (bs, 64, 128, 128)
#         ]

#         self.inputs = tf.keras.layers.InputLayer(input_shape=(input_shape[1], input_shape[2], 1))
#         self.flatten = tf.keras.Sequential(
#             [
#                 tf.keras.layers.Flatten(),
#                 tf.keras.layers.Dense(1000),
#                 tf.keras.layers.LeakyReLU(),
#                 tf.keras.layers.Dense(self.z_dim),
#                 tf.keras.layers.LeakyReLU()
#             ]
#         )
#         self.reshape = tf.keras.Sequential(
#             [
#                 tf.keras.layers.Dense(1000),
#                 tf.keras.layers.LeakyReLU(),
#                 tf.keras.layers.Dense(units=8*16*256),
#                 tf.keras.layers.LeakyReLU(),
#                 tf.keras.layers.Reshape(target_shape=(8, 16, 256))
#             ]
#         )
#         self.concat = tf.keras.layers.Concatenate()
#         self.last = tf.keras.layers.Conv2DTranspose(1, 4, strides=2, padding='same')


#     def call(self, x, z, apply_sigmoid=False):
#         x = self.inputs(x)
#         skips = []
#         for down in self.down_stack:
#             x = down(x)
#             skips.append(x)
#         skips = reversed(skips[:-1])

#         x = self.flatten(x)
#         x = self.concat([x,z])
#         x = self.reshape(x)

#         for up, skip in zip(self.up_stack, skips):
#             x = up(x)
#             x = self.concat([x, skip])

#         x = self.last(x)

#         if apply_sigmoid:
#             x = tf.sigmoid(x)
#         return x



class Generator(tf.keras.Model):

    def __init__(self, input_shape, z_dim):
        super(Generator, self).__init__()
        self.z_dim = z_dim
        self.encoder = tf.keras.Sequential(
            [
                tf.keras.layers.InputLayer(input_shape=(input_shape[1], input_shape[2], 1)),
                tf.keras.layers.Conv2D(filters=64, kernel_size=5, strides=2, padding='same'),
                tf.keras.layers.LeakyReLU(),
                tf.keras.layers.Conv2D(filters=64, kernel_size=5, strides=2, padding='same'),
                tfa.layers.GroupNormalization(),
                tf.keras.layers.LeakyReLU(),
                tf.keras.layers.Conv2D(filters=128, kernel_size=3, strides=2, padding='same'),
                tfa.layers.GroupNormalization(),
                tf.keras.layers.LeakyReLU(),
                tf.keras.layers.Flatten(),
                tf.keras.layers.Dense(1000),
                tf.keras.layers.LeakyReLU(),
                tf.keras.layers.Dense(self.z_dim),
            ]
        )

        self.decoder = tf.keras.Sequential(
            [
            tf.keras.layers.InputLayer(input_shape=(self.z_dim * 2,)),
            tf.keras.layers.Dense(units=16*32*64),
            tf.keras.layers.LeakyReLU(),
            tf.keras.layers.Reshape(target_shape=(16, 32, 64)),
            tf.keras.layers.Conv2DTranspose(filters=128, kernel_size=5, strides=2, padding='same'),
            tfa.layers.GroupNormalization(),
            tf.keras.layers.LeakyReLU(),
            tf.keras.layers.Conv2DTranspose(filters=64, kernel_size=5, strides=2, padding='same'),
            tfa.layers.GroupNormalization(),
            tf.keras.layers.LeakyReLU(),
            tf.keras.layers.Conv2DTranspose(filters=64, kernel_size=5, strides=2, padding='same'),
            tfa.layers.GroupNormalization(),
            tf.keras.layers.LeakyReLU(),
            tf.keras.layers.Conv2DTranspose(filters=1, kernel_size=5, strides=1, padding='same'),
            ]
        )
        self.concat = tf.keras.layers.Concatenate()

    @tf.function
    def call(self, x, z, apply_sigmoid=False):
        x = self.encoder(x)
        x = self.concat([x,z])
        x = self.decoder(x)
        if apply_sigmoid:
            x = tf.sigmoid(x)
            return x
        return x



def make_batch_Descriminator_z(output_dim, activation='softmax'):
    model = tf.keras.Sequential()
    model.add(tf.keras.layers.LeakyReLU())
    model.add(tf.keras.layers.Dropout(0.3))
    # model.add(tf.keras.layers.Dense(100))
    # model.add(tf.keras.layers.LeakyReLU())
    # model.add(tf.keras.layers.Dropout(0.3))
    model.add(tf.keras.layers.Dense(output_dim, activation=activation))
    return model


def make_category_Descriminator_z(output_dim, activation='sigmoid'):
    model = tf.keras.Sequential()
    model.add(tf.keras.layers.LeakyReLU())
    model.add(tf.keras.layers.Dropout(0.3))
    model.add(tf.keras.layers.Dense(500))
    model.add(tf.keras.layers.LeakyReLU())
    model.add(tf.keras.layers.Dropout(0.3))
    model.add(tf.keras.layers.Dense(output_dim, activation=activation))
    return model



def make_Descriminator_x(input_shape, output_dim, activation='sigmoid'):
    model = tf.keras.Sequential()
    model.add(tf.keras.layers.Conv2D(32, 5, strides=2, padding='same', input_shape=[input_shape[1], input_shape[2], 1]))
    model.add(tfa.layers.GroupNormalization())
    model.add(tf.keras.layers.LeakyReLU())
    model.add(tf.keras.layers.Dropout(0.5))

    model.add(tf.keras.layers.Conv2D(64, 5, strides=2, padding='same'))
    model.add(tfa.layers.GroupNormalization())
    model.add(tf.keras.layers.LeakyReLU())
    model.add(tf.keras.layers.Dropout(0.5))

    model.add(tf.keras.layers.Conv2D(128, 5, strides=2, padding='same'))
    model.add(tfa.layers.GroupNormalization())
    model.add(tf.keras.layers.LeakyReLU())
    model.add(tf.keras.layers.Dropout(0.5))

    model.add(tf.keras.layers.Flatten())
    model.add(tf.keras.layers.Dense(1000))
    model.add(tf.keras.layers.LeakyReLU(name='feature_matching_layer'))

    if activation=='sigmoid':
        model.add(tf.keras.layers.Dense(output_dim, activation='sigmoid'))
    elif activation=='softmax':
        model.add(tf.keras.layers.Dense(output_dim, activation='softmax'))
    return model



'''cross entropy loss
def focal loss and cross entropy loss 
'''
def cross_entropy(labels, probs):
    cross_loss = tf.add(tf.math.log(1e-10 + probs) * labels, tf.math.log(1e-10 + (1 - probs)) * (1 - labels))
    loss = tf.negative(tf.reduce_mean(tf.reduce_sum(cross_loss, axis=1)))
    return loss


def focal_loss(labels, probs, alpha=0.25, gamma=2):
    cross_loss = tf.add(alpha * ((1- probs) ** gamma) * tf.math.log(1e-10 + probs) * labels, (1-alpha) * (probs ** gamma)* tf.math.log(1e-10 + (1 - probs)) * (1 - labels))
    loss = tf.negative(tf.reduce_mean(tf.reduce_sum(cross_loss, axis=1)))
    return loss


def missing_label_focal_loss(labels, probs, alpha=0.25, gamma=2):
    loss = 0.0
    i, cond = 0, tf.constant(1)
    probs = tf.cast(probs, tf.float32)
    while cond == 1:
        cond = tf.cond(i >= tf.shape(labels)[0] - 1, lambda: tf.constant(0), lambda: tf.constant(1))
        probs_, Y_ = tf.slice(probs, [i, 0], [1, tf.shape(labels)[1]]), tf.slice(labels, [i, 0],[1, tf.shape(labels)[1]])
        one = tf.constant(1, dtype=tf.float32)
        probs_, Y_ = tf.expand_dims(tf.gather_nd(probs_, tf.where(Y_ <= one)), 0), tf.expand_dims(tf.gather_nd(Y_, tf.where(Y_ <= one)), 0)
        cross_loss = tf.add(alpha * ((1- probs_) ** gamma) * tf.math.log(1e-10 + probs_) * Y_, (1-alpha) * (probs_ ** gamma)* tf.math.log(1e-10 + (1 - probs_)) * (1 - Y_))
        loss += tf.negative(tf.reduce_mean(tf.reduce_sum(cross_loss, axis=1)))
        i += 1
    return 0.1 * loss


def set_training_target_score(label, score=0.9):    
    label = tf.cast(label, tf.float32)
    target_score = tf.ones_like(label) * score
    label = tf.where(label==1, x=target_score, y=label)
    return label



def log_normal_pdf(sample, mean, logvar, raxis=1):
    log2pi = tf.math.log(2. * np.pi)
    return tf.reduce_mean(
        -.5 * ((sample - mean) ** 2. * tf.exp(-logvar) + logvar + log2pi),
        axis=raxis)


'''loss function'''
def conditional_VAE_loss(x, zc_mean, zc_logvar, zc, zb_mean, zb_logvar, zb, x_logit):
    logpx_bc = -tf.reduce_mean(tf.square(x - tf.math.sigmoid(x_logit)))
    logpc = log_normal_pdf(zc, 0., 0.)
    logqc_x = log_normal_pdf(zc, zc_mean, zc_logvar)
    logpb = log_normal_pdf(zb, 0., 0.)
    logqb_x = log_normal_pdf(zb, zb_mean, zb_logvar)
    return -tf.reduce_mean(logpx_bc + logpc - logqc_x + logpb - logqb_x)



def L2_loss(x, bf_x):
    loss = tf.reduce_mean(tf.square(x - bf_x))
    return loss



def L1_loss(x, bf_x):
    loss = tf.reduce_mean(tf.abs(x - bf_x))
    return loss



def feature_matching_loss(x_feature, bf_x_feature):
    loss = tf.reduce_mean(tf.square(x_feature - bf_x_feature))
    return loss



def batch_discriminator_loss(zb_batch, zc_batch, batch):
    zb_loss = cross_entropy(batch, zb_batch)
    zc_loss = cross_entropy(batch, zc_batch)
    return tf.reduce_mean(zb_loss + 5 * zc_loss)



def category_discriminator_loss(zb_category, zc_category, category):
    zb_loss = focal_loss(category, zb_category)
    zc_loss = focal_loss(category, zc_category)
    return tf.reduce_mean(0 * zb_loss + zc_loss)



def Zb_generator_loss(zb_batch, zb_category, batch, category):
    zb_category_loss = focal_loss(tf.zeros_like(category), zb_category)
    zb_batch_loss = cross_entropy(batch, zb_batch)
    return tf.reduce_mean(zb_category_loss + zb_batch_loss)



def Zc_generator_loss(zc_batch, zc_category, batch, category):
    zc_batch_loss = cross_entropy(tf.zeros_like(batch), zc_batch)
    zc_category_loss = focal_loss(category, zc_category)
    return tf.reduce_mean(zc_batch_loss + zc_category_loss)



def bf_discriminator_loss(x_output, bf_x_output, batch):
    x_loss = cross_entropy(batch, x_output)
    bf_x_loss = cross_entropy(batch, bf_x_output)
    return tf.reduce_mean(x_loss + bf_x_loss)



def bf_generator_loss(bf_x_batch, bf_x_category, category):
    batch_loss = cross_entropy(tf.zeros_like(bf_x_batch), bf_x_batch)
    category_loss = focal_loss(category, bf_x_category)
    return tf.reduce_mean(batch_loss + category_loss)



def bf_classifier_loss(x_output, category):    
    x_loss = focal_loss(category, x_output)
    return x_loss
