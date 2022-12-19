import time
import datetime
import tensorflow as tf
import numpy as np
import os

from data_preprocess import load_data
from Batch_Free_GAN import *
from plot_data import *
from custom_metrics import accuracy_score, micro_average, label_ranking_average_precision, k_neighbours_batch_effect

import sys 
sys.stdout = open('train.log', mode = 'w') 


'''
switch GPU
'''
gpus = tf.config.experimental.list_physical_devices("GPU")
if gpus:
    gpu = gpus[0]
    tf.config.experimental.set_memory_growth(gpu, True)
    tf.config.experimental.set_visible_devices([gpu],"GPU")
    print(gpu)

'''
hyperparameters
z_dim: laten layer size
'''
z_dim = 100
batch_size = 32
epochs = 200
# lamda1, lamda2, lamda3, lamda4, lamda5 = 5, 25, 2, 5, 2
# lamda1, lamda2, lamda3, lamda4, lamda5 = 0.5, 500, 5, 10, 5
# lamda1, lamda2, lamda3, lamda4, lamda5 = 0.5, 150, 5, 15, 10
# lamda1, lamda2, lamda3, lamda4, lamda5 = 0.5, 250, 5, 20, 10
lamda1, lamda2, lamda3, lamda4, lamda5 = 0.5, 1000, 5, 20, 10



'''load data'''
x_train, b_train, c_train, x_test, b_test, c_test = load_data()
x_train = np.expand_dims(x_train, axis=3)
x_test = np.expand_dims(x_test, axis=3)

num_examples_to_generate = 16
random_vector_for_generation = tf.random.normal(
    shape=[num_examples_to_generate, z_dim])
sample_for_virtualization = x_test[0:4,]
sample_for_tsne = x_test[0:100,]

train_dataset = (tf.data.Dataset.from_tensor_slices((x_train, b_train, c_train))
                 .shuffle(1000).batch(batch_size))
test_dataset = (tf.data.Dataset.from_tensor_slices((x_test, b_test, c_test))
                .shuffle(1000).batch(batch_size))



'''create models'''
bf_generator = BF_Generator(x_train.shape, z_dim)
generator = Generator(x_train.shape, z_dim)
batch_encoder = Batch_Encoder(x_train.shape, z_dim)
bf_descriminator = make_Descriminator_x(x_train.shape, b_train.shape[1], activation='softmax')
bf_classifier = make_Descriminator_x(x_train.shape, c_train.shape[1], activation='sigmoid')
category_discriminator = make_category_Descriminator_z(c_train.shape[1])
batch_discriminator = make_batch_Descriminator_z(b_train.shape[1])


'''create optimizer'''
lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(
        2e-5,
        decay_steps=2000,
        decay_rate=0.96,
        staircase=True)
bf_generator_optimizer = tf.keras.optimizers.Adam(lr_schedule)
generator_optimizer = tf.keras.optimizers.Adam(lr_schedule)
batch_encoder_optimizer = tf.keras.optimizers.Adam(lr_schedule)
batch_discriminator_optimizer = tf.keras.optimizers.Adam(lr_schedule)
category_discriminator_optimizer = tf.keras.optimizers.Adam(lr_schedule)
bf_discriminator_optimizer = tf.keras.optimizers.Adam(lr_schedule)
bf_classifier_optimizer = tf.keras.optimizers.Adam(lr_schedule)



train_gen_loss = tf.keras.metrics.Mean(name='train_genetator_loss')
train_disc_loss = tf.keras.metrics.Mean(name='train_discriminator_loss')
test_gen_loss = tf.keras.metrics.Mean(name='test_genetator_loss')
test_disc_loss = tf.keras.metrics.Mean(name='test_discriminator_loss')

train_zb_batch_accuracy = accuracy_score(name='train_zb_batch_accuracy')
train_zb_category_accuracy = accuracy_score(name='train_zb_category_accuracy')
train_zc_batch_accuracy = accuracy_score(name='train_zc_batch_accuracy')
train_zc_category_accuracy = accuracy_score(name='train_zc_category_accuracy')
train_x_batch_accuracy = accuracy_score(name='train_x_batch_accuracy')
train_x_category_accuracy = accuracy_score(name='train_x_category_accuracy')
train_bf_x_batch_accuracy = accuracy_score(name='train_bf_x_batch_accuracy')
train_bf_x_category_accuracy = accuracy_score(name='train_bf_x_category_accuracy')
test_zb_batch_accuracy = accuracy_score(name='test_zb_batch_accuracy')
test_zb_category_accuracy = accuracy_score(name='test_zb_category_accuracy')
test_zc_batch_accuracy = accuracy_score(name='test_zc_batch_accuracy')
test_zc_category_accuracy = accuracy_score(name='test_zc_category_accuracy')
test_x_batch_accuracy = accuracy_score(name='test_x_batch_accuracy')
test_x_category_accuracy = accuracy_score(name='test_x_category_accuracy')
test_bf_x_batch_accuracy = accuracy_score(name='test_bf_x_batch_accuracy')
test_bf_x_category_accuracy = accuracy_score(name='test_bf_x_category_accuracy')

train_zb_batch_LRAP = label_ranking_average_precision(name='train_zb_batch_LRAP')
train_zb_category_LRAP = label_ranking_average_precision(name='train_zb_category_LRAP')
train_zc_batch_LRAP = label_ranking_average_precision(name='train_zc_batch_LRAP')
train_zc_category_LRAP = label_ranking_average_precision(name='train_zc_category_LRAP')
train_x_batch_LRAP = label_ranking_average_precision(name='train_x_batch_LRAP')
train_x_category_LRAP = label_ranking_average_precision(name='train_x_category_LRAP')
train_bf_x_batch_LRAP = label_ranking_average_precision(name='train_bf_x_batch_LRAP')
train_bf_x_category_LRAP = label_ranking_average_precision(name='train_bf_x_category_LRAP')
test_zb_batch_LRAP = label_ranking_average_precision(name='test_zb_batch_LRAP')
test_zb_category_LRAP = label_ranking_average_precision(name='test_zb_category_LRAP')
test_zc_batch_LRAP = label_ranking_average_precision(name='test_zc_batch_LRAP')
test_zc_category_LRAP = label_ranking_average_precision(name='test_zc_category_LRAP')
test_x_batch_LRAP = label_ranking_average_precision(name='test_x_batch_LRAP')
test_x_category_LRAP = label_ranking_average_precision(name='test_x_category_LRAP')
test_bf_x_batch_LRAP = label_ranking_average_precision(name='test_bf_x_batch_LRAP')
test_bf_x_category_LRAP = label_ranking_average_precision(name='test_bf_x_category_LRAP')

train_zb_category_micro_average = micro_average(name='train_zb_category_micro_average')
train_zc_category_micro_average = micro_average(name='train_zc_category_micro_average')
train_x_category_micro_average = micro_average(name='train_x_category_micro_average')
train_bf_x_category_micro_average = micro_average(name='train_bf_x_category_micro_average')
test_zb_category_micro_average = micro_average(name='test_zb_category_micro_average')
test_zc_category_micro_average = micro_average(name='test_zc_category_micro_average')
test_x_category_micro_average = micro_average(name='test_x_category_micro_average')
test_bf_x_category_micro_average = micro_average(name='test_bf_x_category_micro_average')

train_gen_loss_results = []
train_disc_loss_results = []
test_gen_loss_results = []
test_disc_loss_results = []



'''save checkpoint'''
checkpoint_dir = '/vol1/cuipeng_group/qiankun/GAN-VAE/training_checkpoints_v5/'
ckpt = tf.train.Checkpoint(
    bf_generator_optimizer=bf_generator_optimizer, 
    generator_optimizer=generator_optimizer, 
    batch_encoder_optimizer=batch_encoder_optimizer, 
    batch_discriminator_optimizer=batch_discriminator_optimizer, 
    category_discriminator_optimizer=category_discriminator_optimizer, 
    bf_discriminator_optimizer=bf_discriminator_optimizer, 
    bf_classifier_optimizer=bf_classifier_optimizer, 
    bf_generator=bf_generator,
    generator=generator, 
    batch_encoder=batch_encoder, 
    bf_descriminator=bf_descriminator, 
    bf_classifier=bf_classifier, 
    category_discriminator=category_discriminator, 
    batch_discriminator=batch_discriminator
    )
ckpt_manager = tf.train.CheckpointManager(ckpt, checkpoint_dir, max_to_keep=50)
if not os.path.exists(checkpoint_dir):
     os.makedirs(checkpoint_dir)

        
@tf.function
def train_step(x, batch, category):
    with tf.GradientTape(persistent=True) as tape:
        #generate batch free x
        zc_mean, zc_logvar = bf_generator.encode(x)
        zc = bf_generator.reparameterize(zc_mean, zc_logvar)
        bf_x = bf_generator.decode(zc, apply_sigmoid=True)
        #batch info encode
        zb_mean, zb_logvar = batch_encoder.encode(x)
        zb = batch_encoder.reparameterize(zb_mean, zb_logvar)
        #conditional Unets to restore orignal x
        x_logits = generator(bf_x, zb)
        #predict batch
        bf_x_batch = bf_descriminator(bf_x)
        x_batch = bf_descriminator(x)
        #predict category
        bf_x_class = bf_classifier(bf_x)
        x_class = bf_classifier(x)
        #predict category on latent z_dim
        zb_category = category_discriminator(zb)
        zc_category = category_discriminator(zc)
        #predict batch on latent z_dim
        zb_batch = batch_discriminator(zb)
        zc_batch = batch_discriminator(zc)
        
        #compute loss
        c_VAE_loss = conditional_VAE_loss(x, zc_mean, zc_logvar, zc, zb_mean, zb_logvar, zb, x_logits)
        bf_loss = L2_loss(x, bf_x)
        zb_gen_loss = Zb_generator_loss(zb_batch, zb_category, batch, category)
        zc_gen_loss = Zc_generator_loss(zc_batch, zc_category, batch, category)
        bf_gen_loss = bf_generator_loss(bf_x_batch, bf_x_class, category)

        batch_disc_loss = batch_discriminator_loss(zb_batch, zc_batch, batch)
        category_disc_loss = category_discriminator_loss(zb_category, zc_category, category)
        bf_disc_loss = bf_discriminator_loss(x_batch, bf_x_batch, batch)
        bf_class_loss = bf_classifier_loss(x_class, category)

        total_gen_loss = lamda1 * c_VAE_loss + lamda2 * bf_loss + lamda3 * zb_gen_loss + lamda4 * zc_gen_loss + lamda5 * bf_gen_loss
        total_disc_loss = batch_disc_loss + category_disc_loss + bf_disc_loss + 0.1 * bf_class_loss

    #compute gradients and apply on optimizers
    generator_gradients = tape.gradient(total_gen_loss,
                                             generator.trainable_variables)                                              
    bf_generator_gradients = tape.gradient(total_gen_loss,
                                           bf_generator.trainable_variables)
    batch_encoder_gradients = tape.gradient(total_gen_loss,
                                            batch_encoder.trainable_variables) 
    bf_classifier_gradients = tape.gradient(total_disc_loss,
                                            bf_classifier.trainable_variables)                                     
    bf_descriminator_gradients = tape.gradient(total_disc_loss,
                                               bf_descriminator.trainable_variables)
    batch_discriminator_gradients = tape.gradient(total_disc_loss,
                                                  batch_discriminator.trainable_variables)
    category_discriminator_gradients = tape.gradient(total_disc_loss,
                                                     category_discriminator.trainable_variables)


    bf_discriminator_optimizer.apply_gradients(zip(bf_descriminator_gradients, 
                                                  bf_descriminator.trainable_variables))
    batch_discriminator_optimizer.apply_gradients(zip(batch_discriminator_gradients,
                                                        batch_discriminator.trainable_variables))
    category_discriminator_optimizer.apply_gradients(zip(category_discriminator_gradients,
                                                category_discriminator.trainable_variables))
    bf_classifier_optimizer.apply_gradients(zip(bf_classifier_gradients, 
                                                    bf_classifier.trainable_variables))      
    generator_optimizer.apply_gradients(zip(generator_gradients, 
                                            generator.trainable_variables))                                
    bf_generator_optimizer.apply_gradients(zip(bf_generator_gradients, 
                                            bf_generator.trainable_variables))
    batch_encoder_optimizer.apply_gradients(zip(batch_encoder_gradients, 
                                            batch_encoder.trainable_variables))

    train_gen_loss(total_gen_loss)
    train_disc_loss(total_disc_loss)
    
    train_zb_batch_accuracy.update_state(batch, zb_batch)
    train_zc_batch_accuracy.update_state(batch, zc_batch)
    train_zb_category_accuracy.update_state(category, zb_category)
    train_zc_category_accuracy.update_state(category, zc_category)
    train_x_batch_accuracy.update_state(batch, x_batch)
    train_bf_x_batch_accuracy.update_state(batch, bf_x_batch)
    train_x_category_accuracy.update_state(category, x_class)
    train_bf_x_category_accuracy.update_state(category, bf_x_class)
    
    train_zb_batch_LRAP.update_state(batch, zb_batch)
    train_zc_batch_LRAP.update_state(batch, zc_batch)
    train_zb_category_LRAP.update_state(category, zb_category)
    train_zc_category_LRAP.update_state(category, zc_category)
    train_x_batch_LRAP.update_state(batch, x_batch)
    train_bf_x_batch_LRAP.update_state(batch, bf_x_batch)
    train_x_category_LRAP.update_state(category, x_class)
    train_bf_x_category_LRAP.update_state(category, bf_x_class)

    train_zb_category_micro_average.update_state(category, zb_category)
    train_zc_category_micro_average.update_state(category, zc_category)
    train_x_category_micro_average.update_state(category, x_class)
    train_bf_x_category_micro_average.update_state(category, bf_x_class)


@tf.function
def test_step(x, batch, category):
    zc_mean, zc_logvar = bf_generator.encode(x)
    zc = bf_generator.reparameterize(zc_mean, zc_logvar)
    bf_x = bf_generator.decode(zc)
    #batch info encode
    zb_mean, zb_logvar = batch_encoder.encode(x)
    zb = batch_encoder.reparameterize(zb_mean, zb_logvar)
    #conditional Unets to restore orignal x
    x_logits = generator(bf_x, zb)
    #predict batch
    bf_x_batch = bf_descriminator(bf_x)
    x_batch = bf_descriminator(x)
    #predict category
    bf_x_class = bf_classifier(bf_x)                                            
    x_class = bf_classifier(x)
    #predict category on latent z_dim
    zb_category = category_discriminator(zb)
    zc_category = category_discriminator(zc)
    #predict batch on latent z_dim
    zb_batch = batch_discriminator(zb)
    zc_batch = batch_discriminator(zc)

    
    #compute loss
    c_VAE_loss = conditional_VAE_loss(x, zc_mean, zc_logvar, zc, zb_mean, zb_logvar, zb, x_logits)
    bf_loss = L1_loss(x, bf_x)
    zb_gen_loss = Zb_generator_loss(zb_batch, zb_category, batch, category)
    zc_gen_loss = Zc_generator_loss(zc_batch, zc_category, batch, category)
    bf_gen_loss = bf_generator_loss(bf_x_batch, bf_x_class, category)

    batch_disc_loss = batch_discriminator_loss(zb_batch, zc_batch, batch)
    category_disc_loss = category_discriminator_loss(zb_category, zc_category, category)
    bf_disc_loss = bf_discriminator_loss(x_batch, bf_x_batch, batch)
    bf_class_loss = bf_classifier_loss(x_class, category)

    total_gen_loss = lamda1 * c_VAE_loss + lamda2 * bf_loss + lamda3 * zb_gen_loss + lamda4 * zc_gen_loss + lamda5 * bf_gen_loss
    total_disc_loss = batch_disc_loss + category_disc_loss + bf_disc_loss +  0.1 * bf_class_loss

    test_gen_loss(total_gen_loss)
    test_disc_loss(total_disc_loss)

    test_zb_batch_accuracy.update_state(batch, zb_batch)
    test_zc_batch_accuracy.update_state(batch, zc_batch)
    test_zb_category_accuracy.update_state(category, zb_category)
    test_zc_category_accuracy.update_state(category, zc_category)
    test_x_batch_accuracy.update_state(batch, x_batch)
    test_bf_x_batch_accuracy.update_state(batch, bf_x_batch)
    test_x_category_accuracy.update_state(category, x_class)
    test_bf_x_category_accuracy.update_state(category, bf_x_class)
    
    test_zb_batch_LRAP.update_state(batch, zb_batch)
    test_zc_batch_LRAP.update_state(batch, zc_batch)
    test_zb_category_LRAP.update_state(category, zb_category)
    test_zc_category_LRAP.update_state(category, zc_category)
    test_x_batch_LRAP.update_state(batch, x_batch)
    test_bf_x_batch_LRAP.update_state(batch, bf_x_batch)
    test_x_category_LRAP.update_state(category, x_class)
    test_bf_x_category_LRAP.update_state(category, bf_x_class)
    
    test_zb_category_micro_average.update_state(category, zb_category)
    test_zc_category_micro_average.update_state(category, zc_category)
    test_x_category_micro_average.update_state(category, x_class)
    test_bf_x_category_micro_average.update_state(category, bf_x_class)

generate_and_save_images(bf_generator, 0, random_vector_for_generation)
comparing_real_and_generated_data(bf_generator, batch_encoder, generator, sample_for_virtualization, 4, 0)
comparing_pca_between_x_and_bfx(sample_for_tsne, bf_generator, batch_encoder, generator, 0)
n = 0
for epoch in range(1, epochs+1):
    start_time = time.time()
    train_gen_loss.reset_states()
    train_disc_loss.reset_states()
    test_gen_loss.reset_states()
    test_disc_loss.reset_states()

    train_zb_batch_accuracy.reset_states()
    train_zc_batch_accuracy.reset_states()
    train_zb_category_accuracy.reset_states()
    train_zc_category_accuracy.reset_states()
    train_x_batch_accuracy.reset_states()
    train_bf_x_batch_accuracy.reset_states()
    train_x_category_accuracy.reset_states()
    train_bf_x_category_accuracy.reset_states()
    test_zb_batch_accuracy.reset_states()
    test_zc_batch_accuracy.reset_states()
    test_zb_category_accuracy.reset_states()
    test_zc_category_accuracy.reset_states()
    test_x_batch_accuracy.reset_states()
    test_bf_x_batch_accuracy.reset_states()
    test_x_category_accuracy.reset_states()
    test_bf_x_category_accuracy.reset_states()
    
    train_zb_batch_LRAP.reset_states()
    train_zc_batch_LRAP.reset_states()
    train_zb_category_LRAP.reset_states()
    train_zc_category_LRAP.reset_states()
    train_x_batch_LRAP.reset_states()
    train_bf_x_batch_LRAP.reset_states()
    train_x_category_LRAP.reset_states()
    train_bf_x_category_LRAP.reset_states()
    test_zb_batch_LRAP.reset_states()
    test_zc_batch_LRAP.reset_states()
    test_zb_category_LRAP.reset_states()
    test_zc_category_LRAP.reset_states()
    test_x_batch_LRAP.reset_states()
    test_bf_x_batch_LRAP.reset_states()
    test_x_category_LRAP.reset_states()
    test_bf_x_category_LRAP.reset_states()
    
    train_zb_category_micro_average.reset_states()
    train_zc_category_micro_average.reset_states()
    train_x_category_micro_average.reset_states()
    train_bf_x_category_micro_average.reset_states()
    test_zb_category_micro_average.reset_states()
    test_zc_category_micro_average.reset_states()
    test_x_category_micro_average.reset_states()
    test_bf_x_category_micro_average.reset_states()

    for batch, (train_x, train_b, train_c) in enumerate(train_dataset):
        n += 1
        train_step(train_x, train_b, train_c)
        end_time = time.time()
        tf.summary.scalar("train_gen_loss",train_gen_loss.result(),step=n)
        tf.summary.scalar("train_disc_loss",train_disc_loss.result(),step=n)
        tf.summary.histogram("bf_gen_top_layer/kernel", bf_generator.variables[0],step=n)
        tf.summary.histogram("bf_gen_top_layer/bias", bf_generator.variables[1],step=n)
        tf.summary.histogram("batch_encoder_top_layer/kernel", batch_encoder.variables[0],step=n)
        tf.summary.histogram("batch_encoder_top_layer/bias", batch_encoder.variables[1],step=n)
        tf.summary.histogram("bf_class_top_layer/kernel", bf_classifier.variables[0],step=n)
        tf.summary.histogram("bf_class_top_layer/bias", bf_classifier.variables[1],step=n)
        tf.summary.histogram("bf_disc_top_layer/kernel", bf_descriminator.variables[0],step=n)
        tf.summary.histogram("bf_disc_top_layer/bias", bf_descriminator.variables[1],step=n)

        # if (batch % 100) == 0:
        #     print('Epoch: {}, Batch: {}, Train G Loss: {}, Train D loss {}'
        #         .format(epoch, batch, train_gen_loss.result(), train_disc_loss.result()))
        #     print('zb batch acc: {}, zb category acc: {}, zc batch acc: {}, zc category acc: {}'
        #           .format(train_zb_batch_accuracy.result(), train_zb_category_accuracy.result(),
        #                   train_zc_batch_accuracy.result(), train_zc_category_accuracy.result()))
        #     print('x batch acc：{}, x class acc：{}, bf_x batch acc：{}, bf_x class acc：{}'
        #           .format(train_x_batch_accuracy.result(), train_x_category_accuracy.result(),
        #                   train_bf_x_batch_accuracy.result(), train_bf_x_category_accuracy.result()))
    print('Train zb batch LRAP: {}, Train zb category LRAP: {}, Train zc batch LRAP: {}, Train zc category LRAP: {}'
                .format(train_zb_batch_LRAP.result(), train_zb_category_LRAP.result(),
                        train_zc_batch_LRAP.result(), train_zc_category_LRAP.result()))
    print('Train x batch LRAP：{}, Train x class LRAP：{}, Train bf_x batch LRAP：{}, Train bf_x class LRAP：{}'
                .format(train_x_batch_LRAP.result(), train_x_category_LRAP.result(),
                        train_bf_x_batch_LRAP.result(), train_bf_x_category_LRAP.result()))
    print('Train zb batch LRAP: {}, Train zb category LRAP: {}, Train zc batch LRAP: {}, Train zc category LRAP: {}'
                .format(train_zb_batch_LRAP.result(), train_zb_category_LRAP.result(),
                        train_zc_batch_LRAP.result(), train_zc_category_LRAP.result()))
    print('Train x batch LRAP：{}, Train x class LRAP：{}, Train bf_x batch LRAP：{}, Train bf_x class LRAP：{}'
                .format(train_x_batch_LRAP.result(), train_x_category_LRAP.result(),
                        train_bf_x_batch_LRAP.result(), train_bf_x_category_LRAP.result()))
    print('Train zb category precision: {}, Train zc category precision: {}'
            .format(train_zb_category_micro_average.result()[0], train_zc_category_micro_average.result()[0]))
    print('Train x class precision：{}, Train bf_x class precision：{}'
            .format(train_x_category_micro_average.result()[0], train_bf_x_category_micro_average.result()[0]))
    print('Train zb category recall: {}， Train zc category recall: {}'
            .format(train_zb_category_micro_average.result()[1], train_zc_category_micro_average.result()[1]))
    print('Train x class recall：{}, Train bf_x class recall：{}'
            .format(train_x_category_micro_average.result()[1], train_bf_x_category_micro_average.result()[1]))
    print('Train zb category F1: {}, Train zc category F1: {}'
            .format(train_zb_category_micro_average.result()[2], train_zc_category_micro_average.result()[2]))
    print('Train x class F1：{}, Train bf_x class F1：{}'
            .format(train_x_category_micro_average.result()[2], train_bf_x_category_micro_average.result()[2]))                

    for test_x, test_b, test_c in test_dataset:
        test_step(test_x, test_b, test_c)

    print('Test zb batch acc: {}, Test zb category acc: {}, Test zc batch acc: {}, Test zc category acc: {}'
          .format(test_zb_batch_accuracy.result(), test_zb_category_accuracy.result(),
                  test_zc_batch_accuracy.result(), test_zc_category_accuracy.result()))
    print('Test x batch acc：{}, Test x class acc：{}, Test bf_x batch acc：{},Test bf_x class acc：{}'
          .format(test_x_batch_accuracy.result(), test_x_category_accuracy.result(),
                  test_bf_x_batch_accuracy.result(), test_bf_x_category_accuracy.result()))
    print('Test zb batch LRAP: {}, Test zb category LRAP: {}, Test zc batch LRAP: {}, Test zc category LRAP: {}'
          .format(test_zb_batch_LRAP.result(), test_zb_category_LRAP.result(),
                  test_zc_batch_LRAP.result(), test_zc_category_LRAP.result()))
    print('Test x batch LRAP：{}, Test x class LRAP：{}, Test bf_x batch LRAP：{}, Test bf_x class LRAP：{}'
          .format(test_x_batch_LRAP.result(), test_x_category_LRAP.result(),
                  test_bf_x_batch_LRAP.result(), test_bf_x_category_LRAP.result()))
    print('Test zb category precision: {}, Test zc category precision: {}'
          .format(test_zb_category_micro_average.result()[0], test_zc_category_micro_average.result()[0]))
    print('Test x class precision：{}, Test bf_x class precision：{}'
          .format(test_x_category_micro_average.result()[0], test_bf_x_category_micro_average.result()[0]))
    print('Test zb category recall: {}, Test zc category recall: {}'
          .format(test_zb_category_micro_average.result()[1], test_zc_category_micro_average.result()[1]))
    print('Test x class recall：{}, Test bf_x class recall：{}'
          .format(test_x_category_micro_average.result()[1], test_bf_x_category_micro_average.result()[1]))
    print('Test zb category F1: {}, Test zc category F1: {}'
          .format(test_zb_category_micro_average.result()[2], test_zc_category_micro_average.result()[2]))
    print('Test x class F1：{}, Test bf_x class F1：{}'
          .format(test_x_category_micro_average.result()[2], test_bf_x_category_micro_average.result()[2]))
    
    print('Epoch: {}, Train G Loss: {}, Test G Loss: {}, Train D loss {}, Test D loss {}, time elapse for current epoch: {}'
          .format(epoch, train_gen_loss.result(), test_gen_loss.result(), train_disc_loss.result(),
                  test_disc_loss.result(), (end_time - start_time)))

    ckpt_save_path = ckpt_manager.save()
    print('Saving checkpoint for epoch {} at {}'.format(epoch, ckpt_save_path))
    
    train_gen_loss_results.append(train_gen_loss.result())
    train_disc_loss_results.append(train_disc_loss.result())
    test_gen_loss_results.append(test_gen_loss.result())
    test_disc_loss_results.append(test_disc_loss.result())

    if epoch % 2 == 0:
        generate_and_save_images(bf_generator, epoch, random_vector_for_generation)
        comparing_real_and_generated_data(bf_generator, batch_encoder, generator, sample_for_virtualization, 4, epoch)
        comparing_pca_between_x_and_bfx(sample_for_tsne, bf_generator, batch_encoder, generator, epoch)


