import os, sys
sys.path.append('/notebooks/01-Code/')

import time
import datetime
import pickle
import argparse
import math
import numpy as np
import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()

import utils
import tflib as lib
import tflib.ops.linear
import tflib.ops.conv1d
import tflib.plot
import data.filesystem as fs
import data.timestamp as ts
import models

from pathlib import Path

lines=None


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument('--training-data', '-t',
                        required=True,
                        dest='training_data',
                        help='Path to training data directory')

    parser.add_argument('--sample-every', '-s',
                        type=int,
                        default=5000,
                        dest='sample_every',
                        help='Sample und evaluate Fake Passwords after this many iterations (default: 5000)')

    parser.add_argument('--iters', '-i',
                        type=int,
                        default=5000,
                        dest='iters',
                        help='The number of training iterations (default: 200000)')

    parser.add_argument('--batch-size', '-b',
                        type=int,
                        default=64,
                        dest='batch_size',
                        help='Batch size (default: 64).')
        
    parser.add_argument('--fake-samples', '-f',
                        type=int,
                        default=10000000,
                        dest='fake_samples',
                        help='The number of fake samples (default: 1000)')
    
    parser.add_argument('--layer-dim', '-d',
                        type=int,
                        default=128,
                        dest='layer_dim',
                        help='The hidden layer dimensionality for the generator and discriminator (default: 128)')
    
    parser.add_argument('--critic-iters', '-c',
                        type=int,
                        default=10,
                        dest='critic_iters',
                        help='The number of discriminator weight updates per generator update (default: 10)')
    
    parser.add_argument('--lambda', '-p',
                        type=int,
                        default=10,
                        dest='lamb',
                        help='The gradient penalty lambda hyperparameter (default: 10)')
    
    parser.add_argument('--restart', '-r',
                        action="store_true",
                        dest='restart',
                        help='Delete old Checkpoints and Restart from Scratch (default: 10)')
    
    return parser.parse_args()


        

# Dataset iterator
def inf_train_gen(batch_size, charmap):
    global lines

    while True:
        #np.random.shuffle(lines)
        for i in range(0, len(lines)-batch_size+1, batch_size):
            yield np.array(
                [[charmap[c] if c in charmap.keys() else charmap["unk"] for c in l] for l in lines[i:i+batch_size]],
                dtype='int32'
            )


if __name__ == "__main__":
    
    # Arguments
    args = parse_args()

    batch_size = args.batch_size
    layer_dim = args.layer_dim
    iters = args.iters
    fake_pw_count = args.fake_samples
    c_iter = args.critic_iters
    ckpt_iter = args.sample_every
    lamb = args.lamb
    restart = args.restart
    ckpt = round(iters/10)
    
    # Directories
    root_dir = '/notebooks/02-Data/' + args.training_data + '/'
    
    
    # Training Password Set
    train_pw_filepath = fs.find_pw_file(root_dir, "Train")
    train_pw_filedir = fs.get_pw_filedir(train_pw_filepath)
    train_pw_filename = fs.get_pw_filename(train_pw_filepath)
    train_pw_name = fs.get_pw_name(train_pw_filename)
    train_pw_length = fs.get_pw_length(train_pw_filename)
    train_pw_minlength = fs.get_pw_minlength(train_pw_filename)
    train_pw_maxlength = fs.get_pw_maxlength(train_pw_filename)
    train_pw_count = fs.get_pw_count(train_pw_filename)
    

    
    
    # Load Training PW
    print(ts.timestamp(), "Loading Training Passwords")
    lines, charmap, inv_charmap = utils.load_dataset(train_pw_filepath, train_pw_maxlength)
    print(ts.timestamp(), "Training Passwords loaded")


    
    real_inputs_discrete = tf.placeholder(tf.int32, shape=[batch_size, train_pw_maxlength])
    real_inputs = tf.one_hot(real_inputs_discrete, len(charmap))
    fake_inputs = models.Generator(batch_size, train_pw_maxlength, layer_dim, len(charmap))
    fake_inputs_discrete = tf.argmax(fake_inputs, fake_inputs.get_shape().ndims-1)

    disc_real = models.Discriminator(real_inputs, train_pw_maxlength, layer_dim, len(charmap))
    disc_fake = models.Discriminator(fake_inputs, train_pw_maxlength, layer_dim, len(charmap))

    disc_cost = tf.reduce_mean(disc_fake) - tf.reduce_mean(disc_real)
    gen_cost = -tf.reduce_mean(disc_fake)

    # WGAN lipschitz-penalty
    alpha = tf.random_uniform(
        shape=[batch_size,1,1],
        minval=0.,
        maxval=1.
    )

    differences = fake_inputs - real_inputs
    interpolates = real_inputs + (alpha*differences)
    gradients = tf.gradients(models.Discriminator(interpolates, train_pw_maxlength, layer_dim, len(charmap)), [interpolates])[0]
    slopes = tf.sqrt(tf.reduce_sum(tf.square(gradients), reduction_indices=[1,2]))
    gradient_penalty = tf.reduce_mean((slopes-1.)**2)
    disc_cost += lamb * gradient_penalty

    gen_params = lib.params_with_name('Generator')
    disc_params = lib.params_with_name('Discriminator')

    gen_train_op = tf.train.AdamOptimizer(learning_rate=1e-4, beta1=0.5, beta2=0.9).minimize(gen_cost, var_list=gen_params)
    disc_train_op = tf.train.AdamOptimizer(learning_rate=1e-4, beta1=0.5, beta2=0.9).minimize(disc_cost, var_list=disc_params)




    with tf.Session() as session:
        
        print()
        print(ts.timestamp(), "--- TRAINING STARTED ---") 
        train_start = time.time()
        
        # Generator Function for Fake Passwords
        def generate_samples():
            samples = session.run(fake_inputs)
            samples = np.argmax(samples, axis=2)
            decoded_samples = []
            for i in range(len(samples)):
                decoded = []
                for j in range(len(samples[i])):
                    decoded.append(inv_charmap[samples[i][j]])
                decoded_samples.append(tuple(decoded))
            return decoded_samples

        # Generator Function for Training Passwords
        gen = inf_train_gen(batch_size, charmap)
        



        session.run(tf.global_variables_initializer())
        iteration_start = 1
        print(ts.timestamp(), "Training started from scratch")
        print()
        
        

        

        
        
        
        ckpt_start = time.time()
        
        iters_time = []

        for iteration in range(iteration_start, iters+1):
            start_time = time.time()

            # Train generator
            if iteration > 0:
                _ = session.run(gen_train_op)

            # Train critic
            for i in range(c_iter):
                _data = next(gen)
                _disc_cost, _ = session.run(
                    [disc_cost, disc_train_op],
                    feed_dict={real_inputs_discrete:_data}
                )



            if iteration % ckpt == 0:
                ckpt_stop = time.time()
                ckpt_time = round(ckpt_stop - ckpt_start, 1)
                iter_time = round((ckpt_stop - ckpt_start)/ckpt, 3)
                ckpt_start = time.time()
                iters_time.append(iter_time)
                log = "{} Iteration: {}/{} ({}%), Checkpoint Time: {}sec, Iteration Time: {}sec".format(ts.timestamp(), iteration, iters, round(iteration/iters*100.0), ckpt_time, iter_time)
                print(log)
            
            




    train_stop = round(time.time() - train_start, 1)
    
    iters_time = iters_time[1:]
    avg_iter_time = round(sum(iters_time)/len(iters_time),3)
    train_stop_log = "{} Training Time: {}sec, Average Iteration Time (sec): {}".format(ts.timestamp(), train_stop, avg_iter_time)
    
    print()
    print(ts.timestamp(), "--- TRAINING FINISHED ---")
    print(train_stop_log)
    print()