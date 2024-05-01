# PassGAN
#
# This code is based on https://github.com/beta6/PassGAN/tree/main
#
# Copyright (c) 2017 Ishaan Gulrajani
#           (c) 2017 Brannon Dorsey (PassGAN modification and modularization)
#
# Please see the attached LICENSE file for additional licensing information.

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

    parser.add_argument('--save-every', '-s',
                        type=int,
                        default=5000,
                        dest='save_every',
                        help='Save Model after this many iterations (default: 5000)')

    parser.add_argument('--iters', '-i',
                        type=int,
                        default=200000,
                        dest='iters',
                        help='The number of training iterations (default: 200000)')

    parser.add_argument('--batch-size', '-b',
                        type=int,
                        default=64,
                        dest='batch_size',
                        help='Batch size (default: 64).')
            
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

# Check if dirs exist and make dir if not
def initdirs(dirs):
    for d in dirs:
        if not os.path.isdir(d):
            os.makedirs(d)

    
        
def savemaps(charmap_filedir, charmap, inv_charmap):

    # pickle to avoid encoding errors with json
    with open(os.path.join(charmap_filedir, 'charmap.pickle'), 'wb') as f:
        pickle.dump(charmap, f)

    with open(os.path.join(charmap_filedir, 'inv_charmap.pickle'), 'wb') as f:
        pickle.dump(inv_charmap, f)

        

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
    
    training_data = args.training_data
    batch_size = args.batch_size
    layer_dim = args.layer_dim
    iters = args.iters
    c_iter = args.critic_iters
    ckpt_iter = args.save_every
    lamb = args.lamb
    restart = args.restart
        
    
    # Directories
    root_dir = '/notebooks/02-Data/' + training_data + '/'
    ckpt_filedir = root_dir + 'Ckpt/'
    charmap_filedir = root_dir + 'Charmap/'
    plot_filedir = root_dir + 'Plot/'
    fake_pw_filedir = root_dir + 'Fake/'
    
    dirs = (ckpt_filedir, charmap_filedir, plot_filedir, fake_pw_filedir)
    
    
    # Training Password Set
    train_pw_filepath = fs.find_pw_file(root_dir, "Train")
    train_pw_filedir = fs.get_pw_filedir(train_pw_filepath)
    train_pw_filename = fs.get_pw_filename(train_pw_filepath)
    train_pw_name = fs.get_pw_name(train_pw_filename)
    train_pw_length = fs.get_pw_length(train_pw_filename)
    train_pw_minlength = fs.get_pw_minlength(train_pw_filename)
    train_pw_maxlength = fs.get_pw_maxlength(train_pw_filename)
    train_pw_count = fs.get_pw_count(train_pw_filename)
    
    
    # Testing Password Set
    test_pw_filepath = fs.find_pw_file(root_dir, "Test")
    test_pw_filedir = fs.get_pw_filedir(test_pw_filepath)
    test_pw_filename = fs.get_pw_filename(test_pw_filepath)
    test_pw_name = fs.get_pw_name(test_pw_filename)
    test_pw_length = fs.get_pw_length(test_pw_filename)
    test_pw_minlength = fs.get_pw_minlength(test_pw_filename)
    test_pw_maxlength = fs.get_pw_maxlength(test_pw_filename)
    test_pw_count = fs.get_pw_count(test_pw_filename)
    
    # Training Log
    log_filename = fs.create_pw_filename('Log', train_pw_name, train_pw_minlength, train_pw_maxlength, train_pw_count)
    log_filepath = os.path.join(root_dir, log_filename + '.log')
    csv_filepath = os.path.join(root_dir, log_filename + '.csv')
        

    # If Restart delete previous files and directories
    if restart:
        fs.delete_directory(ckpt_filedir)
        fs.delete_directory(charmap_filedir)
        fs.delete_directory(plot_filedir)
        fs.delete_directory(fake_pw_filedir)
        if Path(log_filepath).exists():
                os.remove(log_filepath)
        if Path(csv_filepath).exists():
                os.remove(csv_filepath)
        

    
    # Initialize Directories
    initdirs(dirs)
    
        
    # Load Training PW
    print()
    print(ts.timestamp(), "--- TRAINING STARTED ---") 
    print(ts.timestamp(), "Loading Training Passwords")
    lines, charmap, inv_charmap = utils.load_dataset(train_pw_filepath, train_pw_maxlength)
    print(ts.timestamp(), "Training Passwords loaded")
    print()

    # Save Charmap
    savemaps(charmap_filedir, charmap, inv_charmap)

    
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
        

        # Check if previous Checkpoints are available
        last_ckpt = tf.train.latest_checkpoint(ckpt_filedir)
        if last_ckpt:
            ckpt_iteration = int(last_ckpt[last_ckpt.rfind('_')+1:last_ckpt.rfind('.')])
            iteration_start = ckpt_iteration + 1
            # Restore from last Checkpoint
            model_saver = tf.train.import_meta_graph(last_ckpt+'.meta')
            model_saver.restore(session, last_ckpt)
            
            # Jump to next Training Passwords
            for _ in range (0, ckpt_iteration*10):
                next(gen)
            
            print(ts.timestamp(), "Progress restored from: " + last_ckpt)
            print()
        else:
            # Initialize CSV
            with open(csv_filepath, 'a') as file:
                eval_csv_header= "Time;PW Set;Iteration;Total Iterations;Checkpoint Time (min);G Cost;D Cost"
                file.write(eval_csv_header + '\n')
                
            model_saver = tf.train.Saver(max_to_keep=None)
            session.run(tf.global_variables_initializer())
            iteration_start = 1
            print(ts.timestamp(), "Training started from scratch")
            print()
        


        ckpt_start = time.time()

        for iteration in range(iteration_start, iters+1):
            start_time = time.time()

            # Train generator
            if iteration > 0:
                g_cost, _ = session.run([gen_cost, gen_train_op])

            # Train critic
            for i in range(c_iter):
                _data = next(gen)
                d_cost, _ = session.run(
                    [disc_cost, disc_train_op],
                    feed_dict={real_inputs_discrete:_data}
                )


            lib.plot.output_dir = plot_filedir
            lib.plot.plot('time', time.time() - start_time)
            lib.plot.plot('train disc cost', d_cost)


            if iteration % ckpt_iter == 0 and iteration > 0:
                # Save Model
                model_saver.save(session, os.path.join(ckpt_filedir, 'checkpoint_{}.ckpt').format(iteration))
                
                ckpt_stop = time.time()
                ckpt_time = round((ckpt_stop - ckpt_start)/60, 1)
                ckpt_start = time.time()
        
                log = "{} Iteration: {}/{} ({}%), Checkpoint Time: {}min, G Cost: {}, D Cost: {}".format(ts.timestamp(), iteration, iters, round(iteration/iters*100.0), ckpt_time, g_cost, d_cost)
                
                csv_log = "{};{};{};{};{};{};{}".format(ts.timestamp()[1:-1], training_data, iteration, iters, ckpt_time, g_cost, d_cost)
                csv_log = csv_log.replace('.', ',')
                
                print(log)
                
                # Write Log to Log File
                with open(log_filepath, 'a') as file:
                    file.write(log + '\n')
            
                # Write Log to CSV File
                with open(csv_filepath, 'a') as file:
                    file.write(csv_log + '\n')
                

            if iteration % 100 == 0:
                lib.plot.flush()


            lib.plot.tick()
    
    print()
    print(ts.timestamp(), "--- TRAINING FINISHED ---")
    print()
