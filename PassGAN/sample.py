import os
import time
import math
import pickle
import argparse


import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()

import numpy as np
import tflib as lib
import tflib.ops.linear
import tflib.ops.conv1d
import utils
import models
import data.timestamp as ts
import data.filesystem as fs

from pathlib import Path

def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument('--training-data', '-t',
                        required=True,
                        dest='training_data',
                        help='Path to training data directory')
    
    parser.add_argument('--checkpoint', '-c',
                        type=int,
                        default=None,
                        dest='ckpt',
                        help='Model checkpoint to use for sampling.')
                        
    parser.add_argument('--iteration_start', '-start',
                        type=int,
                        default=None,
                        dest='iteration_start',
                        help='The iteration to start with')
                        
    parser.add_argument('--iteration_end', '-end',
                        type=int,
                        default=None,
                        dest='iteration_end',
                        help='The number of training iterations')
                        
    parser.add_argument('--iteration_step', '-step',
                        type=int,
                        default=None,
                        dest='iteration_step',
                        help='Sample and Evaluate Fake Passwords after this many iterations')

    parser.add_argument('--fake-samples', '-f',
                        type=int,
                        default=10000000,
                        dest='fake_samples',
                        help='The number of fake samples (default: 10000000)')

    parser.add_argument('--batch-size', '-b',
                        type=int,
                        default=64,
                        dest='batch_size',
                        help='Batch size (default: 64).')
        
    parser.add_argument('--layer-dim', '-d',
                        type=int,
                        default=128,
                        dest='layer_dim',
                        help='The hidden layer dimensionality for the generator. Use the same value that you did for training (default: 128)')
                        
    parser.add_argument('--restart', '-r',
                        action="store_true",
                        dest='restart',
                        help='Delete old Checkpoints and Restart from Scratch (default: 10)')
    
    args = parser.parse_args()


    return args





if __name__ == "__main__":

    # Arguments
    args = parse_args() 
    training_data = args.training_data
    batch_size = args.batch_size
    layer_dim = args.layer_dim
    fake_samples_count = args.fake_samples
    restart = args.restart
    
    if not args.ckpt == None:
        iteration_start = args.ckpt
        iteration_end = args.ckpt
        iteration_step = 1
        
    if not args.iteration_start == None and not args.iteration_end == None and not args.iteration_step == None:
        iteration_start = args.iteration_start
        iteration_end = args.iteration_end
        iteration_step = args.iteration_step
    
    
    # Directories
    root_dir = '/notebooks/02-Data/' + training_data + '/'
    ckpt_filedir = root_dir + 'Ckpt/'
    charmap_filedir = root_dir + 'Charmap/'
    fake_pw_filedir = root_dir + 'Fake/'

    # Testing Password Set
    test_pw_filepath = fs.find_pw_file(root_dir, "Test")
    test_pw_filedir = fs.get_pw_filedir(test_pw_filepath)
    test_pw_filename = fs.get_pw_filename(test_pw_filepath)
    test_pw_name = fs.get_pw_name(test_pw_filename)
    test_pw_length = fs.get_pw_length(test_pw_filename)
    test_pw_minlength = fs.get_pw_minlength(test_pw_filename)
    test_pw_maxlength = fs.get_pw_maxlength(test_pw_filename)
    test_pw_count = fs.get_pw_count(test_pw_filename)
    
       
    # Evaluation Log
    log_filename = fs.create_pw_filename('Sample', test_pw_name, test_pw_minlength, test_pw_maxlength, fake_samples_count)
    log_filepath = os.path.join(root_dir, log_filename + '.log')
    csv_filepath = os.path.join(root_dir, log_filename + '.csv')
    
    
    # If Restart delete previous files and directories
    if restart:
        if Path(fake_pw_filedir).exists():
            fs.delete_directory(fake_pw_filedir)
        if Path(log_filepath).exists():
            os.remove(log_filepath)
        if Path(csv_filepath).exists():
            os.remove(csv_filepath)
    
	# Initialize Fake Passwords Directory
    if not os.path.isdir(fake_pw_filedir):
        os.makedirs(fake_pw_filedir)

    # Initialize Eval CSV
    with open(csv_filepath, 'a') as file:
        sample_csv_header= "Time;PW Set;Fake Samples Count;Iteration;Total Iterations;Checkpoint Time (min);Unique Fake PW Count;Unique Fake PW %;Match Count;Match %"
        file.write(sample_csv_header + '\n')
    
    # Check Charmap
    if not os.path.exists(os.path.join(charmap_filedir, 'charmap.pickle')):
        parser.error('charmap.pickle doesn\'t exist in {}, are you sure that directory is a trained model directory'.format(args.input_dir))

    if not os.path.exists(os.path.join(charmap_filedir, 'inv_charmap.pickle')):
        parser.error('inv_charmap.pickle doesn\'t exist in {}, are you sure that directory is a trained model directory'.format(args.input_dir))
    
    # Get Charmap
    with open(os.path.join(charmap_filedir, 'charmap.pickle'), 'rb') as f:
        charmap = pickle.load(f)

    with open(os.path.join(charmap_filedir, 'inv_charmap.pickle'), 'rb') as f:
        inv_charmap = pickle.load(f)

    # Generator
    fake_inputs = models.Generator(batch_size, test_pw_maxlength, layer_dim, len(charmap))

    with tf.Session() as session:

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

            
    
        print()
        print( ts.timestamp(), "--- SAMPLING STARTED ---")
        print()
        
        # Load Test Passwords
        print(ts.timestamp(), "Loading Test Passwords")
        test_pw_list, test_pw_charmap, test_pw_inv_charmap = utils.load_dataset(test_pw_filepath, test_pw_maxlength)
        test_pw_ndarray = np.array(test_pw_list)
        test_pw_list = None
        print(ts.timestamp(), "Test Passwords loaded")
        print()
        
        for iteration in range(iteration_start, iteration_end+1, iteration_step):
            start_time = time.time()
            print("{} Checkpoint: {}/{} ({}%)".format(ts.timestamp(), iteration, iteration_end, round(iteration/iteration_end*100.0)))
            
            # Restore Checkpoint
            ckpt_filepath = os.path.join(ckpt_filedir, 'checkpoint_{}.ckpt').format(iteration)
            fake_pw_filepath = fs.find_pw_file(fake_pw_filedir, "Fake#{}".format(iteration))
            saver = tf.train.Saver()
            saver.restore(session, ckpt_filepath)

            if fake_pw_filepath:
                fake_pw_filename = fs.get_pw_filename(fake_pw_filepath)
                fake_pw_count = fs.get_pw_count(fake_pw_filename)
                fake_pw_percent = round(fake_pw_count*100/math.ceil(fake_samples_count / batch_size)*batch_size)

            else:
                # Generate Fake Password Samples
                print(ts.timestamp(), "Generating Fake Password Samples")
                sampling_start = time.time()
                samples = []
                
                for i in range(math.ceil(fake_samples_count / batch_size)):
                    samples.extend(generate_samples())
                
                # Deduplicate Fake Password Samples
                fake_pw_list = list(dict.fromkeys(samples))
                fake_pw_count = len(fake_pw_list)
                fake_pw_percent = round(((fake_pw_count*100)/(math.ceil(fake_samples_count / batch_size)*batch_size)),2)
                samples = None
                
                # Write Fake Passwords to Disc
                fake_pw_filename = fs.create_pw_filename('Fake#' +  str(iteration), test_pw_name, test_pw_minlength, test_pw_maxlength, fake_pw_count)
                fake_pw_filepath = os.path.join(fake_pw_filedir, fake_pw_filename + '.txt')
                
                if Path(fake_pw_filepath).exists():
                    os.remove(fake_pw_filepath)
                
                with open(fake_pw_filepath, 'w') as f:
                    for fake_pw in fake_pw_list:
                        fake_pw = "".join(fake_pw).replace('´', '')
                        f.write(fake_pw + "\n")
                
                fake_pw_list = None
                print(ts.timestamp(), "Fake Password Samples written to Disc. Sampling Time (min):", round((time.time() - sampling_start)/60, 1))
            																														 
			
            # Load Fake Passwords
            print(ts.timestamp(), "Loading Fake Passwords")
            fake_pw_list, fake_pw_charmap, fake_pw_inv_charmap = utils.load_dataset(fake_pw_filepath, test_pw_maxlength)
            fake_pw_ndarray = np.array(fake_pw_list)
            fake_pw_list = None
            print(ts.timestamp(), "Fake Passwords loaded")
            
                                 
            # Comparing Fake Password Samples with Test Passwords
            print(ts.timestamp(), "Comparing Passwords")
            pw_match_count = len(np.intersect1d(fake_pw_ndarray, test_pw_ndarray, assume_unique=False))
            fake_pw_ndarray = None
            pw_match_percent = round((pw_match_count * 100) / fake_pw_count, 2)              
            print(ts.timestamp(), "Passwords compared")          
            
            
            # Write and Print Log
            iteration_time = round((time.time() - start_time)/60, 1)
            start_time = time.time()
            
            sample_log = "{} Checkpoint: {}/{} ({}%), Checkpoint Time: {}min, Matches: {} ({}%)".format(ts.timestamp(), iteration, iteration_end, round(iteration/iteration_end*100.0), iteration_time, pw_match_count, pw_match_percent)
            sample_csv = "{};{};{};{};{};{};{};{};{};{}".format(ts.timestamp()[1:-1], training_data, fake_samples_count, iteration, iteration_end, iteration_time, fake_pw_count, fake_pw_percent, pw_match_count, pw_match_percent)
            sample_csv = sample_csv.replace('.', ',')
            
            # Write Evaluation Log to Eval Log File
            with open(log_filepath, 'a') as file:
                file.write(sample_log + '\n')
                
            # Write Evaluation Log to Eval CSV File
            with open(csv_filepath, 'a') as file:
                file.write(sample_csv + '\n')


            print(sample_log)
            print()
            
                    
        print(ts.timestamp(), "--- SAMPLING FINISHED ---")
        print()
        
        
                        
        
