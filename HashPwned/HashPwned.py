import time
import numpy as np
import collections
import numpy as np
import re, os, sys
import gc, random

import multiprocessing as mp
import filesystem as fs
import timestamp as ts

from pathlib import Path

MP_CPU_CORES = 8
P_CKPT = 100000
SHA1_LENGTH = 40
TEST_HASH_FILEDIR = 'D:\\gazdaa93\\04-Passwords\\HaveIBeenPwned\\Buckets'



def hashcompare(fake_hash_5, fake_hashes_35, p_step, p_steps, p_lock):
    # Comparing Test Password Samples with Fake Passwords 
    match_hash_count = 0
    test_hash_filepath = os.path.join(TEST_HASH_FILEDIR, 'Hash_' + str(fake_hash_5) + '.txt')

    with open(test_hash_filepath, 'r') as test_hash_file:
        test_hash_list = list(map(lambda x: x[:35], test_hash_file.readlines()))
        
        for fake_hash_35 in fake_hashes_35:
            if fake_hash_35 in test_hash_list:
                    match_hash_count += 1
    
    with p_lock:
        # Increase the Progress Step Counter
        if p_step.value % P_CKPT == 0:
            print("{} Step: {} / {} ({}%)".format(ts.timestamp(), p_step.value, p_steps, round(p_step.value*100/p_steps)))
        p_step.value += 1

    return  match_hash_count  




if __name__ == "__main__":
    
    # Arguments  
    fake_hash_filepath = sys.argv[1]
    
    # Fake Password Attributes
    fake_hash_filedir = fs.get_pw_filedir(fake_hash_filepath)
    fake_hash_filename = fs.get_pw_filename(fake_hash_filepath)
    fake_hash_name = fs.get_pw_name(fake_hash_filename)
    fake_hash_minlength = fs.get_pw_minlength(fake_hash_filename)
    fake_hash_maxlength = fs.get_pw_maxlength(fake_hash_filename)
    fake_hash_count = fs.get_pw_count(fake_hash_filename)

    
    print()
    print(ts.timestamp(), "--- HASH COMPARING STARTED ---")
    print(ts.timestamp(), "Fake Hashes:", fake_hash_filepath)
    print(ts.timestamp(), "Cores:", MP_CPU_CORES)
    start_time = time.time()    
       
    
    # Load Fake Passwords
    print(ts.timestamp(), "Loading Fake Hashes")
    #fake_hash_ndarray = np.genfromtxt(fake_hash_filepath, dtype=str, delimiter=SHA1_LENGTH, comments=None)
    fake_hash_list=list(map(lambda x: x.strip("\n"), open(fake_hash_filepath, 'r').readlines()))
    
    fake_hash_dict = {}
    for fake_hash in fake_hash_list:
        fake_hash_5 = fake_hash[:5]
        fake_hash_35 = fake_hash[5:]
        if fake_hash_5 in fake_hash_dict.keys():
            fake_hash_dict[fake_hash_5] += [fake_hash_35]
        else:
            fake_hash_dict.update({fake_hash_5 : [fake_hash_35]})
    
    fake_hash_list = list(fake_hash_dict.items())
    print(ts.timestamp(), "Fake Hashes loaded")    

    

    with mp.Manager() as p_manager:
        
        # Progress Step Counter
        p_lock = p_manager.Lock()
        p_step = p_manager.Value('i', 1)
        p_steps = len(fake_hash_list)
        print(ts.timestamp(), "Steps:", p_steps)

        # Prepare Process Arguments
        p_args = []
        for args in fake_hash_list:
            #a = args + (pstep, p_lock)
            p_args.append(args + (p_step, p_steps, p_lock))
               
        start_time = time.time()
        match_pw_count = []
        with mp.Pool(MP_CPU_CORES) as p_pool:
            hashcompare_result = p_pool.starmap(hashcompare, p_args)
            #match_hash_count = sum(hashcompare_result.get())
            match_hash_count = sum(hashcompare_result)


    #match_pw_count = len(match_pw_list)
    match_hash_percent = int(round((match_hash_count * 100) / fake_hash_count)) 
    print(ts.timestamp(), "Hashes compared")
   
    print()
    print(ts.timestamp(), "--- COMPARING FINISHED ---")
    print(ts.timestamp(), "Matches: {} ({}%)".format(match_hash_count, match_hash_percent))
    print(ts.timestamp(), "Comparing Time (min):", round((time.time() - start_time)/60, 1))
    print()

    
