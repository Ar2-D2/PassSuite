# PassFilter - Password Filter tool
#
# This tool is based on StatsGen, which is part of PACK (Password Analysis and Cracking Kit)
#
# Copyright (C) 2013 Peter Kacherginsky
# All rights reserved.
#
# Please see the attached LICENSE file for additional licensing information.


import os
import string
import time
import multiprocessing as mp
import filesystem as fs
import timestamp as ts

from pathlib import Path
from optparse import OptionParser, OptionGroup

MP_CPU_CORES = 20
MP_CHUNK_SIZE_MB = 100


"""
FILE IMPORT
--------------------------------------------------------------------------------------------------------------------------------
"""

def get_chunk_args(input_pw_filepath, output_pw_filepath, chunk_size):

    file_size = os.path.getsize(input_pw_filepath)

    # Arguments for each chunk (eg. [('input.txt', 0, 32), ('input.txt', 32, 64)])
    chunk_args = []
    
    with open(input_pw_filepath, 'r', errors="replace") as input_pw_file:
        def is_start_of_line(position):
            if position == 0:
                return True
            # Check whether the previous character is EOL
            input_pw_file.seek(position - 1)
            return input_pw_file.read(1) == '\n'

        def get_next_line_position(position):
            # Read the current line till the end
            input_pw_file.seek(position)
            input_pw_file.readline()
            # Return a position after reading the line
            return input_pw_file.tell()

        chunk_start = 0
        # Iterate over all chunks and construct arguments for `process_chunk`
        while chunk_start < file_size:
            chunk_end = min(file_size, chunk_start + chunk_size)

            # Make sure the chunk ends at the beginning of the next line
            while not is_start_of_line(chunk_end):
                chunk_end -= 1

            # Handle the case when a line is too long to fit the chunk size
            if chunk_start == chunk_end:
                chunk_end = get_next_line_position(chunk_end)

            # Save `process_chunk` arguments
            args = (input_pw_filepath, output_pw_filepath, chunk_start, chunk_end)
            chunk_args.append(args)

            # Move to the next chunk and increase process step
            chunk_start = chunk_end

        return chunk_args



"""
PASSWORD ANALYSIS
--------------------------------------------------------------------------------------------------------------------------------
"""
def password_charset(password):

    # Character-set and policy counters
    digit = 0
    lower = 0
    upper = 0
    special = 0

    # Detect Char-Set
    for letter in password:

        if letter in string.digits:
            digit += 1

        elif letter in string.ascii_lowercase:
            lower += 1

        elif letter in string.ascii_uppercase:
            upper += 1

        else:
            special += 1

    # Determine character-set
    if   digit and not lower and not upper and not special: charset = 'numeric'
    elif not digit and lower and not upper and not special: charset = 'loweralpha'
    elif not digit and not lower and upper and not special: charset = 'upperalpha'
    elif not digit and not lower and not upper and special: charset = 'special'

    elif not digit and lower and upper and not special:     charset = 'mixedalpha'
    elif digit and lower and not upper and not special:     charset = 'loweralphanum'
    elif digit and not lower and upper and not special:     charset = 'upperalphanum'
    elif not digit and lower and not upper and special:     charset = 'loweralphaspecial'
    elif not digit and not lower and upper and special:     charset = 'upperalphaspecial'
    elif digit and not lower and not upper and special:     charset = 'specialnum'

    elif not digit and lower and upper and special:         charset = 'mixedalphaspecial'
    elif digit and not lower and upper and special:         charset = 'upperalphaspecialnum'
    elif digit and lower and not upper and special:         charset = 'loweralphaspecialnum'
    elif digit and lower and upper and not special:         charset = 'mixedalphanum'
    else:                                                   charset = 'all'

    return (charset)



def password_simplemask(password):

    simplemask = list()

    # Detect simple and advanced masks
    for letter in password:

        if letter in string.digits:
            if not simplemask or not simplemask[-1] == 'digit': simplemask.append('digit')

        elif letter in string.ascii_lowercase:
            if not simplemask or not simplemask[-1] == 'string': simplemask.append('string')


        elif letter in string.ascii_uppercase:
            if not simplemask or not simplemask[-1] == 'string': simplemask.append('string')

        else:
            if not simplemask or not simplemask[-1] == 'special': simplemask.append('special')

    # String representation of masks
    simplemask_string = ''.join(simplemask) if len(simplemask) <= 3 else 'othermask'


    return simplemask_string


def password_advancedmask(password):

    advancedmask_string = ""

    # Detect simple and advanced masks
    for letter in password:

        if letter in string.digits:
            advancedmask_string += "?d"

        elif letter in string.ascii_lowercase:
            advancedmask_string += "?l"

        elif letter in string.ascii_uppercase:
            advancedmask_string += "?u"

        else:
            advancedmask_string += "?s"


    return advancedmask_string


def password_basiclatin(password):
    for char in password:
        unicode = ord(char)
        if unicode<=31 or unicode>=127:
            return False
    return True
    


"""
FILTERING
--------------------------------------------------------------------------------------------------------------------------------
"""

def filter_by_length(input_pw_filepath, output_pw_filepath, chunk_start, chunk_end, pstep, psteps, p_lock, filter_arg):
    chunk_results = []

    pw_minlength, pw_maxlength = filter_arg

    # Read Passwords from Chunk and filter them
    with open(input_pw_filepath, 'r', errors="replace") as input_file:
        input_file.seek(chunk_start)
        for line in input_file:
            password = line.rstrip("\r\n")
            if len(password) >= pw_minlength and len(password) <= pw_maxlength:
                chunk_results.append(password)  
            chunk_start += len(line)
            if chunk_start >= chunk_end:
                break
            
    with p_lock:
        # Write filtered Passwords to tmp-File
        if chunk_results:
            with open(output_pw_filepath, 'a', newline='\n') as file:            
                file.write("\n".join(chunk_results)+"\n")

        # Increase the Progress Step Counter
        print(ts.timestamp(), " Step: ", pstep.value, "/", psteps, sep='')
        pstep.value += 1

    # Return the Password Count
    return len(chunk_results)
    
            


def filter_by_charset(input_pw_filepath, output_pw_filepath, chunk_start, chunk_end, pstep, psteps, p_lock, filter_arg):
    chunk_results = []

    # Read Passwords from Chunk and filter them
    with open(input_pw_filepath, 'r', errors="replace") as input_file:
        input_file.seek(chunk_start)
        for line in input_file:
            password = line.rstrip("\r\n")
            if password_charset(password) == filter_arg:
                chunk_results.append(password)
            chunk_start += len(line)
            if chunk_start >= chunk_end:
                break
            
    with p_lock:
        # Write filtered Passwords to tmp-File
        if chunk_results:
            with open(output_pw_filepath, 'a', newline='\n') as file:            
                file.write("\n".join(chunk_results)+"\n")

        # Increase the Progress Step Counter
        print(ts.timestamp(), " Step: ", pstep.value, "/", psteps, sep='')
        pstep.value += 1

    # Return the Password Count
    return len(chunk_results)                 



def filter_by_simplemask(input_pw_filepath, output_pw_filepath, chunk_start, chunk_end, pstep, psteps, p_lock, filter_arg):
    chunk_results = []

    # Read Passwords from Chunk and filter them
    with open(input_pw_filepath, 'r', errors="replace") as input_file:
        input_file.seek(chunk_start)
        for line in input_file:
            password = line.rstrip("\r\n")
            if password_simplemask(password) == filter_arg:
                chunk_results.append(password)
            chunk_start += len(line)
            if chunk_start >= chunk_end:
                break
            
    with p_lock:
        # Write filtered Passwords to tmp-File
        if chunk_results:
            with open(output_pw_filepath, 'a', newline='\n') as file:            
                file.write("\n".join(chunk_results)+"\n")

        # Increase the Progress Step Counter
        print(ts.timestamp(), " Step: ", pstep.value, "/", psteps, sep='')
        pstep.value += 1

    # Return the Password Count
    return len(chunk_results)
 
 
def filter_by_advancedmask(input_pw_filepath, output_pw_filepath, chunk_start, chunk_end, pstep, psteps, p_lock, filter_arg):
    chunk_results = []

    # Read Passwords from Chunk and filter them
    with open(input_pw_filepath, 'r', errors="replace") as input_file:
        input_file.seek(chunk_start)
        for line in input_file:
            password = line.rstrip("\r\n")
            if password_advancedmask(password) == filter_arg:
                chunk_results.append(password)
            chunk_start += len(line)
            if chunk_start >= chunk_end:
                break
            
    with p_lock:
        # Write filtered Passwords to tmp-File
        if chunk_results:
            with open(output_pw_filepath, 'a', newline='\n') as file:            
                file.write("\n".join(chunk_results)+"\n")

        # Increase the Progress Step Counter
        print(ts.timestamp(), " Step: ", pstep.value, "/", psteps, sep='')
        pstep.value += 1

    # Return the Password Count
    return len(chunk_results)






def filter_by_basiclatin(input_pw_filepath, output_pw_filepath, chunk_start, chunk_end, pstep, psteps, p_lock, filter_arg):
    chunk_results = []

    # Read Passwords from Chunk and filter them
    with open(input_pw_filepath, 'r', errors="replace") as input_file:
        input_file.seek(chunk_start)
        for line in input_file:
            password = line.rstrip("\r\n")
            if password_basiclatin(password):
                chunk_results.append(password)
            chunk_start += len(line)
            if chunk_start >= chunk_end:
                break
    
    with p_lock:
        # Write filtered Passwords to tmp-File
        if chunk_results:
            with open(output_pw_filepath, 'a', newline='\n') as file:            
                file.write("\n".join(chunk_results)+"\n")

        # Increase the Progress Step Counter
        print(ts.timestamp(), " Step: ", pstep.value, "/", psteps, sep='')
        pstep.value += 1

    # Return the Password Count
    return len(chunk_results)
   




"""
MAIN
--------------------------------------------------------------------------------------------------------------------------------
"""

if __name__ == "__main__":

    # Parse Command Line Arguments
    parser = OptionParser()
    filters = OptionGroup(parser, "Password Filters")
    filters.add_option("-l", dest="pw_length", help="Password Length (min-max)", metavar="0-20")            # -l: Length
    filters.add_option("-c", dest="pw_charset", help="Password Character-Set", metavar="loweralpha")        # -c: Character-Set
    filters.add_option("-s", dest="pw_simplemask",help="Password Simple Mask", metavar="stringdigit")       # -s: Simple Mask
    filters.add_option("-a", dest="pw_advancedmask",help="Password Advanced Mask", metavar="?l?l?l?l?l?l")  # -a: Advanced Mask
    filters.add_option("-b", action = "store_true", dest="pw_basiclatin", help="Basic Latin")               # -b: Basic Latin
    filters.add_option("-p", dest="mp_cpu_cores", type="int", help="Multiprocessing CPU Cores")             # -m: Multiprocessing CPU Cores
    filters.add_option("-m", dest="mp_chunk_size_mb", type="int", help="Multiprocessing Chunk Size (MB)")   # -m: Multiprocessing Chunk Size (MB)
    parser.add_option_group(filters)
    (options, args) = parser.parse_args()



    # Set Multiprocessing Parameters CPU Cores & Chunk Size (MB)
    mp_cpu_cores = MP_CPU_CORES             # Default Value
    mp_chunk_size_mb = MP_CHUNK_SIZE_MB     # Default Value

    if not options.mp_cpu_cores == None:
        mp_cpu_cores = options.mp_cpu_cores

    if not options.mp_chunk_size_mb == None:
        mp_chunk_size_mb = options.mp_chunk_size_mb

    mp_chunk_size = mp_chunk_size_mb * 1000000

    # Get Filepath, Password Count and Password Length of the Input Password List
    if len(args) != 1:
        parser.error("no passwords file specified")
        exit(1) 
    input_pw_filepath = args[0]
    
    # Get Password Attributes
    input_pw_filedir = fs.get_pw_filedir(input_pw_filepath)
    input_pw_filename = fs.get_pw_filename(input_pw_filepath)
    input_pw_name = fs.get_pw_name(input_pw_filename)
    input_pw_length = fs.get_pw_length(input_pw_filename)
    input_pw_minlength = fs.get_pw_minlength(input_pw_filename)
    input_pw_maxlength = fs.get_pw_maxlength(input_pw_filename)
    input_pw_count = fs.get_pw_count(input_pw_filename)

    
    # Output Password List
    output_pw_name = input_pw_name
    output_pw_minlength = input_pw_minlength
    output_pw_maxlength = input_pw_maxlength
    output_pw_count = 0
    

    # Temp Password List
    tmp_pw_filepath = input_pw_filepath +  "-pwfilter.temp"
    if Path(tmp_pw_filepath).exists():
        os.remove(tmp_pw_filepath)


    # Filter Passwords by Length
    if not options.pw_length == None:
        output_pw_minlength, output_pw_maxlength = options.pw_length.split('-')
        output_pw_minlength = int(output_pw_minlength)
        output_pw_maxlength = int(output_pw_maxlength)
        
        filter = filter_by_length
        filter_arg = (output_pw_minlength, output_pw_maxlength)


    # Filter Passwords by Char-Set
    elif not options.pw_charset == None: 
        filter = filter_by_charset
        filter_arg = options.pw_charset
        output_pw_name = input_pw_name + '-c=' + options.pw_charset

    # Filter Passwords by Simple Mask
    elif not options.pw_simplemask == None: 
        filter = filter_by_simplemask
        filter_arg = options.pw_simplemask
        output_pw_name = input_pw_name + '-s=' + options.pw_simplemask

    # Filter Passwords by Advanced Mask
    elif not options.pw_advancedmask == None: 
        filter = filter_by_advancedmask
        filter_arg = options.pw_advancedmask
        output_pw_advancedmask = options.pw_advancedmask.replace('?', '')
        output_pw_name = input_pw_name + '-a=' + output_pw_advancedmask
        output_pw_minlength = len(output_pw_advancedmask)
        output_pw_maxlength = len(output_pw_advancedmask)
    
    # Filter Passwords by Basic Latin
    elif not options.pw_basiclatin == None: 
        filter = filter_by_basiclatin
        filter_arg = None
        output_pw_name = input_pw_name + "-BL"   


    with mp.Manager() as p_manager:
        
        # Progress Step Counter
        p_lock = p_manager.Lock()
        pstep = p_manager.Value('i', 1)


        # Get Chunk Args
        chunk_args = get_chunk_args(input_pw_filepath, tmp_pw_filepath, mp_chunk_size)
        psteps = len(chunk_args)


        # Prepare Process Arguments
        p_args = []
        for args in chunk_args:
            #a = args + (pstep, p_lock)
            p_args.append(args + (pstep, psteps, p_lock, filter_arg))
        

        # Start filtering Passwords
        print()
        print(ts.timestamp(), "--- FILTERING STARTED ---")
        print(ts.timestamp(), "Filter:", filter_arg)
        print(ts.timestamp(), "Filtering Passwords:", input_pw_filename)
        print(ts.timestamp(), "Cores:", mp_cpu_cores, "/ Chunk Size:", mp_chunk_size_mb, "MB")
        print(ts.timestamp(), "Number of Steps:", psteps)
        print()
        
        start_time = time.time()
        chunk_results_pw_count = []
        with mp.Pool(mp_cpu_cores) as p_pool:
            chunk_results_pw_count = p_pool.starmap_async(filter, p_args)
            output_pw_count = sum(chunk_results_pw_count.get())

        stop_time = round((time.time() - start_time)/60, 1)
        #stop_time = round((time.time() - start_time), 1)

    print()
    print(ts.timestamp(), "--- FILTERING FINISHED ---")
    print(ts.timestamp(), "Execution Time (min):", stop_time)

    # Check if there are any Output Passwords and save them to Disc
    if output_pw_count:

        # Get and Create Output Password Attributes
        output_pw_filename = fs.create_pw_filename('Full', output_pw_name, output_pw_minlength, output_pw_maxlength, output_pw_count)
        output_pw_filedir = os.path.join(input_pw_filedir, output_pw_filename[5:])
        output_pw_filepath = os.path.join(output_pw_filedir, output_pw_filename + '.txt')

        # Create Output Directory and move/rename the tmp-File
        fs.check_and_create_dir(output_pw_filedir)
        if Path(output_pw_filepath).exists():
            os.remove(output_pw_filepath)
        os.rename(tmp_pw_filepath, output_pw_filepath)                
        print(ts.timestamp(), "Passwords filtered:", output_pw_count)              
        print(ts.timestamp(), "Passwords saved to:", output_pw_filepath)
        print()
        
    else:
        if Path(tmp_pw_filepath).exists():
            os.remove(tmp_pw_filepath)
        print(ts.timestamp(), "No Passwords found with this Filter")
        print()

