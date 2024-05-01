import os
import random
import glob

DIR_SEPARATOR='/'

# Delete the directory including files and subdirectories
def delete_directory(directory):
    for item in os.listdir(directory):
        item_path = os.path.join(directory, item)
        
        # Check if the item is a file and delete it
        if os.path.isfile(item_path):
            os.remove(item_path)

        # Check if the item is a directory and delete all subdirectories and files recursively
        elif os.path.isdir(item_path):
            delete_directory(item_path)
    
    # Delete the directory
    os.rmdir(directory)


# Create the directory if it doesn't exist
def check_and_create_dir(directory):
    if not os.path.exists(directory):    
        os.makedirs(directory)


# Find the given Password Set in a given Directory
def find_pw_file(directory, pw_set):
    file = glob.glob(directory + pw_set + '*')
    if file:
        return file[0]
    else:
        return None



# Get the File Directory from a given Filepath: | /Tests/RockYou_12x10000/Train_RockYou-loweralphanum_0-12x8000.txt |
# --------------------------------------------- | /Tests/RockYou_12x10000/                                          |
def get_pw_filedir(filepath):
    return filepath[:filepath.rfind(DIR_SEPARATOR)+1]


# Get the File Name from a given Filepath: | /Tests/RockYou_12x10000/Train_RockYou-loweralphanum_0-12x8000.txt |
# ---------------------------------------- |                         Train_RockYou-loweralphanum_0-12x8000     |
def get_pw_filename(filepath):
    return filepath[filepath.rfind(DIR_SEPARATOR)+1:-4]


# Get the Password Set from a given Filepath: | Train_RockYou-loweralphanum_0-12x8000 |
# ------------------------------------------- | Train                                 |
def get_pw_set(file_name):
    return file_name[:file_name.find('_')]


# Get the Password Name from a given Filepath: | Train_RockYou-loweralphanum_0-12x8000 |
# -------------------------------------------- |       RockYou-loweralphanum           |
def get_pw_name(file_name):
    return file_name[file_name.find('_')+1:file_name.rfind('_')]


# Get the Password Length from a given Filepath: | Train_RockYou-loweralphanum_0-12x8000 |
# ---------------------------------------------- |                             0-12      |
def get_pw_length(file_name):
    return file_name[file_name.rfind('_')+1:file_name.rfind('x')]


# Get the Password Length from a given Filepath: | Train_RockYou-loweralphanum_0-12x8000 |
# ---------------------------------------------- |                             0         |
def get_pw_minlength(file_name):
    return int(file_name[file_name.rfind('_')+1:file_name.rfind('-')])


# Get the Password Length from a given Filepath: | Train_RockYou-loweralphanum_0-12x8000 |
# ---------------------------------------------- |                               12      |
def get_pw_maxlength(file_name):
    return int(file_name[file_name.rfind('-')+1:file_name.rfind('x')])


# Get the Password Count from a given Filepath: | Train_RockYou-loweralphanum_0-12x8000 |
# --------------------------------------------- |                                  8000 |
def get_pw_count(file_name):
    return int(file_name[file_name.rfind('x')+1:])


# Add an addition to the password Name: | RockYou-loweralphanum,stringdigit |
# ------------------------------------- | RockYou-loweralphanum-stringdigit |
def add_pw_name(pw_name, addition):
    return pw_name + '-' + addition


# Create the File Name: | Train,RockYou-loweralphanum,0,12,8000 |
# --------------------- | Train_RockYou-loweralphanum_0-12x8000 |
def create_pw_filename(pw_set, pw_name, pw_minlength, pw_maxlength, pw_count):
    return "{}_{}_{}-{}x{}".format(pw_set, pw_name, pw_minlength, pw_maxlength, pw_count)



def create_pw_filedir(filedir, pw_name, pw_length, pw_count):
    return filedir + pw_name + '_' + str(pw_length) + 'x' + str(pw_count) + DIR_SEPARATOR