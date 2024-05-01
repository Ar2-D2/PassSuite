import os, sys
import random
import filesystem as fs
import timestamp as ts
import time



# Save passwords to file


# Main function
def main():
    # Check if the password list file is provided as a command-line argument
    if len(sys.argv) != 2:
        print("Usage: python script_name.py passwordlist_file.txt")
        sys.exit(1)
    
    input_pw_filepath = sys.argv[1]
    input_pw_filedir = fs.get_pw_filedir(input_pw_filepath)
    input_pw_filename = fs.get_pw_filename(input_pw_filepath)
    input_pw_name = fs.get_pw_name(input_pw_filename)
    input_pw_minlength = fs.get_pw_minlength(input_pw_filename)
    input_pw_maxlength = fs.get_pw_maxlength(input_pw_filename)
    input_pw_count = fs.get_pw_count(input_pw_filename)
   
    output_pw_filedir = os.path.join(input_pw_filedir, input_pw_filename[5:])
    fs.check_and_create_dir(output_pw_filedir)

    
    print()
    print(ts.timestamp(), "--- SPLITTING TEST AND TRAINING PASSWORDS STARTED ---")
    print(ts.timestamp(), "Input Passwords:", input_pw_filename)
    
    start_time = time.time()

    # Load passwords from file
    with open(input_pw_filepath, 'r') as file:
        input_pw = file.read().splitlines()
    
    print(ts.timestamp(), "Input Passwords loaded from file")
    

    # Shuffle the list of passwords
    for i in range(3):
        random.shuffle(input_pw)
        print(ts.timestamp(), "Input Passwords shuffled:", i+1)
    
    # Calculate the number of passwords for each file
    train_pw_count = int(0.8 * input_pw_count)
    test_pw_count = input_pw_count - train_pw_count
    
    # Split the list of passwords
    train_pw = input_pw[:train_pw_count]
    print(ts.timestamp(), "Splitting Training Passwords completed")

    test_pw = input_pw[train_pw_count:]
    print(ts.timestamp(), "Splitting Test Passwords completed")


    # Save passwords to files
    train_pw_filename = fs.create_pw_filename("Train", input_pw_name, input_pw_minlength, input_pw_maxlength, train_pw_count)
    test_pw_filename = fs.create_pw_filename("Test", input_pw_name, input_pw_minlength, input_pw_maxlength, test_pw_count)
    train_pw_filepath = os.path.join(output_pw_filedir, train_pw_filename + '.txt')
    test_pw_filepath = os.path.join(output_pw_filedir, test_pw_filename + '.txt')
    
    # Write Training Passwords to Disc
    with open(train_pw_filepath, 'w', newline='\n') as file:
        for password in train_pw:
            file.write(password + '\n')
    print(ts.timestamp(), "Training Passwords written to disc")

    # Write Test Passwords to Disc
    with open(test_pw_filepath, 'w', newline='\n') as file:
        for password in test_pw:
            file.write(password + '\n')
    print(ts.timestamp(), "Test Passwords written to disc")

    stop_time = time.time()
    print()
    print(ts.timestamp(), "--- SPLITTING TEST AND TRAINING PASSWORDS FINISHED ---")
    print(ts.timestamp(), "Input Passwords:", input_pw_filename)
    print(ts.timestamp(), "Training Passwords saved to:", train_pw_filepath)
    print(ts.timestamp(), "Test Passwords saved to:", test_pw_filepath)
    print(ts.timestamp(), "Execution Time: ", round((stop_time - start_time)/60 ,1), "min", sep='')
    print()

if __name__ == "__main__":
    main()
