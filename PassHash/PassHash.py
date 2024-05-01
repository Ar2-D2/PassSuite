import os, sys
import filesystem as fs
import timestamp as ts
import time
import hashlib as h


# Main function
def main():
    # Check if the password list file is provided as a command-line argument
    if len(sys.argv) != 2:
        print("Usage: python script_name.py passwordlist_file.txt")
        sys.exit(1)
    
    input_pw_filepath = sys.argv[1]
    input_pw_filedir = fs.get_pw_filedir(input_pw_filepath)
    input_pw_filename = fs.get_pw_filename(input_pw_filepath)
    input_pw_set = fs.get_pw_set(input_pw_filename)
    input_pw_name = fs.get_pw_name(input_pw_filename)
    input_pw_minlength = fs.get_pw_minlength(input_pw_filename)
    input_pw_maxlength = fs.get_pw_maxlength(input_pw_filename)
    input_pw_count = fs.get_pw_count(input_pw_filename)
 
    output_pw_filename = fs.create_pw_filename("Hash" + input_pw_set[4:], input_pw_name, input_pw_minlength, input_pw_maxlength, input_pw_count)
    output_pw_filepath = os.path.join(input_pw_filedir, output_pw_filename + '.txt')


   
    print()
    print(ts.timestamp(), "--- HASHING PASSWORDS STARTED ---")
    print(ts.timestamp(), "Input Passwords:", input_pw_filename)
    
    start_time = time.time()  

    
    # Write Fake Password Hashes to Disc
    input_line_count = 0
    ckpt = int(input_pw_count/100)

    with open(input_pw_filepath, 'r', errors="replace") as input_file:
        with open(output_pw_filepath, 'w', newline='\n') as tmp_file:         
            for line in input_file:
                sha1 = h.sha1()
                sha1.update(str(line.rstrip("\r\n")).encode('utf-8'))
                output = sha1.hexdigest().upper()
                tmp_file.write(output + '\n')
                input_line_count += 1
                if input_line_count % ckpt == 0:
                    print("{} Passwords processed: {}%".format(ts.timestamp(), int(input_line_count/ckpt)))            
                



    print()
    print(ts.timestamp(), "--- HASHING PASSWORDS FINISHED ---")
    print(ts.timestamp(), "Input Passwords:", input_line_count)
    print(ts.timestamp(), "Passwords saved to:", output_pw_filepath)
    print(ts.timestamp(), "Execution Time: ", round((time.time() - start_time)/60 ,1), "min", sep='')
    print()

if __name__ == "__main__":
    main()
