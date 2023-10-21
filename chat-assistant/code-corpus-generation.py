import os

def merge_py_files(directory, output_file):
    with open(output_file, 'a') as outfile:
        for root, dirs, files in os.walk(directory):
            for filename in files:
                if filename.endswith(".py"):
                    with open(os.path.join(root, filename), 'r') as infile:
                        outfile.write("Here is the code content for " + filename + ":\n")
                        outfile.write(infile.read())
                        outfile.write("\n\n")

# --------- text ------------
with open('pyhealth-code.txt', 'w') as f:
    f.write("The code content for pyhealth")
    
merge_py_files('../pyhealth', 'pyhealth-code.txt')
merge_py_files('../examples', 'pyhealth-code.txt')
merge_py_files('/home/chaoqiy2/pyhealth-tutorial', 'pyhealth-code.txt')