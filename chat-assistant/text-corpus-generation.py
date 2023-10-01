import re
import os

def parse_text(text):
    pattern = r"(pyhealth\.\w+\.\w+)[\s\S]*?===\n\n([\s\S]*?)\n\n.. autoclass"
    match = re.search(pattern, text)
    if match:
        # Remove any leading/trailing whitespace and replace multiple spaces/newlines with a single space
        description = ' '.join(match.group(2).split())
        return match.group(1) + ": " + description
    else:
        return "No match found"
    
# --------- text ------------
with open('pyhealth-text.txt', 'w') as f:
    f.write("The text content for pyhealth")
    
with open('pyhealth-text.txt', 'a') as f:
    # for readme.rst, write readme.rst into the pyhealth-text.txt file
    f.write("Here is the readme.rst file content: \n\n")
    readme = open('../README.rst', 'r')
    f.write(readme.read())

    # for docs/api/data
    f.write("Here is the docs/api/data content: \n\n")
    for path in os.listdir('../docs/api/data'):
        content = parse_text(open(f'../docs/api/data/{path}', 'r').read())
        f.write(content + "\n")
        f.write("\n")
    
    # for docs/api/datasets
    f.write("Here is the docs/api/datasets content: \n\n")
    for path in os.listdir('../docs/api/datasets'):
        if "Dataset" not in path: continue
        content = parse_text(open(f'../docs/api/datasets/{path}', 'r').read())
        f.write(content + "\n")
        f.write("\n")
        
    # for docs/api/models
    f.write("Here is the docs/api/models content: \n\n")
    for path in os.listdir('../docs/api/models'):
        content = parse_text(open(f'../docs/api/models/{path}', 'r').read())
        f.write(content + "\n")
        f.write("\n")
        
    # for dos/about.rst
    f.write("Here is the docs/about.rst content: \n\n")
    about = open('../docs/about.rst', 'r')
    f.write(about.read())
    f.write("\n")
    
    # for docs/advance_tutorial.rst
    f.write("Here is the docs/advance_tutorials.rst content: \n\n")
    advance_tutorial = open('../docs/advance_tutorials.rst', 'r')
    f.write(advance_tutorial.read())
    f.write("\n")
    
    # for docs/index.rst
    f.write("Here is the docs/index.rst content: \n\n")
    index = open('../docs/index.rst', 'r')
    f.write(index.read())
    f.write("\n")
    
    # for docs/live.rst
    f.write("Here is the docs/live.rst content: \n\n")
    live = open('../docs/live.rst', 'r')
    f.write(live.read())
    f.write("\n")
    
    # for docs/log.rsts
    f.write("Here is the docs/log.rst content: \n\n")
    log = open('../docs/log.rst', 'r')
    f.write(log.read())
    f.write("\n")
    
    # for docs/requirements.txt
    f.write("Here is the docs/requirements.txt content: \n\n")
    requirements = open('../docs/requirements.txt', 'r')
    f.write(requirements.read())
    f.write("\n")
    
    # for docs/tutorials.rst
    f.write("Here is the jupyter docs/tutorials.rst content: \n\n")
    tutorials = open('../docs/tutorials.rst', 'r')
    f.write(tutorials.read())
    f.write("\n")
    
    

    
    
    
    