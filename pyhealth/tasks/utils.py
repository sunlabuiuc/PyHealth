import re

def clean_text(x):
    y = re.sub(r'\[(.*?)\]', '', x)  # remove de-identified brackets"
    y = re.sub(r'[0-9]+\.', '', y)  # remove 1.2. since the segmenter segments based on this
    y = re.sub(r'dr\.', 'doctor', y)
    y = re.sub(r'm\.d\.', 'md', y)
    y = re.sub(r'admission date:', '', y)
    y = re.sub(r'discharge date:', '', y)
    y = re.sub(r'--|__|==', '', y)
    return y


def embedding_generator(term, model='clinicalbert'):
    pass