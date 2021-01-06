set -x

# NOTE 1: for actual applications, you may want to change:
#   train batch size to 4-8 (depending on the GPU you are using)
#   eval batch size to 16-32 (depending on the GPU you are using)
#   accumulation steps to 4-8 (depending on the GPU you are using)
#   num train epochs to 10-15

# NOTE 2: do not forget to activate your conda environment!

# NOTE 3: these experiments run on toy datasets, the output metrics
#   do not have any actual comparison value.

python download.py --model='bert-base-uncased'
python download.py --model='general_character_bert'
python download.py --model='medical_character_bert'

# NER with BERT
python main.py \
    --task='sequence_labelling' \
    --embedding='bert-base-uncased' \
    --do_lower_case \
    --do_train \
    --do_predict

# NER with medical CharacterBERT
python main.py \
    --task='sequence_labelling' \
    --embedding='medical_character_bert' \
    --do_lower_case \
    --do_train \
    --do_predict

# Sentiment Analysis with general CharacterBERT
python main.py \
    --task='classification' \
    --embedding='general_character_bert' \
    --do_lower_case \
    --do_train \
    --do_predict

set +x
