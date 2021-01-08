import pytorch_pretrained_bert
from pytorch_pretrained_bert import PYTORCH_PRETRAINED_BERT_CACHE
from pytorch_pretrained_bert.file_utils import PYTORCH_PRETRAINED_BERT_CACHE
from pytorch_pretrained_bert.modeling import BertModel, BertConfig, WEIGHTS_NAME, CONFIG_NAME
from pytorch_pretrained_bert.tokenization import BertTokenizer
from pytorch_pretrained_bert.optimization import BertAdam
from pyhealth.utils.characterbertmain.modeling.character_bert import CharacterBertModel
from pyhealth.utils.characterbertmain.utils.character_cnn import CharacterIndexer
import os

def get_embedding(embed_type):
    if embed_type == 'BioBERT':
        model_loc = './auxiliary/pretrained_bert_tf/biobert_pretrain_output_all_notes_150000/'
        tokenizer = BertTokenizer.from_pretrained(model_loc, do_lower_case=True)
        cache_dir = os.path.join(PYTORCH_PRETRAINED_BERT_CACHE, 'distributed_{}'.format(-1))
        model = BertModel.from_pretrained(model_loc, cache_dir=cache_dir)
        indexer = None
    elif embed_type == 'BERT':
        model_loc = './auxiliary/pretrained_bert_tf/bert_pretrain_output_all_notes_150000/'
        tokenizer = BertTokenizer.from_pretrained(model_loc, do_lower_case=True)
        cache_dir = os.path.join(PYTORCH_PRETRAINED_BERT_CACHE, 'distributed_{}'.format(-1))
        model = BertModel.from_pretrained(model_loc, cache_dir=cache_dir)
        indexer = None
    elif embed_type == 'CharBERT':
        model_loc = './auxiliary/pretrained_character_bert/general_character_bert/'
        model = CharacterBertModel.from_pretrained(model_loc)
        tokenizer = BertTokenizer.from_pretrained('./auxiliary/pretrained_bert_tf/bert_pretrain_output_all_notes_150000/')
        indexer = CharacterIndexer() 
    elif embed_type == 'BioCharBERT':
        model_loc = './auxiliary/pretrained_character_bert/medical_character_bert/'
        model = CharacterBertModel.from_pretrained(model_loc)
        tokenizer = BertTokenizer.from_pretrained('./auxiliary/pretrained_bert_tf/biobert_pretrain_output_all_notes_150000/')
        indexer = CharacterIndexer()        
    return indexer, tokenizer, model


