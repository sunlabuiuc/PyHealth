# Readme file for mortality prediction run examples here.




# Multimodality

## Model Variants

nohup python examples/mortality_prediction/multimodal_embedding_mamba_mimic4_cxr.py --batch-size 1 --device cuda:3 > ../logs/multimodal_embedding_mamba_mimic4_cxr_b1.log &

nohup python examples/mortality_prediction/multimodal_embedding_mlp_mimic4_cxr.py --batch-size 1 --device cuda:3 > ../logs/multimodal_embedding_mlp_mimic4_cxr_b1.log &

nohup python examples/mortality_prediction/multimodal_embedding_rnn_mimic4_cxr.py --batch-size 1 --device cuda:3 > ../logs/multimodal_embedding_rnn_mimic4_cxr_b1.log &

nohup python examples/mortality_prediction/multimodal_embedding_transformer_mimic4_cxr.py --batch-size 1 --device cuda:3 > ../logs/multimodal_embedding_transformer_mimic4_cxr_b1.log &

nohup python examples/mortality_prediction/multimodal_embedding_jamba_mimic4_cxr.py --batch-size 1 --device cuda:3 > ../logs/multimodal_embedding_jamba_mimic4_cxr_b1.log &