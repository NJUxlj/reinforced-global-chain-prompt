


# accelerate launch --config_file default_config.yaml baas_prompt.py --model_name roberta-large --dataset_name race --classes_initiate_method lda

# accelerate launch --config_file default_config.yaml baas_prompt.py --model_name roberta-large --dataset_name race --classes_initiate_method normal
# accelerate launch --config_file default_config.yaml baas_prompt.py --model_name roberta-large --dataset_name race --classes_initiate_method cluster


# accelerate launch --config_file default_config.yaml prompt_tuning.py --model_name Qwen2.5-1.5B --dataset_name race --train_size 5000


# accelerate launch --config_file three_gpu.yaml prompt_tuning.py --model_name Qwen2.5-1.5B --dataset_name race --train_size 5000 --batch_size 4



# accelerate launch --config_file four_gpu.yaml baas_prompt.py --model_name roberta-large --dataset_name dream --classes_initiate_method cluster --batch_size 8

# accelerate launch --config_file four_gpu.yaml baas_prompt.py --model_name roberta-large --dataset_name commonsense_qa --classes_initiate_method cluster --batch_size 8

# accelerate launch --config_file four_gpu.yaml prompt_tuning.py --model_name bert-base-uncased --dataset_name commonsense_qa --classes_initiate_method cluster --batch_size 8


# accelerate launch --config_file default_config.yaml p_tuning.py --model_name bert-large-uncased --dataset_name race --classes_initiate_method cluster --batch_size 16 --num_epochs 10
# rm -rf ~/.cache/huggingface/datasets/*
# accelerate launch --config_file default_config.yaml p_tuning.py --model_name bert-large-uncased --dataset_name sciq --classes_initiate_method cluster --batch_size 16 --num_epochs 10
# rm -rf ~/.cache/huggingface/datasets/
# # accelerate launch --config_file default_config.yaml p_tuning.py --model_name bert-large-uncased --dataset_name dream --classes_initiate_method cluster --batch_size 16 --num_epochs 10
# # rm -rf ~/.cache/huggingface/datasets/
# accelerate launch --config_file default_config.yaml p_tuning.py --model_name bert-large-uncased --dataset_name commonsense_qa --classes_initiate_method cluster --batch_size 16 --num_epochs 10
# rm -rf ~/.cache/huggingface/datasets/



# accelerate launch --config_file default_config.yaml p_tuning.py --model_name roberta-large --dataset_name race --classes_initiate_method cluster --batch_size 16 --num_epochs 10
# rm -rf ~/.cache/huggingface/datasets/
# accelerate launch --config_file default_config.yaml p_tuning.py --model_name roberta-large --dataset_name sciq --classes_initiate_method cluster --batch_size 16 --num_epochs 10
# rm -rf ~/.cache/huggingface/datasets/
# accelerate launch --config_file default_config.yaml p_tuning.py --model_name roberta-large --dataset_name dream --classes_initiate_method cluster --batch_size 16 --num_epochs 10
# rm -rf ~/.cache/huggingface/datasets/
# accelerate launch --config_file default_config.yaml p_tuning.py --model_name roberta-large --dataset_name commonsense_qa --classes_initiate_method cluster --batch_size 16 --num_epochs 10
# rm -rf ~/.cache/huggingface/datasets/


# accelerate launch --config_file default_config.yaml p_tuning.py --model_name bert-base-uncased --dataset_name race --classes_initiate_method cluster --batch_size 16 --num_epochs 10
# rm -rf ~/.cache/huggingface/datasets/
# accelerate launch --config_file default_config.yaml p_tuning.py --model_name bert-base-uncased --dataset_name sciq --classes_initiate_method cluster --batch_size 16 --num_epochs 10
# rm -rf ~/.cache/huggingface/datasets/
# accelerate launch --config_file default_config.yaml p_tuning.py --model_name bert-base-uncased --dataset_name dream --classes_initiate_method cluster --batch_size 16 --num_epochs 10
# rm -rf ~/.cache/huggingface/datasets/
# accelerate launch --config_file default_config.yaml p_tuning.py --model_name bert-base-uncased --dataset_name commonsense_qa --classes_initiate_method cluster --batch_size 16 --num_epochs 10
# rm -rf ~/.cache/huggingface/datasets/


# -------------------------------

# accelerate launch --config_file default_config.yaml p_tuning_v2.py --model_name bert-large-uncased --dataset_name race --classes_initiate_method cluster --batch_size 4 --num_epochs 10
# rm -rf ~/.cache/huggingface/datasets/
# accelerate launch --config_file default_config.yaml p_tuning_v2.py --model_name bert-large-uncased --dataset_name sciq --classes_initiate_method cluster --batch_size 4 --num_epochs 10
# rm -rf ~/.cache/huggingface/datasets/
# accelerate launch --config_file default_config.yaml p_tuning_v2.py --model_name bert-large-uncased --dataset_name dream --classes_initiate_method cluster --batch_size 4 --num_epochs 10
# rm -rf ~/.cache/huggingface/datasets/
# accelerate launch --config_file default_config.yaml p_tuning_v2.py --model_name bert-large-uncased --dataset_name commonsense_qa --classes_initiate_method cluster --batch_size 4 --num_epochs 10
# rm -rf ~/.cache/huggingface/datasets/



# accelerate launch --config_file default_config.yaml p_tuning_v2.py --model_name roberta-large --dataset_name race --classes_initiate_method cluster --batch_size 4 --num_epochs 10
# rm -rf ~/.cache/huggingface/datasets/
# accelerate launch --config_file default_config.yaml p_tuning_v2.py --model_name roberta-large --dataset_name sciq --classes_initiate_method cluster --batch_size 4 --num_epochs 10
# rm -rf ~/.cache/huggingface/datasets/
# accelerate launch --config_file default_config.yaml p_tuning_v2.py --model_name roberta-large --dataset_name dream --classes_initiate_method cluster --batch_size 4 --num_epochs 10
# rm -rf ~/.cache/huggingface/datasets/
# accelerate launch --config_file default_config.yaml p_tuning_v2.py --model_name roberta-large --dataset_name commonsense_qa --classes_initiate_method cluster --batch_size 4 --num_epochs 10
# rm -rf ~/.cache/huggingface/datasets/


# accelerate launch --config_file default_config.yaml p_tuning_v2.py --model_name bert-base-uncased --dataset_name race --classes_initiate_method cluster --batch_size 4 --num_epochs 10
# rm -rf ~/.cache/huggingface/datasets/
# accelerate launch --config_file default_config.yaml p_tuning_v2.py --model_name bert-base-uncased --dataset_name sciq --classes_initiate_method cluster --batch_size 4 --num_epochs 10
# rm -rf ~/.cache/huggingface/datasets/
# accelerate launch --config_file default_config.yaml p_tuning_v2.py --model_name bert-base-uncased --dataset_name dream --classes_initiate_method cluster --batch_size 4 --num_epochs 10
# rm -rf ~/.cache/huggingface/datasets/
# accelerate launch --config_file default_config.yaml p_tuning_v2.py --model_name bert-base-uncased --dataset_name commonsense_qa --classes_initiate_method cluster --batch_size 4 --num_epochs 10
# rm -rf ~/.cache/huggingface/datasets/



# # Qwen2.5-0.5B
# accelerate launch --config_file default_config.yaml baas_prompt.py --model_name Qwen2.5-0.5B --dataset_name race --classes_initiate_method cluster --batch_size 2 --num_epochs 10 --mixed_precision fp16
# accelerate launch --config_file default_config.yaml baas_prompt.py --model_name Qwen2.5-0.5B --dataset_name sciq --classes_initiate_method cluster --batch_size 4 --num_epochs 10
# accelerate launch --config_file default_config.yaml baas_prompt.py --model_name Qwen2.5-0.5B --dataset_name dream --classes_initiate_method cluster --batch_size 4 --num_epochs 10
# accelerate launch --config_file default_config.yaml baas_prompt.py --model_name Qwen2.5-0.5B --dataset_name commonsense_qa --classes_initiate_method cluster --batch_size 4 --num_epochs 10


# accelerate launch --config_file default_config.yaml baas_prompt.py --model_name Qwen2.5-1.5B --dataset_name race --classes_initiate_method cluster --batch_size 4 --num_epochs 10
# accelerate launch --config_file default_config.yaml baas_prompt.py --model_name Qwen2.5-1.5B --dataset_name sciq --classes_initiate_method cluster --batch_size 4 --num_epochs 10
# accelerate launch --config_file default_config.yaml baas_prompt.py --model_name Qwen2.5-1.5B --dataset_name dream --classes_initiate_method cluster --batch_size 4 --num_epochs 10
# accelerate launch --config_file default_config.yaml baas_prompt.py --model_name Qwen2.5-1.5B --dataset_name commonsense_qa --classes_initiate_method cluster --batch_size 4 --num_epochs 10


# accelerate launch --config_file default_config.yaml baas_prompt.py --model_name Qwen2.5-3B --dataset_name race --classes_initiate_method cluster --batch_size 4 --num_epochs 10
# accelerate launch --config_file default_config.yaml baas_prompt.py --model_name Qwen2.5-3B --dataset_name sciq --classes_initiate_method cluster --batch_size 4 --num_epochs 10
# accelerate launch --config_file default_config.yaml baas_prompt.py --model_name Qwen2.5-3B --dataset_name dream --classes_initiate_method cluster --batch_size 4 --num_epochs 10
# accelerate launch --config_file default_config.yaml baas_prompt.py --model_name Qwen2.5-3B --dataset_name commonsense_qa --classes_initiate_method cluster --batch_size 4 --num_epochs 10


# accelerate launch --config_file default_config.yaml baas_prompt.py --model_name Qwen2.5-7B --dataset_name race --classes_initiate_method cluster --batch_size 4 --num_epochs 10
# accelerate launch --config_file default_config.yaml baas_prompt.py --model_name Qwen2.5-7B --dataset_name sciq --classes_initiate_method cluster --batch_size 4 --num_epochs 10
# accelerate launch --config_file default_config.yaml baas_prompt.py --model_name Qwen2.5-7B --dataset_name dream --classes_initiate_method cluster --batch_size 4 --num_epochs 10
# accelerate launch --config_file default_config.yaml baas_prompt.py --model_name Qwen2.5-7B --dataset_name commonsense_qa --classes_initiate_method cluster --batch_size 4 --num_epochs 10





# accelerate launch --config_file six_gpu.yaml baas_prompt.py --model_name Qwen2.5-0.5B --dataset_name race --classes_initiate_method cluster --batch_size 2 --num_epochs 10 --mixed_precision bf16 --train_size 2000






# Ablation Study
# rm -rf ~/.cache/huggingface/datasets/
# accelerate launch --config_file four_gpu.yaml baas_prompt.py --model_name bert-base-uncased --dataset_name race --classes_initiate_method cluster --batch_size 4 --num_epochs 10 --suffix_ratio 1
# accelerate launch --config_file four_gpu.yaml baas_prompt.py --model_name bert-base-uncased --dataset_name race --classes_initiate_method cluster --batch_size 4 --num_epochs 10 --suffix_ratio 5
# accelerate launch --config_file four_gpu.yaml baas_prompt.py --model_name bert-base-uncased --dataset_name race --classes_initiate_method cluster --batch_size 4 --num_epochs 10 --suffix_ratio 10
# accelerate launch --config_file four_gpu.yaml baas_prompt.py --model_name bert-base-uncased --dataset_name race --classes_initiate_method cluster --batch_size 4 --num_epochs 10 --suffix_ratio 20
# accelerate launch --config_file four_gpu.yaml baas_prompt.py --model_name bert-base-uncased --dataset_name race --classes_initiate_method cluster --batch_size 4 --num_epochs 10 --suffix_ratio 50


# accelerate launch --config_file four_gpu.yaml baas_prompt.py --model_name bert-base-uncased --dataset_name sciq --classes_initiate_method cluster --batch_size 4 --num_epochs 10 --suffix_ratio 1
# accelerate launch --config_file four_gpu.yaml baas_prompt.py --model_name bert-base-uncased --dataset_name sciq --classes_initiate_method cluster --batch_size 4 --num_epochs 10 --suffix_ratio 5
# accelerate launch --config_file four_gpu.yaml baas_prompt.py --model_name bert-base-uncased --dataset_name sciq --classes_initiate_method cluster --batch_size 4 --num_epochs 10 --suffix_ratio 10
# accelerate launch --config_file four_gpu.yaml baas_prompt.py --model_name bert-base-uncased --dataset_name sciq --classes_initiate_method cluster --batch_size 4 --num_epochs 10 --suffix_ratio 20
# accelerate launch --config_file four_gpu.yaml baas_prompt.py --model_name bert-base-uncased --dataset_name sciq --classes_initiate_method cluster --batch_size 4 --num_epochs 10 --suffix_ratio 50


# accelerate launch --config_file four_gpu.yaml baas_prompt.py --model_name bert-base-uncased --dataset_name dream --classes_initiate_method cluster --batch_size 4 --num_epochs 10 --suffix_ratio 1
# accelerate launch --config_file four_gpu.yaml baas_prompt.py --model_name bert-base-uncased --dataset_name dream --classes_initiate_method cluster --batch_size 4 --num_epochs 10 --suffix_ratio 5
# accelerate launch --config_file four_gpu.yaml baas_prompt.py --model_name bert-base-uncased --dataset_name dream --classes_initiate_method cluster --batch_size 4 --num_epochs 10 --suffix_ratio 10
# accelerate launch --config_file four_gpu.yaml baas_prompt.py --model_name bert-base-uncased --dataset_name dream --classes_initiate_method cluster --batch_size 4 --num_epochs 10 --suffix_ratio 20
# accelerate launch --config_file four_gpu.yaml baas_prompt.py --model_name bert-base-uncased --dataset_name dream --classes_initiate_method cluster --batch_size 4 --num_epochs 10 --suffix_ratio 50


# accelerate launch --config_file four_gpu.yaml baas_prompt.py --model_name bert-base-uncased --dataset_name commonsense_qa --classes_initiate_method cluster --batch_size 4 --num_epochs 10 --suffix_ratio 1
# accelerate launch --config_file four_gpu.yaml baas_prompt.py --model_name bert-base-uncased --dataset_name commonsense_qa --classes_initiate_method cluster --batch_size 4 --num_epochs 10 --suffix_ratio 5
# accelerate launch --config_file four_gpu.yaml baas_prompt.py --model_name bert-base-uncased --dataset_name commonsense_qa --classes_initiate_method cluster --batch_size 4 --num_epochs 10 --suffix_ratio 10
# accelerate launch --config_file four_gpu.yaml baas_prompt.py --model_name bert-base-uncased --dataset_name commonsense_qa --classes_initiate_method cluster --batch_size 4 --num_epochs 10 --suffix_ratio 20
# accelerate launch --config_file four_gpu.yaml baas_prompt.py --model_name bert-base-uncased --dataset_name commonsense_qa --classes_initiate_method cluster --batch_size 4 --num_epochs 10 --suffix_ratio 50






# ## Bert-Large
# rm -rf ~/.cache/huggingface/datasets/
# accelerate launch --config_file four_gpu.yaml baas_prompt.py --model_name bert-large-uncased --dataset_name race --classes_initiate_method cluster --batch_size 4 --num_epochs 10 --suffix_ratio 1
# accelerate launch --config_file four_gpu.yaml baas_prompt.py --model_name bert-large-uncased --dataset_name race --classes_initiate_method cluster --batch_size 4 --num_epochs 10 --suffix_ratio 5
# accelerate launch --config_file four_gpu.yaml baas_prompt.py --model_name bert-large-uncased --dataset_name race --classes_initiate_method cluster --batch_size 4 --num_epochs 10 --suffix_ratio 10
# accelerate launch --config_file four_gpu.yaml baas_prompt.py --model_name bert-large-uncased --dataset_name race --classes_initiate_method cluster --batch_size 4 --num_epochs 10 --suffix_ratio 20
# accelerate launch --config_file four_gpu.yaml baas_prompt.py --model_name bert-large-uncased --dataset_name race --classes_initiate_method cluster --batch_size 4 --num_epochs 10 --suffix_ratio 50


# accelerate launch --config_file four_gpu.yaml baas_prompt.py --model_name bert-large-uncased --dataset_name sciq --classes_initiate_method cluster --batch_size 4 --num_epochs 10 --suffix_ratio 1
# accelerate launch --config_file four_gpu.yaml baas_prompt.py --model_name bert-large-uncased --dataset_name sciq --classes_initiate_method cluster --batch_size 4 --num_epochs 10 --suffix_ratio 5
# accelerate launch --config_file four_gpu.yaml baas_prompt.py --model_name bert-large-uncased --dataset_name sciq --classes_initiate_method cluster --batch_size 4 --num_epochs 10 --suffix_ratio 10
# accelerate launch --config_file four_gpu.yaml baas_prompt.py --model_name bert-large-uncased --dataset_name sciq --classes_initiate_method cluster --batch_size 4 --num_epochs 10 --suffix_ratio 20
# accelerate launch --config_file four_gpu.yaml baas_prompt.py --model_name bert-large-uncased --dataset_name sciq --classes_initiate_method cluster --batch_size 4 --num_epochs 10 --suffix_ratio 50


# accelerate launch --config_file four_gpu.yaml baas_prompt.py --model_name bert-large-uncased --dataset_name dream --classes_initiate_method cluster --batch_size 4 --num_epochs 10 --suffix_ratio 1
# accelerate launch --config_file four_gpu.yaml baas_prompt.py --model_name bert-large-uncased --dataset_name dream --classes_initiate_method cluster --batch_size 4 --num_epochs 10 --suffix_ratio 5
# accelerate launch --config_file four_gpu.yaml baas_prompt.py --model_name bert-large-uncased --dataset_name dream --classes_initiate_method cluster --batch_size 4 --num_epochs 10 --suffix_ratio 10
# accelerate launch --config_file four_gpu.yaml baas_prompt.py --model_name bert-large-uncased --dataset_name dream --classes_initiate_method cluster --batch_size 4 --num_epochs 10 --suffix_ratio 20
# accelerate launch --config_file four_gpu.yaml baas_prompt.py --model_name bert-large-uncased --dataset_name dream --classes_initiate_method cluster --batch_size 4 --num_epochs 10 --suffix_ratio 50


# accelerate launch --config_file four_gpu.yaml baas_prompt.py --model_name bert-large-uncased --dataset_name commonsense_qa --classes_initiate_method cluster --batch_size 4 --num_epochs 10 --suffix_ratio 1
# accelerate launch --config_file four_gpu.yaml baas_prompt.py --model_name bert-large-uncased --dataset_name commonsense_qa --classes_initiate_method cluster --batch_size 4 --num_epochs 10 --suffix_ratio 5
# accelerate launch --config_file four_gpu.yaml baas_prompt.py --model_name bert-large-uncased --dataset_name commonsense_qa --classes_initiate_method cluster --batch_size 4 --num_epochs 10 --suffix_ratio 10
# accelerate launch --config_file four_gpu.yaml baas_prompt.py --model_name bert-large-uncased --dataset_name commonsense_qa --classes_initiate_method cluster --batch_size 4 --num_epochs 10 --suffix_ratio 20
# accelerate launch --config_file four_gpu.yaml baas_prompt.py --model_name bert-large-uncased --dataset_name commonsense_qa --classes_initiate_method cluster --batch_size 4 --num_epochs 10 --suffix_ratio 50






# ## Roberta
# rm -rf ~/.cache/huggingface/datasets/
# accelerate launch --config_file four_gpu.yaml baas_prompt.py --model_name roberta-large --dataset_name race --classes_initiate_method cluster --batch_size 4 --num_epochs 10 --suffix_ratio 1
# accelerate launch --config_file four_gpu.yaml baas_prompt.py --model_name roberta-large --dataset_name race --classes_initiate_method cluster --batch_size 4 --num_epochs 10 --suffix_ratio 5
# accelerate launch --config_file four_gpu.yaml baas_prompt.py --model_name roberta-large --dataset_name race --classes_initiate_method cluster --batch_size 4 --num_epochs 10 --suffix_ratio 10
# accelerate launch --config_file four_gpu.yaml baas_prompt.py --model_name roberta-large --dataset_name race --classes_initiate_method cluster --batch_size 4 --num_epochs 10 --suffix_ratio 20
# accelerate launch --config_file four_gpu.yaml baas_prompt.py --model_name roberta-large --dataset_name race --classes_initiate_method cluster --batch_size 4 --num_epochs 10 --suffix_ratio 50


# accelerate launch --config_file four_gpu.yaml baas_prompt.py --model_name roberta-large --dataset_name sciq --classes_initiate_method cluster --batch_size 4 --num_epochs 10 --suffix_ratio 1
# accelerate launch --config_file four_gpu.yaml baas_prompt.py --model_name roberta-large --dataset_name sciq --classes_initiate_method cluster --batch_size 4 --num_epochs 10 --suffix_ratio 5
# accelerate launch --config_file four_gpu.yaml baas_prompt.py --model_name roberta-large --dataset_name sciq --classes_initiate_method cluster --batch_size 4 --num_epochs 10 --suffix_ratio 10
# accelerate launch --config_file four_gpu.yaml baas_prompt.py --model_name roberta-large --dataset_name sciq --classes_initiate_method cluster --batch_size 4 --num_epochs 10 --suffix_ratio 20
# accelerate launch --config_file four_gpu.yaml baas_prompt.py --model_name roberta-large --dataset_name sciq --classes_initiate_method cluster --batch_size 4 --num_epochs 10 --suffix_ratio 50


# accelerate launch --config_file four_gpu.yaml baas_prompt.py --model_name roberta-large --dataset_name dream --classes_initiate_method cluster --batch_size 4 --num_epochs 10 --suffix_ratio 1
# accelerate launch --config_file four_gpu.yaml baas_prompt.py --model_name roberta-large --dataset_name dream --classes_initiate_method cluster --batch_size 4 --num_epochs 10 --suffix_ratio 5
# accelerate launch --config_file four_gpu.yaml baas_prompt.py --model_name roberta-large --dataset_name dream --classes_initiate_method cluster --batch_size 4 --num_epochs 10 --suffix_ratio 10
# accelerate launch --config_file four_gpu.yaml baas_prompt.py --model_name roberta-large --dataset_name dream --classes_initiate_method cluster --batch_size 4 --num_epochs 10 --suffix_ratio 20
# accelerate launch --config_file four_gpu.yaml baas_prompt.py --model_name roberta-large --dataset_name dream --classes_initiate_method cluster --batch_size 4 --num_epochs 10 --suffix_ratio 50


# accelerate launch --config_file four_gpu.yaml baas_prompt.py --model_name roberta-large --dataset_name commonsense_qa --classes_initiate_method cluster --batch_size 4 --num_epochs 10 --suffix_ratio 1
# accelerate launch --config_file four_gpu.yaml baas_prompt.py --model_name roberta-large --dataset_name commonsense_qa --classes_initiate_method cluster --batch_size 4 --num_epochs 10 --suffix_ratio 5
# accelerate launch --config_file four_gpu.yaml baas_prompt.py --model_name roberta-large --dataset_name commonsense_qa --classes_initiate_method cluster --batch_size 4 --num_epochs 10 --suffix_ratio 10
# accelerate launch --config_file four_gpu.yaml baas_prompt.py --model_name roberta-large --dataset_name commonsense_qa --classes_initiate_method cluster --batch_size 4 --num_epochs 10 --suffix_ratio 20
# accelerate launch --config_file four_gpu.yaml baas_prompt.py --model_name roberta-large --dataset_name commonsense_qa --classes_initiate_method cluster --batch_size 4 --num_epochs 10 --suffix_ratio 50













# GPT2
# accelerate launch --config_file four_gpu.yaml baas_prompt.py --model_name gpt2 --dataset_name race --classes_initiate_method cluster --batch_size 4 --num_epochs 10 --suffix_ratio 10
accelerate launch --config_file four_gpu.yaml baas_prompt.py --model_name gpt2 --dataset_name sciq --classes_initiate_method cluster --batch_size 4 --num_epochs 10 --suffix_ratio 10
accelerate launch --config_file four_gpu.yaml baas_prompt.py --model_name gpt2 --dataset_name dream --classes_initiate_method cluster --batch_size 4 --num_epochs 10 --suffix_ratio 10
accelerate launch --config_file four_gpu.yaml baas_prompt.py --model_name gpt2 --dataset_name commonsense_qa --classes_initiate_method cluster --batch_size 4 --num_epochs 10 --suffix_ratio 10


accelerate launch --config_file four_gpu.yaml p_tuning_v2.py --model_name gpt2 --dataset_name race --classes_initiate_method cluster --batch_size 4 --num_epochs 10 --suffix_ratio 10
accelerate launch --config_file four_gpu.yaml p_tuning_v2.py --model_name gpt2 --dataset_name sciq --classes_initiate_method cluster --batch_size 4 --num_epochs 10 --suffix_ratio 10
accelerate launch --config_file four_gpu.yaml p_tuning_v2.py --model_name gpt2 --dataset_name dream --classes_initiate_method cluster --batch_size 4 --num_epochs 10 --suffix_ratio 10
accelerate launch --config_file four_gpu.yaml p_tuning_v2.py --model_name gpt2 --dataset_name commonsense_qa --classes_initiate_method cluster --batch_size 4 --num_epochs 10 --suffix_ratio 10

accelerate launch --config_file four_gpu.yaml p_tuning.py --model_name gpt2 --dataset_name race --classes_initiate_method cluster --batch_size 4 --num_epochs 10 --suffix_ratio 10
accelerate launch --config_file four_gpu.yaml p_tuning.py --model_name gpt2 --dataset_name sciq --classes_initiate_method cluster --batch_size 4 --num_epochs 10 --suffix_ratio 10
accelerate launch --config_file four_gpu.yaml p_tuning.py --model_name gpt2 --dataset_name dream --classes_initiate_method cluster --batch_size 4 --num_epochs 10 --suffix_ratio 10
accelerate launch --config_file four_gpu.yaml p_tuning.py --model_name gpt2 --dataset_name commonsense_qa --classes_initiate_method cluster --batch_size 4 --num_epochs 10 --suffix_ratio 10


accelerate launch --config_file four_gpu.yaml prompt_tuning.py --model_name gpt2 --dataset_name race --classes_initiate_method cluster --batch_size 4 --num_epochs 10 --suffix_ratio 10
accelerate launch --config_file four_gpu.yaml prompt_tuning.py --model_name gpt2 --dataset_name sciq --classes_initiate_method cluster --batch_size 4 --num_epochs 10 --suffix_ratio 10
accelerate launch --config_file four_gpu.yaml prompt_tuning.py --model_name gpt2 --dataset_name dream --classes_initiate_method cluster --batch_size 4 --num_epochs 10 --suffix_ratio 10
accelerate launch --config_file four_gpu.yaml prompt_tuning.py --model_name gpt2 --dataset_name commonsense_qa --classes_initiate_method cluster --batch_size 4 --num_epochs 10 --suffix_ratio 10
