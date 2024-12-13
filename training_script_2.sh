


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



accelerate launch --config_file default_config.yaml p_tuning_v2.py --model_name roberta-large --dataset_name race --classes_initiate_method cluster --batch_size 4 --num_epochs 10
rm -rf ~/.cache/huggingface/datasets/
accelerate launch --config_file default_config.yaml p_tuning_v2.py --model_name roberta-large --dataset_name sciq --classes_initiate_method cluster --batch_size 4 --num_epochs 10
rm -rf ~/.cache/huggingface/datasets/
accelerate launch --config_file default_config.yaml p_tuning_v2.py --model_name roberta-large --dataset_name dream --classes_initiate_method cluster --batch_size 4 --num_epochs 10
rm -rf ~/.cache/huggingface/datasets/
accelerate launch --config_file default_config.yaml p_tuning_v2.py --model_name roberta-large --dataset_name commonsense_qa --classes_initiate_method cluster --batch_size 4 --num_epochs 10
rm -rf ~/.cache/huggingface/datasets/


# accelerate launch --config_file default_config.yaml p_tuning_v2.py --model_name bert-base-uncased --dataset_name race --classes_initiate_method cluster --batch_size 4 --num_epochs 10
# rm -rf ~/.cache/huggingface/datasets/
# accelerate launch --config_file default_config.yaml p_tuning_v2.py --model_name bert-base-uncased --dataset_name sciq --classes_initiate_method cluster --batch_size 4 --num_epochs 10
# rm -rf ~/.cache/huggingface/datasets/
# accelerate launch --config_file default_config.yaml p_tuning_v2.py --model_name bert-base-uncased --dataset_name dream --classes_initiate_method cluster --batch_size 4 --num_epochs 10
# rm -rf ~/.cache/huggingface/datasets/
# accelerate launch --config_file default_config.yaml p_tuning_v2.py --model_name bert-base-uncased --dataset_name commonsense_qa --classes_initiate_method cluster --batch_size 4 --num_epochs 10
# rm -rf ~/.cache/huggingface/datasets/



