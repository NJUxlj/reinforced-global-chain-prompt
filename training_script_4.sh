rm -rf ~/.cache/huggingface/datasets/
accelerate launch --config_file four_gpu.yaml baas_prompt.py --model_name bert-base-uncased --dataset_name sciq --classes_initiate_method normal --batch_size 2 --num_epochs 10 --suffix_ratio 1 --train_size 11000
accelerate launch --config_file four_gpu.yaml baas_prompt.py --model_name bert-base-uncased --dataset_name dream --classes_initiate_method normal --batch_size 2 --num_epochs 10 --suffix_ratio 5 --train_size 11000
accelerate launch --config_file four_gpu.yaml baas_prompt.py --model_name bert-base-uncased --dataset_name commonsense_qa --classes_initiate_method normal --batch_size 2 --num_epochs 10 --suffix_ratio 10 --train_size 11000
accelerate launch --config_file four_gpu.yaml baas_prompt.py --model_name bert-base-uncased --dataset_name commonsense_qa --classes_initiate_method lda --batch_size 2 --num_epochs 10 --suffix_ratio 20 --train_size 11000


rm -rf ~/.cache/huggingface/datasets/
accelerate launch --config_file four_gpu.yaml baas_prompt.py --model_name bert-large-uncased --dataset_name race --classes_initiate_method normal --batch_size 2 --num_epochs 10 --suffix_ratio 1 --train_size 11000
accelerate launch --config_file four_gpu.yaml baas_prompt.py --model_name bert-large-uncased --dataset_name sciq --classes_initiate_method normal --batch_size 2 --num_epochs 10 --suffix_ratio 1 --train_size 11000
accelerate launch --config_file four_gpu.yaml baas_prompt.py --model_name bert-large-uncased --dataset_name dream --classes_initiate_method normal --batch_size 2 --num_epochs 10 --suffix_ratio 5 --train_size 11000
accelerate launch --config_file four_gpu.yaml baas_prompt.py --model_name bert-large-uncased --dataset_name commonsense_qa --classes_initiate_method normal --batch_size 2 --num_epochs 10 --suffix_ratio 10 --train_size 11000


rm -rf ~/.cache/huggingface/datasets/
accelerate launch --config_file four_gpu.yaml baas_prompt.py --model_name bert-large-uncased --dataset_name race --classes_initiate_method cluster --batch_size 2 --num_epochs 10 --suffix_ratio 1 --train_size 11000
accelerate launch --config_file four_gpu.yaml baas_prompt.py --model_name bert-large-uncased --dataset_name sciq --classes_initiate_method cluster --batch_size 2 --num_epochs 10 --suffix_ratio 1 --train_size 11000
accelerate launch --config_file four_gpu.yaml baas_prompt.py --model_name bert-large-uncased --dataset_name dream --classes_initiate_method cluster --batch_size 2 --num_epochs 10 --suffix_ratio 5 --train_size 11000
accelerate launch --config_file four_gpu.yaml baas_prompt.py --model_name bert-large-uncased --dataset_name commonsense_qa --classes_initiate_method cluster --batch_size 2 --num_epochs 10 --suffix_ratio 10 --train_size 11000


rm -rf ~/.cache/huggingface/datasets/
accelerate launch --config_file four_gpu.yaml baas_prompt.py --model_name bert-large-uncased --dataset_name race --classes_initiate_method lda --batch_size 2 --num_epochs 10 --suffix_ratio 1 --train_size 11000
accelerate launch --config_file four_gpu.yaml baas_prompt.py --model_name bert-large-uncased --dataset_name sciq --classes_initiate_method lda --batch_size 2 --num_epochs 10 --suffix_ratio 1 --train_size 11000
accelerate launch --config_file four_gpu.yaml baas_prompt.py --model_name bert-large-uncased --dataset_name dream --classes_initiate_method lda --batch_size 2 --num_epochs 10 --suffix_ratio 5 --train_size 11000
accelerate launch --config_file four_gpu.yaml baas_prompt.py --model_name bert-large-uncased --dataset_name commonsense_qa --classes_initiate_method lda --batch_size 2 --num_epochs 10 --suffix_ratio 10 --train_size 11000




rm -rf ~/.cache/huggingface/datasets/

# accelerate launch --config_file four_gpu.yaml baas_prompt.py --model_name bert-base-uncased --dataset_name race --classes_initiate_method cluster --batch_size 2 --num_epochs 10 --suffix_ratio 1 --train_size 11000    
# accelerate launch --config_file four_gpu.yaml baas_prompt.py --model_name bert-base-uncased --dataset_name race --classes_initiate_method cluster --batch_size 2 --num_epochs 10 --suffix_ratio 5 --train_size 11000
# accelerate launch --config_file four_gpu.yaml baas_prompt.py --model_name bert-base-uncased --dataset_name race --classes_initiate_method cluster --batch_size 2 --num_epochs 10 --suffix_ratio 10 --train_size 11000
accelerate launch --config_file four_gpu.yaml baas_prompt.py --model_name bert-base-uncased --dataset_name race --classes_initiate_method cluster --batch_size 2 --num_epochs 10 --suffix_ratio 20 --train_size 11000
accelerate launch --config_file four_gpu.yaml baas_prompt.py --model_name bert-base-uncased --dataset_name race --classes_initiate_method cluster --batch_size 2 --num_epochs 10 --suffix_ratio 50 --train_size 11000

rm -rf ~/.cache/huggingface/datasets/
accelerate launch --config_file four_gpu.yaml baas_prompt.py --model_name bert-base-uncased --dataset_name sciq --classes_initiate_method cluster --batch_size 2 --num_epochs 10 --suffix_ratio 1 --train_size 11000    
accelerate launch --config_file four_gpu.yaml baas_prompt.py --model_name bert-base-uncased --dataset_name sciq --classes_initiate_method cluster --batch_size 2 --num_epochs 10 --suffix_ratio 5 --train_size 11000
# accelerate launch --config_file four_gpu.yaml baas_prompt.py --model_name bert-base-uncased --dataset_name sciq --classes_initiate_method cluster --batch_size 2 --num_epochs 10 --suffix_ratio 10 --train_size 11000
accelerate launch --config_file four_gpu.yaml baas_prompt.py --model_name bert-base-uncased --dataset_name sciq --classes_initiate_method cluster --batch_size 2 --num_epochs 10 --suffix_ratio 20 --train_size 11000
accelerate launch --config_file four_gpu.yaml baas_prompt.py --model_name bert-base-uncased --dataset_name sciq --classes_initiate_method cluster --batch_size 2 --num_epochs 10 --suffix_ratio 50 --train_size 11000

rm -rf ~/.cache/huggingface/datasets/

accelerate launch --config_file four_gpu.yaml baas_prompt.py --model_name bert-base-uncased --dataset_name dream --classes_initiate_method cluster --batch_size 2 --num_epochs 10 --suffix_ratio 1 --train_size 11000    
accelerate launch --config_file four_gpu.yaml baas_prompt.py --model_name bert-base-uncased --dataset_name dream --classes_initiate_method cluster --batch_size 2 --num_epochs 10 --suffix_ratio 5 --train_size 11000
# accelerate launch --config_file four_gpu.yaml baas_prompt.py --model_name bert-base-uncased --dataset_name dream --classes_initiate_method cluster --batch_size 2 --num_epochs 10 --suffix_ratio 10 --train_size 11000
accelerate launch --config_file four_gpu.yaml baas_prompt.py --model_name bert-base-uncased --dataset_name dream --classes_initiate_method cluster --batch_size 2 --num_epochs 10 --suffix_ratio 20 --train_size 11000
accelerate launch --config_file four_gpu.yaml baas_prompt.py --model_name bert-base-uncased --dataset_name dream --classes_initiate_method cluster --batch_size 2 --num_epochs 10 --suffix_ratio 50 --train_size 11000

rm -rf ~/.cache/huggingface/datasets/

accelerate launch --config_file four_gpu.yaml baas_prompt.py --model_name bert-base-uncased --dataset_name commonsense_qa --classes_initiate_method cluster --batch_size 2 --num_epochs 10 --suffix_ratio 1 --train_size 11000    
accelerate launch --config_file four_gpu.yaml baas_prompt.py --model_name bert-base-uncased --dataset_name commonsense_qa --classes_initiate_method cluster --batch_size 2 --num_epochs 10 --suffix_ratio 5 --train_size 11000
# accelerate launch --config_file four_gpu.yaml baas_prompt.py --model_name bert-base-uncased --dataset_name commonsense_qa --classes_initiate_method cluster --batch_size 2 --num_epochs 10 --suffix_ratio 10 --train_size 11000
accelerate launch --config_file four_gpu.yaml baas_prompt.py --model_name bert-base-uncased --dataset_name commonsense_qa --classes_initiate_method cluster --batch_size 2 --num_epochs 10 --suffix_ratio 20 --train_size 11000
accelerate launch --config_file four_gpu.yaml baas_prompt.py --model_name bert-base-uncased --dataset_name commonsense_qa --classes_initiate_method cluster --batch_size 2 --num_epochs 10 --suffix_ratio 50 --train_size 11000