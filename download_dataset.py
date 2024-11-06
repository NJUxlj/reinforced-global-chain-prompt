from datasets import load_dataset  
# from config import Config


# 设置数据集缓存目录（可选）  
import os  
from datasets import config  
# 设置自定义缓存目录  
cache_dir = "./data/super_glue"  # 你想保存数据的本地路径  
# cache_dir = Config["datasets"]["super_glue"]
# os.makedirs(cache_dir, exist_ok=True)  
# config.HF_DATASETS_CACHE = cache_dir  

# 下载所有SuperGLUE任务  
# tasks = ['boolq', 'cb', 'copa', 'multirc', 'record', 'rte', 'wic', 'wsc', 'wsc.fixed', 'axb', 'axg']  
tasks = ['multirc', 'record']  



def download():

    for task in tasks:  
        print(f"Downloading {task}...")  
        # download_mode='force_redownload' 强制重新下载  
        # cache_dir 指定缓存目录 
        os.environ['HF_DATASETS_CACHE'] = cache_dir   
        dataset = load_dataset('super_glue', task)  
        print(f"{task} downloaded successfully!")  
        
        dataset.cache_files
        
        # 打印数据集信息  
        print(f"\nDataset splits for {task}:")  
        for split in dataset.keys():  
            print(f"{split}: {len(dataset[split])} examples")  
        print("-" * 50)


def clean():
    for task in tasks:  
        print(f"Cleaning {task}...")  
        # download_mode='force_redownload' 强制重新下载  
        # cache_dir 指定缓存目录 
        # os.environ['HF_DATASETS_CACHE'] = cache_dir   
        dataset = load_dataset('super_glue', task)  
        print(f"Dataset cache directory before cleanup: {dataset.cache_files}")  
        dataset.cleanup_cache_files()  
        print(f"Dataset cache directory after cleanup: {dataset.cache_files}")
        print(f"{task} cleaned successfully!")  
        
        dataset.cache_files
        
        
        



if __name__ == '__main__':
    clean()