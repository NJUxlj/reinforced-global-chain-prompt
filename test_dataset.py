from load import *

from config import Config



def test():
    # dataset_path = Config['datasets']["sciq"]
    # ds = load_dataset_from_huggingface(dataset_path)
    
    
    dataset_path = Config['datasets']["commonsense_qa"]
    ds = load_dataset_from_huggingface(dataset_path)
    
    
    print(ds['train'][0:10])
    
    
    




if __name__ == '__main__':
    test()