from sklearn.cluster import KMeans  
from transformers import AutoModel, AutoTokenizer  
import torch  
from config import Config

# 初始化词向量模型
model_path = Config['models']["bert-base-uncased"]["model_path"]

tokenizer_emb = AutoTokenizer.from_pretrained(model_path)  
model_emb = AutoModel.from_pretrained(model_path).cuda()  

# 获取词向量  
def get_embeddings(entities):  
    """  
    将实体列表转换为词向量  
    """  
    embeddings = []  
    for token, label in entities:  
        inputs = tokenizer_emb(label, return_tensors="pt", truncation=True, padding=True, max_length=512)
        inputs = inputs.to("cuda")
        outputs = model_emb(**inputs) # shape = (1, seq_len, hidden_size)
        # 获取[CLS]向量  
        embedding = outputs.last_hidden_state[:,0,:].detach().cpu().numpy() # shape = (1, hidden_size) 
        embeddings.append(embedding[0])     
    return embeddings  

# 进行K-Means聚类  
def perform_kmeans(embeddings, k):  
    """  
    对词向量进行K-Means聚类  
    """  
    kmeans = KMeans(n_clusters=k, random_state=0)  
    kmeans.fit(embeddings)  
    return kmeans.cluster_centers_  

# 示例用法  
if __name__ == "__main__":  
    # 假设已提取实体  
    entities = [('Paris', 'LOC'), ('France', 'LOC')]  
    embeddings = get_embeddings(entities)  
    cluster_centers = perform_kmeans(embeddings, k=2)  
    print("Cluster Centers:")  
    print(cluster_centers)  