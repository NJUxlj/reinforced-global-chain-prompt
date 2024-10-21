'''

使用soft-attention 来跨任务选择最有效的prompt-embeddings

后期可以换成别的attention， 比如sharerable attention, flash-attention

1. 首先在不同的任务{RACE, MedQA, LogicQ, ...} 上面训练模型，获取不同的prompt-embeddings块

2. 在推理任务中计算当前Question与不同的prompt-embedding块的attention，选择Top-3个最相关的块做加和

3. 得到加和后的prompt-embedding块作为最终的可训练模块
'''



