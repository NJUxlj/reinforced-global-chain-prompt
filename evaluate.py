


class Evaluator:
    def __init__(self, dataset, model, criterion, config):
        self.config = config
        self.model = model
        self.dataset = dataset
        self.vocab_size = self.config["vocab_size"]
        
    
    def eval(self):
        pass
    
    
    
    def write_stats(self):
        pass
    
    
    
    
    
    def show_stats(self):
        pass