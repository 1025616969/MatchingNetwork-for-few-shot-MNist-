import torch
import os
class BasicModule(torch.nn.Module):
    """
    该类封装了nn.Module,主要提供save和load方法

    """
    def __init__(self):
        super(BasicModule,self).__init__()

    def load(self,path='./results/best_model.pth'):
        self.load_state_dict(torch.load(path))

    def save(self,path='./results'):
        model_path=os.path.join(path,'best_model.pth')
        torch.save(self.state_dict(),model_path)
        return model_path