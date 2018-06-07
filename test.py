import os
import torch
from tqdm import tqdm
from torch.autograd import Variable
import numpy as np
from config import get_parser
from main import init_dataset
from model.MatchingNetwork import MatchingNetwork
def test():
    option=vars(get_parser().parse_args())
    path=os.path.join(option['log_file'],'best_model.pth')

    if not os.path.isfile(path):
        print("未找到最佳保存模型，请先训练好吗！")
        return

    #设定随机数种子
    torch.manual_seed(2018)
    if option['data_cuda']:
        torch.cuda.manual_seed(2018)

    #step 1 加载数据
    testLoader=init_dataset(option,mode='test')

    #step 2 加载模型
    model=MatchingNetwork(keep_prob=0.0,num_channles=1,fce=True,
                          image_size=28,batch_size=1,use_cuda=option['data_cuda'])
    model.load(path)

    if option['data_cuda']:
        model.cuda()


    #step 测试
    model.eval()
    test_acc=[]
    test_loss=[]

    for data,label in tqdm(testLoader):
        data=Variable(data)
        if option['data_cuda']:
            data=data.cuda()

        acc, loss = model(data, option['data_test_way'], option['data_test_shot'], option['data_test_query'])
        test_acc.append(float(acc))
        test_loss.append(float(loss))

    avg_acc=np.mean(test_acc)
    avg_loss=np.mean(test_loss)
    print("%4d 测试结束：avg_acc:%.4f , avg_loss:%.4f" % (option['data_test_episodes'], avg_acc,avg_loss))

if __name__=='__main__':
    test()