import json
import os
import torch
from torch.autograd import Variable
from tqdm import tqdm

from config import get_parser
from data.Omniglot_dataSet import OmniglotDataSet
from data.Batch_sampler import PrototypicalBatchSampler
from model.MatchingNetwork import MatchingNetwork
import numpy as np

def init_dataset(option,mode):
    if mode=='train_val':
        train_dataset=OmniglotDataSet(mode='train',root=option['data_root'])
        val_dataset=OmniglotDataSet(mode='val',root=option['data_root'])

        train_sampler=PrototypicalBatchSampler(labels=train_dataset.y,
                                               classes_per_it=option['data_way'],
                                               num_samples=option['data_shot']+option['data_query'],
                                               num_support=option['data_shot'],
                                               iterations=option['data_train_episodes'])
        val_sampler=PrototypicalBatchSampler(labels=val_dataset.y,
                                             classes_per_it=option['data_test_way'],
                                             num_samples=option['data_test_shot']+option['data_test_query'],
                                             num_support=option['data_test_shot'],
                                             iterations=option['data_train_episodes'])

        train_dataloader=torch.utils.data.DataLoader(train_dataset,batch_sampler=train_sampler)
        val_dataloader=torch.utils.data.DataLoader(val_dataset,batch_sampler=val_sampler)

        return train_dataloader,val_dataloader

    elif mode=='train':
        train_dataset=OmniglotDataSet(mode='trainval',root=option['data_root'])
        train_sampler=PrototypicalBatchSampler(labels=train_dataset.y,
                                               classes_per_it=option['data.way'],
                                               num_samples=option['data_shot']+option['data_query'],
                                               num_support=option['data_shot'],
                                               iterations=option['data_train_episodes'])
        train_dataloader=torch.utils.data.DataLoader(train_dataset,batch_sampler=train_sampler)
        return train_dataloader

    elif mode=='test':
        test_dataset=OmniglotDataSet(mode='test',root=option['data_root'])
        test_sampler=PrototypicalBatchSampler(labels=test_dataset.y,
                                              classes_per_it=option['data_test_way'],
                                              num_samples=option['data_test_shot']+option['data_test_query'],
                                              num_support=option['data_test_shot'],
                                              iterations=option['data_test_episodes'])
        test_dataloader=torch.utils.data.DataLoader(test_dataset,batch_sampler=test_sampler)
        return test_dataloader


def main():
    #获取参数 vars将parser对象转换为字典
    option=vars(get_parser().parse_args())
    #首先判断保存结果的文件夹是否存在，并写入参数到json文件中保存
    if not os.path.isdir(option['log_file']):
        os.makedirs(option['log_file'])

    with open(os.path.join(option['log_file'],'opt.json'),'w') as f:
        json.dump(option,f)
        f.write('\n')
    #定义trace_file文件路径
    trace_file=os.path.join(option['log_file'],'trace.txt')

    #设置随机数种子
    torch.manual_seed(2018)
    if option['data_cuda']:
        torch.cuda.manual_seed(2018)


    #step 1 加载数据集
    if option['run_mode']=='train_val':
        trainLoader,valLoader=init_dataset(option,mode=option['run_mode'])

    #step2 加载模型
    model=MatchingNetwork(keep_prob=0.0,num_channles=1,fce=False,
                          image_size=28,batch_size=1,use_cuda=option['data_cuda'])
    if option['data_cuda']:
        model.cuda()
    #step3 目标函数和优化器
    optimizer=torch.optim.Adam(model.parameters(),lr=option['train_learningrate'])
    lr_schedule=torch.optim.lr_scheduler.StepLR(optimizer=optimizer,gamma=0.5,step_size=option['train_decay_every'])

    #训练
    if os.path.isfile(trace_file):
        os.remove(trace_file)
    best_acc=0.0
    for epoch in range(option['train_epoches']):
        train_loss=[]
        train_acc=[]
        print("------epoch: %2d --------"%epoch)
        model.train()
        for data,label in tqdm(trainLoader):
            data=Variable(data)

            optimizer.zero_grad()
            if(option['data_cuda']):
                data=data.cuda()
            acc,loss=model(data,option['data_way'],option['data_shot'],option['data_query'])

            loss.backward()
            optimizer.step()
            train_loss.append(float(loss))
            train_acc.append(float(acc))

        avg_loss=np.mean(train_loss)
        avg_acc=np.mean(train_acc)
        lr_schedule.step()
        print("epoch %2d 训练结束 ： avg_loss:%.4f , avg_acc:%.4f"%(epoch,avg_loss,avg_acc))

        #下面进入validation阶段

        val_acc=[]
        print("开始进行validation:")
        model.eval()
        for data,label in tqdm(valLoader):
            data=Variable(data)
            if option['data_cuda']:
                data=data.cuda()

            acc_val,_=model(data,option['data_test_way'],option['data_test_shot'],option['data_test_query'])
            val_acc.append(float(acc_val))

        avg_acc_val=np.mean(val_acc)
        print("validation结束 : avg_acc:%.4f"%avg_acc_val)
        if (best_acc<avg_acc_val):
            print("产生目前最佳模型，正在保存......")
            name=model.save(option['log_file'])
            best_acc = avg_acc_val
            print("保存成功，保存在: ",name)
        with open(trace_file,'a') as f:
            f.write('epoch:{:2d} 训练结束：avg_loss:{:.4f} , avg_acc:{:.4f} , validation_acc:{:.4f}'.format(epoch,avg_loss,avg_acc,avg_acc_val))
            f.write('\n')

    print("训练结束，最佳模型的精度为:",best_acc)




if __name__=='__main__':
    main()

