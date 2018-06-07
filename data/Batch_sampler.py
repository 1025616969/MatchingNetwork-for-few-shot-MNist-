# -*- coding:utf-8 -*-
import numpy as np
import torch

class PrototypicalBatchSampler():
    def __init__(self,labels,classes_per_it,num_samples,num_support,iterations):
        super(PrototypicalBatchSampler,self).__init__()

        self.labels=labels
        self.classes_per_it=classes_per_it
        self.num_samples=num_samples
        self.iterations=iterations
        self.num_support=num_support #support set每个class的样本数
        self.num_query=self.num_samples-self.num_support

        self.classes,self.class_counts=np.unique(self.labels,return_counts=True)
        self.classes=torch.LongTensor(self.classes)

        #label_tensor用来保存属于各个label的样本在labels中的index
        #label_lens用来保存各个class样本的数量

        #使用np.empty是因为empty在初始化的时候速度最快
        self.label_tensor=np.empty((len(self.classes),max(self.class_counts)),dtype=int)*np.nan

        self.label_tensor=torch.FloatTensor(self.label_tensor)
        self.label_lens=torch.zeros_like(self.classes)

        #对label_tensor  label_lens进行赋值
        for index,label in enumerate(self.labels):
            label_index=np.argwhere(self.classes==label)[0,0]
            value_index=np.argwhere(np.isnan(self.label_tensor[label_index]))[0,0]
            self.label_tensor[label_index][value_index]=index
            self.label_lens[label_index]+=1

    def __iter__(self):

        """
        按顺序产生 a batch of indexes
        前面是support按类排列，后面是query set按类排列
        11111 22222 33333 44444 55555 111 222 333 444 555 (5 way,5shot 3 query )
        :return:
        """
        for it in range(self.iterations):
            batch_size=self.classes_per_it*self.num_samples
            batch=torch.LongTensor(batch_size)

            #随机选取classes_per_it个类，获得其index
            class_indexes=torch.randperm(len(self.classes))[:self.classes_per_it]
            query_set=torch.LongTensor(self.classes_per_it*self.num_query)

            #对每个类，随机选取num_samples个样本
            for i,c in enumerate(self.classes[class_indexes]):
                s=slice(i*self.num_support,(i+1)*self.num_support)
                sq=slice(i*self.num_query,(i+1)*self.num_query)

                label_index=np.argwhere(self.classes==c)[0,0]
                sample_indexes=torch.randperm(self.label_lens[label_index])[:self.num_samples]

                support_indexes=sample_indexes[:self.num_support]
                query_indexes=sample_indexes[self.num_support:]

                batch[s]=self.label_tensor[label_index][support_indexes]
                #print(batch[s])
                query_set[sq]=self.label_tensor[label_index][query_indexes]
            batch[self.num_support*self.classes_per_it:]=query_set
            yield batch

    def __len__(self):
        return self.iterations


