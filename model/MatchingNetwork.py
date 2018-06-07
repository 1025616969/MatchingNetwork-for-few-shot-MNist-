import torch

from model.BasicModule import BasicModule
import torch.nn as nn
from torch.autograd import Variable
import math
import torch.nn.functional as F

def convLayer(in_channels,out_channels,keep_prob=0.0):
    #3 *3 卷积层，带padding，每次调用通过maxpool尺寸减半
    cnn_seq=nn.Sequential(
        nn.Conv2d(in_channels,out_channels,3,1,1), #尺寸不变
        nn.ReLU(True),
        nn.BatchNorm2d(out_channels),
        nn.MaxPool2d(kernel_size=2,stride=2),
        nn.Dropout(keep_prob)

    )
    return cnn_seq


class Classifier(nn.Module):
    def __init__(self,layer_size=64,num_channels=1,keep_prob=0.0,image_size=28):
        super(Classifier,self).__init__()
        """
        第一层网络：CNN 将输入图片进行embedding
        
        input: batch*channel*height*width   
        
        output:batch*size (64 default)
        
        """
        self.layer1=convLayer(num_channels,layer_size,keep_prob)
        self.layer2=convLayer(layer_size,layer_size,keep_prob)
        self.layer3=convLayer(layer_size,layer_size,keep_prob)
        self.layer4=convLayer(layer_size,layer_size,keep_prob)
        finalSize=int(math.floor(image_size/(2*2*2*2)))
        self.outSize=finalSize*finalSize*layer_size

    def forward(self,image_input):
        """
        :param image_input:  batch*channel*H*W 在这里就是 way*(shot+query), 1, 28,28
        :return: batch* layer_size   : way*(shot+query) , 64
        """
        x=self.layer1(image_input)
        x=self.layer2(x)
        x=self.layer3(x)
        x=self.layer4(x)
        x=x.view(x.size(0),-1)
        #print("输出Embedding的第一个图片看看",x[0])
        return x

class DistanceNetwork(nn.Module):
    """
    该模型计算输入的support set和query set的cos similarity
    """
    def __init__(self):
        super(DistanceNetwork,self).__init__()

    def forward(self,support_set,query_set):
        """
        :param support_set:  (way*shot), 64    5
        :param query_set:    (way*query), 64   25
        :return:  (way*query) , (way* shot) 每行代表第i和query与(way*shot)个support 的相似度
        """
        eps=1e-10
        sum_support=torch.sum(torch.pow(support_set,2),1) #(way*shot),1
        support_manitude=sum_support.clamp(eps,float("inf")).rsqrt() #1/根号x
       # print("support_mani",support_manitude.size())
        similarity=torch.bmm(query_set.unsqueeze(0),support_set.t().unsqueeze(0)).squeeze()  #(way*query),(way*shot)
        similarity=similarity*support_manitude.unsqueeze(0)



        return similarity

class BidirectionalLSTM(nn.Module):
    def __init__(self,layer_size,vector_dim,use_cuda,batch_size=1):
        super(BidirectionalLSTM,self).__init__()

        self.hidden_size=layer_size[0]
        self.vector_dim=vector_dim
        self.num_layer=len(layer_size)
        self.use_cuda=use_cuda
        self.lstm=nn.LSTM(input_size=self.vector_dim,num_layers=self.num_layer,hidden_size=self.hidden_size,bidirectional=True)

        self.batch_size=batch_size

    def init_hidden(self,use_cuda):
        if use_cuda:
            return (Variable(torch.zeros(self.lstm.num_layers * 2, self.batch_size, self.hidden_size)).cuda(),
                    Variable(torch.zeros(self.lstm.num_layers * 2, self.batch_size, self.hidden_size)).cuda())
        else:
            return (Variable(torch.zeros(self.lstm.num_layers * 2, self.batch_size, self.hidden_size)),
                    Variable(torch.zeros(self.lstm.num_layers * 2, self.batch_size, self.hidden_size)))

    # def repackage_hidden(self,h):
    #     if type(h)==Variable:
    #         return Variable(h.data)
    #     else:
    #         return tuple(self.repackage_hidden(v) for v in h)

    def forward(self,input,way,shot):
        """
        :param input:  (support+query) , 64
        :param way:
        :param shot:
        :return: support Embedding : (way*shot),64     (way*query),64
        """
        support_data=input[:way*shot].unsqueeze(1)  #way*shot 1 64  =  seq * batch * hidden_size

        query_data=input[way*shot:].unsqueeze(0)   # 1 way*query 64 =  seq * batch * hidden_size
        hidden=self.init_hidden(self.use_cuda)

        support_output,hidden=self.lstm(support_data,hidden) #support_output : way*shot , 1 , 64
        #print("hidden:",hidden)

        #将通过support data后的LSTM状态用来计算query set获得结果
        query_outputs=[]
        for i in range(query_data.size(1)):
            query_output,_=self.lstm(query_data[:,i,:].unsqueeze(1),hidden) #query_out：1 ，way*query , 64
            query_outputs.append(query_output)
        #print("query_outputs",query_outputs)
        query_outputs=torch.stack(query_outputs,dim=0).squeeze()
        return support_output.squeeze(),query_outputs



def get_one_hot_label(num_classes_per_set, shot_per_class):
    """

    :param num_classes_per_set: way
    :param shot_per_class:   shot
    :return:  one hot矩阵，(way*shot),way
    """
    one_hot=torch.zeros((num_classes_per_set*shot_per_class,num_classes_per_set))
    k=0
    for i in range(num_classes_per_set*shot_per_class):
        one_hot[i][k]=1
        if (i+1)%shot_per_class==0:
            k+=1
    return Variable(one_hot)


class MatchingNetwork(BasicModule):
    def __init__(self,keep_prob=0.0,num_channles=1,fce=False,
                 image_size=28,batch_size=1,use_cuda=False):
        super(MatchingNetwork,self).__init__()
        self.batch_size=batch_size
        self.keep_prob=keep_prob
        self.num_channels=num_channles
        self.fce=fce
        self.use_cuda=use_cuda
        self.image_size=image_size

        self.g=Classifier(layer_size=64,num_channels=num_channles,keep_prob=keep_prob,image_size=image_size)
        self.dn=DistanceNetwork()
        if self.fce:
            self.lstm=BidirectionalLSTM(layer_size=[32],vector_dim=self.g.outSize,use_cuda=use_cuda)

    def forward(self,input_image,num_classes_per_set,shot_per_class,query_per_class,):
        """
        :param input_image:  way*(shot+query) * 1 * 28 * 28

        :return:
        """

        embedding_image=self.g(input_image) #way*(shot+query)*64
        if self.fce:
            support_output,query_output=self.lstm(embedding_image,num_classes_per_set,shot_per_class)
        else:
            support_output=embedding_image[:num_classes_per_set*shot_per_class]
            query_output=embedding_image[num_classes_per_set*shot_per_class:]

        similarities=self.dn(support_output,query_output) # 每一行代表第i个query和各个support的相似度

        softmax=nn.Softmax(dim=1)
        softmax_similarities=softmax(similarities)
        one_hot_label=get_one_hot_label(num_classes_per_set,shot_per_class)
        if(self.use_cuda):
            one_hot_label=one_hot_label.cuda()
        predictions=torch.bmm(softmax_similarities.unsqueeze(0),one_hot_label.unsqueeze(0)).squeeze(0)

        target_y=Variable(torch.arange(num_classes_per_set))
        target_y = target_y.expand(query_per_class, target_y.size(0)).t().contiguous().view(-1)
        if(self.use_cuda):
            target_y=target_y.cuda()
        values,indexes=predictions.max(1)

        acc_val=torch.eq(indexes,target_y.long()).float().mean() #计算query set的正确率
        loss_val=F.cross_entropy(predictions,target_y.long())

        return acc_val,loss_val


