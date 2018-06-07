import os
import torch

import numpy as np
import torch.utils.data as data
from PIL import Image

"""
读取txt中所有类别
return:
[Artanic/character/rot000, Artanic/character/rot090,.....]
"""
def get_classes(root, mode):
    classes=[]
    data_path=os.path.join(root,'splits','vinyals',mode+'.txt')
    with open(data_path) as f:
        for line in f.readlines():
            line=line.strip()#删除头尾的'\n' '\r' '\t' ' '
            classes.append(line)
    return classes

"""
按照classes中存在的class来读取data数据
return:
[("0709_06.png","Artanic/character01","../data/ominiglot/data/Artanic/character01","/rot000"),.....]

"""
def get_all_items(root, classes):
    root_dir=os.path.join(root,'data')
    items=[]
    rots=['/rot000', '/rot090', '/rot180', '/rot270']

    for (root,dirs,files) in os.walk(root_dir):
        for f in files:
            root_split=root.split('/')
            label=os.path.join(root_split[-2],root_split[-1])
            for rot in rots:
                if label+rot in classes and f.endswith('png'):
                    items.append((f,label,root,rot))
    return items

"""
将classes按照 "Anatic/character01/rot000:1"这种格式写入字典
return:
dict

"""
def get_classes_index(classes):
    classes_index={}
    for i in classes:
        if i not in classes_index:
            classes_index[i]=len(classes_index)

    return classes_index

"""
return:
x : [Angelic/character01/0700_01.png/rot000,...]
y : [2,....]
"""
def get_x_y(all_items, classes_index):
    x=[]
    y=[]
    for item in all_items:
        item_y=classes_index[item[1]+item[-1]]
        item_x=os.path.join(item[2],item[0])+item[-1]
        x.append(item_x)
        y.append(item_y)

    return x,y



IMG_CACHE={}


def load_img(path):
    x_dict=[]
    for i in path:
        img_path,rot=i.split('/rot')
        if img_path in IMG_CACHE:
            x=IMG_CACHE[img_path]
        else:
            x=Image.open(img_path)
            IMG_CACHE[img_path]=x
        x=x.rotate(float(rot))
        x=x.resize((28,28))


        shape=1,x.size[0],x.size[1]
        x=np.array(x,np.float32,copy=False)
        x=1.0-torch.from_numpy(x)
        x=x.transpose(0,1).contiguous().view(shape)
        x_dict.append(x)

    return x_dict


class OmniglotDataSet(data.Dataset):

    def __init__(self,mode='trainval',root='./omniglot',transforms=None):
        super(OmniglotDataSet,self).__init__()
        self.root=root
        self.mode=mode
        self.transforms=transforms

        #判断数据集是否存在
        if not self.check_data_exit():
            raise RuntimeError('Omniglot数据集不在data/下，请下载后再使用')
        
        #加载指定数据集的所有类别
        self.classes=get_classes(self.root,self.mode)
        # print(len(self.classes),self.classes)
        self.all_items=get_all_items(self.root,self.classes)
        #print(len(self.all_items),self.all_items[:10])
        self.classes_index=get_classes_index(self.classes)
        #print(self.classes_index)
        path,self.y=get_x_y(self.all_items,self.classes_index)

        self.x=load_img(path)





    def check_data_exit(self):
        return os.path.exists(os.path.join(self.root,'data'))

    def __getitem__(self, index):
        x=self.x[index]
        y=self.y[index]
        if self.transforms:
            x=self.transforms(x)

        return x,y



    def __len__(self):
        return len(self.y)

