import os
import os.path as osp

import torch
import torch.nn as nn
import numpy as np
from PIL import Image

from config import Config as conf
from model import FaceNet

#由于6000个测试用例中，测试用例是有重复的
#所以先获取每一个不重复图片的路径
def unique_image(pair_list) -> set:
    with open(pair_list,'r') as file:
        pairs = file.readlines()
    res = set()
    for pair in pairs:
        path1, path2, _ = pair.split(" ")
        res.add(path1)
        res.add(path2)
    return res

#将图片分组
def group_image(images:set, batch) -> list:
    images = list(images)
    res = []
    for i in range(0,len(images),batch):
        #防止数组越界
        end = min(len(images),i+batch)
        res.append(images[i:end])
    return res

#数据预处理
def _preprocess(images: list, transform) -> torch.Tensor:
    res = []
    for img in images:
        im = Image.open(img)
        im = transform(im)
        res.append(im)
    data = torch.cat(res,dim=0) #shape:[batch,128,128]
    data = data[:, None, :, :] #shape:[batcg,1,128,128]
    return data

#计算特征
#计算一批数据的特征，并返回一个数据字典
def featurize(images:list, transform, net, device) -> dict :
    data = _preprocess(images,transform)
    data = data.to(device)
    net = net.to(device)
    with torch.no_grad():
        feature = net(data)
    res = {img:feature for(img,feature) in zip(images,feature)}
    return res

#余弦距离
#这里采用余弦距离来度量两张人脸的距离，与训练过程相对应
def cosin_metric(x1,x2):
    return np.dot(x1,x2) / (np.linalg.norm(x1) * np.linalg.norm(x2))

#人脸划分
#当A，B两张人脸的距离有多大时，才认为A、B是不同的人
def threshold_search(y_score, y_true):
    y_score = np.asarray(y_score)
    y_true = np.asarray(y_true)
    best_acc = 0
    best_th = 0
    for i in range(len(y_score)):
        th = y_score[i]
        y_test = (y_score >= th)
        acc = np.mean((y_test == y_true).astype(int))
        if acc > best_acc:
            best_acc = acc
            best_th = th
    return best_acc,best_th

#计算准确率
def compute_accuracy(feature_dict, pair_list, test_root):

    with open(pair_list,'r') as f:
        pairs = f.readlines()

    similarities = []
    labels = []
    for pair in pairs:
        img1, img2, label = pair.split()
        img1 = osp.join(test_root,img1)
        img2 = osp.join(test_root,img2)
        feature1 = feature_dict[img1].cpu().numpy()
        feature2 = feature_dict[img2].cpu().numpy()
        label = int(label)

        similarity = cosin_metric(feature1,feature2)
        similarities.append(similarity)
        labels.append(label)

    accuracy, threshold = threshold_search(similarities,labels)
    return accuracy, threshold

if __name__ == '__main__':

    model = FaceNet(conf.embedding_size)
    model = nn.DataParallel(model)
    model.load_state_dict(torch.load(conf.test_model,map_location=conf.device))
    model.eval()

    images = unique_image(conf.test_list)
    images = [osp.join(conf.test_root,img) for img in images]
    groups = group_image(images,conf.test_batcg_size)

    feature_dict = dict()
    for group in groups:
        d = featurize(group,conf.test_transform,model,conf.device)
        feature_dict.update(d)

    accuracy,threshold = compute_accuracy(feature_dict,conf.test_list,conf.test_root)
    print(
        f"Test Model: {conf.test_model}\n"
        f"Accuracy: {accuracy:.3f}\n"
        f"Threshold:{threshold:.3f}\n"
    )