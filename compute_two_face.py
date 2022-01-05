import cv2
import os
import torch
import torch.nn as nn

from config import Config as config
from model import FaceNet
from PIL import Image
import numpy as np

#数据预处理
def _preprocess(images: list, transform) -> torch.Tensor:
    res = []
    for img in images:
        im = Image.open(f"D:\\data_test\\faceX_sdk\\{img}")
        im = transform(im)
        res.append(im)
    data = torch.cat(res, dim=0) #shape:[batch,128,128]
    data = data[:, None, :, :] #shape:[batcg,1,128,128]
    return data

#计算特征
#计算一批数据的特征，并返回一个数据字典
def featurize(images:list, transform, net, device) -> dict :
    res = {}
    if images != []:
        #对传入数据进行预处理
        data = _preprocess(images,transform)
        #print(f"data shape:{data.shape}\n")
        #将数据传入到相应设备上 cuda
        data = data.to(device)
        #将网络传入到相应设备上 cuda
        net = net.to(device)
        #因为是预测，不进行计算图的构建，不会被梯度
        with torch.no_grad():
            #调用网络处理数据，获取到传入的所有图片的特征值
            feature = net(data)
        #调用zip函数，形成img_name->feature映射的字典
        res = {img:feature for(img,feature) in zip(images,feature)}
    return res

#余弦距离
#这里采用余弦距离来度量两张人脸的距离，与训练过程相对应
def cosin_metric(x1,x2):
    return np.dot(x1,x2) / (np.linalg.norm(x1) * np.linalg.norm(x2))

if __name__ == '__main__':
    face_model = FaceNet(config.embedding_size)
    face_model = nn.DataParallel(face_model)
    face_model.load_state_dict(torch.load(config.test_model, map_location=config.device))
    face_model.eval()

    feature_dict = featurize(os.listdir("D:\\data_test\\faceX_sdk"), config.test_transform, face_model,
                             config.device)

    print(cosin_metric(feature_dict["2.jpg"].cpu().numpy(),feature_dict["20175119.jpg"].cpu().numpy()))