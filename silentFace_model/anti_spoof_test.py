from silentFace_model import AntiSpoofPredict
from PIL import Image
import torch
'''
想法：测试模型CelebA_Spoof测试数据集上的准确率
应该做的工作：
1.生成对应的测试列表 Done D:\\data\\CelebA_Spoof\\metas\\intra_test\\test_label.txt
2.分组读入测试数据
3.通过模型计算得到预测值
4.与真实值进行比较，计算得到精确率，召回率指标
'''

#读入label文件，获取到测试数据地址以及测试数据->真实值的字典
def read_test_label():
    test_label_path = "D:\\data\\CelebA_Spoof\\metas\\intra_test\\test_label.txt"
    file = open(test_label_path,'r')
    lines = file.readlines()
    true_value_kv = {}
    images_list = []
    for line in lines:
        img_path,label = line.split(" ")
        images_list.append(img_path)
        true_value_kv[img_path] = int(label)
    return images_list,true_value_kv

#对读入的图片进行分组
def group_image(images:list,batch):
    size = len(images)
    res = []
    for i in range(0,size,batch):
        end = min(i+batch,size)
        res.append(images[i:end])
    return res

#对测试的数据进行预处理
def preprocess(images:list,transform):
    res = []
    for img in images:
        im = Image.open(img)
        im = transform(im)
        res.append(im)
    data = torch.cat(res,dim=0)
    print(data.shape)
    return data





def test():
    # slient face detect
    pred_model = AntiSpoofPredict(0)