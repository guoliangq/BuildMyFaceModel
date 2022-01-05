from torch.utils.data import DataLoader
from torchvision.datasets import ImageFolder

from config import Config as conf

#判断是训练还是测试阶段，使用了conf中的不同参数
#生成了一个ImageFolder对象，而且对数据进行了transform，得到data
#将data传入DataLoader，指定每次迭代的batch_size和其他参数
def load_data(conf, training=True):
    if training:
        dataroot = conf.train_root
        transform = conf.train_transform
        batch_size = conf.train_batch_size
    else:
        dataroot = conf.test_root
        transform = conf.test_transform
        batch_size = conf.test_batch_size

    data = ImageFolder(dataroot,transform=transform)
    class_num = len(data.classes)
    loader = DataLoader(data, batch_size=batch_size,shuffle=True,
                        pin_memory=conf.pin_memory,num_workers=conf.num_workers)
    return loader,class_num