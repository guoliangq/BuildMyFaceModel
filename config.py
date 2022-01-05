import torch
import torchvision.transforms as T

class Config:
    #dataset
    train_root = "D:\\data\\CASIA-WebFace"
    test_root = "D:\\data\\lfw-align-128"
    test_list = "D:\\data\\lfw-align-128-test\\lfw_test_pair.txt"
    test_model = "D:\\code\\BuildMyFaceModel\\checkpoints1\\21.pth"

    #加载数据时的一些参数
    train_batch_size = 64
    test_batch_size = 16

    pin_memory = False #如果内存够大，设置成True，会加快速度
    num_workers = 1 #设置成大于1的数，可以多线程加载数据，但是我电脑上会报错 加载数据给dataloader

    #模型参数
    backbone = 'facenet' #使用的网络结构
    metric = 'arcface' #度量函数 [cosface,arcface]
    embedding_size = 512
    drop_ratio = 0.5

    epoch = 30 #数据要训练多少轮
    optimizer = 'sgd' #['sgd','adam'] 优化方法
    lr = 1e-1 #学习率
    lr_step = 10
    lr_decay = 0.95
    weight_decay = 5e-4
    loss = 'focal_loss' #['focal_loss','cross_entropy']
    device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
    checkpoints = 'checkpoints3' #存放权重文件的文件夹

    log_path = './jobs'

    #数据预处理
    input_shape = [1,128,128]
    train_transform = T.Compose([
        T.Grayscale(),
        T.RandomHorizontalFlip(),
        T.Resize((144,144)),
        T.RandomCrop(input_shape[1:]),
        T.ToTensor(),
        T.Normalize(mean=[0.5],std=[0.5]),
    ])
    test_transform = T.Compose([
        T.Grayscale(),
        T.Resize(input_shape[1:]),
        T.ToTensor(),
        T.Normalize(mean=[0.5],std=[0.5])
    ])
