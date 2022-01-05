import os
import os.path as osp

import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm

from model import FaceNet
from model.metric import  ArcFace,CosFace
from model.loss import FocalLoss
from dataset import load_data
from config import Config as conf
from torch.utils.tensorboard import SummaryWriter
from datetime import datetime

def train():
    #Data Setup
    dataloader, class_num = load_data(conf, training=True)
    embedding_size = conf.embedding_size
    device = torch.device(conf.device)

    print(device)
    #tensorboard
    current_time = datetime.now().strftime('%b%d_%H-%M-%S')
    # 初始化log_path,根据log_path,job_name,current_time进行区分
    log_path = '{}/{}'.format(conf.log_path, current_time)
    writer = SummaryWriter(log_path)
    #Network Setup
    net = FaceNet(embedding_size).to(device)

    if conf.metric == 'arcface':
        metric = ArcFace(embedding_size,class_num).to(device)
    else:
        metric = CosFace(embedding_size,class_num).to(device)

    net = nn.DataParallel(net)
    metric = nn.DataParallel(metric)

    #Training Setup
    if conf.loss == 'focal_loss':
        criterion = FocalLoss(gamma=2)
    else:
        criterion = nn.CrossEntropyLoss()

    if conf.optimizer == 'sgd':
        optimizer = optim.SGD([{'params':net.parameters()},{'params':metric.parameters()}],
                              lr=conf.lr,weight_decay=conf.weight_decay)
    else:
        optimizer = optim.Adam([{'params':net.parameters()},{'params':metric.parameters()}],
                               lr=conf.lr,weight_decay=conf.weight_decay)

    scheduler = optim.lr_scheduler.StepLR(optimizer,step_size=conf.lr_step,gamma=0.1)

    #创建权重文件夹
    os.makedirs(conf.checkpoints,exist_ok=True)

    #start training
    net.train()

    for e in range(conf.epoch):
        for data,labels in tqdm(dataloader,desc=f"Epoch {e}/{conf.epoch}",
                                ascii=True,total=len(dataloader)):
            data = data.to(device)
            labels = labels.to(device)

            optimizer.zero_grad()
            embedding = net(data)
            thetas = metric(embedding,labels)
            loss = criterion(thetas,labels)
            loss.backward()
            optimizer.step()
        writer.add_scalar(
                        'Training/Loss', loss,(e+1))
        print(f'Epoch {e}/{conf.epoch},Loss: {loss}')

        backbone_path = osp.join(conf.checkpoints,f"{e}.pth")
        torch.save(net.state_dict(),backbone_path)
        scheduler.step()

if __name__ == '__main__':
    train()