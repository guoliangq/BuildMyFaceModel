# BuildMyFaceModel
基于MobileFaceNet的静默活体检测系统的设计与实现

1、代码git地址 

https://github.com/gl0513/BuildMyFaceModel.git

2、实验环境

Windows 10（64位）

CPU：AMD Ryzen 7 5800H 

RAM：16G

GPU：NVIDIA RTX3060

开发工具：IntelliJ IDEA以及PyCharm

相关配置及版本：

Chrome          	90.0.4430.212

SpringBoot                   	2.2.6

Java	                         JDK8

MyBatis	                        2.1.1

Mysql	                       8.0.25

Python	                          3.8

pytorch	                        1.7.1

torchvision	                0.8.2

numpy              	       1.18.5

tensorboard                   	2.4.1

pandas           	        1.2.3

cuda	                       11.0.2

cudnn	                         11.2

torch	                        1.8.1

torchvision             	0.9.1

3、模型训练命令

python train.py 

4、模型测试命令

python test.py 


5、运行检测模块命令

python detect.py --source 0

6、简要说明

  该课程作业成功的基于轻量级网络MobileFaceNet实现了检测功能，并基于此搭建了静默活体检测系统，如果代码运行有任何问题，请联系微信13940227331
