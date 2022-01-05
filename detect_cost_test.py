import argparse
import time
import os
from pathlib import Path

import cv2
import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
from PIL import Image
from numpy import random
import numpy as np

from models.experimental import attempt_load
from silentFace_model import AntiSpoofPredict
from silentFace_model.utility import parse_model_name
from silentFace_model.generate_patches import CropImage
from utils.datasets import LoadStreams, LoadImages
from utils.general import check_img_size, check_requirements, check_imshow, non_max_suppression, apply_classifier, \
    scale_coords, xyxy2xywh, strip_optimizer, set_logging, increment_path
from utils.plots import plot_one_box
from utils.torch_utils import select_device, load_classifier, time_synchronized

from config import Config as config
from model import FaceNet

import pymysql
import time

#数据预处理
def _preprocess(images: list, transform) -> torch.Tensor:
    res = []
    for img in images:
        im = Image.open(f"D:\\recordsystem\\facephoto1\\{img}")
        im = transform(im)
        res.append(im)
    data = torch.cat(res,dim=0) #shape:[batch,128,128]
    data = data[:, None, :, :] #shape:[batcg,1,128,128]
    return data

#计算特征
#计算一批数据的特征，并返回一个数据字典
def featurize(images:list, transform, net, device) -> dict :
    if images != []:
        # 对传入数据进行预处理
        data = _preprocess(images, transform)
        # print(f"data shape:{data.shape}\n")
        # 将数据传入到相应设备上 cuda
        data = data.to(device)
        # 将网络传入到相应设备上 cuda
        net = net.to(device)
        # 因为是预测，不进行计算图的构建，不会被梯度
        with torch.no_grad():
            # 调用网络处理数据，获取到传入的所有图片的特征值
            feature = net(data)
        # 调用zip函数，形成img_name->feature映射的字典
        res = {img: feature for (img, feature) in zip(images, feature)}
    return res

#余弦距离
#这里采用余弦距离来度量两张人脸的距离，与训练过程相对应
def cosin_metric(x1,x2):
    return np.dot(x1,x2) / (np.linalg.norm(x1) * np.linalg.norm(x2))

results = {0:'all_cost.txt',1:'yolo_cost.txt',2:'face_cost.txt',3:'spoof_cost.txt'}

def generate_result(kv, num):
    file = open(results[num], 'w')
    for key in kv.keys():
        file.write(f"{key} {kv[key]}\n")
    file.close()


def detect(save_img=False, config=config):
    source, weights, view_img, save_txt, imgsz = opt.source, opt.weights, opt.view_img, opt.save_txt, opt.img_size
    save_img = not opt.nosave and not source.endswith('.txt')  # save inference images
    webcam = source.isnumeric() or source.endswith('.txt') or source.lower().startswith(
        ('rtsp://', 'rtmp://', 'http://'))
    #连接数据库
    db = pymysql.connect(host="localhost", user="root", password="zxd990501", database="recordsystem")
    cursor = db.cursor()
    createdate = time.strftime("%Y-%m-%d", time.localtime())

    #load face model
    face_model = FaceNet(config.embedding_size)
    face_model = nn.DataParallel(face_model)
    face_model.load_state_dict(torch.load(config.test_model, map_location=config.device))
    face_model.eval()

    feature_dict = featurize(os.listdir("D:\\recordsystem\\facephoto1"),config.test_transform,face_model,config.device)

    #slient face detect
    pred_model = AntiSpoofPredict(0)
    image_cropper = CropImage()

    # Directories
    save_dir = Path(increment_path(Path(opt.project) / opt.name, exist_ok=opt.exist_ok))  # increment run
    (save_dir / 'labels' if save_txt else save_dir).mkdir(parents=True, exist_ok=True)  # make dir

    # Initialize
    set_logging()
    device = select_device(opt.device)
    half = device.type != 'cpu'  # half precision only supported on CUDA

    # Load model
    model = attempt_load(weights, map_location=device)  # load FP32 model
    stride = int(model.stride.max())  # model stride
    imgsz = check_img_size(imgsz, s=stride)  # check img_size
    if half:
        model.half()  # to FP16

    # Second-stage classifier
    classify = False
    if classify:
        modelc = load_classifier(name='resnet101', n=2)  # initialize
        modelc.load_state_dict(torch.load('weights/resnet101.pt', map_location=device)['model']).to(device).eval()

    # Set Dataloader
    vid_path, vid_writer = None, None
    if webcam:
        view_img = check_imshow()
        cudnn.benchmark = True  # set True to speed up constant image size inference
        dataset = LoadStreams(source, img_size=imgsz, stride=stride)
    else:
        dataset = LoadImages(source, img_size=imgsz, stride=stride)

    # Get names and colors
    names = model.module.names if hasattr(model, 'module') else model.names
    colors = [[random.randint(0, 255) for _ in range(3)] for _ in names]

    # Run inference
    if device.type != 'cpu':
        model(torch.zeros(1, 3, imgsz, imgsz).to(device).type_as(next(model.parameters())))  # run once
    t0 = time.time()
    all_cost = {}
    yolo_cost = {}
    face_cost = {}
    spoof_cost = {}
    count = 0
    for path, img, im0s, vid_cap in dataset:
        count += 1
        all_cost[count] = 0
        yolo_cost[count] = 0
        face_cost[count] = 0
        spoof_cost[count] = 0
        all_start = time.time()
        img = torch.from_numpy(img).to(device)
        img = img.half() if half else img.float()  # uint8 to fp16/32
        img /= 255.0  # 0 - 255 to 0.0 - 1.0
        if img.ndimension() == 3:
            img = img.unsqueeze(0)

        # Inference
        t1 = time_synchronized()
        yolo_start = time.time()
        pred = model(img, augment=opt.augment)[0]
        yolo_cost[count] = yolo_cost[count] + time.time()-yolo_start
        # Apply NMS
        pred = non_max_suppression(pred, opt.conf_thres, opt.iou_thres, classes=opt.classes, agnostic=opt.agnostic_nms)
        t2 = time_synchronized()

        # Apply Classifier
        if classify:
            pred = apply_classifier(pred, modelc, img, im0s)

        # Process detections
        for i, det in enumerate(pred):  # detections per image
            if webcam:  # batch_size >= 1
                p, s, im0, frame = path[i], '%g: ' % i, im0s[i].copy(), dataset.count
            else:
                p, s, im0, frame = path, '', im0s, getattr(dataset, 'frame', 0)

            p = Path(p)  # to Path
            save_path = str(save_dir / p.name)  # img.jpg
            txt_path = str(save_dir / 'labels' / p.stem) + ('' if dataset.mode == 'image' else f'_{frame}')  # img.txt
            s += '%gx%g ' % img.shape[2:]  # print string
            gn = torch.tensor(im0.shape)[[1, 0, 1, 0]]  # normalization gain whwh
            if len(det):
                # Rescale boxes from img_size to im0 size
                det[:, :4] = scale_coords(img.shape[2:], det[:, :4], im0.shape).round()

                # Print results
                for c in det[:, -1].unique():
                    n = (det[:, -1] == c).sum()  # detections per class
                    s += f"{n} {names[int(c)]}{'s' * (n > 1)}, "  # add to string

                # Write results
                #调用yolov5处理后获取到人脸的bbox
                for *xyxy, conf, cls in reversed(det):
                    #crop
                    #print(f"img0 shape:{im0.shape}\n")
                    #初始化prediction
                    prediction = np.zeros((1, 3))
                    #根据bbox裁剪出人脸的照片
                    face_img = im0[int(xyxy[1]):int(xyxy[3]), int(xyxy[0]):int(xyxy[2])]
                    #将图片大小重置为活体检测的输入大小
                    rf_img = cv2.resize(face_img, (80, 80))
                    #将array类型的数据转化成PIL格式
                    face_img = Image.fromarray(face_img, 'RGB')
                    #对人脸图片进行处理
                    face_img = config.test_transform(face_img)
                    #扩展维度，以符合人脸检测模型的输入维度
                    face_img = face_img[:,None,:,:]
                    #print(f"face_img shape:{face_img.shape}")
                    #将数据加载到设备上 cuda:0
                    face_data = face_img.to(conf.device)
                    #调用人脸检测模型获取到该图片的特征值
                    face_start = time.time()
                    _feature = face_model(face_data)  # 获取特征
                    face_cost[count] = face_cost[count] + time.time()-face_start
                    #转化成numpy数组
                    _feature = _feature.data.cpu().numpy()
                    #初始化label
                    label = "none"
                    #获取数据库中所存的所有照片
                    list = os.listdir("D:\\recordsystem\\facephoto1")
                    #初始化max_f
                    max_f = 0
                    #初始化t
                    t = 0
                    #print("start test.\n")
                    #遍历数据库中的所有图片，求取传入图片人脸与数据库中图片的余弦距离
                    for i, each in enumerate(list):
                        #求取传入图片与数据库中图片的余弦距离
                        t = cosin_metric(feature_dict[each].cpu().numpy(), _feature[0])
                        #print(i)
                        #如果当前余弦距离大于保存的距离，则将max_f设置为当前值，并保存当前数据库中该图片的名称
                        if t > max_f:
                            max_f = t
                            max_n = each
                        #print(max_n,max_f)
                        #如果最大余弦距离大于阈值，则根据与其对应的数据库图片名称获取到传入图片的人脸对应的名称
                        if (max_f > 0.44):
                            #根据图片名字获取到label
                            label = max_n.split(".")[0]
                    print(xyxy,label)
                    print("end test.\n")
                    '''
                    for model_name in os.listdir("weights/anti_spoof_models"):
                        print(model_test.predict(img, os.path.join(model_dir, model_name)))

                        prediction += pred_model.predict(rf_img, os.path.join("weights\\anti_spoof_models", model_name))
                    '''
                    #遍历目录下所有活体检测模型，多模型融合的方法

                    for model_name in os.listdir("weights/anti_spoof_models"):
                        #根据模型的需要去获取对应的输入数据
                        h_input, w_input, model_type, scale = parse_model_name(model_name)
                        param = {
                            "org_img": rf_img,"bbox": xyxy,"scale": scale,"out_w": w_input,
                            "out_h": h_input,"crop": True,
                        }
                        if scale is None:
                            param["crop"] = False
                        img = image_cropper.crop(**param)
                        #各模型预测值相加
                        spoof_start = time.time()
                        prediction += pred_model.predict(img, os.path.join("weights\\anti_spoof_models", model_name))
                        spoof_cost[count] = spoof_cost[count] + time.time() - spoof_start
                    rf_label = np.argmax(prediction)
                    value = prediction[0][rf_label] / len(os.listdir("weights/anti_spoof_models"))
                    print(rf_label, value)
                    #if rf_label == 1 and value > 0.90:
                    if rf_label == 1 and value > 0.80:
                        # 在识别出真脸并且该人脸是数据字典库中的人脸时，连接数据库，保存该人脸的此次打卡记录
                        if label != 'none':
                            updatedate = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())
                            cursor.execute(
                                f"update record set success=1,updatedate='{updatedate}' where account ='{label}' and createdate='{createdate}'")
                            db.commit()
                        label += "_success"
                    else:
                        label += "_fail"
                    plot_one_box(xyxy, im0, label=label, color=colors[int(cls)], line_thickness=3)

            # Print time (inference + NMS)
            print(f'{s}Done. ({t2 - t1:.3f}s)')

            # Stream results
            if view_img:
                cv2.imshow(str(p), im0)
                cv2.waitKey(1)  # 1 millisecond

            # Save results (image with detections)
            if save_img:
                if dataset.mode == 'image':
                    cv2.imwrite(save_path, im0)
                else:  # 'video' or 'stream'
                    if vid_path != save_path:  # new video
                        vid_path = save_path
                        if isinstance(vid_writer, cv2.VideoWriter):
                            vid_writer.release()  # release previous video writer
                        if vid_cap:  # video
                            fps = vid_cap.get(cv2.CAP_PROP_FPS)
                            w = int(vid_cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                            h = int(vid_cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                        else:  # stream
                            fps, w, h = 30, im0.shape[1], im0.shape[0]
                            save_path += '.mp4'
                        vid_writer = cv2.VideoWriter(save_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (w, h))
                    vid_writer.write(im0)
        all_cost[count] += all_cost[count] + time.time() - all_start
    generate_result(all_cost,0)
    generate_result(yolo_cost,1)
    generate_result(face_cost,2)
    generate_result(spoof_cost,3)
    if save_txt or save_img:
        s = f"\n{len(list(save_dir.glob('labels/*.txt')))} labels saved to {save_dir / 'labels'}" if save_txt else ''
        print(f"Results saved to {save_dir}{s}")

    print(f'Done. ({time.time() - t0:.3f}s)')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--weights', nargs='+', type=str, default='yolov5s.pt', help='model.pt path(s)')
    parser.add_argument('--source', type=str, default='data/images', help='source')  # file/folder, 0 for webcam
    parser.add_argument('--img-size', type=int, default=640, help='inference size (pixels)')
    parser.add_argument('--conf-thres', type=float, default=0.25, help='object confidence threshold')
    parser.add_argument('--iou-thres', type=float, default=0.45, help='IOU threshold for NMS')
    parser.add_argument('--device', default='', help='cuda device, i.e. 0 or 0,1,2,3 or cpu')
    parser.add_argument('--view-img', action='store_true', help='display results')
    parser.add_argument('--save-txt', action='store_true', help='save results to *.txt')
    parser.add_argument('--save-conf', action='store_true', help='save confidences in --save-txt labels')
    parser.add_argument('--nosave', action='store_true', help='do not save images/videos')
    parser.add_argument('--classes', nargs='+', type=int, help='filter by class: --class 0, or --class 0 2 3')
    parser.add_argument('--agnostic-nms', action='store_true', help='class-agnostic NMS')
    parser.add_argument('--augment', action='store_true', help='augmented inference')
    parser.add_argument('--update', action='store_true', help='update all models')
    parser.add_argument('--project', default='runs/detect', help='save results to project/name')
    parser.add_argument('--name', default='exp', help='save results to project/name')
    parser.add_argument('--exist-ok', action='store_true', help='existing project/name ok, do not increment')
    opt = parser.parse_args()
    print(opt)
    check_requirements(exclude=('pycocotools', 'thop'))

    with torch.no_grad():
        if opt.update:  # update all models (to fix SourceChangeWarning)
            for opt.weights in ['yolov5s.pt', 'yolov5m.pt', 'yolov5l.pt', 'yolov5x.pt']:
                detect()
                strip_optimizer(opt.weights)
        else:
            detect()
