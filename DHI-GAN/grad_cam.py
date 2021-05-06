import os
from PIL import Image
import torch
import numpy as np
import cv2
import matplotlib.pyplot as plt
import os,sys,argparse
import config
import resnet
import DenseNet
from test import load_model, load_image
import metrics
import torch.nn as nn
import torchvision
from classifier import Classfier, TI_classifier
from msse import MSCANet



def draw_CAM(model, classifier, img_path, save_path, label, visual_heatmap=False):
    '''
    绘制 Class Activation Map
    :param model: 加载好权重的Pytorch model
    :param img_path: 测试图片路径
    :param save_path: CAM结果保存路径
    :param transform: 输入图像预处理方法
    :param visual_heatmap: 是否可视化原始heatmap（调用matplotlib）
    :return:
    '''
    # 图像加载&预处理
    img = load_image(img_file, True)
    img = torch.from_numpy(img).cuda()
    label = torch.from_numpy(np.array([label])).cuda()

    # 获取模型输出的feature/score
    features = model(img)

    cls_a, cls_b = classifier
    cls_feature = cls_a(features.view((1, -1)))
    output =  cls_b((cls_feature,label))

    # 为了能读取到中间梯度定义的辅助函数
    def extract(g):
        global features_grad
        features_grad = g

    # 预测得分最高的那一类对应的输出score
    pred = torch.argmax(output).item()
    pred_class = output[:, pred]

    features.register_hook(extract)
    pred_class.backward()  # 计算梯度

    grads = features_grad  # 获取梯度


    pooled_grads =  grads #torch.nn.functional.adaptive_avg_pool2d(grads, (1, 1)) ### mod by lyi #

    # 此处batch size默认为1，所以去掉了第0维（batch size维）
    pooled_grads = pooled_grads[0]#.view(-1)

    features = features[0]#.view(-1)
    # 512是最后一层feature的通道数
    for i in range(features.size(0)):
        features[i, ...] *= pooled_grads[i, ...]

    # 以下部分同Keras版实现
    heatmap = features.detach().cpu().numpy()
    heatmap = np.mean(heatmap, axis=0)

    heatmap = np.maximum(heatmap, 0)
    heatmap /= np.max(heatmap)

    # 可视化原始热力图
    if visual_heatmap:
        plt.matshow(heatmap)
        plt.show()

    img = cv2.imread(img_path)  # 用cv2加载原始图像
    h,w,c = img.shape #850 2100 3

    if configs.crop:
        wf, hf = 5, 4
        wf_per, hf_per = w // wf, h // hf
        rois = (wf_per, hf_per, w - wf_per, h - hf_per)
        img = img[hf_per:h - hf_per,wf_per:w - wf_per,:]

    #img = cv2.resize(img, (128,128))


    #heatmap = cv2.resize(heatmap, (img.shape[1], img.shape[0]))  # 将热力图的大小调整为与原始图像相同
    heatmap = cv2.resize(heatmap, (img.shape[1], img.shape[0]))  # 将热力图的大小调整为与原始图像相同

    heatmap = np.uint8(255 * heatmap)  # 将热力图转换为RGB格式
    heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)  # 将热力图应用于原始图像

    superimposed_img = heatmap * 0.8 + img*0.3  # 这里的0.4是热力图强度因子
    output_imgs = np.array([img, superimposed_img]).transpose((0,3,1,2))


    gridOfFakeImages = torchvision.utils.make_grid(torch.from_numpy(output_imgs), nrow=1, normalize = True)
    torchvision.utils.save_image(gridOfFakeImages,save_path)


    #cv2.imwrite(save_path, superimposed_img)  # 将图像保存到硬盘
    print('save gradcam result in {}'.format(save_path))



def parse_arguments(argv):
    parser = argparse.ArgumentParser()

    parser.add_argument('--savepath', help='savepath', type=str, required=True)
    parser.add_argument('--model', help='model type', type=str, default='res')
    parser.add_argument('--loss', help='loss', type=str, default='arc')
    parser.add_argument('--XR', help='XR path', type=str, default="../TMI_TI/data/Gray/label/Train.txt")
    parser.add_argument('--XT', help='XT path', type=str,  default="../TMI_TI/data/Gray/label/Train.txt")
    parser.add_argument('--crop', help='if crop image', action='store_true', default=False)


    return parser.parse_args(argv)

def load_descfile(desc_file):
    img_paths, labels = [],[]
    with open(desc_file, 'r') as fr:
        for line in fr:
            img_file, label = line.strip().split()
            img_file = './' + img_file.replace('\\', '/')
            img_file = os.path.join("../TMI_TI/data/", img_file)

            img_paths.append(img_file)
            labels.append(int(label))
    return img_paths, labels





if __name__ == '__main__':
    configs = parse_arguments(sys.argv[1:])

    opt = config.Config()
    opt.FirstFile = configs.savepath
    opt.FilenameXR = configs.XR
    opt.FilenameXT = configs.XT

    opt.checkpoints_path = os.path.join(opt.FirstFile, r"./savefile/checkpoints")
    opt.test_model_list = os.path.join(opt.FirstFile, r"./savefile/checkpoints")
    opt.backbone = configs.model
    embedding_size = 1024
    opt.metric = configs.loss


    print("*" * 30, opt.FirstFile, opt.backbone)

    export_dir =  os.path.join(opt.FirstFile, 'heatmap')
    if not os.path.exists(export_dir):
        os.makedirs(export_dir)

    device = torch.device("cuda:" + opt.gpu)
    if opt.backbone == 'dense':
        model = DenseNet.DenseNet121(embedding_size)
    elif opt.backbone == 'res':
        model = resnet.resnet_face18(embedding=embedding_size)
    else:
        model = MSCANet(input_dim=1, hidden_dim=32, reduction=4)
    model.to(device)

    if opt.metric == 'add':
        metric_fc = (metrics.AddMarginProduct(embedding_size, opt.num_classes, s=30, m=0.30))
    elif opt.metric == 'arc':
        metric_fc = (metrics.ArcMarginProduct(embedding_size,opt.num_classes))
        pass
    elif opt.metric =='sphere':
        metric_fc = (metrics.SphereProduct(embedding_size,opt.num_classes))
        pass
    else:
        metric_fc = (nn.Linear(embedding_size, opt.num_classes))
    metric_fc.to(device)

    models_list = []


    XR_files, XR_labels = load_descfile(opt.FilenameXR)
    XT_files, XT_labels = load_descfile(opt.FilenameXT)



    for root,dirs,files in os.walk(opt.test_model_list):
        for file in files:
            if not file.endswith('.pth') or  not file.startswith(opt.backbone):
                continue
            model_path = os.path.join(root,file)
            models_list.append(model_path)

    models_list = sorted(models_list, key=lambda x: os.path.getmtime(x), reverse=True)

    for model_path in models_list:
        try:
            load_model(model, model_path)
        except Exception as e:
            print('load embedding model error', model_path, str(e))
            continue

        try:
            cls_path = model_path.replace(opt.backbone, 'cls')
            load_model(metric_fc, cls_path)
        except Exception as e:
            print('load classifier model error', model_path, str(e))
            continue

        model.eval()
        metric_fc.eval()

        model_name = os.path.basename(model_path).replace('.pth', '')
        exp_dir =os.path.join(export_dir, model_name)
        if not os.path.exists(exp_dir):
            os.makedirs(exp_dir)
        for img_file, img_label in zip(XT_files, XT_labels):

            fb_name = os.path.basename(img_file)
            hm_save = os.path.join(exp_dir, fb_name)
            if not os.path.exists(img_file):
                continue
            draw_CAM(model.feature, (model.cls, metric_fc),img_file, hm_save,img_label)







