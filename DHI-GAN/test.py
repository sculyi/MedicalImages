from __future__ import print_function
import os,csv

import torch
import numpy as np
import config

from PIL import Image
from msse import MSCANet

counta = 0
count = 0
counta10 = 0
count10 = 0

def get_test_list(pair_list):
    with open(pair_list, 'r',encoding="utf-8") as fd:
        pairs = fd.readlines()
    data_list = []
    label_list = []
    cur_path = os.getcwd()

    for pair in pairs:
        splits = pair.split()
        img_path = './' + splits[0].replace('\\', '/')
        img_path = os.path.join(os.path.join(cur_path, "../TMI_TI/data/"), img_path)
        if not os.path.exists(img_path):
            continue

        data_list.append(img_path)
        label_list.append(splits[-1])
    print("*"*30, pair_list, len(data_list))
    return data_list, label_list


def load_image(img_path, crop=False):
    image = Image.open(img_path).convert("L")#cv2.imread(img_path, 0)
    if crop:
        w, h = image.size
        wf, hf = 5, 4
        wf_per, hf_per = w // wf, h // hf
        rois = (wf_per, hf_per, w - wf_per, h - hf_per)
        image = image.crop(rois)


    image = image.resize((128,128)) #cv2.resize(image,(128,128))
    if image is None:
        return None
    image = np.reshape(image, (128,128,1))
    image = image.transpose((2, 0, 1))
    image = image[:, np.newaxis, :, :]
    image = image.astype(np.float32, copy=False)
    #归一化，可有可无
    image -= 127.5
    image /= 127.5

    return image
def get_featurs(model, test_list, batch_size):
    images = None
    features = None
    cnt = 0
    for i, img_path in enumerate(test_list):
        image = load_image(img_path, configs.crop)
        if image is None:
            print('read {} error'.format(img_path))

        if images is None:
            images = image
        else:
            images = np.concatenate((images, image), axis=0)

        if images.shape[0] % batch_size == 0 or i == len(test_list) - 1:
            cnt += 1

            data = torch.from_numpy(images)
            data = data.to(device)

            output = model(data)
            output = output.view((output.size(0), -1)) #flatten

            output = output.data.cpu().numpy()

            feature = output
            if features is None:
                features = feature
            else:
                features = np.vstack((features, feature))

            images = None

    return features, cnt

def load_model(model, model_path):
    checkpoint = torch.load(model_path)
    model.load_state_dict(checkpoint['model'])
    return model

def get_feature_dict(test_list, features):
    fe_dict = {}
    # f = open(r"./fea_dict.csv", "w", newline="")
    # headers = ["pic", "feature"]
    # f_csv = csv.DictWriter(f, headers)
    # f_csv.writeheader()
    for i, each in enumerate(test_list):
        # key = each.split('/')[1]
        fe_dict[each] = features[i]
        # rows = [{"pic":each,"feature":features[i]}]
        # f_csv.writerows(rows)
    return fe_dict

def cosinDistance(probVectors,gallaryVector,labelXT,labelXR):
    """
    :param probVectors: XT测试集特征
    :param gallaryVector: XR注册集特征
    :param labelXT: 测试集标签
    :param labelXR: 注册集标签
    :return:
    """
    global count
    global counta
    global count10
    global counta10
    probVectors = np.vstack((probVectors))
    gallaryVector = gallaryVector.T #(512,5)
    # print(probVectors.shape) # (512,1)
    ga_id = None
    for j in range(probVectors.shape[1]):
        probstemp = probVectors[:,j]  # (512,1)
        probstempT = probstemp.T  # (1,512)
        lrs = np.zeros((1, gallaryVector.shape[1]))  # 1x
        lrs2 = np.zeros((1, gallaryVector.shape[1]))
        # print(lrs.shape,lrs2.shape)
        for i in range(gallaryVector.shape[1]):
            gallarystemp = gallaryVector[:,i]
            # rows = [{"pic": str(i), "feature": gallarystemp}]
            # f_csv.writerows(rows)
            gallarytempT = gallarystemp.T #(1,512)
            num = np.dot(probstempT, gallarystemp)  # (1,1)
            denum = np.sqrt(np.dot(probstempT, probstemp) * np.dot(gallarytempT, gallarystemp))
            cos = 1 - num / denum
            lrs[0][i] = cos
            lrs2[0][i] = 1-cos

        # top1
        minD = np.argmin(lrs,axis=1) # smaller -> better
        if labelXR[minD[0]] == labelXT:
            counta += 1
            ga_id = minD[0]
        else:#error recog
            pass
        #     print("top1预测：", labelXR[minD[0]], "真实", labelXT)
        # else:
        #     print("top1预测：", labelXR[minD[0]], "真实", labelXT)
        count += 1
        distance = np.array(lrs2[0])
        minD10 = np.argpartition(distance, -10)[-10:]  # 余弦相似度最大的下标
        for x in range(10):
            if labelXR[minD10[x]] == labelXT:
                counta10 += 1
                break
            """ # 需要时打印
            #else:
                #print("预测top10：", labelXR[minD10[x]], "真实", labelXT[i])
            """
        count10 += 1  # top10比较次数加1

    return lrs, lrs2, ga_id
import time
import sys, argparse
def parse_arguments(argv):
    parser = argparse.ArgumentParser()

    parser.add_argument('--savepath', help='savepath', type=str, required=True)
    parser.add_argument('--XR', help='XR path', type=str, default="../TMI_TI/label/testXR.txt")
    parser.add_argument('--XT', help='XT path', type=str, default="../TMI_TI/label/testXT.txt")
    parser.add_argument('--model', help='savepath', type=str, default='res')
    parser.add_argument('--crop', help='if crop image',  action='store_true', default=False)


    return parser.parse_args(argv)
from acc_on_far import cal_acc_and_far_from_file
if __name__ == '__main__':
    configs = parse_arguments(sys.argv[1:])

    opt = config.Config()
    opt.FirstFile = configs.savepath

    opt.checkpoints_path = os.path.join(opt.FirstFile, r"./savefile/checkpoints")
    opt.test_model_list = os.path.join(opt.FirstFile, r"./savefile/checkpoints")
    opt.backbone = configs.model
    embedding_size = 1024
    print('-'*30, configs)
    print("*" * 30, opt.FirstFile, opt.backbone)

    opt.FilenameXR = configs.XR
    opt.FilenameXT = configs.XT


    device = torch.device("cuda:"+ opt.gpu)

    model = MSCANet(input_dim=1, hidden_dim=32, reduction=4)

    model.to(device)
    imgpath_XR, label_XR = get_test_list(opt.FilenameXR)
    imgpath_XT, label_XT = get_test_list(opt.FilenameXT)
    models_list = []
    acc_file = open(os.path.join(opt.checkpoints_path,r"acc_test.txt"),"w+",encoding="utf-8")
    test_idx = 0
    for root,dirs,files in os.walk(opt.test_model_list):
        for file in files:
            if not file.endswith('.pth') or  not file.startswith(opt.backbone):
                continue
            model_path = os.path.join(root,file)
            models_list.append(model_path)

    models_list = sorted(models_list, key=lambda x: os.path.getmtime(x), reverse=True)

    if True:
        for model_path in models_list:
            try:
                load_model(model, model_path)
            except Exception as e:
                print('load model error', model_path, str(e))
                continue
            model.eval()
            # TODO calc similarity for all sample pairs and obtain the min/max thresold

            sim_path = model_path.replace('.pth','.sim')
            sim_file = open(sim_path,'w+', encoding="utf-8")

            features_XR, _ = get_featurs(model, imgpath_XR, opt.test_batch_size)

            # fe_dict_XR = get_feature_dict(imgpath_XR,features_XR)
            features_XT, _ = get_featurs(model, imgpath_XT, opt.test_batch_size)
            # fe_dict_XT = get_feature_dict(imgpath_XT,features_XT)
            test_idx += 1
            for i,t_path in enumerate(imgpath_XT):
                lrs, lrs2, ga_id = cosinDistance(features_XT[i], features_XR, label_XT[i], label_XR)
                if ga_id is  None:
                    print('***ERROR', t_path )

                for real_idx in range(len(imgpath_XR)):
                    r_path, cos_sim = imgpath_XR[real_idx], lrs[0][real_idx]
                    sim_file.write("{},{},{}\n".format( cos_sim, label_XT[i], label_XR[real_idx]))




            sim_file.close()
            #acc_file.write(file+"\n")
            #print("testing model:",file)
            #print("true:", counta, "try num:", count)
            acc = float(counta) / count  # 判断正确/比较次数
            #print("Accuracy(Top1):", acc)
            ## TOP5
            acc10 = float(counta10) / count10  # 判断正确/比较次数
            #print("Accuracy(Top10):", acc10)
            cur_time = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())
            accs = cal_acc_and_far_from_file(sim_path)
            strinfo = "{},({}/{}),{},{},{},{}\n".format(cur_time, test_idx, len(models_list), model_path, acc, acc10, accs)
            print(strinfo.strip())
            acc_file.write(strinfo)
            """下一个模型，重新附初值"""
            counta = 0
            count = 0
            counta10 = 0
            count10 = 0
        acc_file.close()

        #finally run the acc_on_far
        #print('run acc_on_far.py here...')
        #os.system('python acc_on_far.py {}'.format(opt.FirstFile))


            

