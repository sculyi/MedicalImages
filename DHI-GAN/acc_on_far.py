'''
author: lyi
date 20210225
desc calc the FAR

'''

import sys,os
def PostiveRate(threshold, PostiveScore):
    count = 0
    counta = 0
    for x in PostiveScore:
        if x <= threshold:
            counta += 1
        count += 1
    tpr = float(counta) / count
    return tpr

def NegtiveRate(threshold, NegtiveScore):
    count = 0
    counta = 0
    for x in NegtiveScore:
        if x <= threshold:
            counta += 1
        count += 1
    far = float(counta) / count
    return far

def cal_acc_and_far_from_file(sim_path, thres=1e-4):
    PostiveScore, NegtiveScore = readScoreFromFile(sim_path)
    ret =  cal_acc_and_far(PostiveScore, NegtiveScore,thres)
    #print(sim_path, ret)
    return ret


def readScoreFromFile(sim_path):
    PostiveScore, NegtiveScore = [],[]
    with open(sim_path,'r') as fr:
        for line in fr:
            sim_,cls1,cls2 = line.strip().split(',') #fp1,fp2,
            if cls1 == cls2:
                PostiveScore.append(float(sim_))
            else:
                NegtiveScore.append(float(sim_))

    return PostiveScore, NegtiveScore
def cal_acc_and_far(PostiveScore, NegtiveScore, thres):
    fac = 1000.
    thresholdList = list(range(0, int(fac)))
    thresholdx = [float(x) / fac for x in thresholdList]
    vals = []
    thres1, thres2, thres3 = 1e-6, 1e-5, 1e-4
    acc1, acc2, acc3 = None, None, None
    record_log = open('rrr.txt', 'w+')
    for x in thresholdx:
        Tpr = PostiveRate(x, PostiveScore)
        Far = NegtiveRate(x, NegtiveScore)

        vals.append((x, Tpr, Far))
        #print(x, Tpr, Far)
        record_log.write('{},{},{}\n'.format(x, Tpr, Far))
        if Far >= thres1 and acc1 is  None:
            acc1 = round(Tpr*100,3)
            #print(1,thres1, Tpr)
        elif Far >= thres2 and acc2 is  None:
            acc2 = round(Tpr*100,3)
            #print(2,thres2, Tpr)
        elif Far >= thres3 and acc3 is  None:
            acc3 = round(Tpr*100,3)
            #print(3,thres3, Tpr)
            #break
    record_log.close()
    return acc1, acc2, acc3


def sk_acc_far(filepath):
    scores, labels = [],[]
    import sklearn
    import numpy as np
    import sklearn.metrics
    with open(filepath,'r') as fr:
        for line in fr:
            segs = line.strip().split(',')
            scores.append(float(segs[0]))
            labels.append(int(segs[1]==segs[2]))
    fpr, tpr, thresholds = sklearn.metrics.roc_curve(labels, scores, pos_label=True)
    thres1, thres2, thres3 = 1e-6, 1e-5, 1e-4

    for t_ in [thres1, thres2, thres3]:
        idx1 = np.where(fpr>t_)[0][0]
        print(fpr[idx1], tpr[idx1], thresholds[idx1], t_)

    #print(filepath, fpr, tpr, thresholds)
    exit()



if __name__ == '__main__':
    #baiwan 1e-6
    #wan 1e-4
    model_path = os.path.join(sys.argv[1],'./savefile/checkpoints/')
    print('*'*30, model_path)
    all_files = os.listdir(model_path)
    #thres = float(sys.argv[2])
    for af in all_files:
        if not af.endswith('.sim'):
            continue
        ffp = os.path.join(model_path, af)
        #sk_acc_far(ffp)
        vals = cal_acc_and_far_from_file(ffp)
        #print(af, vals)
        exit()






