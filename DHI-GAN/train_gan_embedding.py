'''
author: lyi
date 20210129
desc main file for DHI-GAN

'''


from __future__ import print_function
import os
import dataset
from torch.utils import data
import torch.nn.functional as F
import metrics
import torchvision
import torch
import numpy as np

import time
import config
from torch.optim.lr_scheduler import StepLR
import torch.nn as nn

import sys, argparse


from discriminator import TI_Dis
from generator import TI_Gen, TI_GenEmd_DS
from msse import MSCANet
import torch.distributed as distributed

def save_model(model, save_path, name, iter_cnt, optimizer, epoch, loss):
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    save_name = os.path.join(save_path, name + '_' + str(iter_cnt) + '.pth')
    state = {'model': model.state_dict(), 'optimizer': optimizer.state_dict(), 'epoch': epoch, 'loss': loss}

    torch.save(state, save_name)

    return save_name

def load_model(model, save_path):
    if os.path.isfile(save_path):
        checkpoint = torch.load(save_path)
        start_epoch = checkpoint['epoch']
        model.load_state_dict(checkpoint['model'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        loss = checkpoint['loss']
    else:
        raise FileNotFoundError("Can't find %s"%save_path)

    return model, optimizer, start_epoch, loss

def parse_arguments(argv):
    parser = argparse.ArgumentParser()

    parser.add_argument('--savepath', help='savepath', type=str, default='')

    parser.add_argument('--trainfile', help='desc file for train', type=str, default='../TMI_TI/label/train.txt')
    parser.add_argument('--XR', help='XR file for test', type=str, default='../TMI_TI/label/testXT.txt')
    parser.add_argument('--XT', help='XT file for test', type=str, default='../TMI_TI/label/testXR.txt')


    parser.add_argument('--model', help='model type', type=str, default='res')
    parser.add_argument('--loss', help='loss type', type=str, default='focal_loss')
    parser.add_argument('--metric', help='metrics', type=str, default='arc')
    parser.add_argument('--lr', help='metrics', type=float, default=5e-4)
    parser.add_argument('--bs', help='metrics', type=int, default=16)
    parser.add_argument('--embedding', type=int, help='embedding dimension', default=1024)
    parser.add_argument('--cof', type=float, help='confidence to update the classifier', default=0.15)
    parser.add_argument('--adv', type=float, help='adv', default=0.1)
    parser.add_argument('--crop', help='if crop image',  action='store_true', default=False)

    return parser.parse_args(argv)

def file_print(info, logfilename='log.txt',savepath = None, debug=True):
    t = time.localtime(int(time.time()))
    ts = time.strftime('%m-%d %H:%M:%S ', t)
    if savepath is not None:
        if not os.path.exists(savepath):
            os.mkdir(savepath)
        fullfilename = os.path.join(savepath,logfilename)
    else:
        fullfilename = logfilename

    if debug:
        print(ts + info)

    with open(fullfilename,'a') as f:
        f.write(ts+info+'\n')


if __name__ == '__main__':
    configs = parse_arguments(sys.argv[1:])


    opt = config.Config()
    ##overlapped parameter from cmdline
    opt.train_list = configs.trainfile
    opt.FilenameXT = configs.XT
    opt.FilenameXR = configs.XR


    opt.loss = configs.loss
    opt.backbone = configs.model
    opt.metric = configs.metric
    opt.lr = configs.lr
    opt.FirstFile = configs.savepath
    if opt.FirstFile == '':
        cur_time = time.strftime("%Y%m%d%H%M%S", time.localtime())
        opt.FirstFile = "./lyi/{}_{}/".format(os.path.basename(__file__).replace(".py", ''), cur_time)

    if not os.path.exists(opt.FirstFile):
        os.makedirs(opt.FirstFile)
    genimg_path = os.path.join(opt.FirstFile, "./content/gridOfFakeImages/")
    if not os.path.exists(genimg_path):
        os.makedirs(genimg_path)

    opt.checkpoints_path = os.path.join(opt.FirstFile , r"./savefile/checkpoints")
    opt.test_model_list = os.path.join(opt.FirstFile , r"./savefile/checkpoints")
    opt.max_epoch *= 2 #for gan training

    embedding_size = configs.embedding
    log_file = os.path.join(opt.FirstFile,'train.log')
    file_print('{}\n'.format(opt.__dict__),log_file)



    train_dataset = dataset.Dataset(opt.train_list, phase='train', input_shape=opt.input_shape)
    dp_last = True if train_dataset.__len__() % opt.train_batch_size == 1 else False
    trainloader = data.DataLoader(train_dataset,
                                  batch_size=opt.train_batch_size,
                                  shuffle=True,
                                  num_workers=opt.num_workers,drop_last=dp_last)

    file_print('{} train iters per epoch:'.format(len(trainloader)),log_file)
    if opt.loss == 'focal_loss':
        criterion = metrics.FocalLoss(gamma=2)
    else:
        criterion = torch.nn.CrossEntropyLoss()

    model = MSCANet(input_dim=1, hidden_dim=32, reduction=4)

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


    print(model)


    model = model.cuda()
    metric_fc = metric_fc.cuda()
    hidden_dim, image_channel = 64, 1
    noise_dim, input_embedding_dim = 100, 512
    advWeight = configs.adv
    confidenceThresh = configs.cof


    generator = TI_GenEmd_DS(input_dim = noise_dim, embedding_dim = input_embedding_dim, hidden_dim=hidden_dim, out_dim=image_channel).cuda()

    discriminator = TI_Dis(input_dim=image_channel, hidden_dim=hidden_dim).cuda()
    file_print('cof:{},adv:{}\n'.format(confidenceThresh, advWeight), log_file)


    if torch.cuda.device_count() > 1:
        distributed.init_process_group(
            backend='nccl',
            init_method='tcp://localhost:{}2345'.format(3),
            rank=0,
            world_size=1
        )
        model = torch.nn.parallel.DistributedDataParallel(model, find_unused_parameters=True)
        metric_fc = torch.nn.parallel.DistributedDataParallel(metric_fc, find_unused_parameters=True)

        generator = torch.nn.parallel.DistributedDataParallel(generator, find_unused_parameters=True)
        discriminator = torch.nn.parallel.DistributedDataParallel(discriminator, find_unused_parameters=True)


    gan_loss = nn.BCELoss()

    if opt.pretrained:
        if opt.continue_train:
            model, optimizer, start_epoch, loss = load_model(model, opt.load_model_path, opt.continue_train)
            print('加载 epoch {} 成功！'.format(start_epoch))
        else:
            checkpoint = torch.load(opt.load_model_path)
            pretrained_dict = checkpoint['model']
            model_dict = model.state_dict()
            pretrained_dict = {k: v for k, v in pretrained_dict.items() if (k in model_dict)}
            for k, v in model.named_parameters():
                if k != 'fc.weight' and k != 'fc.bias'and k != 'bn.weight' and k != 'bn.bias':
                    v.requires_grad = False  # 固定参数
            for k, v in model.named_parameters():
                print(k, v.requires_grad)
            #         print(v.requires_grad)  # 理想状态下，冻结层值都是False
            model_dict.update(pretrained_dict)
            model.load_state_dict(model_dict)
            start_epoch = 0
    else:
        start_epoch = 0

    if opt.optimizer == "sgd": # filter(lambda p: p.requires_grad, model.parameters())
        optimizer = torch.optim.SGD([{'params': model.parameters()}, {'params': metric_fc.parameters()}],
                                    lr=opt.lr, weight_decay=opt.weight_decay)
        optim_gen = torch.optim.SGD(generator.parameters(), lr=opt.lr, weight_decay=opt.weight_decay)
        optim_dis = torch.optim.SGD(discriminator.parameters(), lr=opt.lr, weight_decay=opt.weight_decay)
    elif opt.optimizer == "momentum":
        optimizer = torch.optim.SGD([{'params': model.parameters()}, {'params': metric_fc.parameters()}],
                                    lr=opt.lr, momentum=opt.momentum) # [model.fc5.weight, model.fc5.bias]
        optim_gen = torch.optim.SGD(generator.parameters(), lr=opt.lr, momentum=opt.momentum)
        optim_dis = torch.optim.SGD(discriminator.parameters(), lr=opt.lr, momentum=opt.momentum)

    elif opt.optimizer == "rmsprop":
        optimizer = torch.optim.RMSprop([{'params': model.parameters()}, {'params': metric_fc.parameters()}],
                                        lr=opt.lr,alpha=0.9)
        optim_gen = torch.optim.RMSprop(generator.parameters(), lr=opt.lr, alpha=0.9)
        optim_dis = torch.optim.RMSprop(discriminator.parameters(), lr=opt.lr, alpha=0.9)

    else:
        optimizer = torch.optim.Adam([{'params': model.parameters()}, {'params': metric_fc.parameters()}],
                                     lr=opt.lr, weight_decay=opt.weight_decay)
        optim_gen = torch.optim.Adam(generator.parameters(), lr=opt.lr, weight_decay=opt.weight_decay)
        optim_dis = torch.optim.Adam(discriminator.parameters(), lr=opt.lr, weight_decay=opt.weight_decay)


    scheduler = StepLR(optimizer, step_size=opt.lr_step, gamma=0.5)
    sch_gen = StepLR(optim_gen, step_size=opt.lr_step, gamma=0.5)
    sch_dis = StepLR(optim_dis, step_size=opt.lr_step, gamma=0.5)


    start = time.time()
    torch.manual_seed(np.random.randint(1000000))
    for epoch in range(start_epoch, opt.max_epoch+1): #用于继续训练


        model.train()
        for ii, data in enumerate(trainloader):
            data_input, label, data_aug, embedding = data
            data_input, data_aug = data_input.cuda(), data_aug.cuda()

            label = label.cuda().long()


            tmpBatchSize = len(label)
            #step 1: generate the images
            true_label = torch.ones(tmpBatchSize, 1).view(-1).cuda()
            fake_label = torch.zeros(tmpBatchSize, 1).view(-1).cuda()

            r = torch.randn(tmpBatchSize, noise_dim, 1, 1).cuda()
            embedding = embedding.view(tmpBatchSize, input_embedding_dim,1, 1).cuda()
            fakeImageBatch = generator(r, embedding)


            # train discriminator on real images
            optim_dis.zero_grad()  ########
            predictionsReal = discriminator(data_input)

            lossDiscriminator = gan_loss(predictionsReal, true_label)  # labels = 1
            lossDiscriminator.backward(retain_graph=True)

            # train discriminator on fake images
            predictionsFake = discriminator(fakeImageBatch)
            lossFake = gan_loss(predictionsFake, fake_label)  # labels = 0
            lossFake.backward(retain_graph=True)
            optim_dis.step()  # update discriminator parameters

            # train generator
            optim_gen.zero_grad()
            predictionsFake = discriminator(fakeImageBatch)
            lossGenerator = gan_loss(predictionsFake, true_label)  # labels = 1
            lossGenerator.backward(retain_graph=True)
            optim_gen.step()

            torch.autograd.set_detect_anomaly(True)
            fakeImageBatch = fakeImageBatch.detach().clone()



            # step train classifer with real data

            output = model(data_input)
            if opt.metric in ['add', 'arc', 'sphere']:
                output = metric_fc((output, label))
            else:
                output = metric_fc(output)

            loss = criterion(output, label)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # train with aug data
            output = model(data_aug)
            if opt.metric in ['add', 'arc', 'sphere']:
                output = metric_fc((output, label))
            else:
                output = metric_fc(output)

            loss = criterion(output, label)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            #train with generated data when meets the conditions
            # update the classifer on fake data
            predictionsFake = model(fakeImageBatch)
            if opt.metric in ['add', 'arc', 'sphere']:
                predictionsFake = metric_fc((predictionsFake, label))
            else:
                predictionsFake = metric_fc(predictionsFake)

            # get a tensor of the labels that are most likely according to model
            predictedLabels = torch.argmax(predictionsFake, 1)  # -> [0 , 5, 9, 3, ...]


            # psuedo labeling threshold
            probs = F.softmax(predictionsFake, dim=1)
            mostLikelyProbs = np.asarray([probs[i, predictedLabels[i]].item() for i in range(len(probs))])
            toKeep = mostLikelyProbs > confidenceThresh
            if sum(toKeep) != 0:
                cur_time = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())
                file_print('{} training netC using generated images...'.format(cur_time),log_file,debug=False)
                fakeClassifierLoss = criterion(predictionsFake[toKeep], predictedLabels[toKeep]) * advWeight
                fakeClassifierLoss.backward()
                optimizer.step()

            optimizer.zero_grad()
            optim_gen.zero_grad()
            optim_dis.zero_grad()


            iters = epoch * len(trainloader) + ii

            if iters % opt.print_freq == 0:
                output = output.data.cpu().numpy()
                output = np.argmax(output, axis=1)
                label = label.data.cpu().numpy()
                # print(output)
                # print(label)
                acc = np.mean((output == label).astype(int))
                speed = opt.print_freq / (time.time() - start)
                time_str = time.asctime(time.localtime(time.time()))
                strprint = '{} train epoch {} iter {} {:.6f} iters/s lr {:.6f} loss {:.6f} acc {}'.format(
                    time_str, epoch, ii, speed, scheduler.get_lr()[0], loss.item(), acc)
                file_print(strprint,log_file)


                start = time.time()
        scheduler.step()
        sch_dis.step()
        sch_gen.step()

        gridOfFakeImages = torchvision.utils.make_grid(fakeImageBatch.cpu())
        torchvision.utils.save_image(gridOfFakeImages,
                                     os.path.join(genimg_path, str(epoch) + '_' + str(epoch) + '_F.png'))

        # save  real image
        gridofRealImages = torchvision.utils.make_grid(data_input.cpu())
        torchvision.utils.save_image(gridofRealImages,
                                     os.path.join(genimg_path, str(epoch) + '_' + str(epoch) + '_R.png'))


        if epoch % opt.save_interval == 0 or epoch == opt.max_epoch:
            save_model(model, opt.checkpoints_path, opt.backbone, epoch, optimizer, epoch, loss)
            save_model(generator, opt.checkpoints_path, "gen", epoch, optim_gen, epoch, lossGenerator)
            save_model(discriminator, opt.checkpoints_path, "dis", epoch, optim_dis, epoch, lossDiscriminator)
            save_model(metric_fc, opt.checkpoints_path, "cls", epoch, optimizer, epoch, loss)



