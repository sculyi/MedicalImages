import os,cv2,sys,random
import numpy as np
from PIL import Image
import torch,torchvision
from torch.utils import data
from torchvision import transforms as T
import h5py


class Dataset(data.Dataset):
    def __init__(self,data_list_file,phase="train", input_shape=(1,128,128), crop=False):
        self.phase = phase
        self.input_shape = input_shape
        self.crop = crop
        self.imgs = []
        cur_path = os.getcwd()
        with open(os.path.join(data_list_file),"r",encoding="utf-8") as fd:
            #imgs = fd.readlines()
            for line in fd:
                splits = line.strip().split()
                img_path = './' + splits[0].replace('\\', '/')
                img_path = os.path.join(os.path.join(cur_path, "../TMI_TI/data/"), img_path)
                if not os.path.exists(img_path):
                    continue
                self.imgs.append("{} {}".format(img_path, splits[1]))

        print(phase,'data num', len(self.imgs))
        self.imgs = np.random.permutation(self.imgs)

        normalize = T.Normalize(mean=[0.5],std=[0.5])

        if self.phase == "train":
            self.transforms = T.Compose([
                # T.CenterCrop((650,1700)),
                # T.RandomHorizontalFlip(),
                T.Resize((128,128)),
                T.ToTensor(),
                normalize
            ])
        else:
            self.transforms = T.Compose([
                # T.CenterCrop((650,1700)),
                T.Resize((128,128)),
                T.ToTensor(),
                normalize
            ])

    def __getitem__(self, index):


        sample = self.imgs[index]

        splits = sample.strip().split()
        img_path = splits[0]

        data = Image.open(img_path).convert("L")#(2100,850)
        # to crop the image
        if self.crop:
            w, h = data.size
            wf, hf = 5, 4
            wf_per, hf_per = w//wf, h//hf
            rois = (wf_per, hf_per, w-wf_per,h-hf_per)
            data = data.crop(rois)  ###20210224  to test

        data = self.transforms(data)
        data = data.float()

        # 随机遮挡图片
        _,height, width = data.shape
        boxh = random.randrange(15, 30) # size500: (30,50)
        boxw = random.randrange(15, 30)
        lh = random.randrange(20, height - 40) # size500: (100,100)
        lw = random.randrange(20, width - 40)
        image_erase = np.ones((height, width,1), np.float)
        image_erase[lh:lh + boxh, lw:lw + boxw,:] = 0
        totensor = T.ToTensor()
        image_erase = totensor(image_erase)
        image_erase = image_erase.float()
        data_aug = data * image_erase

        label = np.int32(splits[-1])


        #load embedding from h5 file
        embedding_h5 = img_path.replace('.jpg','_feat.h5')
        if os.path.exists(embedding_h5):
            with h5py.File(embedding_h5, 'r') as f:
                embedding = f['data'][()]
        else:
            embedding = data
            #print('no embedding found...', embedding_h5)

        return data, label, data_aug, embedding

    def __len__(self):
        return len(self.imgs)



if __name__ == '__main__':
    dataset = Dataset(root=r'G:\Tooth_Project\ToothImg\PytorchImg\Train',
                      data_list_file=r'G:\Tooth_Project\Code\Preprocess\new.txt',
                      phase='train',
                      input_shape=(1, 128, 128))

    trainloader = data.DataLoader(dataset,batch_size=1)
    for i, (data, label) in enumerate(trainloader):
        # imgs, labels = data
        # print imgs.numpy().shape
        # print data.cpu().numpy()
        # if i == 0:
        img = torchvision.utils.make_grid(data).numpy()
        # img = torchvision.utils.save_image(data,"./1.jpg")
        # print img.shape
        # print label.shape
        # chw -> hwc
        img = np.transpose(img, (1, 2, 0))
        # img *= np.array([0.229, 0.224, 0.225])
        # img += np.array([0.485, 0.456, 0.406])
        img += np.array([1, 1, 1])
        img *= 127.5
        img = img.astype(np.uint8)
        img = img[:, :, [2, 1, 0]]

        cv2.imshow('img', img)
        cv2.waitKey()
        # break
        # dst.decode_segmap(labels.numpy()[0], plot=True)