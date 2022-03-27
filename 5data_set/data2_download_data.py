"""
数据预处理流程
1、 image resize ：224*224 for ResNet 18
2、 data argumentation
rotate crop
3、 normalize
mean std
4、totensor

"""
import random

import torch
import os, csv, glob
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image


class Getdata(Dataset):
    def __init__(self, root, save_root, resize, mode):
        super(Getdata, self).__init__()

        self.root = root
        self.resize = resize
        self.save_root = save_root
        # 将文件夹的名字转为label，猫：0，狗：1存在一个元组中
        self.name2label = {}
        for name in sorted(os.listdir(os.path.join(root))):
            # os.listdir:用于返回一个由文件名和目录名组成的列表
            if not os.path.isdir(os.path.join(root, name)):
                # os.path.isdir()用于判断对象是否为一个目录。本代码只需要文件名
                continue
            self.name2label[name] = len(self.name2label.keys())
        print(self.name2label)

        # 将image，label 下载下来 存储在self.images,self.labels
        self.images, self.labels = self.load_csv('images.csv')

        if mode == 'train':  # 60%
            self.images = self.images[:int(0.8 * len(self.images))]
            self.labels = self.labels[:int(0.8 * len(self.labels))]
        elif mode == 'val':  # 60% ==> 80%
            self.images = self.images[int(0.8 * len(self.images)):]
            self.labels = self.labels[int(0.8 * len(self.labels)):]
        else:
            print("please input train or val")

    def load_csv(self, filename):
        # 首先建立image_path和image_label的csv文件，然后通过csv.reader（）将images_path和label读取出来
        if not os.path.exists(os.path.join(self.save_root, filename)):
            images = []
            for name in self.name2label.keys():
                # r"F:\CNN\CNN_classification\ml2021spring-hw3\food-11\training\labeled\00\00.jpg
                images += glob.glob(os.path.join(self.root, name, '*.jpg'))
                images += glob.glob(os.path.join(self.root, name, '*.png'))
                images += glob.glob(os.path.join(self.root, name, '*.jpeg'))
            print(len(images), images)
            # 将image_path,和label储存在csv文件中
            random.shuffle(images)
            with open(os.path.join(self.save_root, filename), mode='w', newline='') as f:
                writer = csv.writer(f)
                for img in images:
                    name = img.split(os.sep)[-2]
                    # 00\\0_104.jpg' ,0
                    label = self.name2label[name]
                    # r"'F:\\CNN\\CNN_classification\\ml2021spring-hw3\\food-11\\training\\labeled\\00\\0_104.jpg', ,0
                    writer.writerow([img, label])
                print("successfully write into csv file :", filename)
        else:
            print('{} has been written'.format(filename))

        # 从csv文件中加载img_path,label
        images, labels = [], []
        with open(os.path.join(self.save_root, filename)) as f:
            reader = csv.reader(f)
            for row in reader:
                img, label = row
                label = int(label)

                images.append(img)
                labels.append(label)
        assert len(images) == len(labels)
        return images, labels

    def denormalize(self, x_heat):
        # 考虑到transform之后会使得图像可视化看不清楚，加入反normalize模块
        mean = [0.485, 0.456, 0.406]
        std = [0.229, 0.224, 0.225]

        # x_head = (x - mean)/std
        # x = x_head * std + mean
        # x = [c, h, w]
        # mean[3] ==> [3,1,1] 从一维转化成三维
        mean = torch.tensor(mean).unsqueeze(1).unsqueeze(1)
        std = torch.tensor(std).unsqueeze(1).unsqueeze(1)

        x = x_heat * std + mean

        return x

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        # idx = [0,len(self.images)]
        # self.images,self.labels
        # img_path:r"F:\CNN\CNN_classification\ml2021spring-hw3\food-11\training\labeled\00\00.jpg
        # label : 0
        # 将img表示的img_path转化成img图片，同时将label的scaler转化为tensor

        img, label = self.images[idx], self.labels[idx]
        tf = transforms.Compose([
            lambda x: Image.open(x).convert('RGB'),  # string_image_path  ===> image data
            # transforms.Resize((self.resize, self.resize)),
            transforms.Resize((int(1.25 * self.resize), int(1.25 * self.resize))),
            transforms.RandomRotation(15),
            transforms.CenterCrop(self.resize),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])

        img = tf(img)  # 此时的img不再是img_path 而是image_data
        label = torch.tensor(label)
        return img, label


class Gettestdata(Dataset):
    def __init__(self, root, save_root, resize, mode='test'):
        super(Gettestdata, self).__init__()

        self.root = root
        self.resize = resize
        self.save_root = save_root
        # 将文件夹的名字转为label，猫：0，狗：1存在一个元组中
        self.name2label = {}
        for name in sorted(os.listdir(os.path.join(root))):
            # os.listdir:用于返回一个由文件名和目录名组成的列表
            if not os.path.isdir(os.path.join(root, name)):
                # os.path.isdir()用于判断对象是否为一个目录。本代码只需要文件名
                continue
            self.name2label[name] = len(self.name2label.keys())
        # print(self.name2label)

        # 将image，label 下载下来 存储在self.images,self.labels
        self.images, self.labels = self.load_csv('images_test.csv')

        if mode == 'test':  # 60%
            self.images = self.images
            self.labels = self.labels
        else:
            print("please input test")

    def load_csv(self, filename):
        # 首先建立image_path和image_label的csv文件，然后通过csv.reader（）将images_path和label读取出来
        if not os.path.exists(os.path.join(self.save_root, filename)):
            images = []
            for name in self.name2label.keys():
                # r"F:\CNN\CNN_classification\ml2021spring-hw3\food-11\training\labeled\00\00.jpg
                images += glob.glob(os.path.join(self.root, name, '*.jpg'))
                images += glob.glob(os.path.join(self.root, name, '*.png'))
                images += glob.glob(os.path.join(self.root, name, '*.jpeg'))
            print(len(images), images)
            # 将image_path,和label储存在csv文件中
            random.shuffle(images)
            with open(os.path.join(self.save_root, filename), mode='w', newline='') as f:
                writer = csv.writer(f)
                for img in images:
                    name = img.split(os.sep)[-2]
                    # 00\\0_104.jpg' ,0
                    label = self.name2label[name]
                    # r"'F:\\CNN\\CNN_classification\\ml2021spring-hw3\\food-11\\training\\labeled\\00\\0_104.jpg', ,0
                    writer.writerow([img, label])
                print("successfully write into csv file :", filename)
        else:
            print('{} has been written'.format(filename))

        # 从csv文件中加载img_path,label
        images, labels = [], []
        with open(os.path.join(self.save_root, filename)) as f:
            reader = csv.reader(f)
            for row in reader:
                img, label = row
                label = int(label)

                images.append(img)
                labels.append(label)
        assert len(images) == len(labels)
        return images, labels

    def denormalize(self, x_heat):
        # 考虑到transform之后会使得图像可视化看不清楚，加入反normalize模块
        mean = [0.485, 0.456, 0.406]
        std = [0.229, 0.224, 0.225]

        # x_head = (x - mean)/std
        # x = x_head * std + mean
        # x = [c, h, w]
        # mean[3] ==> [3,1,1] 从一维转化成三维
        mean = torch.tensor(mean).unsqueeze(1).unsqueeze(1)
        std = torch.tensor(std).unsqueeze(1).unsqueeze(1)

        x = x_heat * std + mean

        return x

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        # idx = [0,len(self.images)]
        # self.images,self.labels
        # img_path:r"F:\CNN\CNN_classification\ml2021spring-hw3\food-11\training\labeled\00\00.jpg
        # label : 0
        # 将img表示的img_path转化成img图片，同时将label的scaler转化为tensor

        img, label = self.images[idx], self.labels[idx]
        tf = transforms.Compose([
            lambda x: Image.open(x).convert('RGB'),  # string_image_path  ===> image data
            # transforms.Resize((self.resize, self.resize)),
            transforms.Resize((self.resize, self.resize)),
            # transforms.RandomRotation(15),
            # transforms.CenterCrop(self.resize),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])

        img = tf(img)  # 此时的img不再是img_path 而是image_data
        label = torch.tensor(label)
        return img, label


"""
调参模块
"""
def main():
    import visdom
    import time
    viz = visdom.Visdom()

    root = r"F:\CNN\CNN_classification\ml2021spring-hw3\food-11\training\labeled"
    save_root = r"F:\pytorchstudy\5data_set"
    db = Getdata(root, save_root, 224, 'train')
    # print(db.class_to_idx)
    # image, label = next(iter(db))
    # print(image.shape, label)

    # viz.image(db.denormalize(image), win='sample', opts=dict(title='sample_X'))
    train_data = DataLoader(db, batch_size= 16 , shuffle=True)
    image, label = next(iter(train_data))
    # viz.images同时显示多张图片
    for image, label in train_data:
        viz.images(db.denormalize(image), nrow=4, win='batch', opts=dict(title='sample_X'))
        viz.images(image, nrow=4, win='batch1', opts=dict(title='sample_X'))
        viz.text(str(label.numpy()), win='label',opts = dict(title = 'batch_y'))
        time.sleep(10)

main()
