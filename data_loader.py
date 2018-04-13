import torch
import os
import numpy as np
import scipy.io as io
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from torchvision import transforms
from torch.autograd import Variable
from torchvision.datasets import ImageFolder
from PIL import Image       #PIL is the image library of python

torch.manual_seed(1)
torch.cuda.manual_seed(1)

# data_loader with image
def pil_loader(path):
    # open path as file to avoid ResourceWarning (https://github.com/python-pillow/Pillow/issues/835)
    with open(path, 'rb') as f:
        with Image.open(f) as img:
            return img.convert('RGB')


class OfficeDataset(Dataset):
    def __init__(self, image_path, labels, flag_arr=None, transform=None, target_transform=None,
                 loader=pil_loader, show_message=True):
        """
        control the train-dataset by flag_arr, if flag_arr[i] == 0 it will be test sample, pay attention to it^s size
        """
        super(OfficeDataset, self).__init__()
        self.classes = list(os.listdir(image_path))
        self.imgs = []
        if flag_arr is None:
            flag_arr = [True for i in range(5000)]
        cur = 0
        for i in range(len(self.classes)):
            if self.classes[i] not in labels.keys():
                continue
            label = labels[self.classes[i]]
            i_path = os.path.join(image_path, self.classes[i])
            for jpg_name in os.listdir(i_path):
                if flag_arr[cur]:
                    if labels is None:
                        self.imgs.append((os.path.join(i_path, jpg_name), i))
                    else:
                        self.imgs.append((os.path.join(i_path, jpg_name), label))
                cur += 1           # forget it again again again...
        self.image_path = image_path
        self.transform = transform
        self.target_transform = target_transform
        self.loader = loader
        if show_message:
            print(
                self.image_path,
                "classess :\n",
                # self.classes, "\n",
                list(labels.keys()),
                "\nlen :",
                self.__len__(),
            )

    def __getitem__(self, index):
        i_path, target = self.imgs[index]
        img = self.loader(i_path)
        if self.transform is not None:
            img = self.transform(img)
        if self.target_transform is not None:
            target = self.target_transform(target)

        return img, target

    def __len__(self):
        return len(self.imgs)

def get_dataset_length():
    length = {}
    length['amazon'] = 958
    length['caltech'] = 1123
    length['dslr'] = 157
    length['webcam'] = 295
    return length


def get_transforms(train=True, image_size=224):
    if train:
        transform = transforms.Compose([
            transforms.RandomSizedCrop(image_size),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])
    else:
        transform = transforms.Compose([
            transforms.Scale(256),
            transforms.CenterCrop(image_size),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])
    return transform

def get_loader(batch_size, image_size=224, image_path="./office_caltech_10", pp=10, dataset_name='amazon',
               full_test=True, show_message=True, labels=None):
    """Build and return data loader for train and test"""
    length = 5000
    flag_array = np.random.randint(0, pp, length)
    i_path = os.path.join(image_path, dataset_name)
    train_dataset = OfficeDataset(i_path, labels, flag_array > 0, transform=get_transforms(True, image_size),
                                  show_message=show_message)
    test_dataset = OfficeDataset(i_path, labels, flag_array == 0, transform=get_transforms(False, image_size),
                                 show_message=show_message)
    train_dataloader = DataLoader(
        dataset=train_dataset, batch_size=batch_size, shuffle=True, num_workers=2, drop_last=True
    )
    if full_test:
        batch_size = len(test_dataset)
    test_dataloader = DataLoader(
        dataset=test_dataset, batch_size=batch_size, shuffle=False, num_workers=2, drop_last=True
    )
    return train_dataloader, test_dataloader, flag_array


def get_loader_witharr(batch_size, flag_arr, image_size=224, train=True,image_path="./office_caltech_10",
                       dataset_name='amazon', show_message=True, labels=None, dl=True):
    i_path = os.path.join(image_path, dataset_name)
    dataset = OfficeDataset(i_path, labels, flag_arr, transform=get_transforms(train, image_size),
                            show_message=show_message)
    dataloader = DataLoader(
        dataset=dataset, batch_size=batch_size, shuffle=True, num_workers=2, drop_last=dl
    )
    return dataloader


def get_fullloader(image_size=224, batch_size=None, train=True, image_path="./office_caltech_10",
                   dataset_name='amazon', show_message=True, labels_dict=None, shuffle=True, dl=True):
    length = 5000
    i_path = os.path.join(image_path, dataset_name)
    flag_arr = np.ones(length)
    dataset = OfficeDataset(i_path, labels_dict, flag_arr, transform=get_transforms(train, image_size),
                            show_message=show_message)
    if batch_size is None:
        batch_size = len(dataset)
    dataloader = DataLoader(
        dataset=dataset, batch_size=batch_size, shuffle=shuffle, num_workers=2, drop_last=dl
    )
    return dataloader


# dataloader with features
class FeaturesDataset(Dataset):
    def __init__(self, features_path, feas_name='fts', transform=None, target_transform=None,show_message=True):
        """
        FeaturesDataset, Compare the performance with other Adversarial-based model
        It is recommend to use sklearn to divide the datasets for train and test,
        But other model don^t divide it, so do us.
        """
        data = io.loadmat(features_path)
        self.feas = data[feas_name].astype(np.float32) # feas
        self.labels = data['labels'].astype(np.int64).reshape(-1)
        if target_transform is None:
            target_transform = lambda x : x - 1
        if transform is not None:
            self.feas = transform(self.feas)
        if target_transform is not None:
            self.labels = target_transform(self.labels)
        self.features_path = features_path
        if show_message:
            print(
                self.features_path,
                "classess :\n",
                np.unique(self.labels),
                "len :",
                self.__len__(),
            )

    def __getitem__(self, index):
        return self.feas[index], self.labels[index]

    def __len__(self):
        return len(self.labels)


def get_featureloader(batch_size=None, dataset_name='amazon', dataset_path='./decaf6', shuffle=True, show_message=True):
    f_path = os.path.join(dataset_path, dataset_name)
    dataset = FeaturesDataset(f_path, transform=None, show_message=show_message)
    if batch_size is None:
        batch_size = len(dataset)
    return DataLoader(dataset=dataset, batch_size=batch_size, shuffle=shuffle, num_workers=2, drop_last=False)


if __name__ == "__main__":
    q = {1:2, 2:3}
    # x = get_featureloader(1, "amazon_fc7.mat")
    # print(len(x))
    # classes = os.listdir("./office_caltech_10/amazon")
    # print(classes)
    # x = Variable(torch.FloatTensor([1, 2, 3]), requires_grad=True)
    # y = torch.matmul(x, x)
    # print(y.detach().requires_grad)
    # d10 = io.loadmat("./decaf6/amazon_decaf.mat")
    # d31 = io.loadmat("./decaf6/amazon_fc7.mat")
    # f1 = np.zeros(4096)
    # f2 = np.zeros(4096)
    # k = 1
    # for i in range(len(d10['labels'])):
    #     if d10['labels'][i] == k:
    #         f1 = f1 + d10['feas'][i]
    # for i in range(len(d31['labels'])):
    #     if d31['labels'][i] == k:
    #         f2 = f2 + d31['fts'][i]
    # print(
    #     f1, f2,
    #     np.unique(d10['labels']),
    #     np.unique(d31['labels'])
    # )
