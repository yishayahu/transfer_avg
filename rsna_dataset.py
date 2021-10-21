from torch.utils.data import Dataset
import torch
import numpy as np
import os
from torchvision import transforms
import random
import matplotlib
import matplotlib.pyplot as plt
import glob
import cv2
import PIL
from PIL import Image
import pandas as pd
import pydicom
import tqdm

matplotlib.use('Agg')


class RSNADataset(Dataset):
    """
    torch dataLoader for masks or images
    """

    def __init__(self, list_IDs, image_dir, settings, labels, use_smaller_datasize=False, train=True):
        """
                Initialize this dataset class. for train mode
                """

        self.image_dir = image_dir
        self.list_IDs = list_IDs
        self.labels = labels

        if use_smaller_datasize:
            self.list_IDs = self.list_IDs[:500]


        self.window_center_dict = {'1': 50, '2': 40, '3': 300}
        self.window_width_dict = {'1': 150, '2': 80, '3': 1500}

        # self.filter_noise_images()
        self.labels = {key: self.labels[key] for key in self.list_IDs}
        self.input_size = settings.input_size
        self.batch_size = settings.batch_size
        self.n_channels = settings.n_channels
        self.crop_x_min = 64
        self.crop_x_max = 448
        self.crop_y_min = 64
        self.crop_y_max = 448
        if train:
            self.transforms = transforms.Compose([
                transforms.RandomHorizontalFlip(p=0.2),
                transforms.RandomRotation(degrees=(0, 15)),
                transforms.CenterCrop(384),
                transforms.ToTensor()]
            )
        else:
            self.transforms = transforms.Compose([
                transforms.CenterCrop(384),
                transforms.ToTensor()]
            )

    def __len__(self):
        return len(self.list_IDs)

    @staticmethod
    def get_id(img_dicom):
        return str(img_dicom.SOPInstanceUID)

    @staticmethod
    def get_first_of_dicom_field_as_int(x):
        if type(x) == pydicom.multival.MultiValue:
            return int(x[0])
        return int(x)

    @staticmethod
    def get_metadata_from_dicom(img_dicom):
        metadata = {
            "window_center": img_dicom.WindowCenter,
            "window_width": img_dicom.WindowWidth,
            "intercept": img_dicom.RescaleIntercept,
            "slope": img_dicom.RescaleSlope,

        }
        return {k: RSNADataset.get_first_of_dicom_field_as_int(v) for k, v in metadata.items()}

    def window_image(self, img, key, window_center, window_width, intercept, slope):
        img = img * slope + intercept
        # img_min = window_center - window_width // 2
        # img_max = window_center + window_width // 2
        img_min = self.window_center_dict[key] - self.window_width_dict[key] // 2
        img_max = self.window_center_dict[key] + self.window_width_dict[key] // 2
        img[img < img_min] = img_min
        img[img > img_max] = img_max
        return img

    def crop_center(self, img):
        img = img[self.crop_x_min:self.crop_x_max, self.crop_y_min:self.crop_y_max]
        return img

    def resize(self, img):
        img = PIL.Image.fromarray(img.astype(np.int8), mode="L")
        return img.resize((self.input_size[1], self.input_size[2]), resample=PIL.Image.BICUBIC)

    def resize_cv2(self, img):
        return cv2.resize(img, (384, 384), interpolation=cv2.INTER_LINEAR)

    @staticmethod
    def normalize_minmax(img):
        mi, ma = img.min(), img.max()
        return (img - mi) / (ma - mi)

    def prepare_image(self, img_path):
        img_dicom = pydicom.read_file(img_path)
        img_id = RSNADataset.get_id(img_dicom)
        metadata = RSNADataset.get_metadata_from_dicom(img_dicom)
        img_1 = self.window_image(img_dicom.pixel_array, key='1', **metadata)
        img_2 = self.window_image(img_dicom.pixel_array, key='2', **metadata)
        img_3 = self.window_image(img_dicom.pixel_array, key='3', **metadata)
        img_1 = np.expand_dims(RSNADataset.normalize_minmax(img_1), axis=-1)
        img_2 = np.expand_dims(RSNADataset.normalize_minmax(img_2), axis=-1)
        img_3 = np.expand_dims(RSNADataset.normalize_minmax(img_3), axis=-1)
        # img = self.crop_center(img)
        # img_pil = np.expand_dims(RSNADataset.resize_cv2(self, img), axis=-1)
        img_pil = np.concatenate((img_1,img_2,img_3), axis=-1)
        img_pil = Image.fromarray(np.uint8(img_pil * 255))

        return img_id, img_pil

    def __getitem__(self, idx):
        tensor_transform = transforms.ToTensor()
        img_id = self.list_IDs[idx]
        img_path = os.path.join(self.image_dir, 'ID_{}.dcm'.format(img_id))
        _, image = self.prepare_image(img_path)
        label = self.labels[img_id]
        #image = tensor_transform(img_pil)
        image = self.transforms(image)

        return {'image': image.float(), 'label': label}

    def filter_noise_images(self):
        filtered_list = []
        for img_id in self.list_IDs:
            img_path = os.path.join(self.image_dir, 'ID_{}.dcm'.format(img_id))
            img_dicom = pydicom.read_file(img_path)
            metadata = RSNADataset.get_metadata_from_dicom(img_dicom)
            img = self.window_image(img_dicom.pixel_array, **metadata)
            if img.max() > img.min():
                filtered_list.append(img_id)

        self.list_IDs = filtered_list
        print('len id list: {}'.format(len(self.list_IDs)))
