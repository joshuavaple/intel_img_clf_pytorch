import torch
import pandas as pd
import cv2
import os
from  matplotlib import pyplot as plt
import random


class IntelDataset(torch.utils.data.Dataset):
    def __init__(self, annot_df, validation_proportion=0, transform=None):
        self.annot_df = annot_df
        self.root_dir = "" # root directory of images, leave "" if using the image path column in the __getitem__ method
        self.transform = transform
        self.dataset = self.create_validation_dataset(validation_proportion=validation_proportion)[0]
        self.validation_set = self.create_validation_dataset(validation_proportion=validation_proportion)[1]
        self.dataset_size = self.create_validation_dataset(validation_proportion=validation_proportion)[2]
        self.validation_size = self.create_validation_dataset(validation_proportion=validation_proportion)[3]

    def __len__(self):
        return len(self.annot_df) # return length (numer of rows) of the dataframe

    def __getitem__(self, idx): 
        image_path = self.annot_df.iloc[idx, 1] #use image path column (index = 1) in csv file
        image = cv2.imread(image_path) # read image by cv2
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB) # convert from BGR to RGB for matplotlib
        class_name = self.annot_df.iloc[idx, 2] # use class name column (index = 2) in csv file
        class_index = self.annot_df.iloc[idx, 3] # use class index column (index = 3) in csv file
        if self.transform:
            image = self.transform(image)
        return image, class_name, class_index # when accessing an instance via index, 3 outputs are returned - the image, class name and class index

    def visualize(self, number_of_img = 10, output_width = 12, output_height = 6):
        plt.figure(figsize=(output_width,output_height))
        for i in range(number_of_img):
            idx = random.randint(0, len(self.annot_df))
            image, class_name, class_index = self.__getitem__(idx)
            ax=plt.subplot(2, 5, i+1) # create an axis
            ax.title.set_text(class_name + '-' + str(class_index)) # create a name of the axis based on the img name
            if self.transform == None:
                plt.imshow(image) 
            else:
                plt.imshow(image.permute(1, 2, 0))

    def create_validation_dataset(self, validation_proportion):
        if (validation_proportion > 1) or (validation_proportion < 0):
            return "The proportion of the validation set must be between 0 and 1"
        else:
            dataset_size = int((1-validation_proportion)*len(self.annot_df))
            validation_size = len(self.annot_df) - dataset_size
            dataset, validation_set = torch.utils.data.random_split(self,[dataset_size, validation_size])
            return dataset, validation_set, dataset_size, validation_size