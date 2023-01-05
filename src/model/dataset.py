import torch
import pandas as pd
import cv2
import os
from  matplotlib import pyplot as plt
import random


class IntelDataset(torch.utils.data.Dataset): # inheritin from Dataset class
    def __init__(self, annot_df, transform=None):
        self.annot_df = annot_df
        self.root_dir = "" # root directory of images, leave "" if using the image path column in the __getitem__ method
        self.transform = transform

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
        return image, class_name, class_index

    def visualize(self, number_of_img, output_width, output_height):
        plt.figure(figsize=(output_width,output_height))
        for i in range(number_of_img):
            idx = random.randint(0, len(self.annot_df))
            image, class_name, class_index = self.__getitem__(idx)
            ax=plt.subplot(2, 5, i+1) # create an axis
            ax.title.set_text(class_name + '-' + str(class_index)) # create a name of the axis based on the img name
            plt.imshow(image) # show the img