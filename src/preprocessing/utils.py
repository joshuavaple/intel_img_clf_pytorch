import csv
import os
import pandas as pd


def build_annotation_csv(image_location, annot_location, output_csv_name):
    """Builds a csv file for pytorch training from a directory of folders of images.
    Install csv module if not already installed.
    Args: 
    image_location: image directory path, e.g. r'.\data\train'
    annot_location: annotation directory path
    output_csv_name: string of output csv file name, e.g. 'train.csv'
    Returns:
    csv file with file names, file paths, class names and class indices
    """
    class_lst = os.listdir(image_location) #returns a LIST containing the names of the entries (folder names in this case) in the directory.
    class_lst.sort() #IMPORTANT 
    with open(os.path.join(annot_location, output_csv_name), 'w', newline='') as csvfile:
        writer = csv.writer(csvfile, delimiter=',')
        writer.writerow(['file_name', 'file_path', 'class_name', 'class_index']) #create column names
        for class_name in class_lst:
            class_path = os.path.join(image_location, class_name) #concatenates various path components with exactly one directory separator (‘/’) except the last path component. 
            file_list = os.listdir(class_path) #get list of files in class folder
            for file_name in file_list:
                file_path = os.path.join(image_location, class_name, file_name) #concatenate class folder dir, class name and file name
                writer.writerow([file_name, file_path, class_name, class_lst.index(class_name)]) #write the file path and class name to the csv file
    return pd.read_csv(os.path.join(annot_location, output_csv_name))

def check_annot_dataframe(annot_df):
    class_zip = zip(annot_df['class_index'], annot_df['class_name'])
    my_list = list()
    for index, name in class_zip:
        my_list.append(tuple((index, name)))
    unique_list = list(set(my_list))
    return unique_list


