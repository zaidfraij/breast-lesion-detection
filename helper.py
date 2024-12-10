import os
from PIL import Image
import matplotlib.pyplot as plt
import json
import pandas as pd
import numpy as np
import csv
from tqdm import tqdm
import numpy as np

def load_json(json_file_path):
    # open and load the JSON file
    with open(json_file_path, 'r') as file:
        data = json.load(file)
    return data

def load_images(folder_path, num_images=None):
    image_files = []
    for root, dirs, files in os.walk(folder_path):
        for file in files:
            if file.endswith(('.png', '.jpg', '.jpeg')):
                image_files.append(os.path.join(root, file))
                if (num_images) and (len(image_files) == num_images):
                    break
    return image_files

def display_images(image_files):
    for i, image_file in enumerate(image_files):
        img = Image.open(image_file)
        plt.figure(figsize=(3, 3))
        plt.imshow(img )
        plt.title(os.path.basename(image_file))
        plt.axis('off')
        plt.show()

def load_and_display_images(folder_path,  num_images=None):
    """
    Load and display images from the specified folder.

    :param folder_path: Path to the folder containing images.
    :param num_images: Number of images to display.
    """
    image_files = load_images(folder_path, num_images)
    display_images(image_files)


def load_and_display_images_with_bb(training_json_dict, video_id, dataset_path,  num_images=None):
    
    training_keys = list(training_json_dict.keys())
    dict_image_data = training_json_dict[training_keys[2]]
    dict_annotation_data = training_json_dict[training_keys[3]]

    images_df = pd.DataFrame(dict_image_data)
    images_df = images_df[images_df['video_id'] == video_id]

    annotation_df = pd.DataFrame(dict_annotation_data)


    images_list = list(images_df.file_name)
    if not num_images:
        num_images = len(images_list)

    image_arrays = []
    for image_name in images_list[:num_images]:
        image_id = int(images_df[images_df['file_name'] == image_name].id)
        bbox = list(annotation_df[annotation_df['image_id'] == image_id]['bbox'])[0]
        print(bbox)
        image_array = np.array(Image.open(os.path.join(dataset_path, image_name)))
        #print(image_id, bbox)
        image_arrays.append(image_array)
        plt.figure(figsize=(6, 6))
        plt.imshow(image_array)
        plt.gca().add_patch(plt.Rectangle((bbox[0], bbox[1]), bbox[2], bbox[3], edgecolor='red', facecolor='none', linewidth=2))
        plt.title(f'Image ID: {image_id}')
        plt.axis('off')
        plt.show()

def buv_json_to_csv(buv_json_file, output_csv_file, dataset_path, two_points_format=True, only_train_frames=False):	
    json_dict = load_json(buv_json_file)
    dict_keys = list(json_dict.keys())
    dict_image_data = json_dict[dict_keys[2]]
    dict_annotation_data = json_dict[dict_keys[3]]

    images_df = pd.DataFrame(dict_image_data)
    if only_train_frames:
        images_df = images_df[images_df['is_vid_train_frame'] == True]

    annotation_df = pd.DataFrame(dict_annotation_data)


    images_list = list(images_df.file_name)
    print('image_list_size: ', len(images_list))

    images_to_ignore_list = ['benign/x66ef02e7f1b9a0ef', 'benign/x63c9ba1377f35bf6', 'benign/x5a1c46ec6377e946']

    with open(output_csv_file, 'w', newline='') as csvfile:
        fieldnames = ['img_file', 'x1', 'y1', 'x2', 'y2', 'class_name']
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
    
        for image_name in tqdm(images_list):

            ignore = False
            for image_to_ignore in images_to_ignore_list:
                if image_to_ignore in image_name:
                    ignore = True
            if ignore:
                print(f'Image {image_name} is ignored')
                continue

            image_id = int(images_df[images_df['file_name'] == image_name].id)
            bbox = list(annotation_df[annotation_df['image_id'] == image_id]['bbox'])[0]

            if two_points_format:
                writer.writerow({
                    'img_file': os.path.join(dataset_path, image_name),
                    'x1': bbox[0] if bbox[2]>=0 else bbox[0] + bbox[2],
                    'y1': bbox[1] if bbox[3]>=0 else bbox[1] + bbox[3],
                    'x2': bbox[0] + bbox[2] if bbox[2]>=0 else bbox[0],
                    'y2': bbox[1] + bbox[3] if bbox[3]>=0 else bbox[1],
                    'class_name': 'Lesion'
                })
            else:
                writer.writerow({
                            'img_file': os.path.join(dataset_path, image_name),
                            'x1': bbox[0],
                            'y1': bbox[1],
                            'x2': bbox[2],
                            'y2': bbox[3],
                            'class_name': 'Lesion'
                        })
   
