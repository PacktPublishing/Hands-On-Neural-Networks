#!/usr/bin/env python
# coding: utf-8

import os
import pandas as pd
from PIL import Image
import numpy as np

# Pixel values range from 0 to 255 (0 is normally black and 255 is white)
basedir = os.path.join('..', 'data', 'raw')
file_origin = os.path.join(basedir, 'fer2013.csv')
data_raw = pd.read_csv(file_origin)

data_input = pd.DataFrame(data_raw, columns=['emotion', 'pixels', 'Usage'])

data_input.rename({'Usage': 'usage'}, inplace=True)
data_input.head()

label_map = {
    0: '00_Anger',
    1: '01_Disgust',
    2: '02_Fear',
    3: '03_Happy',
    6: '04_Neutral',
    4: '05_Sad',
    5: '06_Surprise'
}


def to_image(s):
    image_on_list = np.asarray([int(i) for i in s.split(' ')])
    if image_on_list.max() != 0:
        rescaled = (255.0 / image_on_list.max() *
                    (image_on_list - image_on_list.min())).astype(np.uint8)
        return np.array(rescaled).reshape(48, 48)


def save_image(np_array_flat, file_name):
    try:
        im = Image.fromarray(np_array_flat)
        im.save(file_name)
    except AttributeError as e:
        print('save_image: ', e)
        return


# Creating the folders
output_folders = data_input['Usage'].unique().tolist()
all_folders = []

for folder in output_folders:
    for label in label_map:
        all_folders.append(os.path.join(basedir, folder, label_map[label]))

for folder in all_folders:
    if not os.path.exists(folder):
        os.makedirs(folder)
    else:
        print('Folder {} exists already'.format(folder))

counter_error = 0
counter_correct = 0

for folder in all_folders:

    emotion = folder.split('/')[-1]
    usage = folder.split('/')[-2]

    for key, value in label_map.items():
        if value == emotion:
            emotion_id = key

    df_to_save = data_input.reset_index()[data_input.Usage == usage][
        data_input.emotion == emotion_id]
    print('saving in: ', folder, ' size: ', df_to_save.shape)
    df_to_save['image'] = df_to_save.pixels.apply(to_image)
    df_to_save['file_name'] = folder + '/image_' + df_to_save.index.map(
        str) + '_' + df_to_save.emotion.apply(
        str) + '-' + df_to_save.emotion.apply(
        lambda x: label_map[x]) + '.png'
    df_to_save[['image', 'file_name']].apply(
        lambda x: save_image(x.image, x.file_name), axis=1)
    df_to_save.apply(lambda x: save_image(x.pixels, os.path.join(basedir, x.file_name)), axis=1)
