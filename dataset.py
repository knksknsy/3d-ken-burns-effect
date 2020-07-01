'''Copy this script to a path and execute it in order to download the dataset into the path.'''

from enum import Enum
import re
import urllib.request
import numpy as np
import sys
import os.path
from os import path
from pathlib import Path
import getopt
from zipfile import ZipFile
import pandas as pd
from tqdm import tqdm
import json

class Category(Enum):
    SCENE = 0
    MODE = 1
    COLOR = 2
    DEPTH = 3
    NORMAL = 4
    SIZE_COLOR = 5
    SIZE_DEPTH = 6
    URL_COLOR = 7
    URL_DEPTH = 8

def get_dataset():
    dataset_txt = "dataset.txt"
    dataset = []

    # Read dataset.txt file
    with open(dataset_txt) as f:
        content = f.readlines() 

    # Remove trailing whitespace in lines
    content = [x.strip() for x in content]

    # Cleansing of each line and initializing the dataset array
    for i, c in enumerate(content):
        c = c.replace(' ', '')
        c_split = c.split('|')
        data = []
        for j, s in enumerate(c_split):
            if j > 0 and j < len(c_split) - 1:
                data.append(s)
        dataset.append(data)

    # Calculate total filesize of dataset including categories: color, depth
    filesizes = []
    for i, data in enumerate(dataset):
        # Extract file size e.g.: [5.2 GB] => 5.2
        color = re.search('\[(.+?)\]', data[Category.COLOR.value])
        depth = re.search('\[(.+?)\]', data[Category.DEPTH.value])
        non_decimal = re.compile(r'[^\d.]+')

        size_color = float(non_decimal.sub('', color.group(1)))
        size_depth = float(non_decimal.sub('', depth.group(1)))
        
        filesizes.append(size_color)
        filesizes.append(size_depth)

        dataset[i].append(size_color)
        dataset[i].append(size_depth)

        # Extract url e.g.: (https://.../.zip) => https://.../.zip
        url_color = re.search('\((.+?)\)', data[Category.COLOR.value])
        url_depth = re.search('\((.+?)\)', data[Category.DEPTH.value])

        dataset[i].append(url_color.group(1))
        dataset[i].append(url_depth.group(1))
    
    return dataset, filesizes

def download_dataset(dataset):
    # Start download
    for i, data in enumerate(dataset):
        # Extract meta-information
        scene = data[Category.SCENE.value]
        mode = data[Category.MODE.value]
        color = data[Category.COLOR.value]
        depth = data[Category.DEPTH.value]

        filename_color = scene + '-' + mode + '.zip'
        filename_depth = scene + '-' + mode + '-depth.zip'

        # Set saving path
        file_path_color = os.path.join(dataset_path, filename_color)
        file_path_depth = os.path.join(dataset_path, filename_depth)

        # Download COLOR datasets. Skip already downloaded datasets
        if not path.exists(file_path_color):
            print(f'Downloading {filename_color} ({data[Category.SIZE_COLOR.value]} GB)')
            url_color = data[Category.URL_COLOR.value]
            urllib.request.urlretrieve(url_color, file_path_color)

        # # Download DEPTH datasets. Skip already downloaded datasets
        if not path.exists(file_path_depth):
            print(f'Downloading {filename_depth} ({data[Category.SIZE_DEPTH.value]} GB)')
            url_depth = data[Category.URL_DEPTH.value]
            urllib.request.urlretrieve(url_depth, file_path_depth)

def main(dataset_path):
    dataset, filesizes = get_dataset()
    print(f'Total dataset size: {np.sum(filesizes)} GB')

    # Create 3d-ken-burns-dataset directory if it does not exist
    if not path.exists(dataset_path):
        os.mkdir(dataset_path)
    
    # User input for starting or canceling download procedure
    print('Start download? Type [N] to cancel. Press return to start download.')
    proceed = input()
    if proceed == 'N':
        print('Download terminated.')
        sys.exit()

    download_dataset(dataset)
    print(f'Datasets downloaded into path: {dataset_path}')

def create_csv(dataset_path):
    print('Creating CVS...')
    dataset_zip_color_paths = [f for f in os.listdir(dataset_path) if f.endswith('.zip') and '-depth' not in f]
    dataset_zip_depth_paths = [f for f in os.listdir(dataset_path) if f.endswith('.zip') and '-depth' in f]

    dataset_zip_color_paths.sort()
    dataset_zip_depth_paths.sort()

    df_columns = ['zip_image_path', 'zip_depth_path', 'image_path', 'depth_path', 'fltFov']

    for i, (zip_color_path, zip_depth_path) in enumerate(zip(dataset_zip_color_paths, dataset_zip_depth_paths)):
        # Data frame containing information to a single dataset
        data_frame = pd.DataFrame(columns=df_columns)

        absolute_zip_color_path = os.path.join(dataset_path, zip_color_path)
        absolute_zip_depth_path = os.path.join(dataset_path, zip_depth_path)

        image_color_paths = [image for image in ZipFile(absolute_zip_color_path).namelist() if image.endswith('.png')]
        image_color_paths = np.array(image_color_paths, dtype=object)

        image_depth_paths = [depth for depth in ZipFile(absolute_zip_depth_path).namelist() if depth.endswith('.exr')]
        image_depth_paths = np.array(image_depth_paths, dtype=object)

        meta_paths = [meta for meta in ZipFile(absolute_zip_color_path).namelist() if meta.endswith('.json')]
        meta_paths = np.array(meta_paths, dtype=object)
        # Extend meta_paths to length of image_color_paths/image_depth_paths
        meta_paths = np.tile(meta_paths, 4)
        meta_paths = np.sort(meta_paths)

        # Read json attribute 'fltFov' from meta file
        meta_archive = ZipFile(absolute_zip_color_path, 'r')
        for j, (m1, m2, m3, m4) in enumerate(zip(*[iter(meta_paths)]*4)):
            meta_json = meta_archive.read(m1).decode("utf-8")
            meta_json = json.loads(meta_json)
            meta_paths[j*4:(j*4)+4] = float(meta_json['fltFov'])

        progress_bar = tqdm(zip(image_color_paths, image_depth_paths, meta_paths))
        for j, (c, d, m) in enumerate(progress_bar):
            progress_bar.set_description(f'Preparing CSV for {zip_color_path}: {j+1}/{len(image_color_paths)+1}')
            series = pd.Series([zip_color_path, zip_depth_path, c, d, m], index=df_columns)
            data_frame = data_frame.append(series, ignore_index=True)

        # Create CSV
        data_frame.to_csv(f'dataset_{zip_color_path}.csv', index=False, encoding='utf-8')

    # Concat every CSV file
    pwd = os.path.dirname(os.path.realpath(__file__))
    csv_paths = [f for f in os.listdir(pwd) if f.endswith('.csv')]
    # Data frame containing information for all datasets
    data_frames = []
    for f in csv_paths:
        data_frame = pd.read_csv(f, index_col=None, header=0)
        data_frames.append(data_frame)

    # Save concatenated csv
    data_frame = pd.concat(data_frames, axis=0, ignore_index=True)
    data_frame.to_csv('dataset.csv', index=False, encoding='utf-8')
    print('Saved CSV to ./dataset.csv')

if __name__ == "__main__":
    dataset_dir_name = '3d-ken-burns-dataset'
    dataset_path = './'
    csv = False

    for strOption, strArgument in getopt.getopt(sys.argv[1:], '', [ 'path=', 'csv' ])[0]:
        if strOption == '--path' and strArgument != '':
            dataset_path = strArgument # path to the datasets
        if strOption == '--csv':
            csv = True # Flag to create csv of downloaded datasets

    if dataset_dir_name not in dataset_path:
        dataset_path = os.path.join(dataset_path, dataset_dir_name)

    main(dataset_path)

    if csv:
        create_csv(dataset_path)
