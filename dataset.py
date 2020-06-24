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

def main():
    dataset_path = '3d-ken-burns-dataset'
    arguments_path = './'
    dataset_txt = "dataset.txt"
    dataset = []

    for strOption, strArgument in getopt.getopt(sys.argv[1:], '', [ strParameter[2:] + '=' for strParameter in sys.argv[1::2] ])[0]:
	    if strOption == '--path' and strArgument != '': arguments_path = strArgument # path to the datasets

    dataset_path = os.path.join(arguments_path, dataset_path)

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

    print(f'Total dataset size: {np.sum(filesizes)} GB')

    # Create 3d-ken-burns-dataset directory if it does not exist
    if not path.exists(os.path.join(arguments_path, dataset_path)):
        os.mkdir(dataset_path)
    
    # User input for starting or canceling download procedure
    print('Start download? Type [N] to cancel. Press return to start download.')
    proceed = input()
    if proceed == 'N':
        print('Download terminated.')
        sys.exit()

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

if __name__ == "__main__":
    main()
