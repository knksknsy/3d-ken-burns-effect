from enum import Enum
import re
import urllib.request
import numpy as np
import sys
import os.path
from os import path

dataset_txt = "dataset.txt"
dataset_path = '3d-ken-burns-dataset'
dataset = []

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
    with open(dataset_txt) as f:
        content = f.readlines() 

    content = [x.strip() for x in content]

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

    if not path.exists(dataset_path):
        os.mkdir(dataset_path)
    
    print('Start download? Type [N] to cancel. Press return to start download.')
    proceed = input()
    if proceed == 'N':
        print('Download terminated.')
        sys.exit()

    for i, data in enumerate(dataset):
        scene = data[Category.SCENE.value]
        mode = data[Category.MODE.value]
        color = data[Category.COLOR.value]
        depth = data[Category.DEPTH.value]

        filename = scene + '-' + mode + '.zip'

        if not path.exists(filename):
            print(f'Downloading {filename} ({filesizes[i]} GB)')
            url_color = data[Category.URL_COLOR.value]
            url_depth = data[Category.URL_DEPTH.value]
            file_path = dataset_path + '/' + filename

            urllib.request.urlretrieve(url_color, file_path)

if __name__ == "__main__":
    main()