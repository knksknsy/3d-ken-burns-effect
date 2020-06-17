'''Generate a video for each data (stereo-vision pictures) in 3d-ken-burns-dataset'''
'''Place this script into the 3d-ken-burns-dataset directory and execute it in order to generate videos'''

from cv2 import cv2
from tqdm import tqdm
import os
import numpy
import sys
from zipfile import ZipFile

class Images2Video:
    def __init__(self, data_path):
        self.data_path = data_path
        self.output_path = 'videos'

        if not os.path.exists(self.output_path):
            os.mkdir(self.output_path)

    def generate(self):
        video_name = self.data_path.replace('.zip', '') + '.avi'
        video_path = self.output_path + video_name

        if os.path.exists(video_path):
            print(f'Skip gererating duplicate {video_name}')
            return
        print(f'Generating {video_name}...')
        
        images = [img for img in ZipFile(self.data_path).namelist() if img.endswith('.png')]
        if len(images) == 0:
            print(f'Skip gererating {video_name}\nTODO: implement generating videos from exr files')
            return
        archive = ZipFile(self.data_path, 'r')

        frame_data = archive.read(images[0])
        frame = cv2.imdecode(numpy.frombuffer(frame_data, numpy.uint8), 1)
        height, width, layers = frame.shape

        video = cv2.VideoWriter(video_path, 0, 5, (width,height))
        images.sort()
        pbar = tqdm(zip(*[iter(images)]*4))

        for bl, br, tl, tr in pbar:
            pbar.set_description(f'Processing video {video_name}: Total images {len(images)//4}: Processed Images: ')

            image_bl_data = archive.read(bl)
            image_bl = cv2.imdecode(numpy.frombuffer(image_bl_data, numpy.uint8), 1)

            image_br_data = archive.read(br)
            image_br = cv2.imdecode(numpy.frombuffer(image_br_data, numpy.uint8), 1)

            image_tl_data = archive.read(tl)
            image_tl = cv2.imdecode(numpy.frombuffer(image_tl_data, numpy.uint8), 1)

            image_tr_data = archive.read(tr)
            image_tr = cv2.imdecode(numpy.frombuffer(image_tr_data, numpy.uint8), 1)

            image_tr_br = cv2.vconcat([image_tr, image_br])
            image_tl_bl = cv2.vconcat([image_tl, image_bl])
            # Combine 4 images
            image = cv2.hconcat([image_tr_br, image_tl_bl])
            # Downscale image by factor 2
            image = cv2.resize(image, (frame.shape[0], frame.shape[1])) 
            video.write(image)

        cv2.destroyAllWindows()
        video.release()

def select_dataset(dirs, pwd):
    print("Type dataset's name to generate video from it. Type [all] to select all. Press Ctrl+C to cancel.")
    data_path = input()
    dataset = []

    if data_path in dirs:
        dataset.append(pwd + data_path)
    elif data_path == 'all':
        dataset = [pwd + d for d in dirs]
    else:
        print('Dataset not found.')
        return select_dataset(dirs, pwd)
    
    return dataset

def main():
    pwd = './'
    dirs = [d for d in os.listdir(pwd) if d.endswith('.zip')] #or os.path.isdir(os.path.join(pwd, d))]

    print('Available dataset:')
    for d in dirs:
        print(f'\t- {d}')

    selected_data = select_dataset(dirs, pwd)

    for sd in selected_data:
        img2vid = Images2Video(sd)
        img2vid.generate()

if __name__ == "__main__":
    main()