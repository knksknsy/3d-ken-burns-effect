'''Usage e.g.: python image_to_video.py --path D:\3d-ken-burns-dataset'''

from cv2 import cv2
from tqdm import tqdm
import os
import numpy
import sys
from zipfile import ZipFile
from pathlib import Path
import getopt

class Images2Video:
    def __init__(self, data_path, pwd):
        self.data_path = data_path
        self.pwd = pwd
        self.output_path = Path(os.path.join(self.pwd, 'videos'))

        if not os.path.exists(self.output_path):
            os.mkdir(self.output_path)

    def generate(self):
        video_name = str(self.data_path).split(os.path.sep)[-1].replace('.zip', '.mp4')
        video_path = Path(os.path.join(self.output_path,video_name))
        print(f'video_path: {video_path}')

        if os.path.exists(video_path):
            print(f'Skip gererating duplicate {video_name}')
            return
        print(f'Generating {video_name}')

        images = [img for img in ZipFile(self.data_path).namelist() if img.endswith('.png')]
        images_exr = [img for img in ZipFile(self.data_path).namelist() if img.endswith('.exr')]

        if len(images) > 0:
            self.process_image(images, video_name, str(video_path))
        if len(images_exr) > 0:
            self.process_exr(images_exr, video_name, str(video_path))

    def process_image(self, images, video_name, video_path):
        archive = ZipFile(self.data_path, 'r')

        frame_data = archive.read(images[0])
        frame = cv2.imdecode(numpy.frombuffer(frame_data, numpy.uint8), 1)
        height, width, layers = frame.shape

        video = cv2.VideoWriter(video_path, cv2.VideoWriter_fourcc(*'MP4V'), 5, (width,height))
        images.sort()
        pbar = tqdm(zip(*[iter(images)]*4))

        for i, (bl, br, tl, tr) in enumerate(pbar):
            pbar.set_description(f'Processing video {video_name}: Total images {len(images)//4}: Processed Images: ')

            image_bl_data = archive.read(bl)
            image_bl = cv2.imdecode(numpy.frombuffer(image_bl_data, numpy.uint8), cv2.IMREAD_COLOR)
            image_br_data = archive.read(br)
            image_br = cv2.imdecode(numpy.frombuffer(image_br_data, numpy.uint8), cv2.IMREAD_COLOR)
            image_tl_data = archive.read(tl)
            image_tl = cv2.imdecode(numpy.frombuffer(image_tl_data, numpy.uint8), cv2.IMREAD_COLOR)
            image_tr_data = archive.read(tr)
            image_tr = cv2.imdecode(numpy.frombuffer(image_tr_data, numpy.uint8), cv2.IMREAD_COLOR)

            image_stitched = self.stitch_images(image_tl, image_tr, image_bl, image_br, shape=(height, width))
            video.write(image_stitched)

        cv2.destroyAllWindows()
        video.release()

    def process_exr(self, images, video_name, video_path):
        archive = ZipFile(self.data_path, 'r')

        frame_data = archive.read(images[0])
        frame = cv2.imdecode(numpy.frombuffer(frame_data, numpy.uint8), cv2.IMREAD_ANYDEPTH)
        height, width = frame.shape

        fltFocal = max(height, width) / 2.0
        fltBaseline = 40.0

        video = cv2.VideoWriter(video_path, 0, 5, (width,height), isColor=False)
        images.sort()
        pbar = tqdm(zip(*[iter(images)]*4))

        for i, (bl, br, tl, tr) in enumerate(pbar):
            pbar.set_description(f'Processing video {video_name}: Total images {len(images)//4}: Processed Images: ')

            image_bl = self.float_depth_map(archive, bl, fltFocal, fltBaseline)
            image_br = self.float_depth_map(archive, br, fltFocal, fltBaseline)
            image_tl = self.float_depth_map(archive, tl, fltFocal, fltBaseline) 
            image_tr = self.float_depth_map(archive, tr, fltFocal, fltBaseline)
            image_stitched = self.stitch_images(image_tl, image_tr, image_bl, image_br, shape=(height, width))
            image = self.depth_map(image_stitched)
            '''
            if i == 0:
                print(image_stitched)
                print(image)
                cv2.imwrite('image_stitched.png', image_stitched)
                cv2.imwrite('image.png', image)
            '''
            video.write(image)

        cv2.destroyAllWindows()
        video.release()

    def float_depth_map(self, archive, data, focal, baseline):
        image = archive.read(data)
        image = cv2.imdecode(numpy.frombuffer(image, numpy.uint8), cv2.IMREAD_ANYDEPTH)
        image = (focal * baseline) / (image + 0.0000001)
        return image

    def stitch_images(self, image_tl, image_tr, image_bl, image_br, shape):
        image_tr_br = cv2.vconcat([image_tr, image_br])
        image_tl_bl = cv2.vconcat([image_tl, image_bl])
        # Stitch 4 images
        image = cv2.hconcat([image_tr_br, image_tl_bl])
        # Downscale image by factor 2
        image = cv2.resize(image, shape)
        return image

    def depth_map(self, image):
        # Convert float32 into uint8 in order to generate correct images
        image_normalized = (image - numpy.min(image)) / (numpy.max(image) - numpy.min(image))
        print(f'image normalized: {image_normalized}')
        image = (image_normalized * 255).astype(numpy.uint8) # set is to a range from 0 till 255
        print(f'image gray: {image}')
        return image

def select_dataset(dirs, pwd):
    print("Type dataset's name to generate video from it.\nType [all] to select all,\n[--color] to select color dataset,\n[--depth] to select depth dataset,\nPress Ctrl+C to cancel.")
    data_path = input()
    dataset = []

    if data_path in dirs:
        dataset.append(os.path.join(pwd, data_path))
    elif data_path == 'all':
        dataset = [os.path.join(pwd, d) for d in dirs]
    elif data_path == '--color':
        dataset = [os.path.join(pwd, d) for d in dirs if '-depth' not in d]
    elif data_path == '--depth':
        dataset = [os.path.join(pwd, d) for d in dirs if '-depth' in d]
    else:
        print('Dataset not found.')
        return select_dataset(dirs, pwd)

    # Convert path strings to path object
    dataset = [Path(d) for d in dataset]
    return dataset

def main():
    arguments_path = './'

    for strOption, strArgument in getopt.getopt(sys.argv[1:], '', [ strParameter[2:] + '=' for strParameter in sys.argv[1::2] ])[0]:
	    if strOption == '--path' and strArgument != '': arguments_path = strArgument # path to the datasets

    dirs = [d for d in os.listdir(arguments_path) if d.endswith('.zip')] #or os.path.isdir(os.path.join(pwd, d))]

    print('Available dataset:')
    for d in dirs:
        print(f'\t- {d}')

    selected_data = select_dataset(dirs, Path(arguments_path))

    for sd in selected_data:
        img2vid = Images2Video(sd, Path(arguments_path))
        img2vid.generate()

if __name__ == "__main__":
    main()