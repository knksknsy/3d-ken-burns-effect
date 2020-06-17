from cv2 import cv2
from tqdm import tqdm
import os
import numpy

image_folder = 'city-flying'
video_name = 'video.avi'

images = [img for img in os.listdir(image_folder) if img.endswith(".png")]
frame = cv2.imread(os.path.join(image_folder, images[0]))
height, width, layers = frame.shape
print(frame.shape)

video = cv2.VideoWriter(video_name, 0, 5, (width,height))

images.sort()

pbar = tqdm(zip(*[iter(images)]*4))

for bl, br, tl, tr in pbar:
    pbar.set_description(f'Processing video')
    image_bl = cv2.imread(os.path.join(image_folder, bl))
    image_br = cv2.imread(os.path.join(image_folder, br))
    image_tl = cv2.imread(os.path.join(image_folder, tl))
    image_tr = cv2.imread(os.path.join(image_folder, tr))

    image_tr_br = cv2.vconcat([image_tr, image_br])
    image_tl_bl = cv2.vconcat([image_tl, image_bl])
    # Combine 4 images
    image = cv2.hconcat([image_tr_br, image_tl_bl])
    # Downscale image by factor 2
    image = cv2.resize(image, (frame.shape[0], frame.shape[1])) 
    #cv2.imwrite("image.png", image)
    video.write(image)

cv2.destroyAllWindows()
video.release()