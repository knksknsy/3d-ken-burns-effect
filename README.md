# 3D Ken Burns Effect from a Single Image
This is a reference implementation of 3D Ken Burns Effect from a Single Image [1] using PyTorch. Given a single input image, it animates this still image with a virtual camera scan and zoom subject to motion parallax. Should you be making use of our work, please cite our paper [1].

<a href="https://arxiv.org/abs/1909.05483" rel="Paper"><img src="http://content.sniklaus.com/kenburns/paper.jpg" alt="Paper" width="100%"/></a>

## Setup
To download the pre-trained models, run:
```
bash scripts/download_models.bash
```

Make sure to install the required Python packages listed in the `requirements.txt` with:

```
conda install -c conda-forge --file requirements.txt
```

Several functions are implemented in CUDA using CuPy, which is why CuPy is a required dependency. It can be installed using `pip install cupy` or alternatively using one of the provided binary packages as outlined in the CuPy repository. Please also make sure to have the `CUDA_HOME` environment variable configured.

## Download IPYNB notebooks and assets
For running the notebooks locally on your machine, download the files from [Google Drive](https://drive.google.com/file/d/1w8KcFs88MMbD1ZHww4Q3ieMkffmj5XdC/view?usp=sharing).
Unpack the `notebooks.tar.gz` file by executing the following command in the terminal:
```
tar -xzf notebooks.tar.gz
```
It is recommended to unpack the files into the root directory of the project.

Instructions for downloading the dataset and training the model can be found in chapters [Dataset](#dataset) and [Training](#training) respectively.

## Usage
To run it on an image and generate the 3D Ken Burns effect fully automatically, use the following command.

```
python autozoom.py --in ./images/doublestrike.jpg --out ./videos/autozoom.mp4
```

To start the interface that allows you to manually adjust the camera path, use the following command. You can then navigate to `http://localhost:8080/` and load an image using the button on the bottom right corner. Please be patient when loading an image and saving the result, there is a bit of background processing going on.

```
python interface.py
```

To run the depth estimation to obtain the raw depth estimate, use the following command. Please note that this script does not perform the depth adjustment, see [#22](https://github.com/sniklaus/3d-ken-burns/issues/22) for information on how to add it.

```
python depthestim.py --in ./images/doublestrike.jpg --out ./depthestim.npy
```

To benchmark the depth estimation, run `python scripts/benchmark.py`. You can use it to easily verify that the provided implementation runs as expected.

<a id='dataset'></a>
## Dataset
This dataset is licensed under the [Creative Commons Attribution-NonCommercial-ShareAlike 4.0 International Public License (CC BY-NC-SA 4.0)](https://creativecommons.org/licenses/by-nc-sa/4.0/legalcode) and may only be used for non-commercial purposes. Please see the LICENSE file for more information.

The dataset can be downloaded with the following line:
`python scripts/download_dataset.py --path <DATASET_DESTINATION> --csv`

This script will also generate two CSV files necessary for training the model.

<a id='training'></a>
## Training
The training of the following networks can be triggered with:

**Disparity Estimation Network:**
```
python training/train_disparity_estimation.py
```

**Disparity Refinement Network:**
```
python training/train_disparity_refinement.py
```

**Context and Inpainting Network:**
```
python training/train_pointcloud_inpainting.py
```

## License
This is a project by Adobe Research. It is licensed under the [Creative Commons Attribution-NonCommercial-ShareAlike 4.0 International Public License (CC BY-NC-SA 4.0)](https://creativecommons.org/licenses/by-nc-sa/4.0/legalcode) and may only be used for non-commercial purposes. Please see the LICENSE file for more information.

## References
```
[1]  @article{Niklaus_TOG_2019,
         author = {Simon Niklaus and Long Mai and Jimei Yang and Feng Liu},
         title = {3D Ken Burns Effect from a Single Image},
         journal = {ACM Transactions on Graphics},
         volume = {38},
         number = {6},
         pages = {184:1--184:15},
         year = {2019}
     }
```

## Acknowledgment
This is the implementation of <a href="https://arxiv.org/abs/1909.05483">3D Ken Burns Effect from a Single Image</a> from Niklaus et. al.
