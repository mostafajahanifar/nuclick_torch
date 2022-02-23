# NuClick (pytorch implementation)
Pytorch implementation of NuClick for interactive cell segmentation.

![img1]("docs/11.gif")
![img1]("docs/33.gif")
![img1]("docs/Picture2.gif")

"NuClick is a CNN-based approach to speed up collecting annotations for microscopic objects requiring minimum interaction from the annotator."

For more information about netowrk architecture and training process, refer to the following papers (please consider citing these if you use NuClick in your research):

> Navid Alemi, Mostafa Jahanifar, et al. "NuClick: a deep learning framework for interactive segmentation of microscopic images." Medical Image Analysis 65 (2020): 101771.

> Jahanifar, Mostafa, Navid Alemi Koohbanani, and Nasir Rajpoot. "Nuclick: From clicks in the nuclei to nuclear boundaries." arXiv preprint arXiv:1909.03253 (2019).

## Install dependencies
First, please install PyTorch based on official guidelines [here](https://pytorch.org/get-started/locally/). Make sure if you want to use GPU computations, you have the appropriate version of Cuda Toolkit installed before installing PyTorch. After that, simply run the following command to install allt the requirements:
```bash
pip install -r requirements.txt
```
> This repository has been tested with PyTorch 1.9 and 1.10, but should work with newer versions as well.

## Using pretrained NuClick

We release pretrained model weights for two different network architectures implemented in this repository. Both architectures use the sample principle for insteractive segmentation. You can download these weights using the following links.

- [NuClick architecture](https://drive.google.com/file/d/1JBK3vWsVC4DxbcStukwnKNZm-vCSLdOb/view?usp=sharing): The original NuClick architecture introduced in the paper.
- [UNet architecture](https://drive.google.com/file/d/1d_ypVYTsXoMrTVJaEfVRGS5CfLkxyViK/view?usp=sharing): An imporoved UNet model trained for interactive segmentation.

Having the pretrained weights downloaded, you can call the `predict.py` CLI and provide it with paths to input image, point set, and expected output:

```consol
python predict.py --model nuclick -w "checkpoints/NuClick_Nuclick_40xAll.pth" -i input_image.png -p input_points.csv -o output_results.png
```
where the `input_points.csv` should be a list of point coordinates in `(x,y)` format (without header).


Alternatively, you can do nuclei boundary prediction on multiple images using a single command, by proving a path to directories which contain your images and point annotations:

```consol
python predict.py --model nuclick -w "checkpoints/NuClick_Nuclick_40xAll.pth" -imgdir "path/to/images/" -pntdir "path/to/points/" -o "path/to/save/"
```
You only need to make sure that for each image in the `imgdir` exists a point set (in csv format) in the `pntdir`. 

To see all the options you have with `predict.py` function, simply run:
`python predict.py -h`

## Train NuClick on your data
The input images and target masks should be in the `data/imgs` and `data/masks` folders respectively (note that the `imgs` and `masks` folder should not contain any sub-folder or any other files, due to the greedy data-loader).


## Training

```console
> python train.py -h
usage: train.py [-h] [--epochs E] [--batch-size B] [--learning-rate LR]
                [--load LOAD] [--scale SCALE] [--validation VAL] [--amp]

Train the UNet on images and target masks

optional arguments:
  -h, --help            show this help message and exit
  --epochs E, -e E      Number of epochs
  --batch-size B, -b B  Batch size
  --learning-rate LR, -l LR
                        Learning rate
  --load LOAD, -f LOAD  Load model from a .pth file
  --scale SCALE, -s SCALE
                        Downscaling factor of the images
  --validation VAL, -v VAL
                        Percent of the data that is used as validation (0-100)
  --amp                 Use mixed precision
```

By default, the `scale` is 0.5, so if you wish to obtain better results (but use more memory), set it to 1.

Automatic mixed precision is also available with the `--amp` flag. [Mixed precision](https://arxiv.org/abs/1710.03740) allows the model to use less memory and to be faster on recent GPUs by using FP16 arithmetic. Enabling AMP is recommended.




## Weights & Biases

The training progress can be visualized in real-time using [Weights & Biases](https://wandb.ai/).  Loss curves, validation curves, weights and gradient histograms, as well as predicted masks are logged to the platform.

When launching a training, a link will be printed in the console. Click on it to go to your dashboard. If you have an existing W&B account, you can link it
 by setting the `WANDB_API_KEY` environment variable.