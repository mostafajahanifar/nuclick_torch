import argparse
import logging
import os

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torchvision import transforms
from skimage.color import label2rgb
import matplotlib.pyplot as plt

from data.dataset_generator import BasicDataset
from models import UNet, NuClick_NN
from utils.visualisation import plot_img_and_mask
from config import DefaultConfig
from utils.utils import getCoordinatesFromCSV, getClickMapAndBoundingBox, getPatchs, postProcessing, generateInstanceMap


def predict_img(net,
                full_img,
                device,
                points_csv,
                scale_factor=1,
                out_threshold=0.5):
    net.eval()

    #Get click coordinates from CSV file
    cx, cy = getCoordinatesFromCSV(points_csv)
    imgWidth = full_img.width
    imgHeight = full_img.height
    #Get click map and bounding box
    clickMap, boundingBoxes = getClickMapAndBoundingBox(cx, cy, imgHeight, imgWidth)

    #Convert full_img to numpy array (3, imgHeight, imgWidth)
    image = np.asarray(full_img)[:, :, :3]
    image = np.moveaxis(image, 2, 0)

    #Generate patchs, inclusion and exlusion maps
    patchs, nucPoints, otherPoints = getPatchs(image, clickMap, boundingBoxes, cx, cy, imgHeight, imgWidth)
    #Divide patchs by 255
    patchs = patchs / 255

    #Concatenate input to model
    input = np.concatenate((patchs, nucPoints, otherPoints), axis=1, dtype=np.float32)
    input = torch.from_numpy(input)
    input.to(device=device, dtype=torch.float32)

    #Predict
    with torch.no_grad():
        output = net(input) #(no.patchs, 1, 128, 128)
        output = torch.sigmoid(output)
        output = torch.squeeze(output, 1)   #(no.patchs, 128, 128)
        preds = output.numpy()

    try:
        masks = postProcessing(preds, thresh=0.5, minSize=10, minHole=30, doReconstruction=True, nucPoints=nucPoints)
    except:
        masks = postProcessing(preds, thresh=0.5, minSize=10, minHole=30, doReconstruction=False, nucPoints=nucPoints)
       
    #Generate instanceMap
    instanceMap = generateInstanceMap(masks, boundingBoxes, imgHeight, imgWidth)
    return instanceMap


def get_args():
    parser = argparse.ArgumentParser(description='Predict masks from input images')
    
    parser.add_argument('--model', '-m', metavar='PATH', required=True,
                        help='Specify the file in which the model is stored')
    parser.add_argument('--input', '-i', metavar='PATH', nargs='+', help='Path to the input images', required=True)
    parser.add_argument('--points', '-p', metavar='PATH', nargs='+', help='Path to CSV files containing point location in x,y format', required=True)
    #parser.add_argument('--output', '-o', metavar='PATH', nargs='+', help='Filenames of output images')
    parser.add_argument('--viz', '-v', action='store_true',
                        help='Visualize the images as they are processed')
    parser.add_argument('--no-save', '-n', action='store_true', help='Do not save the output masks')
    parser.add_argument('--mask-threshold', '-t', type=float, default=DefaultConfig.mask_thresh,
                        help='Minimum probability value to consider a mask pixel white')
    parser.add_argument('--scale', '-s', type=float, default=DefaultConfig.img_scale,
                        help='Scale factor for the input images')

    return parser.parse_args()


#Generate an output name
def get_output_filename(fn):
    split = os.path.splitext(fn)
    return f'{split[0]}_OUT{split[1]}'


#Generate a list of tuples (path to image, path to points) from args
def get_images_points(args):
    images_points = []
    if len(args.input) != len(args.points):
        raise ValueError("Images and points do not match")
    else:
        for i in range(len(args.input)):
            entry = (args.input[i], args.points[i])
            images_points.append(entry)
    return images_points


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
    args = get_args()
    images_points = get_images_points(args)
    # out_files = get_output_filenames(args)

    net = NuClick_NN(n_channels=5, n_classes=1)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logging.info(f'Loading model {args.model}')
    logging.info(f'Using device {device}')

    net.to(device=device)
    net.load_state_dict(torch.load(args.model, map_location=device))

    logging.info('Model loaded!')

    for i, image_point in enumerate(images_points):
        imagePath = image_point[0]
        pointsPath = image_point[1]
        logging.info(f'\nPredicting image {imagePath} ...')
        img = Image.open(imagePath)

        instanceMap = predict_img(net=net,
                           full_img=img,
                           scale_factor=args.scale,
                           out_threshold=args.mask_threshold,
                           points_csv=pointsPath,
                           device=device)

        if not args.no_save:
            #Save instance map
            out_filename = get_output_filename(imagePath)
            Image.fromarray((instanceMap * 255).astype(np.uint8)).save(out_filename)
            logging.info(f'Instance map saved')

        if args.viz:
            #Visualise instance map
            logging.info(f'Visualizing results for image {imagePath}, close to continue...')
            instanceMap_RGB = label2rgb(instanceMap, image=np.asarray(img)[:, :, :3], alpha=0.3, bg_label=0, bg_color=(0, 0, 0), image_alpha=1,kind='overlay')
            plt.figure(), plt.imshow(instanceMap_RGB)
            plt.show()
            