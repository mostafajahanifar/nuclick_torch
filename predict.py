import argparse
import logging
import os

import numpy as np
import torch
import cv2
from PIL import Image
from skimage.color import label2rgb
import matplotlib.pyplot as plt

from models import UNet, NuClick_NN
from config import DefaultConfig
from utils.process import post_processing, gen_instance_map
from utils.misc import get_coords_from_csv, get_clickmap_boundingbox, get_output_filename, get_images_points
from utils.guiding_signals import get_patches_and_signals

def predict_img(net,
                full_img,
                device,
                points_csv,
                scale_factor=1,
                out_threshold=0.5):
    net.eval()

    #Get click coordinates from CSV file
    cx, cy = get_coords_from_csv(points_csv)
    imgWidth = full_img.width
    imgHeight = full_img.height
    #Get click map and bounding box
    clickMap, boundingBoxes = get_clickmap_boundingbox(cx, cy, imgHeight, imgWidth)

    #Convert full_img to numpy array (3, imgHeight, imgWidth)
    image = np.asarray(full_img)[:, :, :3]
    image = np.moveaxis(image, 2, 0)

    #Generate patchs, inclusion and exlusion maps
    patchs, nucPoints, otherPoints = get_patches_and_signals(image, clickMap, boundingBoxes, cx, cy, imgHeight, imgWidth)
    #Divide patchs by 255
    patchs = patchs / 255

    #Concatenate input to model
    input = np.concatenate((patchs, nucPoints, otherPoints), axis=1, dtype=np.float32)
    input = torch.from_numpy(input)
    input = input.to(device=device, dtype=torch.float32)

    #Predict
    with torch.no_grad():
        output = net(input) #(no.patchs, 1, 128, 128)
        output = torch.sigmoid(output)
        output = torch.squeeze(output, 1)   #(no.patchs, 128, 128)
        preds = output.cpu().numpy()


    masks = post_processing(preds, thresh=out_threshold, minSize=10, minHole=30, doReconstruction=True, nucPoints=nucPoints)
    
    #Generate instanceMap
    instanceMap = gen_instance_map(masks, boundingBoxes, imgHeight, imgWidth)
    return instanceMap


def get_args():
    parser = argparse.ArgumentParser(description='Predict masks from input images')

    parser.add_argument('--model', '-m', metavar='NAME', required=True, help='Name of the model')
    parser.add_argument('--pretrained_weights', '-w', metavar='PATH', required=True,
                        help='Path to the pretrained weights')
    
    imageGroup = parser.add_mutually_exclusive_group(required=True)
    imageGroup.add_argument('--image', '-i', metavar='PATH', nargs='+', help='Path to the input images')
    imageGroup.add_argument('-imgdir', metavar='PATH', help='Path to the directory containing input images')

    pointsGroup = parser.add_mutually_exclusive_group(required=True)
    pointsGroup.add_argument('--points', '-p', metavar='PATH', nargs='+', help='Path to the CSV files containing points')
    pointsGroup.add_argument('-pntdir', metavar='PATH', help='Path to the directory containing the CSV files')
    
    parser.add_argument('--output', '-o', metavar='PATH', help='Directory where the instance maps will be saved into')
    parser.add_argument('--viz', '-v', action='store_true',
                        help='Visualize the images as they are processed')
    parser.add_argument('--no-save', '-n', action='store_true', help='Do not save the output masks')
    parser.add_argument('--mask-threshold', '-t', type=float, default=DefaultConfig.mask_thresh,
                        help='Minimum probability value to consider a mask pixel white')
    parser.add_argument('--scale', '-s', type=float, default=DefaultConfig.img_scale,
                        help='Scale factor for the input images')
    parser.add_argument('--gpu', '-g', metavar='GPU', default=None, help='ID of GPUs to use (based on `nvidia-smi`)')

    return parser.parse_args()


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
    args = get_args()
    
    # setting gpus
    if args.gpu is not None:
        os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
        print(args.gpu)
        print(type(args.gpu))
    
    # getting images and points
    images_points = get_images_points(args)

    if (args.model.lower() == 'nuclick'):
        net = NuClick_NN(n_channels=5, n_classes=1)
    elif (args.model.lower() == 'unet'):
        net = UNet(n_channels=5, n_classes=1)
    else:
        raise ValueError('Invalid model type. Acceptable networks are UNet or NuClick')

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logging.info(f'Loading model {args.model}')
    logging.info(f'Using device {device}')

    net.to(device=device)
    net.load_state_dict(torch.load(args.pretrained_weights, map_location=device))

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
            out_filename = get_output_filename(imagePath, args.output)

            cv2.imwrite(out_filename, instanceMap)
            logging.info(f'Instance map saved as {out_filename}')

        if args.viz:
            #Visualise instance map
            logging.info(f'Visualizing results for image {imagePath}, close to continue...')
            instanceMap_RGB = label2rgb(instanceMap, image=np.asarray(img)[:, :, :3], alpha=0.3, bg_label=0, bg_color=(0, 0, 0), image_alpha=1,kind='overlay')
            plt.figure(), plt.imshow(instanceMap_RGB)
            plt.show()
            