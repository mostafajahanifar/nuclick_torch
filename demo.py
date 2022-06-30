import numpy as np
from skimage.io import imsave
import matplotlib.pyplot as plt
from models import UNet, NuClick_NN
import torch
import logging
from skimage.color import label2rgb

import os
from config import DemoConfig as config
from utils.process import post_processing, gen_instance_map

from utils.misc import readImageAndGetClicks, get_clickmap_boundingbox
from utils.guiding_signals import get_patches_and_signals



def main():
    logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
    model_type = config.network #'NuClick'
    weights_path = config.weights_path[0]
    print(weights_path)

    # loading models
    if (model_type.lower() == 'nuclick'):
        net = NuClick_NN(n_channels=5, n_classes=1)
    elif (model_type.lower() == 'unet'):
        net = UNet(n_channels=5, n_classes=1)
    else:
        raise ValueError('Invalid model type. Acceptable networks are UNet or NuClick')

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logging.info(f'Loading model {model_type}')
    logging.info(f'Using device {device}')

    net.to(device=device)
    net.load_state_dict(torch.load(weights_path, map_location=device))

    logging.info('Model loaded!')

    ##Reading images
    # Select one image input paradigm
    # img, cx, cy = readImageAndCentroids(path,name)
    # img, cx, cy = readImageFromPathAndGetClicks (path,name,ext='.bmp')
    if config.application in ['Cell', 'Nucleus']:
        img, cx, cy, imgPath = readImageAndGetClicks(os.getcwd())
        m, n = img.shape[0:2]
        img = np.asarray(img)[:, :, :3]
        img = np.moveaxis(img, 2, 0)
        clickMap, boundingBoxes = get_clickmap_boundingbox(cx, cy, m, n)
        patchs, nucPoints, otherPoints = get_patches_and_signals(img, clickMap, boundingBoxes, cx, cy, m, n)
        patchs = patchs / 255

        input = np.concatenate((patchs, nucPoints, otherPoints), axis=1, dtype=np.float32)
        input = torch.from_numpy(input)
        input = input.to(device=device, dtype=torch.float32)
        # prediction with test time augmentation

        #Predict
        with torch.no_grad():
            output = net(input) #(no.patchs, 1, 128, 128)
            output = torch.sigmoid(output)
            output = torch.squeeze(output, 1)   #(no.patchs, 128, 128)
            preds = output.cpu().numpy()
        logging.info("Original images prediction, DONE!")

        masks = post_processing(preds, thresh=config.threshold, minSize=10, minHole=30, doReconstruction=True, nucPoints=nucPoints)

        #Generate instanceMap
        instanceMap = gen_instance_map(masks, boundingBoxes, m, n)
        img = np.moveaxis(img, 0, 2)
        instanceMap_RGB = label2rgb(instanceMap, image=np.asarray(img)[:, :, :3], alpha=0.75, bg_label=0, bg_color=(0, 0, 0), image_alpha=1,kind='overlay')

        imsave(imgPath[:-4]+'_overlay.png',instanceMap_RGB)
        imsave(imgPath[:-4] + '_instances.png', instanceMap)

        plt.subplot(121), plt.imshow(img)
        plt.subplot(122), plt.imshow(instanceMap_RGB)
        plt.show()

if __name__=='__main__':
    main()
