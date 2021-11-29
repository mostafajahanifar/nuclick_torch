import cv2
from utils.mask_operations import maskRelabeling, mask2clickMap
from scipy.io import savemat
import numpy as np
from tqdm import tqdm
from config import DefaultConfig

bb = DefaultConfig.patch_size

def patch_extract_save(imgPath, maskPath, save_path, dataset=None, subset=None, pid=None):
    '''Extract image, nuclei mask, and others mask patches for nuclick'''
    ext = '.' + imgPath.split('.')[-1]
    imgName = imgPath.split('/')[-1][:-len(ext)]
    
    # Reading images
    img = cv2.imread(imgPath)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    mask = cv2.imread(maskPath, -1)
    # mask = imread(maskPath)

    m, n = mask.shape[:2]
    clickMap, cx, cy = mask2clickMap(mask)
    # go through nuclei in the image
    if pid is not None:
        this_pbar = tqdm(total=len(cx), desc=imgName, leave=False, ascii=True, position=pid+1)
    for thisCx, thisCy in zip(cx, cy):
        xStart = int(max(thisCx-bb/2,0))
        yStart = int(max(thisCy-bb/2,0))
        xEnd = xStart+bb
        yEnd = yStart+bb
        if xEnd > n:
            xEnd = n
            xStart = n-bb
        if yEnd > m:
            yEnd = m
            yStart = m-bb
        
        patch_name = f'{imgName}_{xStart}_{yStart}'
        if subset is not None:
            patch_name = f'{subset}_' + patch_name
        if dataset is not None:
            patch_name = f'{dataset}_' + patch_name

        maskVal = mask[thisCy,thisCx]
        if maskVal==0:
            continue
        
        maskPatch = mask[yStart:yEnd, xStart:xEnd]
        imgPatch = img[yStart:yEnd, xStart:xEnd, :]
        
        thisObject = np.uint8(maskPatch==maskVal)
        otherObjects = (1-thisObject)*maskPatch
        otherObjects = np.uint8(maskRelabeling(otherObjects, sizeLimit=5))
        
        mdic = {"img": imgPatch, "mask": thisObject, "others": otherObjects}
        savemat(save_path+patch_name+'.mat', mdic)
        if pid is not None:
            this_pbar.update(1)
    if pid is not None:
        this_pbar.close()