import os, glob
import numpy as np
import cv2
from utils.mask_operations import maskRelabeling, mask2clickMap
from scipy.io import savemat
from skimage.io import imread

from tqdm import tqdm
from multiprocessing import Pool, freeze_support, RLock

bb = 128 
num_processes = 8

# defining the paths
main_path = '/root/user-temp_mostafa-tia/nuclei_instances_database/'
main_save_train_path = '/root/workspace/nuclei_instances_datasets/NuClick/Train/'
main_save_val_path = '/root/workspace/nuclei_instances_datasets/NuClick/Validation/'
mat_save_path = main_save_train_path + 'mat_files/'
mat_save_path_val = main_save_val_path + 'mat_files/'
os.makedirs(mat_save_path, exist_ok=True)
os.makedirs(mat_save_path_val, exist_ok=True)

datasets = {'Colon_Nuclei': '.png', 'CoNSeP': '.png', 'cpm15': '.png', 'cpm17': '.png', 'CRYONUSEG': '.tif', 'Janowczyk': '.tif', 'monusac': '.tif', 'MoNuSeg': '.tif', 'PanNuke': '.png', 'tnbc': '.png'}
sets = {'Train', 'Test', 'Fold 1', 'Fold 2', 'Fold 3'}
val_percents = {'Colon_Nuclei': 0.03, 'CoNSeP': 0.1, 'cpm15': 0.5, 'cpm17': 0, 'CRYONUSEG': 0.1, 'Janowczyk': 0.3, 'monusac': 0.1, 'MoNuSeg': 0.1, 'PanNuke': 0, 'tnbc': 0}

def extract_patches(pid, imgPath, maskPath, set):
    '''Extract image, nuclei mask, and others mask patches for nuclick'''

    # start generating patches for each image in the dataset and save the patches into a numpy  file
    imgName = imgPath.split('/')[-1][:-lenExt]
    # pbar.write('Working on Dataset: {} -- Set: {} -- Image: {}/{} = {}'.format(dataset, set, pid, len(imgsPaths)-1, imgPath))
    
    img = cv2.imread(imgPath)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    # mask = cv2.imread(maskPath)
    mask = imread(maskPath)

    m, n = mask.shape[:2]
    clickMap, cx, cy = mask2clickMap(mask)
    # go through nuclei in the image
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
        
        patch_name = f'{dataset}_{set}_{imgName}_{xStart}_{yStart}'
        maskVal = mask[thisCy,thisCx]
        if maskVal==0:
            continue
        
        maskPatch = mask[yStart:yEnd, xStart:xEnd]
        imgPatch = img[yStart:yEnd, xStart:xEnd, :]
        
        thisObject = np.uint8(maskPatch==maskVal)
        otherObjects = (1-thisObject)*maskPatch
        otherObjects = np.uint8(maskRelabeling(otherObjects, sizeLimit=5))
        
        this_mat_save_path = mat_save_path_val if pid in valIdxs else mat_save_path
        
        mdic = {"img": imgPatch, "mask": thisObject, "others": otherObjects}
        savemat(this_mat_save_path+patch_name+'.mat', mdic)


if __name__ == "__main__":

    for dataset, imgExt in zip(datasets.keys(), datasets.values()):
        lenExt = len(imgExt)
        maskExt = '_mask.png'
        
        # gathering list of images and masks
        imgsPaths = []
        masksPaths = []
        setList = []
        for set in sets:
            imgPath = os.path.join(main_path, dataset, set, 'images/')
            maskPath = os.path.join(main_path, dataset, set, 'masks/')
            
            if not os.path.exists(imgPath):
                continue
            
            thisImgsPaths = glob.glob(imgPath+'*'+imgExt)
            imgsPaths += thisImgsPaths
            thisMasksPaths = [thisPath.replace(imgPath, maskPath) for thisPath in thisImgsPaths]
            thisMasksPaths = [thisPath.replace(imgExt, maskExt) for thisPath in thisMasksPaths]
            masksPaths += thisMasksPaths
            setList += [set for _ in range(len(imgsPaths))]
        
        # selecting random images as validation set
        num_val = round(val_percents[dataset] * len(imgsPaths))
        valIdxs = np.random.choice(len(imgsPaths), num_val, replace=False)

        # Instantiating multiprocessing for this folder
        freeze_support() # For Windows support
        num_jobs = len(imgsPaths)
        pbar = tqdm(total=num_jobs, desc=dataset, ascii=True)
        def update_pbar(*xx):
            pbar.update(1)

        pool = Pool(processes=num_processes, initargs=(RLock(),), initializer=tqdm.set_lock)

        jobs = [pool.apply_async(extract_patches, args=(pid, ip, mp, s), callback=update_pbar()) for pid, (ip, mp, s) in enumerate(zip(imgsPaths, masksPaths, setList))]
        pool.close()
        result_list = [job.get() for job in jobs]
    
    