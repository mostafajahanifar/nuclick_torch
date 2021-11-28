import os, glob
import numpy as np
import cv2
from utils.mask_operations import maskRelabeling, mask2clickMap
from scipy.io import savemat
import tqdm

bb = 128 

main_path = '/root/user-temp_mostafa-tia/nuclei_instances_database/'
main_save_train_path = '/root/workspace/nuclei_instances_datasets/NuClick/Train/'
main_save_val_path = '/root/workspace/nuclei_instances_datasets/NuClick/Validation/'
npy_save_path = main_save_train_path + 'npyfiles/'
npy_save_path_val = main_save_val_path + 'npyfiles/'
mat_save_path = main_save_train_path + 'infos/'
mat_save_path_val = main_save_val_path + 'infos/'
os.makedirs(npy_save_path, exist_ok=True)
os.makedirs(npy_save_path_val, exist_ok=True)
os.makedirs(mat_save_path, exist_ok=True)
os.makedirs(mat_save_path_val, exist_ok=True)

datasets = {'Colon_Nuclei': '.png', 'CoNSeP': '.png', 'cpm15': '.png', 'cpm17': '.png', 'CRYONUSEG': '.tif', 'Janowczyk': '.tif', 'monusac': '.tif', 'MoNuSeg': '.tif', 'PanNuke': '.png', 'tnbc': '.png'}
sets = {'Train', 'Test', 'Fold 1', 'Fold 2', 'Fold 3'}
val_percents = {'Colon_Nuclei': 0.03, 'CoNSeP': 0.1, 'cpm15': 0.5, 'cpm17': 0, 'CRYONUSEG': 0.1, 'Janowczyk': 0.3, 'monusac': 0.1, 'MoNuSeg': 0.1, 'PanNuke': 0, 'tnbc': 0}

for dataset, imgExt in zip(datasets.keys(), datasets.values()):
    lenExt = len(imgExt)
    maskExt = '_mask.png'
    
    # gathering list of images and masks
    imgsPathes = []
    masksPathes = []
    setList = []
    for set in sets:
        imgPath = os.path.join(main_path, dataset, set, 'images/')
        maskPath = os.path.join(main_path, dataset, set, 'masks/')
        
        if not os.path.exists(imgPath):
            continue
        
        thisImgsPathes = glob.glob(imgPath+'*'+imgExt)
        imgsPathes += thisImgsPathes
        thisMasksPathes = [thisPath.replace(imgPath, maskPath) for thisPath in thisImgsPathes]
        thisMasksPathes = [thisPath.replace(imgExt, maskExt) for thisPath in thisMasksPathes]
        masksPathes += thisMasksPathes
        setList += [set for _ in range(len(imgsPathes))]
        
    # selecting random images as validation set
    num_val = round(val_percents[dataset] * len(imgsPathes))
    valIdxs = np.random.choice(len(imgsPathes), num_val, replace=False)
    
    # start generating patches for each image in the dataset and save the patches into a numpy  file
    for i, (imgPath, maskPath, set) in enumerate(zip(imgsPathes, masksPathes, setList)):
        imgName = imgPath.split('/')[-1][:-lenExt]
        print('Working on Dataset: {} -- Set: {} -- Image: {}/{} = {}'.format(dataset, set, i, len(imgsPathes)-1, imgPath))
        
        img = cv2.imread(imgPath)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        mask = cv2.imread(maskPath, 0)

        m, n = mask.shape[:2]
        clickMap, cx, cy = mask2clickMap(mask)
        # go through nuclei in the image
        for thisCx, thisCy in tqdm.tqdm(zip(cx, cy)):
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
            
            this_mat_save_path = mat_save_path_val if i in valIdxs else mat_save_path
            
            mdic = {"img": imgPatch, "mask": thisObject, "others": otherObjects}
            savemat(this_mat_save_path+patch_name+'.mat', mdic)