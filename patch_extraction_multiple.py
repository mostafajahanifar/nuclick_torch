import os, glob
import numpy as np
import cv2
from utils.mask_operations import maskRelabeling, mask2clickMap
from scipy.io import savemat
from data.patch_extractor import patch_extract_save

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

# def extract_patches(imgPath, maskPath, save_path, dataset=None, subset=None):
#     '''Extract image, nuclei mask, and others mask patches for nuclick'''
#     ext = '.' + imgPath.split('.')[-1]
#     imgName = imgPath.split('/')[-1][:-len(ext)]
    
#     # Reading images
#     img = cv2.imread(imgPath)
#     img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
#     mask = cv2.imread(maskPath, -1)
#     # mask = imread(maskPath)

#     m, n = mask.shape[:2]
#     clickMap, cx, cy = mask2clickMap(mask)
#     # go through nuclei in the image
#     for thisCx, thisCy in zip(cx, cy):
#         xStart = int(max(thisCx-bb/2,0))
#         yStart = int(max(thisCy-bb/2,0))
#         xEnd = xStart+bb
#         yEnd = yStart+bb
#         if xEnd > n:
#             xEnd = n
#             xStart = n-bb
#         if yEnd > m:
#             yEnd = m
#             yStart = m-bb
        
#         patch_name = f'{imgName}_{xStart}_{yStart}'
#         if subset is not None:
#             patch_name = f'{subset}_' + patch_name
#         if dataset is not None:
#             patch_name = f'{dataset}_' + patch_name

#         maskVal = mask[thisCy,thisCx]
#         if maskVal==0:
#             continue
        
#         maskPatch = mask[yStart:yEnd, xStart:xEnd]
#         imgPatch = img[yStart:yEnd, xStart:xEnd, :]
        
#         thisObject = np.uint8(maskPatch==maskVal)
#         otherObjects = (1-thisObject)*maskPatch
#         otherObjects = np.uint8(maskRelabeling(otherObjects, sizeLimit=5))
        
#         mdic = {"img": imgPatch, "mask": thisObject, "others": otherObjects}
#         savemat(save_path+patch_name+'.mat', mdic)


if __name__ == "__main__":

    for dataset, imgExt in zip(datasets.keys(), datasets.values()):
        lenExt = len(imgExt)
        maskExt = '_mask.png'
        
        # gathering list of images and masks
        imgsPaths = []
        masksPaths = []
        setList = []
        save_path = []
        for _set in sets:
            imgPath = os.path.join(main_path, dataset, _set, 'images/')
            maskPath = os.path.join(main_path, dataset, _set, 'masks/')
            
            if not os.path.exists(imgPath):
                continue
            
            thisImgsPaths = glob.glob(imgPath+'*'+imgExt)
            imgsPaths += thisImgsPaths
            thisMasksPaths = [thisPath.replace(imgPath, maskPath) for thisPath in thisImgsPaths]
            thisMasksPaths = [thisPath.replace(imgExt, maskExt) for thisPath in thisMasksPaths]
            masksPaths += thisMasksPaths
            setList += [_set for _ in range(len(imgsPaths))]
        
        # selecting random images as validation set
        num_val = round(val_percents[dataset] * len(imgsPaths))
        valIdxs = np.random.choice(len(imgsPaths), num_val, replace=False)
        savePaths = [mat_save_path_val if pid in valIdxs else mat_save_path for pid in range(len(imgsPaths))]

        # Instantiating multiprocessing for this folder
        freeze_support() # For Windows support
        num_jobs = len(imgsPaths)
        pbar = tqdm(total=num_jobs, desc=dataset, ascii=True)
        def update_pbar(*xx):
            pbar.update()

        pool = Pool(processes=num_processes, initargs=(RLock(),), initializer=tqdm.set_lock)

        jobs = []
        pid = 1
        for ip, mp, sp, ss in zip(imgsPaths, masksPaths, savePaths, setList):
            jobs.append(pool.apply_async(patch_extract_save, args=(ip, mp, sp, dataset, ss, pid%num_processes), callback=update_pbar))
            pid += 1
        pool.close()
        result_list = [job.get() for job in jobs]
        pbar.close()
