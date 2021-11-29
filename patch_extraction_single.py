import os, glob
import numpy as np
from data.patch_extractor import patch_extract_save

from tqdm import tqdm
from multiprocessing import Pool, freeze_support, RLock

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

if __name__ == "__main__":
    '''Extracting NuClick patches for a single folder
    
    Assuming that masks have the same name as images with the prefix of '_mask.png'
    '''
    # setting the paths to image, mask , and svaing folders and image and mask extentions.
    imgExt = '.tif'
    maskExt = '_mask.png'

    imgPath = '/root/user-temp_mostafa-tia/nuclei_instances_database/MoNuSeg/Train/images/'
    maskPath = '/root/user-temp_mostafa-tia/nuclei_instances_database/MoNuSeg/Train/masks/'
    mat_save_path = '/root/workspace/nuclei_instances_datasets/NuClick/MoNuSegTrain/'
    os.makedirs(mat_save_path, exist_ok=True)

    # Finding image and paths and creating saveing paths accordingly
    imgsPaths = glob.glob(imgPath+'*'+imgExt)
    masksPaths = [thisPath.replace(imgPath, maskPath) for thisPath in imgsPaths]
    masksPaths = [thisPath.replace(imgExt, maskExt) for thisPath in masksPaths]
    savePaths = [mat_save_path for _ in range(len(imgsPaths))]
      
    # Instantiating multiprocessing for this folder
    freeze_support() # For Windows support
    num_processes = 8
    num_jobs = len(imgsPaths)
    pbar = tqdm(total=num_jobs, ascii=True)
    def update_pbar(*xx):
        pbar.update()

    pool = Pool(processes=num_processes, initargs=(RLock(),), initializer=tqdm.set_lock)

    jobs = []
    pid = 1
    for ip, mp, sp in zip(imgsPaths, masksPaths, savePaths):
        jobs.append(pool.apply_async(patch_extract_save, args=(ip, mp, sp), callback=update_pbar))
        pid += 1
    pool.close()
    result_list = [job.get() for job in jobs]
    pbar.close()
