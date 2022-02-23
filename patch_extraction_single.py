import os, glob
import numpy as np
from data.patch_extractor import patch_extract_save

from tqdm import tqdm
from multiprocessing import Pool, freeze_support, RLock

if __name__ == "__main__":
    '''Extracting NuClick patches for a single folder
    
    Assuming that masks have the same name as images with the prefix of '_mask.png'
    '''
    # setting the paths to image, mask , and svaing folders and image and mask extentions.
    imgExt = '.tif'
    maskExt = '_mask.png'

    imgPath = '/path/to/training/images/'
    maskPath = '/path/to/training/masks/'
    mat_save_path = '/path/to/SAVE/nuclick_patches/'
    os.makedirs(mat_save_path, exist_ok=True)

    # Finding image and paths and creating saveing paths accordingly
    imgsPaths = glob.glob(imgPath+'*'+imgExt)
    masksPaths = [thisPath.replace(imgPath, maskPath) for thisPath in imgsPaths]
    masksPaths = [thisPath.replace(imgExt, maskExt) for thisPath in masksPaths]
    savePaths = [mat_save_path for _ in range(len(imgsPaths))]
      
    # Instantiating multiprocessing for this folder
    freeze_support() # For Windows support
    num_processes = 1 # number of processing cores
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
