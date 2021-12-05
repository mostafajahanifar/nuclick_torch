import warnings
from skimage.measure import regionprops, label
import numpy as np
import cv2

def mask2clickMap (mask):
    clickMap = np.zeros_like(mask)
    cx = []
    cy = []
    stats = regionprops(mask)
    for stat in stats:
        y,x = stat.centroid
        y = int(np.floor(y))
        x = int(np.floor(x))
        cy.append(y)
        cx.append(x)
        clickMap[y, x] = 1
    return clickMap, cx, cy

def maskRelabeling (inMask, sizeLimit=5):
    outMask = np.zeros_like(inMask, dtype=np.uint16)
    uniqueLabels = np.unique(inMask)
    if uniqueLabels[0] == 0:
        uniqueLabels = np.delete(uniqueLabels, 0) 
    
    i = 1
    for l in uniqueLabels:
        thisMask = label(inMask==l, connectivity=1)
        stats = regionprops(thisMask)
        for stat in stats:
            if stat.area > sizeLimit:
                outMask[stat.coords[:,0], stat.coords[:,1]] = i
                i += 1
    return outMask

def binary_reconstruction(marker, mask):
    ret, labels = cv2.connectedComponents(mask, connectivity=8)
    reconstructed_mask = np.zeros_like(mask, dtype=np.uint8)
    for label in range(1,ret):
        this_mask = np.uint8(labels==label)
        residual = this_mask*marker
        if np.any(residual>0):
            reconstructed_mask += this_mask
    return reconstructed_mask

def adaptive_distance_thresholding (mask):
    '''Refining the input mask using adaptive distance thresholding.
    
    Distance map of the input mask is generated and the an adaptive 
    (random) threshold based on the distance map is calculated to
    generate a new mask from distance map based on it.

    Inputs:
        mask (::np.ndarray::): Should be a 2D binary numpy array (uint8)
    Outputs:
        new_mask (::np.ndarray::): the refined mask
        dist (::np.ndarray::): the distance map
    '''
    dist = cv2.distanceTransform(mask, cv2.DIST_L2, 0)
    tempMean = np.mean(dist[dist>0])
    tempStd = np.std(dist[dist>0])
    tempTol = tempStd/2
    low_thresh = np.max([tempMean-tempTol, 0])
    high_thresh = np.min([tempMean+tempTol, np.max(dist)-tempTol])
    if low_thresh>=high_thresh:
        thresh = tempMean
    else:
        thresh = np.random.uniform(low_thresh, high_thresh)
    new_mask = dist>thresh
    if np.all(new_mask == np.zeros_like(new_mask)):
        new_mask = dist>tempMean
    return new_mask, dist