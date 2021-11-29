from skimage.measure import regionprops, label
import numpy as np

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
