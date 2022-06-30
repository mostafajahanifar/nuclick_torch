from skimage.morphology import remove_small_objects, remove_small_holes, reconstruction, disk
import warnings
import numpy as np


#Returns masks
#preds(no.patchs, 128, 128), nucPoints(no.patchs, 1, 128, 128)
def post_processing(preds, thresh=0.33, minSize=10, minHole=30, doReconstruction=False, nucPoints=None):
    masks = preds > thresh
    masks = remove_small_objects(masks, min_size=minSize)
    masks = remove_small_holes(masks, area_threshold=minHole)
    if doReconstruction:
        for i in range(len(masks)):
            thisMask = masks[i]
            thisMarker = nucPoints[i, 0, :, :] > 0
            try:
                thisMask = reconstruction(thisMarker, thisMask)
                masks[i] = np.array([thisMask])
            except:
                warnings.warn('Nuclei reconstruction error #' + str(i))
    return masks    #masks(no.patchs, 128, 128)


#Returns instanceMap
def gen_instance_map(masks, boundingBoxes, m, n):
    instanceMap = np.zeros((m, n), dtype=np.uint16) 
    for i in range(len(masks)):
        thisBB = boundingBoxes[i]
        thisMaskPos = np.argwhere(masks[i] > 0)
        thisMaskPos[:, 0] = thisMaskPos[:, 0] + thisBB[1]
        thisMaskPos[:, 1] = thisMaskPos[:, 1] + thisBB[0]
        instanceMap[thisMaskPos[:, 0], thisMaskPos[:, 1]] = i + 1
    return instanceMap