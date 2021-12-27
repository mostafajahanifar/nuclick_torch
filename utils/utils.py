import numpy as np
import csv
from skimage.morphology import remove_small_objects, remove_small_holes, reconstruction, disk
import warnings

#bb = DefaultConfig.img_rows
bb = 128 #Nucleus

def getClickMapAndBoundingBox(cx, cy, m, n):
    clickMap = np.zeros((m, n), dtype=np.uint8)

    # Removing points out of image dimension (these points may have been clicked unwanted)
    cx_out = [x for x in cx if x >= n]
    cx_out_index = [cx.index(x) for x in cx_out]

    cy_out = [x for x in cy if x >= m]
    cy_out_index = [cy.index(x) for x in cy_out]

    indexes = cx_out_index + cy_out_index
    cx = np.delete(cx, indexes)
    cx = cx.tolist()
    cy = np.delete(cy, indexes)
    cy = cy.tolist()

    clickMap[cy, cx] = 1
    boundingBoxes = []
    for i in range(len(cx)):
        xStart = cx[i] - bb // 2
        yStart = cy[i] - bb // 2
        if xStart < 0:
            xStart = 0
        if yStart < 0:
            yStart = 0
        xEnd = xStart + bb - 1
        yEnd = yStart + bb - 1
        if xEnd > n - 1:
            xEnd = n - 1
            xStart = xEnd - bb + 1
        if yEnd > m - 1:
            yEnd = m - 1
            yStart = yEnd - bb + 1
        boundingBoxes.append([xStart, yStart, xEnd, yEnd])
    return clickMap, boundingBoxes


#Returns a list of x coordinates and a list of y coordinates from the given CSV file
def getCoordinatesFromCSV(filename):
   
    file = open(filename)
    csvReader = csv.reader(file)
    clicks_x = []
    clicks_y = []

    for row in csvReader:
        if len(row) != 2:
            raise ValueError(f"The CSV file: '{filename}' does not have valid entries.")
        v1 = row[0]
        v2= row[1]

        x = int(v1)
        y = int(v2)

        clicks_x.append(x)
        clicks_y.append(y)

    file.close()

    if len(clicks_x) == 0:
        raise ValueError(f"The CSV file '{filename}' is empty")

    return clicks_x, clicks_y


# m: height of img, n: width of img
def getPatchs(img, clickMap, boundingBoxes, cx, cy, m, n):
    # total = number of clicks
    total = len(boundingBoxes)
    img = np.array([img])   #img.shape=(1,3,256,256)
    clickMap = np.array([clickMap])     #clickmap.shape=(1,256,256)

    #clickMap = clickMap[..., np.newaxis]
    clickMap = clickMap[:, np.newaxis, ...]    #clickmap.shape=(1,1,256,256)

    # patchs = np.ndarray((total, bb, bb, 3), dtype=np.uint8)
    # nucPoints = np.ndarray((total, bb, bb, 1), dtype=np.uint8)
    # otherPoints = np.ndarray((total, bb, bb, 1), dtype=np.uint8)
    # PyTorch - channel first
    patchs = np.ndarray((total, 3, bb, bb), dtype=np.uint8)
    nucPoints = np.ndarray((total, 1, bb, bb), dtype=np.uint8)
    otherPoints = np.ndarray((total, 1, bb, bb), dtype=np.uint8)

    # Removing points out of image dimension (these points may have been clicked unwanted)
    cx_out = [x for x in cx if x >= n]
    cx_out_index = [cx.index(x) for x in cx_out]

    cy_out = [x for x in cy if x >= m]
    cy_out_index = [cy.index(x) for x in cy_out]

    indexes = cx_out_index + cy_out_index
    cx = np.delete(cx, indexes)
    cx = cx.tolist()
    cy = np.delete(cy, indexes)
    cy = cy.tolist()

    for i in range(len(boundingBoxes)):
        boundingBox = boundingBoxes[i]
        xStart = boundingBox[0]
        yStart = boundingBox[1]
        xEnd = boundingBox[2]
        yEnd = boundingBox[3]

        #patchs[i] = img[0, yStart:yEnd + 1, xStart:xEnd + 1, :]
        patchs[i] = img[0, :, yStart:yEnd + 1, xStart:xEnd + 1]

        #thisClickMap = np.zeros((1, m, n, 1), dtype=np.uint8)
        #thisClickMap[0, cy[i], cx[i], 0] = 1
        thisClickMap = np.zeros((1, 1, m, n), dtype=np.uint8)
        thisClickMap[0, 0, cy[i], cx[i]] = 1

        othersClickMap = np.uint8((clickMap - thisClickMap) > 0)

        # nucPoints[i] = thisClickMap[0, yStart:yEnd + 1, xStart:xEnd + 1, :]
        # otherPoints[i] = othersClickMap[0, yStart:yEnd + 1, xStart:xEnd + 1, :]
        nucPoints[i] = thisClickMap[0, :, yStart:yEnd + 1, xStart:xEnd + 1]
        otherPoints[i] = othersClickMap[0, :, yStart:yEnd + 1, xStart:xEnd + 1]

    # patchs: (total, 3, 128, 128)
    # nucPoints: (total, 1, 128, 128)
    # otherPoints: (total, 1, 128, 128)
    return patchs, nucPoints, otherPoints   


#preds(no.patchs, 128, 128), nucPoints(no.patchs, 1, 128, 128) 
def postProcessing(preds, thresh=0.33, minSize=10, minHole=30, doReconstruction=False, nucPoints=None):
    masks = preds > thresh
    masks = remove_small_objects(masks, min_size=minSize)
    masks = remove_small_holes(masks, area_threshold=minHole)
    if doReconstruction:
        for i in range(len(masks)):
            thisMask = masks[i]
            # thisMarker = nucPoints[i, :, :, 0] > 0
            thisMarker = nucPoints[i, 0, :, :] > 0
            
            try:
                thisMask = reconstruction(thisMarker, thisMask, footprint=disk(1))
                masks[i] = np.array([thisMask])
            except:
                warnings.warn('Nuclei reconstruction error #' + str(i))
    return masks    #masks(no.patchs, 128, 128)
    


def generateInstanceMap(masks, boundingBoxes, m, n):
    instanceMap = np.zeros((m, n), dtype=np.uint16)
    for i in range(len(masks)):
        thisBB = boundingBoxes[i]
        thisMaskPos = np.argwhere(masks[i] > 0)
        thisMaskPos[:, 0] = thisMaskPos[:, 0] + thisBB[1]
        thisMaskPos[:, 1] = thisMaskPos[:, 1] + thisBB[0]
        instanceMap[thisMaskPos[:, 0], thisMaskPos[:, 1]] = i + 1
    return instanceMap