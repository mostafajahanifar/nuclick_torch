import warnings
import numpy as np
import cv2
from utils.mask_operations import adaptive_distance_thresholding
from config import DefaultConfig

bb = DefaultConfig.patch_size    #128 for nucleus


class GuidingSignal(object):
    '''A generic class for defining guiding signal generators.
    
    This class include some special methods that inclusion and exclusion guiding signals
    for different application can be created based on.
    '''
    def __init__(self, mask: np.ndarray, others: np.ndarray, kernel_size: int = 0) -> None:
        self.mask = self.mask_validator(mask>0.5)
        self.kernel_size = kernel_size
        if kernel_size:
            self.current_mask = self.mask_preprocess(self.mask, kernel_size=self.kernel_size)
        else:
            self.current_mask = self.mask_validator(mask>0.5)
        self.others = others
        
    
    @staticmethod
    def mask_preprocess(mask, kernel_size=3):
        kernel = np.ones((kernel_size,kernel_size),np.uint8)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
        if np.all(mask==np.zeros_like(mask)):
            warnings.warn(f'The kernel_size (radius) of {kernel_size} may be too high, consider checking '
            'the intermediate output for the sanity of generated masks.')
        return mask

    @staticmethod
    def mask_validator(mask):
        '''Validate the input mask be np.uint8 and 2D'''
        assert len(mask.shape)==2, "Mask must be a 2D array (NxM)"
        if not issubclass(type(mask[0, 0]), np.integer):
            mask = np.uint8(mask)
        return mask

    def inclusion_map(self):
        '''A function to generate inclusion gioding signal'''
        raise NotImplementedError

    def exclusion_map(self):
        '''A function to generate exclusion gioding signal'''
        raise NotImplementedError

class PointGuidingSignal(GuidingSignal):
    def __init__(self, mask: np.ndarray, others: np.ndarray, perturb: str = 'None', **kwargs) -> None:
        super().__init__(mask, others, **kwargs)
        if perturb.lower() not in {'none', 'distance', 'inside'}:
            raise ValueError(f'Invalid running perturb type of: {perturb}. Perturn type should be `"None"`, `"inside"`, or `"distance"`.')
        self.perturb = perturb.lower()

    def inclusion_map(self):
        if self.perturb is None: # if there is no purturbation
            indices = np.argwhere(self.current_mask==1) #
            centroid = np.mean(indices, axis=0)
            pointMask = np.zeros_like(self.current_mask)   
            pointMask[int(centroid[0]),int(centroid[1]),0] = 1
            return pointMask, self.current_mask
        elif self.perturb=='distance' and np.any(self.current_mask>0):
            new_mask, _ = adaptive_distance_thresholding(self.current_mask)
        else: # if self.perturb=='inside':
            new_mask = self.current_mask.copy()

        # Creating the point map
        pointMask = np.zeros_like(self.current_mask) 
        indices = np.argwhere(new_mask==1)
        if len(indices)>0:
            rndIdx = np.random.randint(0, len(indices))
            rndX = indices[rndIdx, 1]
            rndY = indices[rndIdx, 0]
            pointMask[rndY, rndX] = 1

        return pointMask

    def exclusion_map(self, random_drop=0, random_jitter=0):
        _, _, _, centroids = cv2.connectedComponentsWithStats(
            self.others, 4, cv2.CV_32S)

        centroids = centroids[1:, :] # removing the first centroid, it's background
        if random_jitter:
            centroids = self.jitterClicks (self.current_mask.shape, centroids, jitter_range=random_jitter)
        if random_drop: # randomly dropping some of the points
            drop_prob = np.random.uniform(0, random_drop)
            num_select = int((1-drop_prob)*centroids.shape[0])
            select_indices = np.random.choice(centroids.shape[0], size=num_select, replace=False)
            centroids = centroids[select_indices, :]
        centroids = np.int64(np.floor(centroids))

        # create the point map
        pointMask = np.zeros_like(self.others)
        pointMask[centroids[:, 1], centroids[:, 0]] = 1

        return pointMask

    @staticmethod
    def jitterClicks (shape, centroids, jitter_range=3):
        ''' Randomly jitter the centroid points
        Points should be an array in (x, y) format while shape is (H, W) of the point map
        '''
        centroids += np.random.uniform(low=-jitter_range, high=jitter_range,
                                            size=centroids.shape)
        centroids[:, 0] = np.clip(centroids[:, 0], 0, shape[1]-1)
        centroids[:, 1] = np.clip(centroids[:, 1], 0, shape[0]-1)
        return centroids


#Returns patchs, nucPoints, otherPoints
# m: height of img, n: width of img
def get_patches_and_signals(img, clickMap, boundingBoxes, cx, cy, m, n):
    # total = number of clicks
    total = len(boundingBoxes)
    img = np.array([img])   #img.shape=(1,3,m,n)
    clickMap = np.array([clickMap])     #clickmap.shape=(1,m,n)    
    clickMap = clickMap[:, np.newaxis, ...]    #clickmap.shape=(1,1,m,n)

    patchs = np.ndarray((total, 3, bb, bb), dtype=np.uint8)
    nucPoints = np.ndarray((total, 1, bb, bb), dtype=np.uint8)
    otherPoints = np.ndarray((total, 1, bb, bb), dtype=np.uint8)

    # Removing points out of image dimension (these points may have been clicked unwanted)
    x_del_indices = set([i for i in range(len(cx)) if cx[i]>=n or cx[i]<0])
    y_del_indices = set([i for i in range(len(cy)) if cy[i]>=m or cy[i]<0])
    del_indices = list(x_del_indices.union(y_del_indices))
    cx = np.delete(cx, del_indices)
    cy = np.delete(cy, del_indices)

    for i in range(len(boundingBoxes)):
        boundingBox = boundingBoxes[i]
        xStart = boundingBox[0]
        yStart = boundingBox[1]
        xEnd = boundingBox[2]
        yEnd = boundingBox[3]

        patchs[i] = img[0, :, yStart:yEnd + 1, xStart:xEnd + 1]

        thisClickMap = np.zeros((1, 1, m, n), dtype=np.uint8)
        thisClickMap[0, 0, cy[i], cx[i]] = 1

        othersClickMap = np.uint8((clickMap - thisClickMap) > 0)

        nucPoints[i] = thisClickMap[0, :, yStart:yEnd + 1, xStart:xEnd + 1]
        otherPoints[i] = othersClickMap[0, :, yStart:yEnd + 1, xStart:xEnd + 1]

    # patchs: (total, 3, m, n)
    # nucPoints: (total, 1, m, n)
    # otherPoints: (total, 1, m, n)
    return patchs, nucPoints, otherPoints   