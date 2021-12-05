import numpy as np
import cv2
from utils.mask_operations import adaptive_distance_thresholding

class GuidingSignal(object):
    '''A generic class for defining guiding signal generators.
    
    This class include some special methods that inclusion and exclusion guiding signals
    for different application can be created based on.
    '''
    def __init__(self, mask: np.ndarray, phase: str, others: np.ndarray = None, kernel_size: int = 3) -> None:
        self.mask = self.mask_validator(mask>0.5)
        self.others = others
        self.phase = phase
        self.kernel_size = kernel_size
        self.current_mask = self.mask_preprocess(self.mask, kernel_size=self.kernel_size)
    
    @staticmethod
    def mask_preprocess(mask, kernel_size=3):
        kernel = np.ones((kernel_size,kernel_size),np.uint8)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
        return mask

    @staticmethod
    def mask_validator(mask):
        '''Validate the input mask be np.uint8 and 2D'''
        assert len(mask.shape)==2, "Mask must be a 2D array (NxM)"
        if not issubclass(mask[0, 0], np.integer):
            mask = np.uint8(mask)
        return mask

    def inclusion_map(self):
        '''A function to generate inclusion gioding signal'''
        raise NotImplementedError

    def exclusion_map(self):
        '''A function to generate exclusion gioding signal'''
        raise NotImplementedError

class PointGuidingSignal(GuidingSignal):
    def __init__(self, mask: np.ndarray, phase: str, perturb: str = 'None') -> None:
        super().__init__(mask, phase)
        if perturb.lower not in {'none', 'distance', 'inside'}:
            raise ValueError(f'Invalid running perturb type of: {perturb}. Perturn type should be `"None"`, `"inside"`, or `"distance"`.')
        self.perturb = perturb.lower()

    def inclusion_map(self):
        if self.perturb is None: # if there is no purturbation
            indices = np.argwhere(self.current_mask==1) #
            centroid = np.mean(indices, axis=0)
            pointMask = np.zeros_like(self.current_mask)   
            pointMask[int(centroid[0]),int(centroid[1]),0] = 1
            return pointMask, self.current_mask
        elif self.perturb=='distance':
            self.current_mask, _ = adaptive_distance_thresholding(self.current_mask)
        
        indices = np.argwhere(self.current_mask==1) 
        rndIdx = np.random.randint(0, len(indices))
        rndX = indices[rndIdx, 1]
        rndY = indices[rndIdx, 0]
        pointMask = np.zeros_like(self.current_mask)   
        pointMask[rndY,rndX,0] = 1

        return pointMask

    def exclusion_map(self, random_drop=0, random_jitter=0):
        _, _, _, centroids = cv2.connectedComponentsWithStats(
            self.others, 4, cv2.CV_32S)

        centroids = centroids[1:, :] # removing the first centroid, it's background
        if random_jitter:
            centroids += np.random.uniform(low=-random_jitter, high=random_jitter,
                                            size=centroids.shape)
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

def jitterClicks (weightMap):
    pointPos = np.argwhere(weightMap[:,:,0]>0)
    if len(pointPos)>0:
        xPos = pointPos[0,1] + np.random.randint(-3,3)
        xPos = np.min([xPos,weightMap.shape[1]-1])
        xPos = np.max([xPos,0])
        yPos = pointPos[0,0] + np.random.randint(-3,3)
        yPos = np.min([yPos,weightMap.shape[0]-1])
        yPos = np.max([yPos,0])
        pointMask = np.zeros_like(weightMap)
        pointMask[yPos,xPos,0] = 1
        return pointMask