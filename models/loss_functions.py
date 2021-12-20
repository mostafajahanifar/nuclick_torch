from abc import ABC,abstractmethod 
import torch.nn as nn
from torch import Tensor
import torch


#Returns a Loss Function instance depending on the loss type
def get_loss_function(loss_type):

    loss_functions = {
        "Dice": Dice_Loss,
        "BCE": BCE_Loss,
        "Weighted_BCE": Weighted_BCE_Loss,
        "BCE_Dice": BCE_Dice_Loss,
        "Weighted_BCE_Dice": Weighted_BCE_Dice_Loss,
        #To add a new loss function, first create a subclass of Loss_Function
        #Then add a new entry here:
        # "<loss_type>": <class name>
    }

    if loss_type in loss_functions:
        return loss_functions[loss_type]()
    else:
        raise ValueError(f"Undefined loss function: {loss_type}")


#Abstract class for loss functions
#All loss functions need to a subclass of this class
class Loss_Function(ABC):

    @abstractmethod
    def name(self):
        pass

    @abstractmethod
    def use_weight(self):
        pass

    @abstractmethod
    def compute_loss(self):
        pass


#-------------------------------------Classes implementing loss functions---------------------------------

#Dice loss
class Dice_Loss(Loss_Function):

    def name(self):
        return 'Dice Loss'

    def use_weight(self):
        return False

    def compute_loss(self, input: Tensor, target: Tensor):
        return dice_loss(input.float(), target.float(), multiclass=False)


#Binary cross entropy loss
class BCE_Loss(Loss_Function):

    def name(self):
        return 'BCE Loss'

    def use_weight(self):
        return False

    def compute_loss(self, input: Tensor, target: Tensor):
        return nn.BCELoss()(input, target.float())


#Weighted binary cross entropy loss
class Weighted_BCE_Loss(Loss_Function):

    def name(self):
        return 'Weighted BCE Loss'

    def use_weight(self):
        return True

    def compute_loss(self, input: Tensor, target: Tensor, weight: Tensor):
        return nn.BCELoss(weight=weight)(input, target.float())

    
#Binary cross entropy + Dice loss
class BCE_Dice_Loss(Loss_Function):

    def name(self):
        return 'BCE + Dice Loss'

    def use_weight(self):
        return False

    def compute_loss(self, input: Tensor, target: Tensor):
        return nn.BCELoss()(input, target.float()) \
                           + dice_loss(input.float(),
                                       target.float(),
                                       multiclass=False)

    
#Weighted binary cross entropy + Dice loss
class Weighted_BCE_Dice_Loss(Loss_Function):

    def name(self):
        return 'Weighted BCE Loss + Dice Loss'

    def use_weight(self):
        return True

    def compute_loss(self, input: Tensor, target: Tensor, weight: Tensor):
        return nn.BCELoss(weight=weight)(input, target.float()) \
                           + dice_loss(input.float(),
                                       target.float(),
                                       multiclass=False)


#------------------------------------------Dice loss functions--------------------------------------
def dice_coeff(input: Tensor, target: Tensor, reduce_batch_first: bool = False, epsilon=1e-6):
    # Average of Dice coefficient for all batches, or for a single mask
    assert input.size() == target.size()
    if input.dim() == 2 and reduce_batch_first:
        raise ValueError(f'Dice: asked to reduce batch but got tensor without batch dimension (shape {input.shape})')

    if input.dim() == 2 or reduce_batch_first:
        inter = torch.dot(input.reshape(-1), target.reshape(-1))
        sets_sum = torch.sum(input) + torch.sum(target)
        if sets_sum.item() == 0:
            sets_sum = 2 * inter

        return (2 * inter + epsilon) / (sets_sum + epsilon)
    else:
        # compute and average metric for each batch element
        dice = 0
        for i in range(input.shape[0]):
            dice += dice_coeff(input[i, ...], target[i, ...])
        return dice / input.shape[0]


def multiclass_dice_coeff(input: Tensor, target: Tensor, reduce_batch_first: bool = False, epsilon=1e-6):
    # Average of Dice coefficient for all classes
    assert input.size() == target.size()
    dice = 0
    for channel in range(input.shape[1]):
        dice += dice_coeff(input[:, channel, ...], target[:, channel, ...], reduce_batch_first, epsilon)

    return dice / input.shape[1]


def dice_loss(input: Tensor, target: Tensor, multiclass: bool = False):
    # Dice loss (objective to minimize) between 0 and 1
    assert input.size() == target.size()
    fn = multiclass_dice_coeff if multiclass else dice_coeff
    return 1 - fn(input, target, reduce_batch_first=True)