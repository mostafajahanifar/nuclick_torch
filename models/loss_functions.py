from abc import ABC,abstractmethod 
import torch.nn as nn
from .dice_loss import dice_loss
from torch import Tensor


#Returns a Loss Function instance depending on the loss type
def Loss_Function_Factory(loss_type):

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