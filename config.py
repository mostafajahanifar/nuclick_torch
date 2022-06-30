from pathlib import Path

class Config:
    def __init__(self,
    seed=0,
    dir='/path/to/nuclick_patches/', #these patches should be generated using "patch_extraction_single.py"
    dir_val=None, # alternatively, you can set a path to validation patches
    dir_checkpoint='./checkpoints/', # path to save checkpoints during the training
    network='UNet', # type of network architecture
    epochs=100, # number of training epochs
    batch_size=64, # batch size for training
    lr=0.001, # learning rate for training
    model_path=None, # path to pretrained weights to resume training
    img_scale=1, # scale to resize image for training
    val_percent=20, # Percentage if validation patches randomly selected from training set
    use_amp=False, # if you want to use half-precision
    loss_type='Weighted_BCE_Dice',  # type of loss function used for training Options: {'Dice', 'BCE', 'Weighted_BCE', 'BCE_DICE', 'Weighted_BCE_Dice'}
    gpu='0', # ID of GPU on you machine to be used for training
    mask_thresh=0.5 # binarization threshold used for post-processing the raw predictions
    ) -> None:
        self.seed = seed
        self.dir = Path(dir)
        self.dir_val = Path(dir_val) if dir_val is not None else None
        self.dir_checkpoint = Path(dir_checkpoint)
        self.epochs = epochs
        self.batch_size = batch_size
        self.lr = lr
        self.model_path = model_path
        self.img_scale = img_scale
        self.val_percent = val_percent if dir_val is None else 0
        self.use_amp = use_amp
        self.loss_type = loss_type
        self.mask_thresh = mask_thresh
        self.patch_size = 128
        self.kernel_size = 3
        self.perturb = 'distance'
        self.drop_rate = 0.5
        self.jitter_range = 3
        self.network = network
        self.gpu = gpu


class TestConfig:
    def __init__(self,
    application = 'Nucleus',
    weights_path = "weights/NuClick_Nuclick_40xAll.pth",
    network = "NuClick",
    threshold = 0.5
    ) -> None:
        self.application = application
        self.network = network
        self.weights_path = weights_path,
        self.threshold = threshold

# Defining the configs
DemoConfig = TestConfig()
DefaultConfig = Config()
