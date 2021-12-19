from pathlib import Path

class Config:
    def __init__(self,
    seed=0,
    dir='/root/workspace/nuclei_instances_datasets/NuClick/MoNuSegTrain/',
    dir_val=None, # '/root/workspace/nuclei_instances_datasets/NuClick/Validation/mat_files/',
    dir_checkpoint='./checkpoints/',
    network='UNet',
    epochs=100,
    batch_size=16,
    lr=0.001,
    model_path=None,
    img_scale=1,
    val_percent=20,
    use_amp=False,
    loss_type='Weighted_BCE_Dice',   #Options: {'Dice', 'BCE', 'Weighted_BCE', 'BCE_DICE', 'Weighted_BCE_Dice'}
    mask_thresh=0.5) -> None:
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

DefaultConfig = Config()

