from pathlib import Path

class Config:
    def __init__(self,
    seed=0,
    dir_img='/root/workspace/nuclei_detection/dataset/compilation/Train/images/MoNuSeg/',
    dir_mask='/root/workspace/nuclei_detection/dataset/compilation/Train/masks/MoNuSeg/',
    dir_checkpoint='./checkpoints/',
    epochs=10,
    batch_size=16,
    lr=0.001,
    model_path=None,
    img_scale=1,
    val_percent=20,
    use_amp=False,
    mask_thresh=0.5) -> None:
        self.seed = seed
        self.dir_img = Path(dir_img)
        self.dir_mask = Path(dir_mask)
        self.dir_checkpoint = Path(dir_checkpoint)
        self.epochs = epochs
        self.batch_size = batch_size
        self.lr = lr
        self.model_path = model_path
        self.img_scale = img_scale
        self.val_percent = val_percent
        self.use_amp = use_amp
        self.mask_thresh = mask_thresh
        self.patch_size = 128
        self.kernel_size = 3
        self.perturb = 'distance'

DefaultConfig = Config()

        
