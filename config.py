from pathlib import Path

class Config:
    def __init__(self,
    seed=0,
    dir='/root/workspace/nuclei_instances_datasets/NuClick/MoNuSegTrain/',
    dir_val='/root/workspace/nuclei_instances_datasets/NuClick/Validation/mat_files/',
    dir_checkpoint='./checkpoints/',
    epochs=10,
    batch_size=32,
    lr=0.001,
    model_path=None,
    img_scale=1,
    val_percent=0,
    use_amp=False,
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
        self.mask_thresh = mask_thresh
        self.patch_size = 128
        self.kernel_size = 3
        self.perturb = 'distance'
        self.drop_rate = 0.5
        self.jitter_range = 3

DefaultConfig = Config()

