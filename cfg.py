from typing import NamedTuple

class DKwargs(NamedTuple):
    class_name: str = "networks.discriminator.ProjectedDiscriminator"

class GKwargs(NamedTuple):
    class_name: str = "networks.generator.Generator"
    z_dim: int = 64
    train_mode: str = "all"
    synthesis_kwargs: dict = {
        "channel_base": 32768,
        "channel_max": 512,
        "num_res_blocks": 2,
        "architecture": "skip"
    }
    img_resolution: int = 32


class Config(NamedTuple):
    D_kwargs: DKwargs = DKwargs()
    G_kwargs: GKwargs = GKwargs()
    G_opt_kwargs: dict = {
        "class_name": "torch.optim.Adam",
        "betas": [
            0,
            0.99
        ],
        "eps": 1e-08,
        "lr": 0.002
    }
    D_opt_kwargs: dict = {
        "class_name": "torch.optim.Adam",
        "betas": [
            0,
            0.99
        ],
        "eps": 1e-08,
        "lr": 0.002
    }

    loss_kwargs: dict = {
        "class_name": "training.loss.ProjectedGANLoss",
        "blur_init_sigma": 32,
        "blur_fade_kimg": 1000,
        "clip_weight": 0.0
    },
    data_loader_kwargs: dict = {
        "pin_memory": True,
        "num_workers": 3,
        "prefetch_factor": 2
    },
    training_set_kwargs: dict = {
        "path": "./data/cifar10-32x32.zip",
        "xflip": False,
        "use_labels": True,
        "class_name": "training.data_zip.ImageFolderDataset",
        "resolution": 32,
        "random_seed": 0
    },
    random_seed: int = 0
    image_snapshot_ticks: int = 100
    network_snapshot_ticks: int = 100
    metrics: list = ["fid50k_full"]
    total_kimg: int = 2500
    kimg_per_tick: int = 4
    batch_size: int = 16
    batch_gpu: int = 4
    ema_kimg: float = 5.0
    run_dir = "./output_dbg/00000-cifar10-32x32@32-lite-gpus1-b16-bgpu4"
