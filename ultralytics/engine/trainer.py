# Ultralytics YOLO 🚀, AGPL-3.0 license
"""
Train a model on a dataset.

Usage:
    $ yolo mode=train model=yolov8n.pt data=coco8.yaml imgsz=640 epochs=100 batch=16
"""

import csv
import gc
import math
import os
import subprocess
import time
import warnings
from copy import deepcopy
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
import torch
from torch import distributed as dist
from torch import nn, optim
from torchvision import transforms
from torchvision.models import vgg16
from ultralytics.cfg import get_cfg, get_save_dir
from ultralytics.data.utils import check_cls_dataset, check_det_dataset
from ultralytics.nn.tasks import attempt_load_one_weight, attempt_load_weights
from ultralytics.utils import (
    DEFAULT_CFG,
    LOGGER,
    RANK,
    TQDM,
    __version__,
    callbacks,
    clean_url,
    colorstr,
    emojis,
    yaml_save,
)
from ultralytics.utils.autobatch import check_train_batch_size
from ultralytics.utils.checks import (
    check_amp,
    check_file,
    check_imgsz,
    check_model_file_from_stem,
    print_args,
)
from ultralytics.utils.dist import ddp_cleanup, generate_ddp_command
from ultralytics.utils.files import get_latest_run
from ultralytics.utils.ops import xywh2xyxy
from ultralytics.utils.torch_utils import (
    EarlyStopping,
    ModelEMA,
    convert_optimizer_state_dict_to_fp16,
    init_seeds,
    one_cycle,
    select_device,
    strip_optimizer,
    torch_distributed_zero_first,
)

csv_file = "vgg_training_cumloss_log.csv"
if os.path.exists(csv_file):
    os.remove(csv_file)


class BaseTrainer:
    """
    BaseTrainer.

    A base class for creating trainers.

    Attributes:
        args (SimpleNamespace): Configuration for the trainer.
        validator (BaseValidator): Validator instance.
        model (nn.Module): Model instance.
        callbacks (defaultdict): Dictionary of callbacks.
        save_dir (Path): Directory to save results.
        wdir (Path): Directory to save weights.
        last (Path): Path to the last checkpoint.
        best (Path): Path to the best checkpoint.
        save_period (int): Save checkpoint every x epochs (disabled if < 1).
        batch_size (int): Batch size for training.
        epochs (int): Number of epochs to train for.
        start_epoch (int): Starting epoch for training.
        device (torch.device): Device to use for training.
        amp (bool): Flag to enable AMP (Automatic Mixed Precision).
        scaler (amp.GradScaler): Gradient scaler for AMP.
        data (str): Path to data.
        trainset (torch.utils.data.Dataset): Training dataset.
        testset (torch.utils.data.Dataset): Testing dataset.
        ema (nn.Module): EMA (Exponential Moving Average) of the model.
        resume (bool): Resume training from a checkpoint.
        lf (nn.Module): Loss function.
        scheduler (torch.optim.lr_scheduler._LRScheduler): Learning rate scheduler.
        best_fitness (float): The best fitness value achieved.
        fitness (float): Current fitness value.
        loss (float): Current loss value.
        tloss (float): Total loss value.
        loss_names (list): List of loss names.
        csv (Path): Path to results CSV file.
    """

    def __init__(self, cfg=DEFAULT_CFG, overrides=None, _callbacks=None):
        """
        Initializes the BaseTrainer class.

        Args:
            cfg (str, optional): Path to a configuration file. Defaults to DEFAULT_CFG.
            overrides (dict, optional): Configuration overrides. Defaults to None.
        """
        self.args = get_cfg(cfg, overrides)
        self.check_resume(overrides)
        self.device = select_device(self.args.device, self.args.batch)
        self.validator = None
        self.metrics = None
        self.plots = {}
        init_seeds(
            self.args.seed + 1 + RANK, deterministic=self.args.deterministic
        )

        # Dirs
        self.save_dir = get_save_dir(self.args)
        self.args.name = self.save_dir.name  # update name for loggers
        self.wdir = self.save_dir / "weights"  # weights dir
        if RANK in {-1, 0}:
            self.wdir.mkdir(parents=True, exist_ok=True)  # make dir
            self.args.save_dir = str(self.save_dir)
            yaml_save(
                self.save_dir / "args.yaml", vars(self.args)
            )  # save run args
        self.last, self.best = (
            self.wdir / "last.pt",
            self.wdir / "best.pt",
        )  # checkpoint paths
        self.save_period = self.args.save_period

        self.batch_size = self.args.batch
        self.epochs = self.args.epochs
        self.start_epoch = 0
        if RANK == -1:
            print_args(vars(self.args))

        # Device
        if self.device.type in {"cpu", "mps"}:
            self.args.workers = 0  # faster CPU training as time dominated by inference, not dataloading

        # Model and Dataset
        self.model = check_model_file_from_stem(
            self.args.model
        )  # add suffix, i.e. yolov8n -> yolov8n.pt
        with torch_distributed_zero_first(
            RANK
        ):  # avoid auto-downloading dataset multiple times
            self.trainset, self.testset = self.get_dataset()
        self.ema = None

        # Optimization utils init
        self.lf = None
        self.scheduler = None

        # Epoch level metrics
        self.best_fitness = None
        self.fitness = None
        self.loss = None
        self.tloss = None
        self.loss_names = ["Loss"]
        self.csv = self.save_dir / "results.csv"
        self.plot_idx = [0, 1, 2]

        # HUB
        self.hub_session = None

        # Callbacks
        self.callbacks = _callbacks or callbacks.get_default_callbacks()
        if RANK in {-1, 0}:
            callbacks.add_integration_callbacks(self)

    def add_callback(self, event: str, callback):
        """Appends the given callback."""
        self.callbacks[event].append(callback)

    def set_callback(self, event: str, callback):
        """Overrides the existing callbacks with the given callback."""
        self.callbacks[event] = [callback]

    def run_callbacks(self, event: str):
        """Run all existing callbacks associated with a particular event."""
        for callback in self.callbacks.get(event, []):
            callback(self)

    def train(self):
        """Allow device='', device=None on Multi-GPU systems to default to device=0."""
        if isinstance(self.args.device, str) and len(
            self.args.device
        ):  # i.e. device='0' or device='0,1,2,3'
            world_size = len(self.args.device.split(","))
        elif isinstance(
            self.args.device, (tuple, list)
        ):  # i.e. device=[0, 1, 2, 3] (multi-GPU from CLI is list)
            world_size = len(self.args.device)
        elif (
            torch.cuda.is_available()
        ):  # i.e. device=None or device='' or device=number
            world_size = 1  # default to device 0
        else:  # i.e. device='cpu' or 'mps'
            world_size = 0

        # Run subprocess if DDP training, else train normally
        if world_size > 1 and "LOCAL_RANK" not in os.environ:
            # Argument checks
            if self.args.rect:
                LOGGER.warning(
                    "WARNING ⚠️ 'rect=True' is incompatible with Multi-GPU training, setting 'rect=False'"
                )
                self.args.rect = False
            if self.args.batch < 1.0:
                LOGGER.warning(
                    "WARNING ⚠️ 'batch<1' for AutoBatch is incompatible with Multi-GPU training, setting "
                    "default 'batch=16'"
                )
                self.args.batch = 16

            # Command
            cmd, file = generate_ddp_command(world_size, self)
            try:
                LOGGER.info(
                    f'{colorstr("DDP:")} debug command {" ".join(cmd)}'
                )
                subprocess.run(cmd, check=True)
            except Exception as e:
                raise e
            finally:
                ddp_cleanup(self, str(file))

        else:
            self._do_train(world_size)

    def _setup_scheduler(self):
        """Initialize training learning rate scheduler."""
        if self.args.cos_lr:
            self.lf = one_cycle(
                1, self.args.lrf, self.epochs
            )  # cosine 1->hyp['lrf']
        else:
            self.lf = (
                lambda x: max(1 - x / self.epochs, 0) * (1.0 - self.args.lrf)
                + self.args.lrf
            )  # linear
        self.scheduler = optim.lr_scheduler.LambdaLR(
            self.optimizer, lr_lambda=self.lf
        )

    def _setup_ddp(self, world_size):
        """Initializes and sets the DistributedDataParallel parameters for training."""
        torch.cuda.set_device(RANK)
        self.device = torch.device("cuda", RANK)
        # LOGGER.info(f'DDP info: RANK {RANK}, WORLD_SIZE {world_size}, DEVICE {self.device}')
        os.environ["TORCH_NCCL_BLOCKING_WAIT"] = "1"  # set to enforce timeout
        dist.init_process_group(
            backend="nccl" if dist.is_nccl_available() else "gloo",
            timeout=timedelta(seconds=10800),  # 3 hours
            rank=RANK,
            world_size=world_size,
        )

    def _setup_train(self, world_size):
        """Builds dataloaders and optimizer on correct rank process."""

        # Model
        self.run_callbacks("on_pretrain_routine_start")
        ckpt = self.setup_model()
        self.model = self.model.to(self.device)
        self.set_model_attributes()

        # Freeze layers
        freeze_list = (
            self.args.freeze
            if isinstance(self.args.freeze, list)
            else (
                range(self.args.freeze)
                if isinstance(self.args.freeze, int)
                else []
            )
        )
        always_freeze_names = [".dfl"]  # always freeze these layers
        freeze_layer_names = [
            f"model.{x}." for x in freeze_list
        ] + always_freeze_names
        for k, v in self.model.named_parameters():
            # v.register_hook(lambda x: torch.nan_to_num(x))  # NaN to 0 (commented for erratic training results)
            if any(x in k for x in freeze_layer_names):
                LOGGER.info(f"Freezing layer '{k}'")
                v.requires_grad = False
            elif (
                not v.requires_grad and v.dtype.is_floating_point
            ):  # only floating point Tensor can require gradients
                LOGGER.info(
                    f"WARNING ⚠️ setting 'requires_grad=True' for frozen layer '{k}'. "
                    "See ultralytics.engine.trainer for customization of frozen layers."
                )
                v.requires_grad = True

        # Check AMP
        self.amp = torch.tensor(self.args.amp).to(self.device)  # True or False
        if self.amp and RANK in {-1, 0}:  # Single-GPU and DDP
            callbacks_backup = (
                callbacks.default_callbacks.copy()
            )  # backup callbacks as check_amp() resets them
            self.amp = torch.tensor(check_amp(self.model), device=self.device)
            callbacks.default_callbacks = callbacks_backup  # restore callbacks
        if RANK > -1 and world_size > 1:  # DDP
            dist.broadcast(
                self.amp, src=0
            )  # broadcast the tensor from rank 0 to all other ranks (returns None)
        self.amp = bool(self.amp)  # as boolean
        self.scaler = torch.cuda.amp.GradScaler(enabled=self.amp)
        if world_size > 1:
            self.model = nn.parallel.DistributedDataParallel(
                self.model, device_ids=[RANK]
            )

        # Check imgsz
        gs = max(
            int(
                self.model.stride.max()
                if hasattr(self.model, "stride")
                else 32
            ),
            32,
        )  # grid size (max stride)
        self.args.imgsz = check_imgsz(
            self.args.imgsz, stride=gs, floor=gs, max_dim=1
        )
        self.stride = gs  # for multiscale training

        # Batch size
        if (
            self.batch_size < 1 and RANK == -1
        ):  # single-GPU only, estimate best batch size
            self.args.batch = self.batch_size = check_train_batch_size(
                model=self.model,
                imgsz=self.args.imgsz,
                amp=self.amp,
                batch=self.batch_size,
            )

        # Dataloaders
        batch_size = self.batch_size // max(world_size, 1)
        self.train_loader = self.get_dataloader(
            self.trainset, batch_size=batch_size, rank=RANK, mode="train"
        )
        if RANK in {-1, 0}:
            # Note: When training DOTA dataset, double batch size could get OOM on images with >2000 objects.
            self.test_loader = self.get_dataloader(
                self.testset,
                batch_size=(
                    batch_size if self.args.task == "obb" else batch_size * 2
                ),
                rank=-1,
                mode="val",
            )
            self.validator = self.get_validator()
            metric_keys = self.validator.metrics.keys + self.label_loss_items(
                prefix="val"
            )
            self.metrics = dict(zip(metric_keys, [0] * len(metric_keys)))
            self.ema = ModelEMA(self.model)
            if self.args.plots:
                self.plot_training_labels()

        # Optimizer
        self.accumulate = max(
            round(self.args.nbs / self.batch_size), 1
        )  # accumulate loss before optimizing
        weight_decay = (
            self.args.weight_decay
            * self.batch_size
            * self.accumulate
            / self.args.nbs
        )  # scale weight_decay
        iterations = (
            math.ceil(
                len(self.train_loader.dataset)
                / max(self.batch_size, self.args.nbs)
            )
            * self.epochs
        )
        self.optimizer = self.build_optimizer(
            model=self.model,
            name=self.args.optimizer,
            lr=self.args.lr0,
            momentum=self.args.momentum,
            decay=weight_decay,
            iterations=iterations,
        )
        # Scheduler
        self._setup_scheduler()
        self.stopper, self.stop = (
            EarlyStopping(patience=self.args.patience),
            False,
        )
        self.resume_training(ckpt)
        self.scheduler.last_epoch = self.start_epoch - 1  # do not move
        self.run_callbacks("on_pretrain_routine_end")

    def _do_train(self, world_size=1):
        """Train completed, evaluate and plot if specified by arguments."""
        vgg, cls_optimizer = get_vgg16_model()

        if world_size > 1:
            self._setup_ddp(world_size)
        self._setup_train(world_size)

        nb = len(self.train_loader)  # number of batches
        nw = (
            max(round(self.args.warmup_epochs * nb), 100)
            if self.args.warmup_epochs > 0
            else -1
        )  # warmup iterations
        last_opt_step = -1
        self.epoch_time = None
        self.epoch_time_start = time.time()
        self.train_time_start = time.time()
        self.run_callbacks("on_train_start")
        LOGGER.info(
            f"Image sizes {self.args.imgsz} train, {self.args.imgsz} val\n"
            f"Using {self.train_loader.num_workers * (world_size or 1)} dataloader workers\n"
            f"Logging results to {colorstr('bold', self.save_dir)}\n"
            f"Starting training for "
            + (
                f"{self.args.time} hours..."
                if self.args.time
                else f"{self.epochs} epochs..."
            )
        )
        if self.args.close_mosaic:
            base_idx = (self.epochs - self.args.close_mosaic) * nb
            self.plot_idx.extend([base_idx, base_idx + 1, base_idx + 2])
        epoch = self.start_epoch
        self.optimizer.zero_grad()  # zero any resumed gradients to ensure stability on train start
        cls_optimizer.zero_grad()
        while True:
            self.epoch = epoch
            self.run_callbacks("on_train_epoch_start")
            with warnings.catch_warnings():
                warnings.simplefilter(
                    "ignore"
                )  # suppress 'Detected lr_scheduler.step() before optimizer.step()'
                self.scheduler.step()

            vgg.train()
            self.model.train()
            if RANK != -1:
                self.train_loader.sampler.set_epoch(epoch)
            pbar = enumerate(self.train_loader)
            # Update dataloader attributes (optional)
            if epoch == (self.epochs - self.args.close_mosaic):
                self._close_dataloader_mosaic()
                self.train_loader.reset()

            if RANK in {-1, 0}:
                LOGGER.info(self.progress_string())
                pbar = TQDM(enumerate(self.train_loader), total=nb)
            self.tloss = None
            for i, batch in pbar:
                self.run_callbacks("on_train_batch_start")
                # Warmup
                ni = i + nb * epoch
                if ni <= nw:
                    xi = [0, nw]  # x interp
                    self.accumulate = max(
                        1,
                        int(
                            np.interp(
                                ni, xi, [1, self.args.nbs / self.batch_size]
                            ).round()
                        ),
                    )
                    for j, x in enumerate(self.optimizer.param_groups):
                        # Bias lr falls from 0.1 to lr0, all other lrs rise from 0.0 to lr0
                        x["lr"] = np.interp(
                            ni,
                            xi,
                            [
                                self.args.warmup_bias_lr if j == 0 else 0.0,
                                x["initial_lr"] * self.lf(epoch),
                            ],
                        )
                        if "momentum" in x:
                            x["momentum"] = np.interp(
                                ni,
                                xi,
                                [
                                    self.args.warmup_momentum,
                                    self.args.momentum,
                                ],
                            )

                # Forward
                with torch.cuda.amp.autocast(self.amp):
                    batch = self.preprocess_batch(batch)
                    original_img_batch = (
                        torch.stack(
                            [
                                torch.tensor(
                                    cv2.cvtColor(
                                        cv2.imread(fp), cv2.COLOR_BGR2RGB
                                    )
                                )
                                .permute(2, 0, 1)
                                .float()
                                for fp in batch["im_file"]
                            ]
                        )
                        / 255.0
                    )
                    self.loss, self.loss_items = self.model(batch)
                    if RANK != -1:
                        self.loss *= world_size
                    self.tloss = (
                        (self.tloss * i + self.loss_items) / (i + 1)
                        if self.tloss is not None
                        else self.loss_items
                    )
                    preds = self.model.predict(batch["img"])
                    y = self.model.model[-1]._inference(preds)

                    from ultralytics.utils import ops

                    bboxes = ops.non_max_suppression(
                        y,
                        0.25,
                        0.7,
                        agnostic=False,
                        max_det=300,
                        classes=None,
                    )
                    del (preds, y)
                    if len(bboxes) == 0:
                        continue
                    else:
                        targets = torch.cat(
                            (
                                batch["batch_idx"].view(-1, 1),
                                batch["cls"].view(-1, 1),
                                batch["bboxes"],
                            ),
                            1,
                        )
                        targets = preprocessing(
                            device=self.device,
                            targets=targets,
                            batch_size=len(batch["img"]),
                            scale_tensor=torch.tensor(
                                [640.0, 640.0, 640.0, 640.0],
                                device=self.device,
                                dtype=torch.float16,
                            ),
                        )
                        cum_loss = torch.tensor(
                            0.0, device=self.device, requires_grad=True
                        )
                        batch_size = 16
                        num_images = len(batch["img"])

                        progress_bar = TQDM(
                            range(0, num_images, batch_size),
                            position=3,
                            leave=False,
                        )

                        normalize = transforms.Normalize(
                            mean=[0.485, 0.456, 0.406],
                            std=[0.229, 0.224, 0.225],
                        )

                        for i in progress_bar:
                            # Load a batch of images
                            batch_imgs = original_img_batch[
                                i : i + batch_size
                            ].to(self.device)
                            batch_targets = targets[i : i + batch_size]
                            batch_bboxes = bboxes[i : i + batch_size]

                            for img_idx, img in enumerate(batch_imgs):
                                filtered_preds = filter_pred_boxes(
                                    batch_targets[img_idx],
                                    batch_bboxes[img_idx],
                                    0.8,
                                )

                                if (
                                    filtered_preds is None
                                    or len(filtered_preds) == 0
                                ):
                                    classifier_loss = torch.tensor(
                                        0.0, device=self.device
                                    )
                                else:
                                    filtered_preds = upscale_boxes(
                                        (640, 640),
                                        filtered_preds,
                                        (640, 640),
                                        padding=False,
                                        device=self.device,
                                    )
                                    cropped_images, cropped_labels = (
                                        process_images_with_filtered_bboxes(
                                            img,
                                            filtered_preds,
                                            device=self.device,
                                        )
                                    )

                                    normalized_images = torch.stack(
                                        [
                                            normalize(img)
                                            for img in cropped_images
                                        ]
                                    ).to(self.device)

                                    # Forward pass through VGG-16
                                    classifier_outputs = vgg(normalized_images)

                                    classifier_loss = classification_loss(
                                        classifier_outputs,
                                        cropped_labels.to(torch.long),
                                    )
                                    cum_loss = torch.add(
                                        cum_loss, classifier_loss
                                    )
                                    del (
                                        filtered_preds,
                                        cropped_images,
                                        cropped_labels,
                                        normalized_images,
                                        classifier_outputs,
                                    )

                                progress_bar.set_postfix(
                                    loss=f"vgg_loss: {cum_loss / (i+1):.4f}"
                                )

                                # Free memory

                                torch.cuda.empty_cache()

                        # os.makedirs("pred_images", exist_ok=True)
                        # for img_idx, img in enumerate(batch["img"]):

                        #     batch_num = i
                        #     plot_image(
                        #         img,
                        #         bboxes[img_idx],
                        #         output_path=f"pred_images/b{batch_num}_i{img_idx}.jpg",
                        #     )
                    # Backward
                    with open(csv_file, mode="a", newline="") as file:
                        writer = csv.writer(file)
                        writer.writerow(
                            [epoch + 1, cum_loss.item() / (len(batch["img"]))]
                        )
                    self.scaler.scale(
                        self.loss + cum_loss / (len(batch["img"]))
                    ).backward()

                # Optimize - https://pytorch.org/docs/master/notes/amp_examples.html
                if ni - last_opt_step >= self.accumulate:
                    self.optimizer_step()
                    cls_optimizer.step()
                    last_opt_step = ni

                    # Timed stopping
                    if self.args.time:
                        self.stop = (time.time() - self.train_time_start) > (
                            self.args.time * 3600
                        )
                        if RANK != -1:  # if DDP training
                            broadcast_list = [self.stop if RANK == 0 else None]
                            dist.broadcast_object_list(
                                broadcast_list, 0
                            )  # broadcast 'stop' to all ranks
                            self.stop = broadcast_list[0]
                        if self.stop:  # training time exceeded
                            break

                # Log
                mem = f"{torch.cuda.memory_reserved() / 1E9 if torch.cuda.is_available() else 0:.3g}G"  # (GB)
                loss_len = self.tloss.shape[0] if len(self.tloss.shape) else 1
                losses = (
                    self.tloss
                    if loss_len > 1
                    else torch.unsqueeze(self.tloss, 0)
                )
                if RANK in {-1, 0}:
                    pbar.set_description(
                        ("%11s" * 2 + "%11.4g" * (2 + loss_len))
                        % (
                            f"{epoch + 1}/{self.epochs}",
                            mem,
                            *losses,
                            batch["cls"].shape[0],
                            batch["img"].shape[-1],
                        )
                    )
                    self.run_callbacks("on_batch_end")
                    if self.args.plots and ni in self.plot_idx:
                        self.plot_training_samples(batch, ni)

                self.run_callbacks("on_train_batch_end")

            self.lr = {
                f"lr/pg{ir}": x["lr"]
                for ir, x in enumerate(self.optimizer.param_groups)
            }  # for loggers
            self.run_callbacks("on_train_epoch_end")
            if RANK in {-1, 0}:
                final_epoch = epoch + 1 >= self.epochs
                self.ema.update_attr(
                    self.model,
                    include=[
                        "yaml",
                        "nc",
                        "args",
                        "names",
                        "stride",
                        "class_weights",
                    ],
                )

                # Validation
                if (
                    self.args.val
                    or final_epoch
                    or self.stopper.possible_stop
                    or self.stop
                ):
                    self.metrics, self.fitness = self.validate()
                self.save_metrics(
                    metrics={
                        **self.label_loss_items(self.tloss),
                        **self.metrics,
                        **self.lr,
                    }
                )
                self.stop |= (
                    self.stopper(epoch + 1, self.fitness) or final_epoch
                )
                if self.args.time:
                    self.stop |= (time.time() - self.train_time_start) > (
                        self.args.time * 3600
                    )

                # Save model
                if self.args.save or final_epoch:
                    self.save_model()
                    self.run_callbacks("on_model_save")
                    torch.save(vgg.state_dict(), "vgg_16_16.pth")
            # Scheduler
            t = time.time()
            self.epoch_time = t - self.epoch_time_start
            self.epoch_time_start = t
            if self.args.time:
                mean_epoch_time = (t - self.train_time_start) / (
                    epoch - self.start_epoch + 1
                )
                self.epochs = self.args.epochs = math.ceil(
                    self.args.time * 3600 / mean_epoch_time
                )
                self._setup_scheduler()
                self.scheduler.last_epoch = self.epoch  # do not move
                self.stop |= epoch >= self.epochs  # stop if exceeded epochs
            self.run_callbacks("on_fit_epoch_end")
            gc.collect()
            torch.cuda.empty_cache()  # clear GPU memory at end of epoch, may help reduce CUDA out of memory errors

            # Early Stopping
            if RANK != -1:  # if DDP training
                broadcast_list = [self.stop if RANK == 0 else None]
                dist.broadcast_object_list(
                    broadcast_list, 0
                )  # broadcast 'stop' to all ranks
                self.stop = broadcast_list[0]
            if self.stop:
                break  # must break all DDP ranks
            epoch += 1

        if RANK in {-1, 0}:
            # Do final val with best.pt
            LOGGER.info(
                f"\n{epoch - self.start_epoch + 1} epochs completed in "
                f"{(time.time() - self.train_time_start) / 3600:.3f} hours."
            )
            self.final_eval()
            if self.args.plots:
                self.plot_metrics()
            self.run_callbacks("on_train_end")
        gc.collect()
        torch.cuda.empty_cache()
        self.run_callbacks("teardown")

    def save_model(self):
        """Save model training checkpoints with additional metadata."""
        import io

        import pandas as pd  # scope for faster 'import ultralytics'

        # Serialize ckpt to a byte buffer once (faster than repeated torch.save() calls)
        buffer = io.BytesIO()
        torch.save(
            {
                "epoch": self.epoch,
                "best_fitness": self.best_fitness,
                "model": None,  # resume and final checkpoints derive from EMA
                "ema": deepcopy(self.ema.ema).half(),
                "updates": self.ema.updates,
                "optimizer": convert_optimizer_state_dict_to_fp16(
                    deepcopy(self.optimizer.state_dict())
                ),
                "train_args": vars(self.args),  # save as dict
                "train_metrics": {**self.metrics, **{"fitness": self.fitness}},
                "train_results": {
                    k.strip(): v
                    for k, v in pd.read_csv(self.csv)
                    .to_dict(orient="list")
                    .items()
                },
                "date": datetime.now().isoformat(),
                "version": __version__,
                "license": "AGPL-3.0 (https://ultralytics.com/license)",
                "docs": "https://docs.ultralytics.com",
            },
            buffer,
        )
        serialized_ckpt = (
            buffer.getvalue()
        )  # get the serialized content to save

        # Save checkpoints
        self.last.write_bytes(serialized_ckpt)  # save last.pt
        if self.best_fitness == self.fitness:
            self.best.write_bytes(serialized_ckpt)  # save best.pt
        if (
            (self.save_period > 0)
            and (self.epoch > 0)
            and (self.epoch % self.save_period == 0)
        ):
            (self.wdir / f"epoch{self.epoch}.pt").write_bytes(
                serialized_ckpt
            )  # save epoch, i.e. 'epoch3.pt'

    def get_dataset(self):
        """
        Get train, val path from data dict if it exists.

        Returns None if data format is not recognized.
        """
        try:
            if self.args.task == "classify":
                data = check_cls_dataset(self.args.data)
            elif self.args.data.split(".")[-1] in {
                "yaml",
                "yml",
            } or self.args.task in {
                "detect",
                "segment",
                "pose",
                "obb",
            }:
                data = check_det_dataset(self.args.data)
                if "yaml_file" in data:
                    self.args.data = data[
                        "yaml_file"
                    ]  # for validating 'yolo train data=url.zip' usage
        except Exception as e:
            raise RuntimeError(
                emojis(f"Dataset '{clean_url(self.args.data)}' error ❌ {e}")
            ) from e
        self.data = data
        return data["train"], data.get("val") or data.get("test")

    def setup_model(self):
        """Load/create/download model for any task."""
        if isinstance(
            self.model, torch.nn.Module
        ):  # if model is loaded beforehand. No setup needed
            return

        cfg, weights = self.model, None
        ckpt = None
        if str(self.model).endswith(".pt"):
            weights, ckpt = attempt_load_one_weight(self.model)
            cfg = weights.yaml
        elif isinstance(self.args.pretrained, (str, Path)):
            weights, _ = attempt_load_one_weight(self.args.pretrained)
        self.model = self.get_model(
            cfg=cfg, weights=weights, verbose=RANK == -1
        )  # calls Model(cfg, weights)
        return ckpt

    def optimizer_step(self):
        """Perform a single step of the training optimizer with gradient clipping and EMA update."""
        self.scaler.unscale_(self.optimizer)  # unscale gradients
        torch.nn.utils.clip_grad_norm_(
            self.model.parameters(), max_norm=10.0
        )  # clip gradients
        self.scaler.step(self.optimizer)
        self.scaler.update()
        self.optimizer.zero_grad()
        if self.ema:
            self.ema.update(self.model)

    def preprocess_batch(self, batch):
        """Allows custom preprocessing model inputs and ground truths depending on task type."""
        return batch

    def validate(self):
        """
        Runs validation on test set using self.validator.

        The returned dict is expected to contain "fitness" key.
        """
        metrics = self.validator(self)
        fitness = metrics.pop(
            "fitness", -self.loss.detach().cpu().numpy()
        )  # use loss as fitness measure if not found
        if not self.best_fitness or self.best_fitness < fitness:
            self.best_fitness = fitness
        return metrics, fitness

    def get_model(self, cfg=None, weights=None, verbose=True):
        """Get model and raise NotImplementedError for loading cfg files."""
        raise NotImplementedError(
            "This task trainer doesn't support loading cfg files"
        )

    def get_validator(self):
        """Returns a NotImplementedError when the get_validator function is called."""
        raise NotImplementedError(
            "get_validator function not implemented in trainer"
        )

    def get_dataloader(
        self, dataset_path, batch_size=16, rank=0, mode="train"
    ):
        """Returns dataloader derived from torch.data.Dataloader."""
        raise NotImplementedError(
            "get_dataloader function not implemented in trainer"
        )

    def build_dataset(self, img_path, mode="train", batch=None):
        """Build dataset."""
        raise NotImplementedError(
            "build_dataset function not implemented in trainer"
        )

    def label_loss_items(self, loss_items=None, prefix="train"):
        """
        Returns a loss dict with labelled training loss items tensor.

        Note:
            This is not needed for classification but necessary for segmentation & detection
        """
        return {"loss": loss_items} if loss_items is not None else ["loss"]

    def set_model_attributes(self):
        """To set or update model parameters before training."""
        self.model.names = self.data["names"]

    def build_targets(self, preds, targets):
        """Builds target tensors for training YOLO model."""
        pass

    def progress_string(self):
        """Returns a string describing training progress."""
        return ""

    # TODO: may need to put these following functions into callback
    def plot_training_samples(self, batch, ni):
        """Plots training samples during YOLO training."""
        pass

    def plot_training_labels(self):
        """Plots training labels for YOLO model."""
        pass

    def save_metrics(self, metrics):
        """Saves training metrics to a CSV file."""
        keys, vals = list(metrics.keys()), list(metrics.values())
        n = len(metrics) + 1  # number of cols
        s = (
            ""
            if self.csv.exists()
            else (("%23s," * n % tuple(["epoch"] + keys)).rstrip(",") + "\n")
        )  # header
        with open(self.csv, "a") as f:
            f.write(
                s
                + ("%23.5g," * n % tuple([self.epoch + 1] + vals)).rstrip(",")
                + "\n"
            )

    def plot_metrics(self):
        """Plot and display metrics visually."""
        pass

    def on_plot(self, name, data=None):
        """Registers plots (e.g. to be consumed in callbacks)"""
        path = Path(name)
        self.plots[path] = {"data": data, "timestamp": time.time()}

    def final_eval(self):
        """Performs final evaluation and validation for object detection YOLO model."""
        for f in self.last, self.best:
            if f.exists():
                strip_optimizer(f)  # strip optimizers
                if f is self.best:
                    LOGGER.info(f"\nValidating {f}...")
                    self.validator.args.plots = self.args.plots
                    self.metrics = self.validator(model=f)
                    self.metrics.pop("fitness", None)
                    self.run_callbacks("on_fit_epoch_end")

    def check_resume(self, overrides):
        """Check if resume checkpoint exists and update arguments accordingly."""
        resume = self.args.resume
        if resume:
            try:
                exists = (
                    isinstance(resume, (str, Path)) and Path(resume).exists()
                )
                last = Path(check_file(resume) if exists else get_latest_run())

                # Check that resume data YAML exists, otherwise strip to force re-download of dataset
                ckpt_args = attempt_load_weights(last).args
                if not Path(ckpt_args["data"]).exists():
                    ckpt_args["data"] = self.args.data

                resume = True
                self.args = get_cfg(ckpt_args)
                self.args.model = self.args.resume = str(
                    last
                )  # reinstate model
                for k in (
                    "imgsz",
                    "batch",
                    "device",
                ):  # allow arg updates to reduce memory or update device on resume
                    if k in overrides:
                        setattr(self.args, k, overrides[k])

            except Exception as e:
                raise FileNotFoundError(
                    "Resume checkpoint not found. Please pass a valid checkpoint to resume from, "
                    "i.e. 'yolo train resume model=path/to/last.pt'"
                ) from e
        self.resume = resume

    def resume_training(self, ckpt):
        """Resume YOLO training from given epoch and best fitness."""
        if ckpt is None or not self.resume:
            return
        best_fitness = 0.0
        start_epoch = ckpt.get("epoch", -1) + 1
        if ckpt.get("optimizer", None) is not None:
            self.optimizer.load_state_dict(ckpt["optimizer"])  # optimizer
            best_fitness = ckpt["best_fitness"]
        if self.ema and ckpt.get("ema"):
            self.ema.ema.load_state_dict(
                ckpt["ema"].float().state_dict()
            )  # EMA
            self.ema.updates = ckpt["updates"]
        assert start_epoch > 0, (
            f"{self.args.model} training to {self.epochs} epochs is finished, nothing to resume.\n"
            f"Start a new training without resuming, i.e. 'yolo train model={self.args.model}'"
        )
        LOGGER.info(
            f"Resuming training {self.args.model} from epoch {start_epoch + 1} to {self.epochs} total epochs"
        )
        if self.epochs < start_epoch:
            LOGGER.info(
                f"{self.model} has been trained for {ckpt['epoch']} epochs. Fine-tuning for {self.epochs} more epochs."
            )
            self.epochs += ckpt["epoch"]  # finetune additional epochs
        self.best_fitness = best_fitness
        self.start_epoch = start_epoch
        if start_epoch > (self.epochs - self.args.close_mosaic):
            self._close_dataloader_mosaic()

    def _close_dataloader_mosaic(self):
        """Update dataloaders to stop using mosaic augmentation."""
        if hasattr(self.train_loader.dataset, "mosaic"):
            self.train_loader.dataset.mosaic = False
        if hasattr(self.train_loader.dataset, "close_mosaic"):
            LOGGER.info("Closing dataloader mosaic")
            self.train_loader.dataset.close_mosaic(hyp=self.args)

    def build_optimizer(
        self,
        model,
        name="auto",
        lr=0.001,
        momentum=0.9,
        decay=1e-5,
        iterations=1e5,
    ):
        """
        Constructs an optimizer for the given model, based on the specified optimizer name, learning rate, momentum,
        weight decay, and number of iterations.

        Args:
            model (torch.nn.Module): The model for which to build an optimizer.
            name (str, optional): The name of the optimizer to use. If 'auto', the optimizer is selected
                based on the number of iterations. Default: 'auto'.
            lr (float, optional): The learning rate for the optimizer. Default: 0.001.
            momentum (float, optional): The momentum factor for the optimizer. Default: 0.9.
            decay (float, optional): The weight decay for the optimizer. Default: 1e-5.
            iterations (float, optional): The number of iterations, which determines the optimizer if
                name is 'auto'. Default: 1e5.

        Returns:
            (torch.optim.Optimizer): The constructed optimizer.
        """

        g = [], [], []  # optimizer parameter groups
        bn = tuple(
            v for k, v in nn.__dict__.items() if "Norm" in k
        )  # normalization layers, i.e. BatchNorm2d()
        if name == "auto":
            LOGGER.info(
                f"{colorstr('optimizer:')} 'optimizer=auto' found, "
                f"ignoring 'lr0={self.args.lr0}' and 'momentum={self.args.momentum}' and "
                f"determining best 'optimizer', 'lr0' and 'momentum' automatically... "
            )
            nc = getattr(model, "nc", 10)  # number of classes
            lr_fit = round(
                0.002 * 5 / (4 + nc), 6
            )  # lr0 fit equation to 6 decimal places
            name, lr, momentum = (
                ("SGD", 0.01, 0.9)
                if iterations > 10000
                else ("AdamW", lr_fit, 0.9)
            )
            self.args.warmup_bias_lr = 0.0  # no higher than 0.01 for Adam

        for module_name, module in model.named_modules():
            for param_name, param in module.named_parameters(recurse=False):
                fullname = (
                    f"{module_name}.{param_name}"
                    if module_name
                    else param_name
                )
                if "bias" in fullname:  # bias (no decay)
                    g[2].append(param)
                elif isinstance(module, bn):  # weight (no decay)
                    g[1].append(param)
                else:  # weight (with decay)
                    g[0].append(param)

        if name in {"Adam", "Adamax", "AdamW", "NAdam", "RAdam"}:
            optimizer = getattr(optim, name, optim.Adam)(
                g[2], lr=lr, betas=(momentum, 0.999), weight_decay=0.0
            )
        elif name == "RMSProp":
            optimizer = optim.RMSprop(g[2], lr=lr, momentum=momentum)
        elif name == "SGD":
            optimizer = optim.SGD(
                g[2], lr=lr, momentum=momentum, nesterov=True
            )
        else:
            raise NotImplementedError(
                f"Optimizer '{name}' not found in list of available optimizers "
                f"[Adam, AdamW, NAdam, RAdam, RMSProp, SGD, auto]."
                "To request support for addition optimizers please visit https://github.com/ultralytics/ultralytics."
            )

        optimizer.add_param_group(
            {"params": g[0], "weight_decay": decay}
        )  # add g0 with weight_decay
        optimizer.add_param_group(
            {"params": g[1], "weight_decay": 0.0}
        )  # add g1 (BatchNorm2d weights)
        LOGGER.info(
            f"{colorstr('optimizer:')} {type(optimizer).__name__}(lr={lr}, momentum={momentum}) with parameter groups "
            f"{len(g[1])} weight(decay=0.0), {len(g[0])} weight(decay={decay}), {len(g[2])} bias(decay=0.0)"
        )
        return optimizer


import numpy as np
import torch
import cv2


def plot_image(image, boxes, output_path):
    """Plots predicted bounding boxes on the image and saves the output to a file."""
    # Move image to CPU and convert to NumPy array
    image = image.cpu()
    image = image.permute(1, 2, 0).numpy()

    # Denormalize the image
    image = (image * 255).astype(np.uint8)

    # Convert image from RGB to BGR for OpenCV
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

    # Move boxes to CPU and convert to NumPy array
    boxes = boxes.detach().cpu().numpy()

    height, width, _ = image.shape

    # Iterate over each box
    for box in boxes:
        x1, y1, x2, y2, confidence, cls = box
        # Draw the rectangle
        cv2.rectangle(
            image, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 1
        )
    # Save the image to a file
    cv2.imwrite(output_path, image)


def preprocessing(
    device,
    targets,
    batch_size,
    scale_tensor,
):
    """Preprocesses the target counts and matches with the input batch size to output a tensor."""
    nl, ne = targets.shape
    if nl == 0:
        out = torch.zeros(batch_size, 0, ne - 1, device=device)
    else:
        i = targets[:, 0]  # image index
        _, counts = i.unique(return_counts=True)
        counts = counts.to(dtype=torch.int32)
        out = torch.zeros(batch_size, counts.max(), ne - 1, device=device)
        for j in range(batch_size):
            matches = i == j
            n = matches.sum()
            if n:
                out[j, :n] = targets[matches, 1:]
        out[..., 1:5] = xywh2xyxy(out[..., 1:5].mul_(scale_tensor))
    return out


def calculate_tensor_iou(gt_boxes, pred_boxes):
    # Compute intersection coordinates
    x1 = torch.max(gt_boxes[:, 1:2], pred_boxes[:, 0:1].T)
    y1 = torch.max(gt_boxes[:, 2:3], pred_boxes[:, 1:2].T)
    x2 = torch.min(gt_boxes[:, 3:4], pred_boxes[:, 2:3].T)
    y2 = torch.min(gt_boxes[:, 4:5], pred_boxes[:, 3:4].T)

    # Calculate intersection area
    intersection = torch.clamp(x2 - x1, min=0) * torch.clamp(y2 - y1, min=0)

    # Calculate area of both boxes
    area_gt = (gt_boxes[:, 3] - gt_boxes[:, 1]) * (
        gt_boxes[:, 4] - gt_boxes[:, 2]
    )
    area_pred = (pred_boxes[:, 2] - pred_boxes[:, 0]) * (
        pred_boxes[:, 3] - pred_boxes[:, 1]
    )

    # Calculate union area
    union = area_gt[:, None] + area_pred - intersection

    # Calculate IoU
    iou = intersection / torch.clamp(union, min=1e-6)  # Avoid division by zero

    return iou


def filter_pred_boxes(gt_boxes, pred_boxes, threshold=0.8):
    iou = calculate_tensor_iou(gt_boxes, pred_boxes)

    # Find indices where IoU >= threshold
    masky = iou >= threshold

    # List to store filtered bounding boxes
    filtered_preds = []

    for gt_idx in range(gt_boxes.shape[0]):
        if masky[gt_idx].any():
            # Get the predicted boxes that meet the IoU threshold
            valid_pred_indices = torch.nonzero(masky[gt_idx], as_tuple=True)[0]
            for pred_idx in valid_pred_indices:
                pred_box = pred_boxes[pred_idx]
                # Replace the class label in pred_box with the class label from gt_box
                filtered_pred = torch.cat(
                    (gt_boxes[gt_idx, :1], pred_box[:5]), dim=0
                )
                filtered_preds.append(filtered_pred)

    if len(filtered_preds) > 0:
        # Concatenate all filtered predictions into a single tensor
        filtered_preds = torch.stack(filtered_preds)
    else:
        # Create an empty tensor on the same device as pred_boxes
        filtered_preds = torch.empty((0, 6), device=pred_boxes.device)

    return filtered_preds


import torch
import torch.nn.functional as F


def process_images_with_filtered_bboxes(
    image_tensor, filtered_preds, output_size=(224, 224), device=None
):
    """
    Processes images with bounding boxes: crops the regions and applies padding.

    Args:
    - image_tensor (torch.Tensor): The input image tensor of shape (3, H, W).
    - filtered_preds (torch.Tensor): A tensor of filtered predictions with shape (N, 6),
                                     where each row is (class, x1, y1, x2, y2, conf).
    - output_size (tuple): The desired output size of the cropped images (default is (224, 224)).
    - device (str): The device to perform operations on ('cuda' or 'cpu').

    Returns:
    - tuple: A tuple containing:
        - list: A list of torch.Tensor objects (cropped and padded images).
        - torch.Tensor: A tensor of class labels corresponding to each cropped image.
    """
    # Ensure the input tensor is of type float
    image_tensor = image_tensor.float()

    _, height, width = image_tensor.shape
    cropped_images = []
    cropped_labels = []

    for prediction in filtered_preds:
        class_label, x1, y1, x2, y2, conf = prediction

        if x1 >= 0 and y1 >= 0 and x2 > x1 and y2 > y1:
            # Calculate bounding box coordinates in pixel values
            left = int(x1)
            top = int(y1)
            right = int(x2)
            bottom = int(y2)

            # Ensure coordinates are within image bounds
            left = max(0, left)
            top = max(0, top)
            right = min(width, right)
            bottom = min(height, bottom)

            # Crop the image
            cropped_img = image_tensor[:, top:bottom, left:right]

            if cropped_img.numel() == 0:
                # Skip if the cropped image is empty
                continue

            # Calculate padding to make the cropped image the desired output size
            cropped_height, cropped_width = cropped_img.shape[1:]
            pad_height = max(output_size[1] - cropped_height, 0)
            pad_width = max(output_size[0] - cropped_width, 0)

            # Apply symmetric padding
            padded_img = F.pad(
                cropped_img,
                (
                    pad_width // 2,
                    pad_width - pad_width // 2,  # Left, Right
                    pad_height // 2,
                    pad_height - pad_height // 2,  # Top, Bottom
                ),
                mode="constant",
                value=0,  # Default padding value is 0 (black)
            )

            # Resize if necessary to ensure final dimensions match output_size
            if padded_img.shape[1:] != output_size:
                padded_img = F.interpolate(
                    padded_img.unsqueeze(0),
                    size=output_size,
                    mode="bilinear",
                    align_corners=False,
                ).squeeze(0)

            cropped_images.append(padded_img)
            cropped_labels.append(class_label.item())

    if cropped_labels:
        # Convert cropped_labels to a tensor
        cropped_labels_tensor = torch.tensor(cropped_labels, device=device)
        return cropped_images, cropped_labels_tensor
    else:
        return [], torch.tensor([], device=device)


def get_vgg16_model():
    """Initializes and returns a VGG-16 model."""
    vgg_model = vgg16(pretrained=True)  # Load pre-trained VGG-16 model
    # Freeze all parameters in the convolutional layers
    for param in vgg_model.parameters():
        param.requires_grad = False

    # Modify the classifier
    num_features = vgg_model.classifier[6].in_features
    classifier = nn.Sequential(
        nn.Linear(num_features, 1024),
        nn.ReLU(inplace=True),
        nn.Dropout(p=0.5),
        nn.Linear(1024, 1024),
        nn.ReLU(inplace=True),
        nn.Dropout(p=0.5),
        nn.Linear(1024, 2),  # Adjust based on your number of classes
    )
    vgg_model.classifier[6] = classifier

    # Move the model to the correct device
    vgg_model = vgg_model.to("cuda")
    cls_optimizer = optim.Adam(
        [
            {"params": vgg_model.classifier.parameters()},
        ],  # parameters of VGG-16 classifier
        lr=2e-6,
        weight_decay=0,
    )
    return vgg_model, cls_optimizer


def classification_loss(outputs, targets):
    """Defines a custom loss function for binary classification."""
    criterion = nn.CrossEntropyLoss()
    loss = criterion(outputs, targets)
    return loss


import torch


def upscale_boxes(
    img1_shape, preds, img0_shape, ratio_pad=None, padding=True, device=None
):
    """
    Rescales bounding boxes from the shape of the image they were originally specified in (img1_shape)
    to the shape of a different image (img0_shape).

    Args:
        img1_shape (tuple): The shape of the image that the bounding boxes are for, (height, width).
        preds (torch.Tensor): The filtered predictions, each row is [class, x1, y1, x2, y2, conf].
        img0_shape (tuple): The shape of the target image, (height, width).
        ratio_pad (tuple): Optional tuple of (ratio, pad) for scaling the boxes.
        padding (bool): If True, assumes the boxes are based on a YOLO-style augmented image. If False, regular rescaling.

    Returns:
        preds (torch.Tensor): The scaled predictions, with updated bounding boxes.
    """
    device = preds.device  # Get the device of the input tensor

    if preds.dim() == 1:  # Ensure preds is at least 2D
        preds = preds.unsqueeze(0)

    new_list = []
    for pred in preds:
        # Extract bounding box and class information
        cls, x1, y1, x2, y2, conf = pred

        # Convert bounding box coordinates to a tensor with requires_grad=True
        box = torch.tensor(
            [x1, y1, x2, y2], dtype=torch.float, requires_grad=True
        ).to(device)

        if ratio_pad is None:
            # Compute scaling ratio and padding based on image dimensions
            scale_ratio = min(
                img1_shape[0] / img0_shape[0], img1_shape[1] / img0_shape[1]
            )
            padding_x = (img1_shape[1] - img0_shape[1] * scale_ratio) / 2
            padding_y = (img1_shape[0] - img0_shape[0] * scale_ratio) / 2
        else:
            scale_ratio = ratio_pad[0][0]
            padding_x, padding_y = ratio_pad[1]

        if padding:
            # Adjust coordinates based on padding
            box = box - torch.tensor(
                [padding_x, padding_y, padding_x, padding_y],
                dtype=torch.float,
                device=device,
            )

        # Scale boxes to new dimensions
        box = box / scale_ratio

        # Clip boxes to be within the new image dimensions
        box = clip_boxes(box.unsqueeze(0), img0_shape).squeeze(0)

        # Append the updated box and other prediction details
        new_list.append(
            torch.cat(
                [
                    cls.unsqueeze(0).to(device),
                    box,
                    conf.unsqueeze(0).to(device),
                ],
                dim=0,
            )
        )

    # Convert the list of new predictions to a tensor
    new_preds = torch.stack(new_list).to(device)

    return new_preds


def clip_boxes(boxes, shape):
    """
    Takes a list of bounding boxes and a shape (height, width) and clips the bounding boxes to the shape.

    Args:
        boxes (torch.Tensor): the bounding boxes to clip
        shape (tuple): the shape of the image

    Returns:
        (torch.Tensor): Clipped boxes
    """
    boxes[..., 0] = boxes[..., 0].clamp(0, shape[1])  # x1
    boxes[..., 1] = boxes[..., 1].clamp(0, shape[0])  # y1
    boxes[..., 2] = boxes[..., 2].clamp(0, shape[1])  # x2
    boxes[..., 3] = boxes[..., 3].clamp(0, shape[0])  # y2
    return boxes


def clip_boxes(boxes, img_shape):
    """
    Clips bounding boxes to ensure they are within the image boundaries.

    Args:
        boxes (torch.Tensor): The bounding boxes to clip.
        img_shape (tuple): The shape of the image, (height, width).

    Returns:
        (torch.Tensor): The clipped bounding boxes.
    """
    height, width = img_shape

    # Ensure boxes are within image boundaries
    if isinstance(boxes, torch.Tensor):
        boxes[..., 0] = boxes[..., 0].clamp(0, width)  # x1
        boxes[..., 1] = boxes[..., 1].clamp(0, height)  # y1
        boxes[..., 2] = boxes[..., 2].clamp(0, width)  # x2
        boxes[..., 3] = boxes[..., 3].clamp(0, height)  # y2
    else:
        boxes[..., [0, 2]] = boxes[..., [0, 2]].clip(0, width)  # x1, x2
        boxes[..., [1, 3]] = boxes[..., [1, 3]].clip(0, height)  # y1, y2

    return boxes
