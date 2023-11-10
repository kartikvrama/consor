"""Script to train ConSOR."""

import os
from datetime import datetime
from pathlib import Path

import yaml
from absl import app
from absl import flags

import torch
import numpy as np
import wandb

from torch.utils.data import DataLoader

import pytorch_lightning as pl
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.callbacks import ModelCheckpoint

from consor_core.model import ConSORTransformer
from consor_core.data_loader import ConSORDataLoader

flags.DEFINE_string(
    "config_file_path",
    "./configs/consor_config.yaml",
    "Path to the config file for training ConSOR.",
)

FLAGS = flags.FLAGS


def main(argv):
    if len(argv) > 1:
        raise app.UsageError("Too many command-line arguments.")

    datetime_obj = datetime.now()
    date_time_stamp = datetime_obj.strftime("%Y_%m_%d_%H_%M_%S")

    # Load config.
    with open(FLAGS.config_file_path, "r") as fconfig:
        config = yaml.safe_load(fconfig)

    # Seed
    np.random.seed(int(config["SYSTEM"]["seed"]))
    torch.random.manual_seed(int(config["SYSTEM"]["seed"]))

    # Create wandb session.
    wandb_logger = WandbLogger(
        name=f"log_{date_time_stamp}", project="ConSOR_IROS_2023"
    )

    rules_list = config["rules"].split(",")

    # Dictionary of training parameters.
    train_params = {
        "num_epochs": int(config["TRAIN"]["num_epochs"]),
        "batch_size": int(config["TRAIN"]["batch_size"]),
        "lrate": float(config["TRAIN"]["lrate"]),
        "wt_decay": float(config["TRAIN"]["wt_decay"]),
        "hidden_layer_size": int(config["MODEL"]["hidden_layer_size"]),
        "output_dimension": int(config["MODEL"]["output_dimension"]),
        "num_heads": int(config["MODEL"]["num_heads"]),
        "num_layers": int(config["MODEL"]["num_layers"]),
        "rules": rules_list,
        "dropout": float(config["MODEL"]["dropout"]),
        "loss_fn": str(config["LOSS"]["loss_fn"]),
    }

    if "triplet_margin" == str(config["LOSS"]["loss_fn"]):
        train_params["triplet_loss_margin"] = float(config["LOSS"]["triplet_margin"])
    else:
        train_params["triplet_loss_margin"] = None

    # Train and validation data filepaths.
    train_filepath = config["DATA"]["train_tensors_path"]
    print(f"Train data path: {train_filepath}")
    val_filepath = config["DATA"]["val_tensors_path"]
    print(f"Val data path: {val_filepath}")

    # Initialize train data loader.
    dataset_train = ConSORDataLoader(is_loading=True, input_file_path=train_filepath)
    dataloader_train = DataLoader(
        dataset_train,
        num_workers=16,
        collate_fn=dataset_train.collate_tensor_batch,
        batch_size=train_params["batch_size"],
        shuffle=True,
    )

    print("Dataloaders are ready!")

    # Update feature dimensionality.
    train_params["node_feature_len"] = dataset_train.get_feature_length()

    print("--Train params--")
    for key, value in train_params.items():
        print(f"{key}: {value}")
    print("----\n")
    wandb.config.update(train_params)

    layer_params = {
        "node_feature_len": train_params["node_feature_len"],
        "hidden_layer_size": train_params["hidden_layer_size"],
        "output_dimension": train_params["output_dimension"],
        "num_layers": train_params["num_layers"],
        "num_heads": train_params["num_heads"],
        "dropout": train_params["dropout"],
    }

    model = ConSORTransformer(
        layer_params=layer_params,
        loss_fn=train_params["loss_fn"],
        batch_size=train_params["batch_size"],
        lrate=train_params["lrate"],
        wt_decay=train_params["wt_decay"],
        train_mode=True,
        triplet_loss_margin=train_params["triplet_loss_margin"],
    ).double()

    # Print model architecture.
    print(f"Model\n{model}")

    # Validation Dataset
    dataset_val = ConSORDataLoader(is_loading=True, input_file_path=val_filepath)
    dataloader_val = DataLoader(
        dataset_val,
        collate_fn=dataset_val.collate_val_data,
        batch_size=len(dataset_val),
    )

    # Create folder to save checkpoints.
    logfolder = os.path.join(
        config["TRAIN"]["log_folder"],
        f"consor_{date_time_stamp}"
    )
    Path(logfolder).mkdir(parents=True)

    # Save checkpoint for the three lowes validation losses.
    val_checkpoint_loss = ModelCheckpoint(
        dirpath=logfolder,
        filename=f"-{date_time_stamp}" + "-{epoch}-{step}-{mean_val_loss:5.3f}.pth",
        monitor="mean_val_loss",
        mode="min",
        save_top_k=3,
    )

    # Save checkpoint for the three highest success rates.
    val_checkpoint_successrate = ModelCheckpoint(
        dirpath=logfolder,
        filename=f"{date_time_stamp}" + "-{epoch}-{step}-{success_rate:4.2f}.pth",
        monitor="success_rate",
        mode="max",
        save_top_k=3,
    )

    # Pytorch lighning trainer
    trainer = pl.Trainer(
        devices=int(config["SYSTEM"]["num_devices"]),
        accelerator=config["SYSTEM"]["device"],
        max_epochs=train_params["num_epochs"],
        default_root_dir=logfolder,
        callbacks=[val_checkpoint_loss, val_checkpoint_successrate],
        logger=wandb_logger,
    )

    # Run one validation step to initialize logging parameters
    trainer.validate(model, dataloader_val)

    # Run train and validation steps across all epochs
    trainer.fit(model, dataloader_train, dataloader_val)

    # Save the last checkpoint.
    trainer.save_checkpoint(
        os.path.join(logfolder, f"consor-{date_time_stamp}-final.ckpt")
    )


if __name__ == "__main__":
    app.run(main)
