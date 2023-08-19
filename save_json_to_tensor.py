"""Script to convert json data into tensor format and save."""

import os
import json
from datetime import datetime
from pathlib import Path

import yaml
from absl import app
from absl import flags

import torch
import numpy as np

from consor_core.data_loader import ConSORDataLoader

flags.DEFINE_string(
    "config_file_path",
    "./data_config.yaml",
    "Path to the config file for generating tensor data.",
)

FLAGS = flags.FLAGS


def convert_json_to_tensor(json_data, obj_pos_encoding_dim, container_pos_encoding_dim):
    """Wraps around ConSOR DataLoader and returns converted tensor data.

    Args:
        json_data: See ConSORDataLoader.
        obj_pos_encoding_dim: See ConSORDataLoader.
        container_pos_encoding_dim: See ConSORDataLoader.
    """

    data_loader = ConSORDataLoader(
        is_loading=False,
        input_json=json_data,
        obj_pos_encoding_dim=obj_pos_encoding_dim,
        container_pos_encoding_dim=container_pos_encoding_dim,
    )
    return data_loader.dataset_scenes


def main(argv):
    if len(argv) > 1:
        raise app.UsageError("Too many command-line arguments.")

    datetime_obj = datetime.now()
    date_time_stamp = datetime_obj.strftime("%Y_%m_%d_%H_%M_%S")

    # Load config
    with open(FLAGS.config_file_path, "r") as fconfig:
        config = yaml.safe_load(fconfig)

    # Seed
    np.random.seed(int(config["seed"]))
    torch.random.manual_seed(int(config["seed"]))

    # Create destination folder.
    Path(config['destination_folder']).mkdir(parents=True, exist_ok=True)

    # Seen objects.
    for mode in ["train", "val", "test"]:
        seen_json_file_path = os.path.join(
            config["json_data_folder"],
            f'consor_{config["json_seen_objects_timestamp"]}_seen_objects_{mode}.json',
        )

        with open(seen_json_file_path, "r") as fjson:
            json_data_seen = json.load(fjson)

        tensor_data = convert_json_to_tensor(
            json_data_seen,
            config["obj_pos_encoding_dim"],
            config["container_pos_encoding_dim"],
        )

        torch.save(
            tensor_data,
            os.path.join(
                config["destination_folder"],
                f"consor_{date_time_stamp}_seen_objects_{mode}.pt"
            ),
        )

        print(f"Saved tensors for seen objects, mode {mode}")

    # Unseen objects.
    unseen_json_file_path = os.path.join(
        config["json_data_folder"],
        f'consor_{config["json_unseen_objects_timestamp"]}_unseen_objects_test.json',
    )

    with open(unseen_json_file_path, "r") as fjson:
        json_data_unseen = json.load(fjson)

    tensor_data = convert_json_to_tensor(
        json_data_unseen,
        config["obj_pos_encoding_dim"],
        config["container_pos_encoding_dim"],
    )

    torch.save(
        tensor_data,
        os.path.join(
            config["destination_folder"],
            f"consor_{date_time_stamp}_unseen_objects_test.pt"
        ),
    )

    print('Saved tensors for unseen objects')


if __name__ == "__main__":
    app.run(main)
