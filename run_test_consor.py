"""Script to evaluate ConSOR."""

import os
from datetime import datetime
from pathlib import Path

import csv
import yaml
from absl import app
from absl import flags

import torch
import numpy as np

from torch.utils.data import DataLoader

from consor_core.model import ConSORTransformer
from consor_core.data_loader import ConSORDataLoader
from helper_eval import calculate_evaluation_metrics

flags.DEFINE_string(
    "config_file_path",
    "./configs/consor_config.yaml",
    "Path to the config file for training ConSOR.",
)

FLAGS = flags.FLAGS


def save_test_results(data_loader, model, file_path):
    """Evaluate individual test batches and save results.

    Args:
        data_loader: DataLoader for the test data.
        model: ConSOR transformer model.
        file_path: Destination file path.

    Returns: Dictionary containing:
        success_rate: Success rate.
        non_zero_sed_mean: Mean of non-zero SEDs.
        non_zero_sed_std: Standard deviation of non-zero SEDs.
    """

    # TODO: Separate test results by rule.
    test_batches = next(iter(data_loader))
    with torch.no_grad():
        test_results = model.test(test_batches, 0)

        with open(file_path, "w") as fsave:
            writer = csv.DictWriter(
                fsave, fieldnames=list(test_results[0].keys())
            )
            writer.writeheader()

            for result in test_results:
                writer.writerow(result)

    (
        success_rate,
        non_zero_sed_mean,
        non_zero_sed_std
    ) = calculate_evaluation_metrics(test_results)

    return {
        "success_rate": success_rate,
        "non_zero_sed_mean": non_zero_sed_mean,
        "non_zero_sed_std": non_zero_sed_std,
        }


def main(argv):
    if len(argv) > 1:
        raise app.UsageError("Too many command-line arguments.")

    datetime_obj = datetime.now()
    date_time_stamp = datetime_obj.strftime("%Y_%m_%d_%H_%M_%S")

    # Load config.
    with open(FLAGS.config_file_path, "r") as fconfig:
        config = yaml.safe_load(fconfig)

    np.random.seed(int(config["SYSTEM"]["seed"]))
    torch.random.manual_seed(int(config["SYSTEM"]["seed"]))

    rules_list = config["rules"].split(",")

    # Dictionary of training parameters.
    test_params = {
        "batch_size": int(config["TRAIN"]["batch_size"]),
        "hidden_layer_size": int(config["MODEL"]["hidden_layer_size"]),
        "output_dimension": int(config["MODEL"]["output_dimension"]),
        "num_heads": int(config["MODEL"]["num_heads"]),
        "num_layers": int(config["MODEL"]["num_layers"]),
        "rules": rules_list,
        "dropout": float(config["MODEL"]["dropout"]),
        "loss_fn": str(config["LOSS"]["loss_fn"]),
    }

    if "triplet_margin" == str(config["LOSS"]["loss_fn"]):
        test_params["triplet_loss_margin"] = float(config["LOSS"]["triplet_margin"])
    else:
        test_params["triplet_loss_margin"] = None

    # Test data filepath.
    seen_test_file_path = config["DATA"]["seen_test_tensors_path"]
    unseen_test_file_path = config["DATA"]["unseen_test_tensors_path"]

    # Initialize test data loaders.
    seen_test_dataset = ConSORDataLoader(
        is_loading=True, input_file_path=seen_test_file_path
    )
    unseen_test_dataset = ConSORDataLoader(
        is_loading=True, input_file_path=unseen_test_file_path
    )

    seen_test_data_loader = DataLoader(
        seen_test_dataset,
        collate_fn=seen_test_dataset.collate_val_data,
        batch_size=len(seen_test_dataset),
    )

    unseen_test_data_loader = DataLoader(
        unseen_test_dataset,
        collate_fn=unseen_test_dataset.collate_val_data,
        batch_size=len(unseen_test_dataset),
    )

    print("Dataloaders are ready!")

    # Update feature dimensionality.
    test_params["node_feature_len"] = seen_test_dataset.get_feature_length()

    # Checkpoint path.
    ckpt_path = config["TEST"]["ckpt"]
    print(f"Checkpoint path: {ckpt_path}")

    layer_params = {
        "node_feature_len": test_params["node_feature_len"],
        "hidden_layer_size": test_params["hidden_layer_size"],
        "output_dimension": test_params["output_dimension"],
        "num_layers": test_params["num_layers"],
        "num_heads": test_params["num_heads"],
        "dropout": test_params["dropout"],
    }

    model = ConSORTransformer.load_from_checkpoint(
        ckpt_path,
        layer_params=layer_params,
        loss_fn=test_params["loss_fn"],
        train_mode=False,
    ).double()

    model.eval()

    # Print model architecture.
    print(f"Model\n{model}")

    # Create the folder to save results.
    results_folder = config["TEST"]["results_folder"]
    Path(results_folder).mkdir(parents=True, exist_ok=True)

    # Save seen and unseen test results.
    seen_file_path = os.path.join(
        results_folder, f"consor_{date_time_stamp}_seen_results.csv"
    )
    seen_eval_results = save_test_results(seen_test_data_loader, model, seen_file_path)
    print("Seen test results:")
    print(f'Success rate: {seen_eval_results["success_rate"]}')
    print(f'Non-zero SED mean: {seen_eval_results["non_zero_sed_mean"]}')
    print(f'Non-zero SED std: {seen_eval_results["non_zero_sed_std"]}')
    np.save(
        os.path.join(results_folder, f"consor_{date_time_stamp}_seen_aggregate.npy"),
        seen_eval_results
    )

    unseen_file_path = os.path.join(
        results_folder, f"consor_{date_time_stamp}_unseen.csv"
    )
    unseen_eval_results = save_test_results(unseen_test_data_loader, model, unseen_file_path)
    print("Unseen test results:")
    print(f'Success rate: {unseen_eval_results["success_rate"]}')
    print(f'Non-zero SED mean: {unseen_eval_results["non_zero_sed_mean"]}')
    print(f'Non-zero SED std: {unseen_eval_results["non_zero_sed_std"]}')
    np.save(
        os.path.join(results_folder, f"consor_{date_time_stamp}_unseen_aggregate.npy"),
        unseen_eval_results
    )


if __name__ == "__main__":
    app.run(main)
