"""Script to train Abdo-CF baseline."""

import os
from datetime import datetime
from pathlib import Path

from absl import app
from absl import flags

import yaml
import torch
import numpy as np
import matplotlib.pyplot as plt

from abdoCf_core.model import OrganizeMyShelves

flags.DEFINE_string(
    "config_file_path",
    "./configs/consor_config.yaml",
    "Path to the config file for training Abdo-CF baseline.",
)

FLAGS = flags.FLAGS


def main(argv):
    if len(argv) > 1:
        raise app.UsageError("Too many command-line arguments.")

    datetime_obj = datetime.now()
    date_time_stamp = datetime_obj.strftime("%Y_%m_%d_%H_%M_%S")

    # Load config.
    with open(FLAGS.config_file_path, "r") as fh:
        config = yaml.safe_load(fh)

    log_folder = os.path.join(
        config["MODEL"]["log_folder"], f"abdoCf_{date_time_stamp}"
    )
    Path(log_folder).mkdir(parents=True, exist_ok=True)

    seed = config["SEED"]
    np.random.seed(seed)
    torch.manual_seed(seed)

    data_folder = config["DATA"]["destination_folder"]
    folder_tag = config["DATA"]["json_seen_objects_timestamp"] + "_seen-objs"
    rating_matrix_file_path = os.path.join(
        data_folder, f"consor_ranking_matrix_train_{folder_tag}.npy"
    )
    print(f"Path to training rank matrix: {rating_matrix_file_path}")

    # Load training ratings matrix.
    ratings_matrix = np.load(rating_matrix_file_path)
    num_pairs, num_schemas = ratings_matrix.shape
    # Extract all non negative indices (x, y).
    nonneg_indices_xy = np.nonzero(ratings_matrix >= 0)

    # Flatten ratings matrix.
    ratings_ravel = ratings_matrix.ravel()
    nonneg_indices_ravel = np.nonzero(ratings_ravel >= 0)[0]

    assert all(
        np.array([x * num_schemas + y for x, y in zip(*nonneg_indices_xy)])
        == nonneg_indices_ravel
    ), "Non negative indices are not calculated properly"

    print(f"Total size of matrix {num_pairs*num_schemas}.")
    print(f"Number of non-negative elements: {len(nonneg_indices_ravel)}")

    ratings_ravel = torch.FloatTensor(ratings_ravel)
    ratings_ravel.requires_grad = False

    # Hyperparameters
    lambda_reg = config["MODEL"]["lambda_reg"]
    learning_rate = config["MODEL"]["learning_rate"]
    hidden_dimension = config["MODEL"]["hidden_dimension"]

    model = OrganizeMyShelves(
        hidden_dimension, num_pairs, num_schemas, lambda_reg=lambda_reg
    )
    optimizer = torch.optim.Adam(params=model.parameters(), lr=learning_rate)

    # Validation dataset.
    val_ratings_matrix_file_path = os.path.join(
        data_folder, f"consor_ranking_matrix_val_{folder_tag}.npy"
    )
    val_ratings_matrix = np.load(val_ratings_matrix_file_path)
    # Extract all non negative indices (x, y).
    val_nonneg_indices_xy = np.nonzero(val_ratings_matrix >= 0)

    # Flatten ratings matrix and convert to tensor
    val_ratings_ravel = val_ratings_matrix.ravel()
    val_nonneg_indices_ravel = np.nonzero(val_ratings_ravel >= 0)[0]

    val_ratings_ravel = torch.FloatTensor(val_ratings_ravel)
    val_ratings_ravel.requires_grad = False

    # Training loop.
    losses = []
    ep = 1
    while True:
        ratings_pred = model.forward()
        mse_loss, total_loss = model.calculate_loss(
            ratings_pred, ratings_ravel, nonneg_indices_ravel, nonneg_indices_xy
        )
        total_loss.backward()
        optimizer.step()
        optimizer.zero_grad()

        save_mse_loss = mse_loss.detach().cpu().numpy()
        losses.append(save_mse_loss)
        if len(losses) >= 2:
            convergence_rate = (losses[-2] - save_mse_loss) / losses[-2]
        else:
            convergence_rate = 1

        print(
            f"Epoch: {ep+1}, Train loss: {save_mse_loss}, Convergence: {convergence_rate}"
        )

        # End training if convergence rate is less than 1e-3.
        if convergence_rate < 1e-3:
            break

        with torch.no_grad():
            ratings_pred = model.forward()
            val_mse_loss, _ = model.calculate_loss(
                ratings_pred,
                val_ratings_ravel,
                val_nonneg_indices_ravel,
                val_nonneg_indices_xy,
            )

            print(f"Epoch: {ep+1}, Val loss: {val_mse_loss.cpu().numpy()}")

        optimizer.zero_grad()
        ep += 1

    print(f"Initial train MSE loss: {losses[0]}")
    print(f"Final train MSE loss: {losses[-1]}")

    # save plot
    plt.plot(losses)
    plt.yticks(np.arange(0, 5, 0.2))
    plt.ylim(0, 5)
    plt.savefig(os.path.join(log_folder, f"train_loss_{date_time_stamp}.png"))

    # tensor to numpy
    biases_obj_pair = model.biases_obj_pair.cpu().detach().numpy()
    biases_schema = model.biases_schema.cpu().detach().numpy()
    obj_preference_matrix = model.obj_preference_matrix.cpu().detach().numpy()
    schema_preference_matrix = model.schema_preference_matrix.cpu().detach().numpy()

    np.savez(
        os.path.join(
            log_folder,
            f"abdoCf-weights-{date_time_stamp}.npz",
        ),
        biases_obj_pair=biases_obj_pair,
        biases_schema=biases_schema,
        obj_preference_matrix=obj_preference_matrix,
        schema_preference_matrix=schema_preference_matrix,
    )


if __name__ == "__main__":
    app.run(main)
