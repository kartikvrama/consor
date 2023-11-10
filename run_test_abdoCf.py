"""Script to evaluate Abdo-CF baseline."""

import os
from datetime import datetime
from pathlib import Path

import csv
import json
from absl import app
from absl import flags

import yaml
import numpy as np
from sklearn.cluster import SpectralClustering

from helper_eval import calculate_scene_edit_distance_lsa, calculate_evaluation_metrics

flags.DEFINE_string(
    "config_file_path",
    "./configs/consor_config.yaml",
    "Path to the config file for training Abdo-CF baseline.",
)

FLAGS = flags.FLAGS


def objcomb2row(obj1, obj2):
    """Returns the row index of the object combination in the ranking matrix.

    Args:
        obj1: Name of first object in the combination.
        obj2: Name of second object in the combination.
    Raises:
        KeyError: If the object combination does not exist.
    """

    global object_combinations

    for i, objtup in enumerate(object_combinations):
        if (obj1, obj2) == objtup or (obj2, obj1) == objtup:
            return i
    raise KeyError(f"{obj1}-{obj2} combination does not exist in list")


def cluster_objects(objects, num_clusters, schema):
    """Determines object grouping based on the learned pairwise ratings.

    Spectral clustering is used to group objects into num_clusters groups.

    Args:
        objects: List of objects in the scene.
        num_clusters: Number of groups to form.
        schema: Desired organizational schema.
    """
    global seed

    # Create a pairwise distance matrix.
    pairwise_rating = np.zeros((len(objects), len(objects)))
    for i, obj_i in enumerate(objects):
        for j, obj_j in enumerate(objects):
            pairwise_rating[i, j] = return_rating(obj_i, obj_j, schema)

    beta = 10
    eps = 1e-6
    pairwise_rating_kernel = (
        np.exp(beta * pairwise_rating / np.std(pairwise_rating)) + eps
    )
    spectral_clustering = SpectralClustering(
        n_clusters=num_clusters,
        random_state=seed,
        affinity="precomputed",
        assign_labels="kmeans",
    ).fit(pairwise_rating_kernel)

    final_labels = [i + 1 for i in spectral_clustering.labels_]  # labels start at 1.

    assert len(set(final_labels)) == max(final_labels), "Missing numbers in the labels"
    final_labels = [str(i) for i in final_labels]

    predicted_scene = dict({i: [] for i in final_labels})
    predicted_scene.update({"table": []})

    for obj, l in zip(objects, final_labels):
        predicted_scene[l].append(obj)

    return dict({"objects": objects, "scene": predicted_scene})


def return_rating(obj1, obj2, schema):
    """Returns the rating for a given object pair and organizational schema.

    This is a wrapper around the objcomb2row function to return the desired
    pairwise rating.

    Args:
        obj1: Name of the first object in the combination.
        obj2: Name of the second object in the combination.
        schema: Desired organizational schema.
    """

    global cf_params
    global schemas2col

    biases_obj_pair = cf_params["biases_obj_pair"]
    biases_schema = cf_params["biases_schema"]
    obj_preference_matrix = cf_params["obj_preference_matrix"]
    schema_preference_matrix = cf_params["schema_preference_matrix"]

    i = objcomb2row(obj1, obj2)
    j = schemas2col(schema)

    result = (
        biases_obj_pair[i, 0]
        + biases_schema[0, j]
        + np.dot(obj_preference_matrix[:, i], schema_preference_matrix[:, j])
    )

    return result


def main(argv):
    if len(argv) > 1:
        raise app.UsageError("Too many command-line arguments.")

    global seed
    global cf_params
    global schemas2col
    global object_combinations

    datetime_obj = datetime.now()
    date_time_stamp = datetime_obj.strftime("%Y_%m_%d_%H_%M_%S")

    # Load config.
    with open(FLAGS.config_file_path, "r") as fh:
        config = yaml.safe_load(fh)

    seed = config["SEED"]
    np.random.seed(seed)

    # Testing only on seen objects.
    with open(config["EVAL"]["test_data_json"], "r") as fh:
        test_examples = json.load(fh)

    # Load object combinations in order.
    with open(config["MODEL"]["object_combinations_file"], "r") as fh:
        rawlines = fh.readlines()

    object_combinations = []
    for line in rawlines:
        object_combinations.append(tuple(line.strip().split(",")))
    print(f"Number of object combinations: {len(object_combinations)}")

    schemas_list = config["DATA"]["schemas"].split(",")
    schemas2col_dict = dict({schema: i for i, schema in enumerate(schemas_list)})
    schemas2col = lambda x: schemas2col_dict[x]

    # Load parameters by schema.
    cf_params = np.load(config["EVAL"]["ratings_matrix_fitted"])

    # Evaluation loop.
    results_array = []
    predictions_dict = {}
    for example_key, example in test_examples.items():
        initial_scene = example["initial"]
        objects = example["objects"]

        # --- PREDICTION ---

        ## Assume schema is provided to abdo
        schema_gt = example["rule"]
        final_pred_scene = cluster_objects(
            objects,
            num_clusters=len(initial_scene["scene"].keys()) - 1,
            schema=schema_gt,
        )

        # --- EVALUATION ---

        # Calculate scene edit distance between prediction and ground truth.
        goal_scene = example["goal"]
        schema_gt = example["rule"]
        pred_goal_ed, _ = calculate_scene_edit_distance_lsa(
            final_pred_scene, goal_scene
        )

        data_result = {
            "data_key": example_key,
            "objects": example["objects"],
            "rule": schema_gt,
            "init_distance_from_goal": example["edit_distance"],
            "sed": pred_goal_ed,
        }

        results_array.append(data_result)
        predictions_dict[example_key] = dict(
            {
                "prediction": final_pred_scene,
                "edit_distance": pred_goal_ed
            }
        )

        # DEBUG
        if len(results_array) % 25 == 0:
            print(
                "Count: {},\tData tag: {},\tEval GED: {},\tGT schema {}".format(
                    len(results_array), example_key, pred_goal_ed, example["rule"]
                )
            )

    # Save results and predictions.
    results_folder = config["EVAL"]["results_folder"]
    Path(results_folder).mkdir(parents=True, exist_ok=True)

    with open(
        os.path.join(results_folder, f"abdoCf_{date_time_stamp}_seen_results.csv"), "w"
    ) as fw:
        writer = csv.DictWriter(fw, fieldnames=list(results_array[0].keys()))
        writer.writeheader()

        for result in results_array:
            writer.writerow(result)

    with open(
        os.path.join(results_folder, f"abdoCf_{date_time_stamp}_seen_predictions.json"),
        "w",
    ) as fw:
        json.dump(predictions_dict, fw, indent=4)

    # Calculate aggregate evaluation metrics and save them.
    (success_rate, non_zero_sed_mean, non_zero_sed_std) = calculate_evaluation_metrics(
        results_array
    )
    eval_results_dict = {
        "success_rate": success_rate,
        "non_zero_sed_mean": non_zero_sed_mean,
        "non_zero_sed_std": non_zero_sed_std,
    }
    print(f"Success rate: {success_rate}")
    print(f"Non-zero SED mean: {non_zero_sed_mean}")
    print(f"Non-zero SED std: {non_zero_sed_std}")
    np.save(
        os.path.join(results_folder, f"abdoCf_{date_time_stamp}_seen_aggregate.npy"),
        eval_results_dict,
    )


if __name__ == "__main__":
    app.run(main)
