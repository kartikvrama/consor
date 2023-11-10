"""Script to evaluate GPT-3 responses and calculate evaluation metrics."""

import os
from datetime import datetime
from pathlib import Path

from copy import copy, deepcopy
from ast import literal_eval

import csv
from absl import app
from absl import flags

import json
import numpy as np
from itertools import chain

from helper_data import display_cluster
from helper_eval import calculate_scene_edit_distance_lsa, calculate_evaluation_metrics

flags.DEFINE_string("responses_file", None, help="Path to the GPT-3 responses file.")

flags.DEFINE_string(
    "test_json_data",
    None,
    help="Path to the json test data.",
)

flags.DEFINE_string(
    "results_folder",
    "./logs",
    help="Path to the folder to save the results."
)

FLAGS = flags.FLAGS


def prompt_to_json_dict(response, json_elem):
    """Converts GPT-3 response to json dict."""

    predicted_scene_objects = []
    predicted_scene_scene = dict({"table": []})

    initial_json_objects = sorted(json_elem["objects"])
    queue_objects_predict = copy(initial_json_objects)

    # print('[START_RESPONSE]'+response+'[END_RESPONSE]')
    newline_split_prompts = response.split("\n")
    newline_split_prompts = [p.strip() for p in newline_split_prompts if len(p) > 0]

    # choose only lines which have the word container or table
    container_wise_prompts = [
        p for p in newline_split_prompts if "Container" == p[:9] or "Table" == p[:5]
    ]

    if len(container_wise_prompts) == 0:
        return deepcopy(json_elem["initial"])

    for cwp in container_wise_prompts:
        cwp = cwp.strip()  # removing leading and trailing spaces
        cwp = cwp.strip("\n")  # removing leading and trailing new line characters

        split_by_semicolon = cwp.replace(" ", "").split(":")

        if len(split_by_semicolon) == 0 or (
            len(split_by_semicolon) == 1 and "empty" not in split_by_semicolon[0]
        ):
            print("weird: ", split_by_semicolon)
            continue

        # if there is a non empty table in the response
        if (
            split_by_semicolon[0] == "Table"
            and len(literal_eval(split_by_semicolon[1])) >= 0
        ):
            objects_with_inst_id = split_by_semicolon[1].split(",")

            # add table objects to table
            for o in objects_with_inst_id:
                o_category = o.split("-")[0]  # separate obj category from instance id
                if o_category in queue_objects_predict:
                    # add to predicted scene
                    predicted_scene_scene["table"].append(o_category)
                    predicted_scene_objects.append(o_category)

                    # remove from queue
                    queue_objects_predict.remove(o_category)

        # if container is empty
        elif (
            split_by_semicolon[0][9].isdigit()
            and split_by_semicolon[0][-7:] == "isempty"
        ):
            cluster_id = int(split_by_semicolon[0][9])

            if cluster_id in predicted_scene_scene.keys() or cluster_id >= len(
                json_elem["initial"]["scene"].keys()
            ):
                continue

            predicted_scene_scene[cluster_id] = []  # empty cluster

        # skip the containers that say <Container X and Container Y are empty>
        elif split_by_semicolon[0][9].isdigit() and "areempty" in split_by_semicolon[0]:
            continue

        else:
            assert split_by_semicolon[0][-1].isdigit()
            cluster_id = int(split_by_semicolon[0][-1])  # Container1 -> 1

            # ignore the cluster ids that are more than number of cluster in initial scene or that have already been read
            if cluster_id in predicted_scene_scene.keys() or cluster_id >= len(
                json_elem["initial"]["scene"].keys()
            ):
                continue

            # create new cluster id
            predicted_scene_scene[cluster_id] = []

            # objects with instance ids
            objects_with_inst_id = split_by_semicolon[1].split(",")

            # add {cluster_id} objects to {cluster_id}
            for o in objects_with_inst_id:
                o_category = o.split("-")[0]  # separate obj category from instance id
                if o_category in queue_objects_predict:
                    # add to predicted scene
                    predicted_scene_scene[cluster_id].append(o_category)
                    predicted_scene_objects.append(o_category)

                    # remove from queue
                    queue_objects_predict.remove(o_category)

    if len(queue_objects_predict) > 0:  # remaining objects go on the table
        for o_category in queue_objects_predict:
            # add to predicted scene
            predicted_scene_scene["table"].append(o_category)
            predicted_scene_objects.append(o_category)

    if len(predicted_scene_scene.keys()) < len(json_elem["initial"]["scene"].keys()):
        print("******************")
        display_cluster(json_elem["initial"]["scene"])
        print(response)
        display_cluster(predicted_scene_scene)
        for k in json_elem["initial"]["scene"].keys():
            if k == "table":
                continue
            if literal_eval(str(k)) not in predicted_scene_scene:
                print(f"{k} not in predicted scene")
                predicted_scene_scene[literal_eval(k)] = []
        print("After adjusting")
        display_cluster(predicted_scene_scene)
        print("*******************")

    assert len(predicted_scene_objects) == len(initial_json_objects)

    return dict({"objects": predicted_scene_objects, "scene": predicted_scene_scene})


def main(argv):
    if len(argv) > 1:
        raise app.UsageError("Too many command-line arguments.")

    datetime_obj = datetime.now()
    date_time_stamp = datetime_obj.strftime("%Y_%m_%d_%H_%M_%S")

    if FLAGS.responses_file is None:
        raise ValueError("Please provide a valid path to the responses file.")

    with open(FLAGS.responses_file, "r") as fread:
        alllines = fread.readlines()

    with open(FLAGS.test_json_data, "r") as fjson:
        test_data = json.load(fjson)

    schemas_list = ["class", "ooe", "affordance", "utility"]

    read = False
    current_key = None

    keys_list = []

    results_array = []

    for lineid, line in enumerate(alllines):
        if lineid == 0 and line[:9] == "Datapath:":
            continue  # dataset

        if "-" in line and line.split("-")[0] in schemas_list:
            read = True
            current_key = line.strip("\n")
            keys_list.append(current_key)
            continue

        if read:
            if "[START_RESPONSE]" in line:
                start_lineid = lineid + 1
                end_lineid = None

                for lineid_read, line_read in enumerate(alllines[start_lineid:]):
                    if "[END_RESPONSE]" in line_read:
                        end_lineid = lineid_read + start_lineid
                        break

                if end_lineid is None:
                    break

                # read between start_lineid and end_lineid
                response = list(chain(*alllines[start_lineid : end_lineid + 1]))

                response_string = ""
                for r in response:
                    response_string += r  # sum response into string
                response_string = response_string.strip("\n")

                response_string = response_string[:-14]
                string_to_response = []
                for rs in response_string:
                    string_to_response.append(rs)

                try:
                    predicted_scene = prompt_to_json_dict(
                        response_string, test_data[current_key]
                    )

                except AssertionError as e:
                    raise e

                assert not np.isinf(
                    calculate_scene_edit_distance_lsa(
                        predicted_scene, test_data[current_key]["initial"]
                    )[0]
                )
                sed, _ = calculate_scene_edit_distance_lsa(
                    predicted_scene, test_data[current_key]["goal"]
                )

                result = {
                    "data_key": current_key,
                    "objects": predicted_scene["objects"],
                    "rule": test_data[current_key]["rule"],
                    "init_distance_from_goal": test_data[current_key]["edit_distance"],
                    "sed": sed,
                }

                results_array.append(result)

                read = False
                current_key = None

    # Save results and predictions.
    results_folder = FLAGS.results_folder
    Path(results_folder).mkdir(parents=True, exist_ok=True)

    if "unseen" in FLAGS.test_json_data:
        seen_tag = "unseen"
    elif "seen" in FLAGS.test_json_data:
        seen_tag = "seen"
    else:
        raise ValueError("Not clear if the test data is seen or unseen.")

    with open(
        os.path.join(results_folder, f"gpt3_{date_time_stamp}_{seen_tag}_results.csv"),
        "w",
    ) as fw:
        writer = csv.DictWriter(fw, fieldnames=list(results_array[0].keys()))
        writer.writeheader()

        for result in results_array:
            writer.writerow(result)

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
        os.path.join(
            results_folder, f"gpt3_{date_time_stamp}_{seen_tag}_aggregate.npy"
        ),
        eval_results_dict,
    )


if __name__ == "__main__":
    app.run(main)
