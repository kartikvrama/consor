"""Script to generate semantic rearrangement initial and goal scenes."""

import os
import json
from pathlib import Path
from typing import Any, Optional

from copy import deepcopy
from datetime import datetime

from absl import app
from absl import flags
import numpy as np
import pandas as pd

from data import clustering_rules
from helper_data import display_cluster
from helper_eval import calculate_scene_edit_distance_lsa

# Arguments for generating data.
flags.DEFINE_integer(
    "min_object_classes",
    4,
    "Minimum number of object classes in a scene.",
)

flags.DEFINE_integer(
    "max_object_classes",
    6,
    "Maximum number of object classes in a scene.",
)

flags.DEFINE_integer(
    "max_object_instances",
    4,
    "Minimum number of instances per object class in a scene.",
)

flags.DEFINE_float(
    "train_test_split",
    0.9,
    "Train/test dataset split for seen objects. No train data for unseen objects.",
)

flags.DEFINE_integer(
    "max_goal_scenes_per_rule",
    2200,
    "Minimum number of goal scenes per schema.",
)

flags.DEFINE_bool(
    "seen",
    True,
    "Flag to generate scenes with seen objects. False means that scenes will be generated with unseen objects.",
)

flags.DEFINE_string(
    "objects_list",
    "/home/kartik/Documents/github_projs/consor/data/objects_annotated.csv",
    "List of objects in the dataset with seen/unseen labels.",
)

flags.DEFINE_string(
    "destination_folder",
    "/home/kartik/Documents/datasets/consor_data",
    "Destination folder for saving data.",
)

# Constants.
SEED = 843846553
FLAGS = flags.FLAGS


def return_scene_pair_json(
    scene_A, scene_A_label, scene_B, scene_B_label, edit_distance
) -> dict[str, Any]:
    """Returns a json object with a paired initial and goal scene.

    The function also performs assertions on the pair of scenes to ensure that
        the initial scene was generated from the goal scene.

    Args:
        scene_A (dict): initial scene
        scene_A_label (str): initial scene label
        scene_B (dict): goal scene
        scene_B_label (str): goal scene label
    """

    # ASSERTIONS:
    # scenes should have the same rule and aliasing.
    assert (
        scene_B["rule"] == scene_A["rule"] and scene_B["aliased"] == scene_A["aliased"]
    )

    # scenes shold have the same list of objects.
    assert scene_B["objects"] == scene_A["objects"]

    # scene labels should have the same root.
    assert scene_B_label.split(".")[0] == scene_A_label.split(".")[0]

    # objects and rule for pair of scenes.
    scene_objects = list(scene_A["objects"])
    scene_rule = str(scene_A["rule"])
    scene_aliasing = str(scene_A["aliased"])

    # typecasting all keys.
    scene_A_grouping = dict()
    for key, value in scene_A["scene"].items():
        if key == "table":
            scene_A_grouping[str(key)] = value
        else:
            scene_A_grouping[int(key)] = value

    scene_B_grouping = dict()
    for key, value in scene_B["scene"].items():
        if key == "table":
            scene_B_grouping[str(key)] = value
        else:
            scene_B_grouping[int(key)] = value

    scene_A_json = dict({"objects": scene_objects, "scene": scene_A_grouping})

    scene_B_json = dict({"objects": scene_objects, "scene": scene_B_grouping})

    elem = dict(
        {
            "objects": scene_objects,
            "initial": scene_A_json,
            "goal": scene_B_json,
            "edit_distance": edit_distance,
            "rule": scene_rule,
            "aliased": scene_aliasing,
        }
    )

    return elem


def save_json_paired(
    goal_scenes_labels,
    goal_scenes,
    pregoal_scenes_labels_dict,
    pregoal_scenes_dict,
    destination_file="save.json",
) -> None:
    """Generates initial-goal scene pairs and saves them in a json file.

    Args:
        goal_scenes_labels: List of goal scene identifiers
        goal_scenes: List of all goal scenes
        pregoal_scenes_labels_dict: Dictionary of all pregoal scene identifiers
            categorized by the distance from goal scene.
        pregoal_scenes_dict: Dictionary of all pregoal scenes categorized by the
            distance from goal scene.
        destination_file: Path to the json file where the data will be saved.
    """

    json_data = {}

    for key in pregoal_scenes_dict.keys():
        pregoal_edit_dist = int(key.split("_")[1])  # depth_{d}.

        pregoal_scenes_labels, pregoal_scenes = (
            pregoal_scenes_labels_dict[key],
            pregoal_scenes_dict[key],
        )

        goal_scenes_copy = deepcopy(goal_scenes)

        for goal_scene, goal_scene_label, pregoal_scene, pregoal_scene_label in zip(
            goal_scenes_copy, goal_scenes_labels, pregoal_scenes, pregoal_scenes_labels
        ):
            assert len(goal_scene["scene"]["table"]
                       ) == 0, f"{goal_scene_label}"

            if pregoal_scene is not None:
                elem = return_scene_pair_json(
                    pregoal_scene,
                    pregoal_scene_label,
                    goal_scene,
                    goal_scene_label,
                    pregoal_edit_dist,
                )

                label = str(
                    goal_scene_label.split(
                        ".")[0] + f".dist{pregoal_edit_dist}"
                )

                print(f"Pairing {pregoal_scene_label} with {goal_scene_label}")
                json_data.update({label: elem})

    print(f"Number of paired examples: {len(json_data.keys())}")

    with open(destination_file, "w") as fh:
        json.dump(json_data, fh, indent=4)

    print("Succesfully saved data to {} !".format(destination_file))


class RearrangeAI2Thor:
    """Class for generating rearrangement scenes from AI2Thor objects."""

    def __init__(self, config, rules_list):
        self.config = config
        self.rules_list = rules_list

    def check_aliasing(self, objects) -> Optional[bool]:
        """Checks whether aliased scenes can be generated from 'objects'.

        The scene generates one goal scene per schema for the given set of
        objects, and checks whether any of these goal scenes overlap. If you can
        generate more than two overlapping goal scenes using different schemas,
        then this combination of objects will yield aliased goal scenes.

        Args:
            objects: List of objects in the scene.
        """

        # separate set of objects for ooe.
        objects_ooe = []
        max_objects_count = max([objects.count(o) for o in objects])

        for o in set(objects):
            objects_ooe += list([o] * max_objects_count)

        assert all(
            [
                objects_ooe.count(o) == objects_ooe.count(objects[0])
                for o in set(objects_ooe)
            ]
        )

        scene_cluster_byrule = dict()
        for rule in self.rules_list:
            if rule == "ooe":
                scene_cluster_byrule[rule] = dict(
                    {
                        "objects": objects_ooe,
                        "scene": clustering_rules.return_clustering_func(rule)(
                            objects_ooe
                        ),
                    }
                )
            else:
                scene_cluster_byrule[rule] = dict(
                    {
                        "objects": objects,
                        "scene": clustering_rules.return_clustering_func(rule)(objects),
                    }
                )

        scene_cluster_byrule_equalinst = dict()
        for rule in self.rules_list:
            scene_cluster_byrule_equalinst[rule] = dict(
                {
                    "objects": objects_ooe,
                    "scene": clustering_rules.return_clustering_func(rule)(objects_ooe),
                }
            )

        # filter scenes that are None.
        if any([v["scene"] is None for v in scene_cluster_byrule.values()]):
            return None

        all_rule_pairs = [
            (r1, r2)
            for i, r1 in enumerate(self.rules_list)
            for r2 in self.rules_list[i + 1:]
        ]

        # goal states are reachable -> distance between goals must be a non-zero positive integer and not inf.
        def ged_condition(x):
            return x > 0 and not np.isinf(x)

        # comparing rule-0 with rule-1 to rule-{N-1} scenes.
        goal_distances = [
            calculate_scene_edit_distance_lsa(
                scene_cluster_byrule[r1], scene_cluster_byrule[r2])[0]
            for r1, r2 in all_rule_pairs
        ]
        comparing_goal_distances = [ged_condition(x) for x in goal_distances]

        goal_distances_equalinst = [
            calculate_scene_edit_distance_lsa(
                scene_cluster_byrule_equalinst[r1], scene_cluster_byrule_equalinst[r2]
            )[0]
            for r1, r2 in all_rule_pairs
        ]
        comparing_goal_distances_equalinst = [
            ged_condition(x) for x in goal_distances_equalinst
        ]

        # if any two goal states are reachable from each other and not same.
        if any(comparing_goal_distances) or any(comparing_goal_distances_equalinst):
            return True

        else:
            return False

    def populate_scenes_from_objects(self, objects_list, is_aliased):
        """Generates a scene from each schema for each set of objects in the list.

        This function is called separately for aliased and unaliased scene
        objects, and the is_aliased label is used to differentiate scenes.

        Args:
            objects_list: List of the objects for each scene.
            is_aliased: Label whether the objects are aliased or not.
        """

        num_pureclass = 0
        goal_scenes = []
        goal_scene_labels = []

        def return_alias_str(x):
            return "aliased" if x == True else "unaliased"

        # constraint- maximum number of scenes with objects in seperate boxes (pure class) = 5%.
        # Defaults to infinity if there are unseen objects, due to scarcity of data.
        max_pureclass_scenes = (
            int(0.05 * self.config["max_goal_scenes_per_rule"] // 2)
            if self.config["seen"]
            else np.inf
        )

        num_scenes_per_rule = 0
        for objects in objects_list:
            # separate set of objects for ooe.
            objects_ooe = []
            max_objects_count = max([objects.count(o) for o in objects])

            num_object_types = len(set(objects))

            for o in set(objects):
                objects_ooe += list([o] * max_objects_count)

            # create a scene for each rule.
            scene_cluster_by_rule = []
            for rule in self.rules_list:
                if rule == "ooe":
                    scene_cluster_by_rule.append(
                        clustering_rules.return_clustering_func(
                            rule)(objects_ooe)
                    )

                else:
                    scene_cluster_by_rule.append(
                        clustering_rules.return_clustering_func(rule)(objects)
                    )

            save = True

            # check for pure class scenes.
            if any(
                [len(s.keys()) == num_object_types +
                 1 for s in scene_cluster_by_rule]
            ):  # if any scene is pure class.
                if num_pureclass < max_pureclass_scenes:
                    num_pureclass += 1
                    save = True

                else:
                    # do not save the scene if there are too many pureclass scenes.
                    save = False

            if save:
                # create json dict from generated goal scene.
                for rule, scene in zip(self.rules_list, scene_cluster_by_rule):
                    if rule == "ooe":
                        scene_dict = dict(
                            {
                                "objects": objects_ooe,
                                "scene": deepcopy(scene),
                                "rule": rule,
                                "aliased": is_aliased,
                            }
                        )

                    else:
                        scene_dict = dict(
                            {
                                "objects": objects,
                                "scene": deepcopy(scene),
                                "rule": rule,
                                "aliased": is_aliased,
                            }
                        )

                    goal_scenes.append(scene_dict)
                    goal_scene_labels.append(
                        f"{rule}-{return_alias_str(is_aliased)}-id{num_scenes_per_rule}.pos"
                    )

                num_scenes_per_rule += 1

            if num_scenes_per_rule % 100 == 0:
                print(
                    "Aliased {}: Finished generating {} out of {}".format(
                        is_aliased,
                        num_scenes_per_rule,
                        self.config["max_goal_scenes_per_rule"] // 2,
                    )
                )

            if num_scenes_per_rule >= self.config["max_goal_scenes_per_rule"] // 2:
                break

        return goal_scenes, goal_scene_labels

    def generate_goal_scenes(self, all_object_names, verify_train_coverage=True):
        """Generates equal number of aliased and unaliased goal scenes for each rule.

        Args:
            all_object_names: List of all object names to be used.
            verify_coverage: If True, checks if all objects appear in the train data.
        """

        num_all_objects_dataset = len(all_object_names)
        print(f"{num_all_objects_dataset} Objects in this dataset")

        # Enumerate positive scenes.
        list_object_sets = []  # list of all sets of unique objects.
        list_scene_objects_aliased = []  # list of all scene objects per scene.
        # list of all scene objects per scene.
        list_scene_objects_unaliased = []

        # maximum number of movable objects in goal scenes.
        self.max_movable_objs = 0

        # threshold on maximum number of goal scenes per rule.
        # seen objects: create double the number of scenes to filter later.
        thershold_max_goal_scenes = (
            2 * self.config["max_goal_scenes_per_rule"]
            if self.config["seen"]
            else int(0.5 * self.config["max_goal_scenes_per_rule"])
        )

        # Generate [max_goal_scenes_per_rule] combinations of objects.
        while True:
            # number of object types sampled between (min_object_classes, max_object_classes).
            num_object_types = np.random.choice(
                range(
                    self.config["min_object_classes"],
                    1 + self.config["max_object_classes"],
                )
            )

            # choose a subset of [num_object_types] objects.
            object_indices = np.random.choice(
                range(num_all_objects_dataset), size=(num_object_types), replace=False
            ).astype(int)

            set_objects = [
                all_object_names[id] for id in object_indices
            ]  # set of unique object types.
            # need to sort for comparing with other positive scenes.
            set_objects = sorted(set_objects)

            # object combinations cannot be repeated.
            if any([set_objects == sc for sc in list_object_sets]):
                continue  # skip example.

            # adding set_objects to the list.
            list_object_sets += [set_objects]

            # choose number of instances per object.
            count_scene_objs = np.random.choice(
                range(1, self.config["max_object_instances"] + 1),
                size=(num_object_types),
                replace=True,
            ).astype(int)

            # measuring the maximum number of (movable) objects in the entire dataset.
            if sum(count_scene_objs) > self.max_movable_objs:
                self.max_movable_objs = sum(count_scene_objs)

            scene_objects = []  # list of object names in the scene.

            for q, obj in zip(count_scene_objs, set_objects):
                for _ in range(q):  # append each obj to the list q times.
                    scene_objects.append(obj)

            # shuffle the order of scene objects before sending it to clustering func.
            np.random.shuffle(scene_objects)

            is_aliased = self.check_aliasing(scene_objects)

            # invalid scene.
            if is_aliased is None:
                continue

            # aliased.
            elif (
                is_aliased
                and len(list_scene_objects_aliased) < thershold_max_goal_scenes
            ):
                list_scene_objects_aliased.append(scene_objects)

            # unaliased.
            elif (
                not is_aliased
                and len(list_scene_objects_unaliased) < thershold_max_goal_scenes
            ):
                list_scene_objects_unaliased.append(scene_objects)

            if len(list_scene_objects_unaliased):
                print(
                    (
                        f"Currently generated {len(list_scene_objects_aliased)}"
                        f" aliased and {len(list_scene_objects_unaliased)}"
                        " unaliased scenes"
                    )
                )

            if (
                len(list_scene_objects_aliased) == thershold_max_goal_scenes
                and len(list_scene_objects_unaliased) == thershold_max_goal_scenes
            ):
                # break while loop once we have a sufficient number of goal scenes.
                break

        (
            aliased_goal_scenes,
            aliased_goal_scene_labels,
        ) = self.populate_scenes_from_objects(
            list_scene_objects_aliased, is_aliased=True
        )
        (
            unaliased_goal_scenes,
            unaliased_goal_scene_labels,
        ) = self.populate_scenes_from_objects(
            list_scene_objects_unaliased, is_aliased=False
        )

        num_aliased = len(aliased_goal_scenes)
        assert num_aliased == len(aliased_goal_scene_labels)
        num_unaliased = len(unaliased_goal_scenes)
        assert num_unaliased == len(unaliased_goal_scene_labels)

        print(
            f"Number of aliased goal scenes (scenes with goal aliasing): {num_aliased}"
        )
        print(
            f"Number of unaliased goal scenes (scenes without goal aliasing): {num_unaliased}"
        )

        # --- Train/val/test split ---
        aliased_index_list = np.arange(len(aliased_goal_scenes)).astype(int)
        unaliased_index_list = np.arange(
            len(unaliased_goal_scenes)).astype(int)

        train_test_split = self.config["train_test_split"]

        # if train_test_split is 0, then use all scenes for test.
        if not train_test_split:
            goal_scenes_split = dict(
                {
                    "train": [],
                    "val": [],
                    "test": aliased_goal_scenes + unaliased_goal_scenes,
                }
            )

            goal_scene_labels_split = dict(
                {
                    "train": [],
                    "val": [],
                    "test": aliased_goal_scene_labels + unaliased_goal_scene_labels,
                }
            )

        else:
            train_lengths = [
                int(train_test_split * len(aliased_index_list))
                - int(train_test_split * len(aliased_index_list))
                % (len(self.rules_list)),
                int(train_test_split * len(unaliased_index_list))
                - int(train_test_split * len(unaliased_index_list))
                % (len(self.rules_list)),
            ]

            val_length = [
                (len(aliased_index_list) - train_lengths[0]) // 2,
                (len(unaliased_index_list) - train_lengths[1]) // 2,
            ]

            val_length = [
                val_length[0] - val_length[0] % (len(self.rules_list)),
                val_length[1] - val_length[1] % (len(self.rules_list)),
            ]

            goal_scenes_split = dict(
                {
                    "train": [
                        aliased_goal_scenes[i1]
                        for i1 in aliased_index_list[: train_lengths[0]]
                    ]
                    + [
                        unaliased_goal_scenes[i2]
                        for i2 in unaliased_index_list[: train_lengths[1]]
                    ],
                    "val": [
                        aliased_goal_scenes[i3]
                        for i3 in aliased_index_list[
                            train_lengths[0]: train_lengths[0] + val_length[0]
                        ]
                    ]
                    + [
                        unaliased_goal_scenes[i4]
                        for i4 in unaliased_index_list[
                            train_lengths[1]: train_lengths[1] + val_length[1]
                        ]
                    ],
                    "test": [
                        aliased_goal_scenes[i5]
                        for i5 in aliased_index_list[train_lengths[0] + val_length[0]:]
                    ]
                    + [
                        unaliased_goal_scenes[i6]
                        for i6 in unaliased_index_list[
                            train_lengths[1] + val_length[1]:
                        ]
                    ],
                }
            )

            goal_scene_labels_split = dict(
                {
                    "train": [
                        aliased_goal_scene_labels[i1]
                        for i1 in aliased_index_list[: train_lengths[0]]
                    ]
                    + [
                        unaliased_goal_scene_labels[i2]
                        for i2 in unaliased_index_list[: train_lengths[1]]
                    ],
                    "val": [
                        aliased_goal_scene_labels[i3]
                        for i3 in aliased_index_list[
                            train_lengths[0]: train_lengths[0] + val_length[0]
                        ]
                    ]
                    + [
                        unaliased_goal_scene_labels[i4]
                        for i4 in unaliased_index_list[
                            train_lengths[1]: train_lengths[1] + val_length[1]
                        ]
                    ],
                    "test": [
                        aliased_goal_scene_labels[i5]
                        for i5 in aliased_index_list[train_lengths[0] + val_length[0]:]
                    ]
                    + [
                        unaliased_goal_scene_labels[i6]
                        for i6 in unaliased_index_list[
                            train_lengths[1] + val_length[1]:
                        ]
                    ],
                }
            )

        object_set_coverage = set()

        if verify_train_coverage:  # Check for missing objects in train dataset.
            for scene in goal_scenes_split["train"]:
                unique_objects = set(scene["objects"])
                object_set_coverage.update(unique_objects)

            print(
                "Train obj count {}, Total obj count {}".format(
                    len(object_set_coverage), num_all_objects_dataset
                )
            )

            if num_all_objects_dataset != len(object_set_coverage):
                print("Increase sampling rate of train dataset!")
                raise ValueError

        else:  # Check for missing objects in test dataset.
            for scene in goal_scenes_split["test"]:
                unique_objects = set(scene["objects"])
                object_set_coverage.update(unique_objects)

            print(
                "Test obj count {}, Total obj count {}".format(
                    len(object_set_coverage), num_all_objects_dataset
                )
            )

            if num_all_objects_dataset != len(object_set_coverage):
                print("Increase sampling rate of test dataset!")
                raise ValueError

        print(
            "Number of positive sgs: Train: {}, Val: {}, Test: {}".format(
                len(goal_scene_labels_split["train"]),
                len(goal_scene_labels_split["val"]),
                len(goal_scene_labels_split["test"]),
            )
        )

        print(
            f"Maximum number of movable objects in the dataset is {self.max_movable_objs}"
        )

        return goal_scenes_split, goal_scene_labels_split

    def generate_pregoal_scenes_depthd(self, scenes_iter, scene_labels_iter, depth):
        """Generates a partially arranged scene by placing one random object at a time on the table.

        The function keeps track of how many objects have been removed from the
        goal scene using the depth variable.

        Args:
            scene_iter: List of scenes, where each scene has the fields
                {'objects', 'scene', 'rule'}.
            scene_labels: list of scene identifiers.
            depth: Current depth of search tree for generating partially
                arranged scenes from goal scenes.
        """

        # Initialize positive negative examples.
        scenes_pregoal = []
        scene_labels_pregoal = []
        pregoal_objects_moved = []

        # Permute each scene from previous depth using one random obj and one random location.
        for scene_iter_posid, scene_orig in enumerate(scenes_iter):
            if scene_iter_posid and scene_iter_posid % 10 == 0:
                print(f"Permuting scene {scene_iter_posid+1} at depth {depth}")

            scene_iter = deepcopy(scene_orig)
            scene_label_iter = scene_labels_iter[scene_iter_posid]

            if scene_iter is None or len(scene_iter["objects"]) == len(
                scene_iter["scene"]["table"]
            ):
                # not required if we start with objects on the table.
                scenes_pregoal.append(None)
                scene_labels_pregoal.append(None)

                continue  # if scene could not be permuted, skip.

            # list of objects not on the table.
            all_query_objects = [
                obj
                for cluster_key in scene_iter["scene"].keys()
                for obj in scene_iter["scene"][cluster_key]
                if cluster_key != "table"
            ]

            assert (
                len(all_query_objects) > 0
            ), f"Did not filter out scenes by depth! -> scene {scene_label_iter} depth {depth}"

            # choose random object to move.
            moved_obj = np.random.choice(all_query_objects)

            # all instances [current locations] of [moved_obj] except table.
            moved_obj_all_prelocs = [
                key
                for key in scene_iter["scene"].keys()
                if moved_obj in scene_iter["scene"][key] and key != "table"
            ]

            # choose random instance (random prelocation) of [moved_obj] to put on the table.
            moved_obj_preloc = int(np.random.choice(moved_obj_all_prelocs))

            scene_pregoal_dict = deepcopy(scene_iter["scene"])

            # Moving [moved_obj] from [moved_obj_preloc] to the table.
            scene_pregoal_dict[moved_obj_preloc].remove(moved_obj)
            scene_pregoal_dict["table"].append(moved_obj)

            # Saving new scene.
            scene_pregoal = dict(
                {
                    "objects": scene_iter["objects"],
                    "scene": deepcopy(scene_pregoal_dict),
                    "rule": scene_iter["rule"],
                    "aliased": scene_iter["aliased"],
                }
            )

            # Validate that edit distance b/w pregoal and original goal equals depth of search tree.
            graph_edit_distance, _ = calculate_scene_edit_distance_lsa(
                scene_pregoal,
                self._goal_scenes[scene_iter_posid],
            )

            if graph_edit_distance != depth:
                print(
                    f"Trying to displace {moved_obj} from {moved_obj_preloc} to table"
                )
                display_cluster(scene_iter)
                display_cluster(scene_pregoal)

                raise ValueError

            scenes_pregoal.append(scene_pregoal)

            scene_label_parent = scene_label_iter.split(".")[0]
            scene_labels_pregoal.append(f"{scene_label_parent}.neg-{depth}")

            pregoal_objects_moved.append(moved_obj)

        assert (
            len(scenes_pregoal) == len(
                scenes_iter) == len(scene_labels_pregoal)
        ), "Did not generate equal number of permuted scenes"

        return scene_labels_pregoal, scenes_pregoal, pregoal_objects_moved

    def permute_dataset(self, goal_scenes, goal_scene_labels):
        """Systematically generate partially arranged scenes from goal scenes,
            labeled with their edit distance from the goal scene.

        Each object in the goal scene is removed from its respective
        bin/container and placed on the table to generate a new partially
        arranged scene (also referred to in the code as pregoal scene). This
        process is continued until all objects are on the table and the
        bins/containers are empty.

        Args:
            goal_scenes: List of goal scenes.
            goal_scen_labels: List of goal scene identifiers.
        """

        # Initializing member variables.
        self._goal_scenes = goal_scenes

        # Depth 1.
        (
            permuted_scene_labels_d,
            permuted_scenes_d,
            moved_objects_scenes_d,
        ) = self.generate_pregoal_scenes_depthd(goal_scenes, goal_scene_labels, depth=1)

        print(
            f"Neg examples generated at depth 1: {len(permuted_scene_labels_d)}")

        permuted_scenes = dict({"depth_1": permuted_scenes_d})
        permuted_scene_labels = dict({"depth_1": permuted_scene_labels_d})
        permuted_scenes_moved_objs = dict({"depth_1": moved_objects_scenes_d})

        for depth in range(2, self.max_movable_objs + 1):  # Depth 2 to N.
            (
                permuted_scene_labels_d,
                permuted_scenes_d,
                moved_objects_scenes_d,
            ) = self.generate_pregoal_scenes_depthd(
                permuted_scenes_d, permuted_scene_labels_d, depth=depth
            )

            print(
                f"Neg examples generated at depth {depth}: {len(permuted_scene_labels_d)}"
            )

            permuted_scenes[f"depth_{depth}"] = permuted_scenes_d
            permuted_scene_labels[f"depth_{depth}"] = permuted_scene_labels_d
            permuted_scenes_moved_objs[f"depth_{depth}"] = moved_objects_scenes_d

        return permuted_scene_labels, permuted_scenes, permuted_scenes_moved_objs


def main(argv):
    if len(argv) > 1:
        raise app.UsageError("Too many command-line arguments.")

    datetime_obj = datetime.now()
    date_time_str = datetime_obj.strftime("%Y_%m_%d_%H_%M_%S")

    rules_list = ["class", "affordance", "ooe", "utility"]

    config = {
        "min_object_classes": FLAGS.min_object_classes,
        "max_object_classes": FLAGS.max_object_classes,
        "max_object_instances": FLAGS.max_object_instances,
        "max_goal_scenes_per_rule": FLAGS.max_goal_scenes_per_rule,
        "seen": FLAGS.seen,
        "objects_list": FLAGS.objects_list,
        "destination_folder": FLAGS.destination_folder,
    }

    # No train and validation split for unseen objects.
    if config["seen"]:
        config["train_test_split"] = FLAGS.train_test_split
    else:
        config["train_test_split"] = 0.0

    np.random.seed(SEED)  # set seed.

    print(f"List of rules: {rules_list}")

    dataset = RearrangeAI2Thor(config, rules_list)

    # read csv file of object names.
    objects_df = pd.read_csv(config["objects_list"])

    if config["seen"]:  # filter seen objects.
        objects_seen_df = objects_df[objects_df["Seen/Unseen"] == "Seen"]
        filtered_objects = list(objects_seen_df["ObjectName"])

    else:  # filter unseen objects.
        objects_unseen_df = objects_df[objects_df["Seen/Unseen"] == "Unseen"]
        filtered_objects = list(objects_unseen_df["ObjectName"])

    # Gathering goal scenes both aliased and unaliased.
    if config["seen"]:  # check that the train dataset covers all objects.
        (
            positive_scenes_split,
            positive_scenes_labels_split,
        ) = dataset.generate_goal_scenes(filtered_objects, verify_train_coverage=True)
    else:  # check that the tes tdataset covers all objects
        (
            positive_scenes_split,
            positive_scenes_labels_split,
        ) = dataset.generate_goal_scenes(filtered_objects, verify_train_coverage=False)

    # Create separate folders for paired data.
    folder_paired = config["destination_folder"]
    if os.path.exists(folder_paired):
        print(f"Folder {folder_paired} exists.")
    else:
        print(f"Creating folder {folder_paired}.")
        Path(folder_paired).mkdir(parents=True)

    for mode in ["train", "val", "test"]:
        # Skip train and val for unseen objects.
        if mode != "test" and not config["seen"]:
            continue

        positive_scenes, positive_scenes_labels = (
            positive_scenes_split[mode],
            positive_scenes_labels_split[mode],
        )

        print(f"Data for {mode} mode")

        negative_scenes_labels, negative_scenes, _ = dataset.permute_dataset(
            positive_scenes, positive_scenes_labels
        )

        if config["seen"]:
            destination_file = os.path.join(
                folder_paired, f"consor_{date_time_str}_seen_objects_{mode}.json"
            )
        else:
            destination_file = os.path.join(
                folder_paired, f"consor_{date_time_str}_unseen_objects_{mode}.json"
            )

        print(f"Destination file: {destination_file}")

        save_json_paired(
            positive_scenes_labels,
            positive_scenes,
            negative_scenes_labels,
            negative_scenes,
            destination_file=destination_file,
        )

        if config["seen"]:
            test_examples_readable_path = os.path.join(
                folder_paired, f"consor_{date_time_str}_seen_objects_test.txt"
            )
        else:
            test_examples_readable_path = os.path.join(
                folder_paired, f"consor_{date_time_str}_unseen_objects_test.txt"
            )

        # print the test examples in a text file for easy readibility.
        with open(test_examples_readable_path, "w") as ftxt:
            for plabel, pscene in zip(
                positive_scenes_labels_split["test"], positive_scenes_split["test"]
            ):
                ftxt.write(f'Key: {plabel}, Rule: {pscene["rule"]}\n')

                for container, objs in pscene["scene"].items():
                    ftxt.write(f"Container: {container}, Objects: {objs}\n")

                ftxt.write("----\n")

    if config["seen"]:
        print(
            f'Number of training positive examples: {len(positive_scenes_labels_split["train"])}'
        )


if __name__ == "__main__":
    app.run(main)
