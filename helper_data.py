"""Helper functions for loading and visualizing semantic rearrangement data."""

import logging
import platform

import numpy as np
import matplotlib.pyplot as plt

from nltk.corpus import wordnet as wn


class ObjectStruct:  # struct of an object
    def __init__(self, object_dict):
        self.name = object_dict["conceptnet_name"]

        self.wn_str = object_dict["wordnet_name"]
        self.wn_syn = wn.synset(object_dict["wordnet_name"])

        self.activity_label = object_dict["activity"]
        self.aff_label = object_dict["affordance"]

        self.flag_stackable = True if object_dict["stackable"] == "y" else False

    def __eq__(self, other):
        """Checks if two objects have the same conceptnet and wordnet name"""

        if isinstance(other, ObjectStruct):
            return self.name == other.name and self.wn_str == other.wn_str

        return False


def return_conceptnet_path():
    if platform.system() == "Linux":
        conceptnet_path = "/home/kartik/Documents/datasets/numberbatch-en.txt"

    elif platform.system() == "Windows":
        conceptnet_path = "C:/Users/karti/Documents/datasets/numberbatch-en.txt"

    else:
        print(f"Unfamiliar OS named {platform.system()}, please add to the list")
        raise ValueError

    return conceptnet_path


def loginfo(data):
    logging.info(data)
    print(data)


def plot_dist_matrix(data, rowlabels, collabels, filename):
    fig, ax = plt.subplots()
    fig.set_figheight(12)
    fig.set_figwidth(12)

    ax.imshow(data, cmap="Blues")

    ax.set_xticks(np.arange(len(collabels)))
    ax.set_yticks(np.arange(len(rowlabels)))
    ax.set_xticklabels(collabels)
    ax.set_yticklabels(rowlabels)

    ax.set_xlabel("columns")
    ax.set_ylabel("rows")

    plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")

    for i in range(len(rowlabels)):
        for j in range(len(collabels)):
            _ = ax.text(
                j,
                i,
                "{}".format(data[i, j]),
                ha="center",
                va="center",
                color="black",
            )
    plt.title(filename)
    plt.savefig("{}.png".format(filename))


def display_cluster(cluster_dict):
    for cluster_id, cluster_stuff in cluster_dict.items():
        print(f"{cluster_id}:{cluster_stuff}")

    print("----")


def display_data(json_data):
    objects = json_data["objects"]
    edges = json_data["edges"]

    immovable_objects = ["table", "box"]

    print("---")

    cluster = dict(
        {obj: [] for obj in objects if obj.split("-")[0] in immovable_objects}
    )

    for i in range(len(edges[0])):
        head, tail = objects[edges[0][i]], objects[edges[1][i]]

        if head.split("-")[0] != "box":
            if tail in cluster:
                cluster[tail].append(head.split("-")[0])

            else:
                cluster[tail] = [head.split("-")[0]]

    display_cluster(cluster)


def json_scene_to_prompt(scene, example_id=None, is_example=True):
    """Converts a JSON scene into a GPT-3 prompt.

    Args:
        scene: A JSON scene.
        example_id: The ID of the example, if this is an example scene.
        is_example: Flag indicating if the given scene is an example scene.
    """

    separate_camel_case = lambda s: "".join(
        map(lambda x: x if x.islower() else " " + x, s)
    )

    prompt = ""

    if example_id is not None:
        prompt += f"Example {example_id}:\n"

    num_containers = len(scene["initial"]["scene"].keys()) - 1  # removing the table

    prompt += (
        f"Task:\nThere are {num_containers} containers on the table. "
        + "They contain the following objects:\n"
    )

    objects = scene["objects"]

    object_inst_ids = dict({o: [*range(objects.count(o))] for o in set(objects)})

    # Initial scene: objects [in] containers.
    for cl in range(1, num_containers + 1):
        prompt += f"Container {cl}: "
        for obj in sorted(scene["initial"]["scene"][str(cl)]):
            obj_noCC = separate_camel_case(obj)
            prompt += f"{obj_noCC}-{object_inst_ids[obj].pop(0)}"

        if len(scene["initial"]["scene"][str(cl)]) == 0:
            prompt = prompt[:-2] + " is empty"

        else:
            prompt = prompt[:-2]

        prompt += "\n"

    # Initial scene: objects [on] the table (surface).
    prompt += "The following objects are on the table: "
    for obj in sorted(scene["initial"]["scene"]["table"]):
        obj_noCC = separate_camel_case(obj)
        prompt += f"{obj_noCC}-{object_inst_ids[obj].pop(0)}"

    prompt = prompt[:-2]
    prompt += "\nHow would you place the objects on the table into the containers?"

    if is_example:  # If this is an example scene prompt
        prompt += "\n\nAnswer:\n"

        object_inst_ids = dict({o: [*range(objects.count(o))] for o in set(objects)})

        # Goal scene: objects [in] containers.
        for cl in range(1, num_containers + 1):
            prompt += f"Container {cl}: "
            for obj in sorted(scene["goal"]["scene"][str(cl)]):
                obj_noCC = separate_camel_case(obj)
                prompt += f"{obj_noCC}-{object_inst_ids[obj].pop(0)}"

            prompt = prompt[:-2]
            prompt += "\n"

    return prompt
