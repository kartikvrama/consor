"""Class to convert json data to tensor for training and evaluation."""

import os
import numpy as np

import torch

import semantic2thor
from helper_eval import return_list_intersection


def return_positional_encoding_generator(max_position, d_model, min_freq=1e-4):
    """Returns a sinusoidal positional encoding generator.

    Args:
        max_position: Maximum number of positions.
        d_model: Positional encoding representation.
        min_freq: Minimum frequency of the sinusoidal positional encoding.

    Returns:
        A function to generate positional embeddings given a position index.
    """
    position = np.arange(max_position)

    freqs = min_freq ** (2 * (np.arange(d_model) // 2) / d_model)

    pos_enc = position.reshape(-1, 1) * freqs.reshape(1, -1)

    pos_enc[:, ::2] = np.cos(pos_enc[:, ::2])
    pos_enc[:, 1::2] = np.sin(pos_enc[:, 1::2])

    def _return_pos_encoding(pos_index: int):
        return pos_enc[pos_index]

    return _return_pos_encoding


class ConSORDataLoader:
    """Dataloader for transformer model.

    Attributes:
        dataset_scenes: List of scenes in the dataset.
    """

    def __init__(
        self,
        is_loading,
        input_json=None,
        input_file_path=None,
        val_batch_size=1,
        obj_pos_encoding_dim=4,
        container_pos_encoding_dim=8,
    ):
        """Initializes data loader class and loads data to memory.

        Args:
            is_loading: Flag to indicate whether data is being loaded for the
                model or saved.
            input_json: JSON data with each object containing
                {'objects', 'initial', 'goal', 'edit_distance', 'rule'}.
                Required if is_loading is False, else defaults to None.
            input_file_path: File path of the saved tensor data. Required if
                is_loading is True, else defaults to None.
            val_batch_size: Batch size of val data, usually set to train batch
                size
            obj_pos_encoding_dim: Dimension of the positional encoding for the
                object.
            container_pos_encoding_dim: Dimension of the positional encoding for
                the container.
        """

        self.val_batch_size = val_batch_size

        if is_loading:
            self.load_tensors(input_file_path)

        else:
            # Positional encoding generators for object and container instance.
            self.position_encoder_object = return_positional_encoding_generator(
                max_position=40, d_model=obj_pos_encoding_dim
            )
            self.position_encoder_container = return_positional_encoding_generator(
                max_position=10, d_model=container_pos_encoding_dim
            )

            self.convert_json_to_tensor(input_json)

    def __len__(self):
        return len(self.dataset_scenes)

    def __getitem__(self, idx):
        return self.dataset_scenes[idx]

    def embed_object(self, object_name):
        """Returns the commonsense embedding of the input object.

        Args:
            object_name: Name of the object to embed. Object is None if
                referring to the container.
        """

        if object_name == "None":  # unit vector of ones if empty box.
            vector = np.ones_like(semantic2thor.load("Box")[
                                  "Conceptnet Embedding"])
            return vector / np.linalg.norm(vector)

        else:
            return semantic2thor.load(object_name)["Conceptnet Embedding"]

    def _to_tensor(self, initial_scene, goal_scene):
        """Helper function to convert json data to tensor.

        Args:
            scene_graph: Scene represented as a dict of type:
                {'objects': list of objects,
                'scene': dictionary of objects organized by container}.
        """

        object_nodes = []
        initial_cluster_asgns = []
        goal_cluster_asgns = []

        for cluster_key in initial_scene["scene"].keys():
            if cluster_key == "table":  # skip the table.
                continue

            # Corresponding initial and goal clusters.
            initial_scene_cluster = initial_scene["scene"][cluster_key]
            goal_scene_cluster = goal_scene["scene"][cluster_key]

            # Note: this is number of elements in GOAL cluster that match with INITIAL cluster.
            matched_elems, mismatched_elems = return_list_intersection(
                goal_scene_cluster, initial_scene_cluster
            )

            assert sorted(matched_elems + mismatched_elems) == sorted(
                goal_scene_cluster
            ), "Something wrong with return_list_intersection called from _to_tensor function"

            if len(matched_elems) == 0:  # [initial_scene_cluster] is empty.
                assert (
                    len(initial_scene_cluster) == 0
                ), "Something wrong in _to_tensor function"

                # Adding placeholder object.
                object_nodes += ["None"]
                initial_cluster_asgns += [int(cluster_key)]
                goal_cluster_asgns += [int(cluster_key)]

            else:  # adding all the matched elements to both lists.
                object_nodes += matched_elems
                initial_cluster_asgns += [int(cluster_key)] * \
                    len(matched_elems)
                goal_cluster_asgns += [int(cluster_key)] * len(matched_elems)

            if (
                len(mismatched_elems) > 0
            ):  # initial and goal clusters do not match perfectly.
                object_nodes += mismatched_elems

                # all mismatched elems in the scene are on the table.
                initial_cluster_asgns += [0] * len(
                    mismatched_elems
                )  # assign 0 to table objects.
                goal_cluster_asgns += [int(cluster_key)] * \
                    len(mismatched_elems)

            else:  # initial and goal clusters match perfectly.
                assert sorted(goal_scene_cluster) == sorted(
                    initial_scene_cluster
                ), "Something wrong with _to_tensor function"

        assert (
            len(object_nodes) == len(
                initial_cluster_asgns) == len(goal_cluster_asgns)
        ), (
            f"Nodes {len(object_nodes)} and cluster asgns: {len(initial_cluster_asgns)}"
            f" & {len(goal_cluster_asgns)} do not match"
        )

        # Object-centric embeddings (using np.array to speed up tensor conversion).
        object_embeddings_cs = torch.Tensor(
            np.array([self.embed_object(ot) for ot in object_nodes])
        )

        # Positional embeddings of cluster.
        object_embeddings_container_pos = torch.Tensor(
            np.array(
                [
                    self.position_encoder_container(ocid)
                    for ocid in initial_cluster_asgns
                ]
            )
        )

        # Positional embedding of object.
        object_embeddings_instance_pos = torch.Tensor(
            np.array(
                [
                    self.position_encoder_object(instid)
                    for instid in range(len(object_nodes))
                ]
            )
        )

        # Final representation.
        object_embeddings = torch.concatenate(
            [
                object_embeddings_cs,
                object_embeddings_container_pos,
                object_embeddings_instance_pos,
            ],
            axis=1,
        ).type(torch.float)

        return (
            object_embeddings,
            initial_cluster_asgns,
            goal_cluster_asgns,
            object_nodes,
        )

    def convert_json_to_tensor(self, input_json_data):
        """Converts json data into transformer batches.

        Args:
            input_json_data: See input_json in init method.
        """

        data_list = []

        for scene_key, scene_json in input_json_data.items():
            (
                object_embeddings,
                initial_cluster_asgns,
                goal_cluster_asgns,
                object_nodes,
            ) = self._to_tensor(scene_json["initial"], scene_json["goal"])

            initial_cluster_asgns_tensor = torch.Tensor(
                initial_cluster_asgns).to(torch.int)
            goal_cluster_asgns_tensor = torch.Tensor(
                goal_cluster_asgns).to(torch.int)

            data_list.append(
                (
                    object_embeddings,
                    initial_cluster_asgns_tensor,
                    goal_cluster_asgns_tensor,
                    object_nodes,
                    (scene_key, scene_json),
                )
            )

        self.dataset_scenes = data_list

    def load_tensors(self, filepath):
        """Loads tensor graphs from file path into memory.

        Args:
            filepath: Path to tensor data of rearrangement scenes.
        """

        if os.path.exists(filepath):
            self.dataset_scenes = torch.load(filepath)

        else:
            raise FileNotFoundError(f"{filepath} does not exist")

        print("Feature length: ", self.dataset_scenes[0][0].size()[1])

    def tensor_to_batch(self, node_embedding_list):
        """Creates a padded batch of scenes from a list of tensors.

        Args:
            node_embedding_list: List of node embeddings, each of size N_{obj} x D_{obj}.
        """

        # max number of objects in any example in batch.
        max_length = max([x.shape[0] for x in node_embedding_list])

        embedding_dim = node_embedding_list[0].shape[1]

        # dimensions of padded data list: 1 x max_length x embedding_dim.
        node_embedding_list_padded = [
            torch.concatenate(
                [x, torch.zeros([max_length - x.shape[0], embedding_dim])], dim=0
            ).unsqueeze(0)
            for x in node_embedding_list
        ]

        # mask out padded values in each row of the batch ->
        # zeros = actual values, ones = padded values.
        # Syntax: https://pytorch.org/docs/stable/generated/torch.nn.Transformer.html#torch.nn.Transformer.
        attention_mask = [
            torch.concatenate(
                [
                    torch.zeros([1, x.shape[0]]),
                    torch.ones([1, max_length - x.shape[0]]),
                ],
                dim=1,
            )
            for x in node_embedding_list
        ]

        batch_tensors = torch.concatenate(
            node_embedding_list_padded, dim=0).to(torch.double)
        batch_attention_masks = torch.concatenate(
            attention_mask, dim=0).type(torch.ByteTensor)

        return batch_tensors, batch_attention_masks

    def collate_tensor_batch(self, tensor_list):
        """Function to convert list of tensors into transformer batch.

        Args:
            tensor_list: List of tuples of the following form:
                (Object-centric embeddings,
                Object cluster assignment in initial scene,
                Object cluster assignment in goal scene,
                List of objects in the scene,
                Scene tuple of key and scene json).
        """

        tensor_list = [tup[0] for tup in tensor_list]

        # batch_input: batch_size x normalized_seq_len x embedding_dim, and
        # batch_attention_mask: batch_size x normalized_seq_len.
        batch_input, batch_attention_mask = self.tensor_to_batch(tensor_list)

        initial_cluster_asgns_list = [
            tup[1] for tup in tensor_list
        ]  # initial cluster labels
        goal_cluster_asgns_list = [tup[2]
                                   for tup in tensor_list]  # goal cluster labels

        batch_initial_cluster_asgns = initial_cluster_asgns_list[0].clone()
        batch_goal_cluster_asgns = goal_cluster_asgns_list[0].clone()

        batch_node_split = [len(goal_cluster_asgns_list[0])]

        for initial_cluster_asgn, goal_cluster_asgn in zip(
            initial_cluster_asgns_list[1:], goal_cluster_asgns_list[1:]
        ):
            # add original goal and initial cluster assignments (no offset)
            batch_goal_cluster_asgns = torch.concatenate(
                [batch_goal_cluster_asgns, goal_cluster_asgn]
            )

            batch_initial_cluster_asgns = torch.concatenate(
                [batch_initial_cluster_asgns, initial_cluster_asgn]
            )

            batch_node_split += [len(goal_cluster_asgn)]

        graph_objects_list = [tup[3] for tup in tensor_list]
        json_list = [tup[4] for tup in tensor_list]

        return (
            batch_input,
            batch_attention_mask,
            batch_initial_cluster_asgns,
            batch_goal_cluster_asgns,
            batch_node_split,
            graph_objects_list,
            json_list,
        )

    def collate_val_data(self, tensor_list):
        """Wraps around collate_tensors method for validation data.

        Args:
            tensor_list: See collate_tensors method.
        """

        batch_size = min(self.val_batch_size, len(tensor_list))

        if batch_size == 1:
            number_of_batches = 1
        else:
            number_of_batches = 1 + len(tensor_list) // batch_size

        val_batch_indices = np.array_split(
            np.arange(len(tensor_list)), number_of_batches
        )

        val_batch_list = []

        for batch_idx in val_batch_indices:
            val_batch = self.collate_tensor_batch(
                [tensor_list[i] for i in batch_idx])

            val_batch_list.append(val_batch)

        return val_batch_list
