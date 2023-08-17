"""ConSOR Transformer model to infer object placement from partial initial scene."""

from copy import deepcopy
import numpy as np

import torch
from torch import nn
from torch.nn import functional as F
from torch.optim import Adam

from pytorch_metric_learning.reducers import DoNothingReducer
from pytorch_metric_learning.losses import TripletMarginLoss, NPairsLoss

import pytorch_lightning as pl
from helper_eval import calculate_ged_lsa


class TransfEmbedderModule(pl.LightningModule):
    """ConSOR Transformer Model to infer goal arrangement from a partially
    arranged initial scene."""

    def __init__(
        self,
        layer_params,
        loss_fn,
        batch_size,
        lrate,
        wt_decay,
        train_mode=False,
        triplet_loss_margin=None,
    ):
        """Initialize the transformer model.

        Args:
            layer_params: Dictionary of layer parameters including node feature
                length, hidden layer size, number of heads, dropout,
                number of layers, and output dimension.
            loss_fn: Type of loss function to use. Currently supports
                'triplet_margin' and 'npairs'.
            batch_size: Batch size for training.
            lrate: Training learning rate.
            wt_decay: Weight decay for model training.
            train_mode: True if the model is being trained, else the model is
                used in inference.
            triplet_loss_margin: Margin for triplet loss function. This must be
                specified if loss_fn is 'triplet_margin'.
        """

        super().__init__()

        # dimensionality of node features
        node_feature_len = layer_params["node_feature_len"]
        hidden_layer_size = layer_params["hidden_layer_size"]
        output_dimension = layer_params["output_dimension"]

        self.encoder_layer = nn.TransformerEncoderLayer(
            d_model=node_feature_len,
            nhead=layer_params["num_heads"],
            activation="relu",
            dim_feedforward=hidden_layer_size,
            dropout=layer_params["dropout"],
            batch_first=True,
            norm_first=False,
        )

        self.encoder = nn.TransformerEncoder(
            encoder_layer=self.encoder_layer, num_layers=layer_params["num_layers"]
        )

        # Linear layer
        self.dropout_linear = nn.Dropout(0)
        self.activation = nn.ReLU()  # nn.LeakyReLU(negative_slope=0.01)
        self.linear = nn.Linear(node_feature_len, output_dimension)

        # Do not reduce loss to mean
        reducer = DoNothingReducer()

        # Contrastive loss function
        if loss_fn == "triplet_margin":  # triple margin loss
            self.contrastive_loss = TripletMarginLoss(
                margin=triplet_loss_margin, reducer=reducer
            )

        elif loss_fn == "npairs":  # npairs loss
            self.contrastive_loss = NPairsLoss(reducer=reducer)

        else:
            raise NotImplementedError(f"{loss_fn} is not implemented")

        self.batch_size = batch_size

        # optimizer hyperparameters
        if train_mode:  # if train mode
            self.optimizer_params = dict({"lrate": lrate, "wtdecay": wt_decay})

        # For debugging purposes only
        self._debug_matrices = dict()

    def forward(self, input_tensor, attention_mask):
        """Forward pass of the transformer model.

        Args:
            input_tensor: Input tensor of shape batch_size x num_columns x d_node.
            attention_mask: Attention mask of shape
                batch_size x num_columns x d_node.

        Returns:
            Output tensor of shape batch_size x num_columns x d_output.
        """

        output = self.encoder(
            input_tensor, src_key_padding_mask=attention_mask)

        # dimension reduction of transformer embeddings.
        batch_len = output.size()[0]

        output = self.linear(
            self.dropout_linear(self.activation(
                output.view(-1, output.size()[-1])))
        )
        output = output.view(batch_len, -1, output.size()[-1])

        output = F.normalize(output, p=2.0, dim=-1)
        return output

    def training_step(self, train_batch, batch_idx):
        """Performs one training step of the transformer model.

        Args:
            train_batch: Sequence of tensors in the following order:
                1. batch input tensors,
                2. batch attention masks,
                3. batch object cluster assignments in initial scene,
                4. batch object cluster assignments in the true goal scene,
                5. list of nodes splitting the batch into scenes,
                6. list of objects in each scene,
                7. list of raw scene jsons loaded from data.
            batch_idx: Default pytorch lightning argument.
        """

        del batch_idx

        (
            batch_input,
            batch_attention_mask,
            _,
            batch_goal_cluster_asgns,
            batch_node_split,
            _,
            _,
        ) = train_batch

        # edge_adj_pred is N x N output, N is num of nodes in batch graph.
        node_embeddings = self.forward(batch_input, batch_attention_mask)

        start = 0
        for scene_num, node_len in enumerate(batch_node_split):
            node_embeddings_sample = node_embeddings[scene_num, :node_len, :]
            goal_cluster_asgns_sample = batch_goal_cluster_asgns[
                start: start + node_len
            ]

            # very special case where there is only one container.
            if torch.unique(goal_cluster_asgns_sample).size()[0] == 1:
                loss_sample = torch.zeros(
                    (1,), dtype=torch.double, requires_grad=True
                ).to("cuda")

            else:
                # Unreduced loss from function.
                loss_dict = self.contrastive_loss(
                    node_embeddings_sample, goal_cluster_asgns_sample
                )

                loss_sample = loss_dict["loss"]["losses"]

            if scene_num == 0:
                loss_array = loss_sample.clone()

            else:
                loss_array = torch.concat([loss_array, loss_sample])

            start += node_len

        # Non zero loss reduction.
        nonzero_loss_indices = torch.nonzero(loss_array)

        if len(nonzero_loss_indices) > 0:
            loss = torch.mean(loss_array[nonzero_loss_indices])

        else:  # all losses are zero, no non zero losses.
            loss = torch.sum(node_embeddings_sample * 0)

        # Record training loss.
        self.log(
            "train_loss",
            loss,
            batch_size=self.batch_size,
            on_step=True,
            on_epoch=True,
            prog_bar=True,
        )

        return loss

    def evaluate_batch(self, batch, return_embeddings=False):
        """Performs evaluation step on input batch.
        Args:
            batch: Sequence of tensors in the following order:
                1. batch input tensors,
                2. batch attention masks,
                3. batch object cluster assignments in initial scene,
                4. batch object cluster assignments in the true goal scene,
                5. list of nodes splitting the batch into scenes,
                6. list of objects in each scene,
                7. list of raw scene jsons loaded from data.
            return_embeddings: If True, returns the raw embeddings of objects
                per scene.
        """
        self.eval()

        # Collect predictions and metrics
        batch_ed = {}
        batch_predictions = {}
        batch_predicted_embeddings = {}

        (
            batch_input,
            batch_attention_mask,
            batch_initial_cluster_asgns,
            _,
            batch_node_split,
            objects_list,
            scene_json_list,
        ) = batch

        # edge_adj_pred is N x N output, N is num of nodes in batch graph.
        batch_node_embeddings = self.forward(batch_input, batch_attention_mask)

        # Unravel the batch into individual graphs.
        start = 0

        for scene_num, node_len in enumerate(batch_node_split):
            # Unraveling the batch.
            pred_node_embeddings = batch_node_embeddings[scene_num, :node_len, :]
            initial_cluster_asgns = batch_initial_cluster_asgns[
                start: start + node_len
            ]

            # list of graph objects.
            graph_objects = objects_list[scene_num]

            # Indexes of arranged objects/clusters and query objects.
            arranged_object_indices = torch.nonzero(initial_cluster_asgns > 0).squeeze(
                1
            )
            query_object_indices = torch.nonzero(
                initial_cluster_asgns == 0).squeeze(1)

            all_cluster_indices = torch.unique(
                initial_cluster_asgns[arranged_object_indices]
            )

            # Initialize cluster means.
            cluster_means = torch.empty(size=(0, pred_node_embeddings.size()[1])).type(
                pred_node_embeddings.type()
            )

            for cidx in all_cluster_indices:
                object_indices_cidx = torch.nonzero(
                    initial_cluster_asgns == cidx)
                mean_cidx = torch.mean(
                    pred_node_embeddings[object_indices_cidx], dim=0)

                cluster_means = torch.concat([cluster_means, mean_cidx], dim=0)

            query_node_embeddings = pred_node_embeddings[query_object_indices]

            # Pairwise similarity matrix between query nodes and cluster means.
            query_mean_distances = torch.matmul(
                query_node_embeddings, cluster_means.t()
            )

            # Choose cluster with maximum similarity.
            query_cluster_asgns = torch.argmax(
                query_mean_distances, dim=1).cpu()
            query_cluster_asgns = query_cluster_asgns.apply_(
                lambda x: all_cluster_indices[x]
            )

            # tuple of (key, scene).
            json_key, json_scene = scene_json_list[scene_num]
            rule = json_scene["rule"]

            save_key = f"{json_key}"

            if return_embeddings:
                batch_predicted_embeddings[save_key] = dict(
                    {
                        "objects": graph_objects,
                        "embeddings": pred_node_embeddings.cpu().numpy(),
                    }
                )

            scene_pred = deepcopy(json_scene["initial"])
            scene_table_objects = scene_pred["scene"]["table"]

            # move objects on the table to their respective boxes.
            for i, oidx in enumerate(query_object_indices):
                object_name = graph_objects[oidx]
                new_cidx = int(query_cluster_asgns[i])

                assert object_name in scene_table_objects

                scene_pred["scene"]["table"].remove(object_name)
                scene_pred["scene"][str(new_cidx)].append(object_name)

            ged, _ = calculate_ged_lsa(scene_pred, json_scene["goal"])

            batch_predictions[save_key] = dict(
                {
                    "initial": deepcopy(json_scene["initial"]),
                    "goal_predicted": scene_pred,
                }
            )

            batch_ed[save_key] = ged

            start += node_len

        if return_embeddings:
            return batch_predictions, batch_ed, batch_predicted_embeddings

        else:
            return batch_predictions, batch_ed

    def validation_step(self, val_batches, batch_idx):
        """Performs evaluation on the validation data.

        Args:
            val_batches: Sequence of batches, each batch being a tuple containing the following:
                1. batch input tensors,
                2. batch attention masks,
                3. batch object cluster assignments in initial scene,
                4. batch object cluster assignments in the true goal scene,
                5. list of nodes splitting the batch into scenes,
                6. list of objects in each scene,
                7. list of raw scene jsons loaded from data.
            batch_idx: Default pytorch lightning argument.
        """

        del batch_idx

        # Collect validation metrics.
        num_val_examples = 0
        nonzero_ged_array = []
        num_successes = 0

        for val_batch in val_batches:
            (
                batch_input,
                batch_attention_mask,
                batch_initial_cluster_asgns,
                batch_goal_cluster_asgns,
                batch_node_split,
                objects_list,
                scene_json_list,
            ) = val_batch

            # edge_adj_pred is N x N output, N is num of nodes in batch graph.
            batch_node_embeddings = self.forward(
                batch_input, batch_attention_mask)

            # Unravel the batch into individual graphs.
            start = 0

            for scene_num, node_len in enumerate(batch_node_split):
                # Unraveling.
                pred_node_embeddings = batch_node_embeddings[scene_num, :node_len, :]
                # unchanged cluster asgns.
                initial_cluster_asgns = batch_initial_cluster_asgns[
                    start: start + node_len
                ]
                goal_cluster_asgns_sample = batch_goal_cluster_asgns[
                    start: start + node_len
                ]

                # if the scene has only one container.
                if torch.unique(goal_cluster_asgns_sample).size()[0] == 1:
                    val_loss_sample = torch.zeros(
                        (1,), dtype=torch.double, requires_grad=True
                    ).to("cuda")

                else:
                    # Calculate unreduced loss using validation data.
                    val_loss_dict = self.contrastive_loss(
                        pred_node_embeddings, goal_cluster_asgns_sample
                    )

                    val_loss_sample = val_loss_dict["loss"]["losses"]

                if scene_num == 0:
                    val_loss_array = val_loss_sample.clone()

                else:
                    val_loss_array = torch.concat(
                        [val_loss_array, val_loss_sample])

                # list of graph objects.
                graph_objects = objects_list[scene_num]

                # Indexes of arranged objects/clusters and query objects.
                arranged_object_indices = torch.nonzero(
                    initial_cluster_asgns > 0
                ).squeeze(1)
                query_object_indices = torch.nonzero(
                    initial_cluster_asgns == 0
                ).squeeze(1)

                all_cluster_indices = torch.unique(
                    initial_cluster_asgns[arranged_object_indices]
                )

                # Initialize cluster means.
                cluster_means = torch.empty(
                    size=(0, pred_node_embeddings.size()[1])
                ).type(pred_node_embeddings.type())

                for cidx in all_cluster_indices:
                    object_indices_cidx = torch.nonzero(
                        initial_cluster_asgns == cidx)
                    mean_cidx = torch.mean(
                        pred_node_embeddings[object_indices_cidx], dim=0
                    )

                    cluster_means = torch.concat(
                        [cluster_means, mean_cidx], dim=0)

                query_node_embeddings = pred_node_embeddings[query_object_indices]

                # Pairwise similarity matrix between query nodes and cluster means.
                query_mean_distances = torch.matmul(
                    query_node_embeddings, cluster_means.t()
                )

                # Choose cluster with maximum similarity.
                query_cluster_asgns = torch.argmax(
                    query_mean_distances, dim=1).cpu()
                query_cluster_asgns = query_cluster_asgns.apply_(
                    lambda x: all_cluster_indices[x]
                )

                # tuple of (key, scene)
                _, json_scene = scene_json_list[scene_num]

                scene_pred = deepcopy(json_scene["initial"])
                scene_table_objects = scene_pred["scene"]["table"]

                # move objects on the table to their respective boxes.
                for i, oidx in enumerate(query_object_indices):
                    object_name = graph_objects[oidx]
                    new_cidx = int(query_cluster_asgns[i])

                    assert object_name in scene_table_objects

                    scene_pred["scene"]["table"].remove(object_name)
                    scene_pred["scene"][str(new_cidx)].append(object_name)

                ged, _ = calculate_ged_lsa(scene_pred, json_scene["goal"])

                # For calculating success rate.
                if ged == 0:
                    num_successes += 1

                # For calculating non-zero average ged.
                else:
                    nonzero_ged_array.append(ged)

                num_val_examples += 1

                start += node_len

        nonzero_ged_avg = np.mean(nonzero_ged_array)
        success_rate = 1.0 * num_successes / num_val_examples

        # Record mean loss, success rate, and average non-zero ged for the validation set.
        mean_val_loss = torch.mean(val_loss_array.clone())
        self.log(
            "mean_val_loss", mean_val_loss, batch_size=num_val_examples, prog_bar=True
        )

        self.log(
            "val_average_nonzero_ged",
            nonzero_ged_avg,
            batch_size=num_val_examples,
            prog_bar=True,
        )

        self.log(
            "success_rate",
            success_rate,
            batch_size=num_val_examples,
            prog_bar=True,
        )

    def test(self, test_batches, batch_idx):
        """Performs evaluation on the test data.

        Args:
            test_batches: Sequence of batches, each batch being a tuple containing the following:
                1. batch input tensors,
                2. batch attention masks,
                3. batch object cluster assignments in initial scene,
                4. batch object cluster assignments in the true goal scene,
                5. list of nodes splitting the batch into scenes,
                6. list of objects in each scene,
                7. list of raw scene jsons loaded from data.
            batch_idx: Default pytorch lightning argument.
        """
        del batch_idx

        self.eval()  # eval mode.

        # Collect test results.
        result_array = []

        for (
            batch_input,
            batch_attention_mask,
            batch_initial_cluster_asgns,
            _,
            batch_node_split,
            objects_list,
            scene_json_list,
        ) in test_batches:
            # edge_adj_pred is N x N output, N is num of nodes in batch graph.
            batch_node_embeddings = self.forward(
                batch_input, batch_attention_mask)

            # Unravel the batch into individual graphs.
            start = 0

            for scene_num, node_len in enumerate(batch_node_split):
                # Unraveling.
                pred_node_embeddings = batch_node_embeddings[scene_num, :node_len, :]
                # unchanged cluster asgns.
                initial_cluster_asgns = batch_initial_cluster_asgns[
                    start: start + node_len
                ]

                # list of graph objects.
                graph_objects = objects_list[scene_num]

                # Indexes of arranged objects/clusters and query objects.
                arranged_object_indices = torch.nonzero(
                    initial_cluster_asgns > 0
                ).squeeze(1)
                query_object_indices = torch.nonzero(
                    initial_cluster_asgns == 0
                ).squeeze(1)

                all_cluster_indices = torch.unique(
                    initial_cluster_asgns[arranged_object_indices]
                )

                # Initialize cluster means
                cluster_means = torch.empty(
                    size=(0, pred_node_embeddings.size()[1])
                ).type(pred_node_embeddings.type())

                for cidx in all_cluster_indices:
                    object_indices_cidx = torch.nonzero(
                        initial_cluster_asgns == cidx)
                    mean_cidx = torch.mean(
                        pred_node_embeddings[object_indices_cidx], dim=0
                    )

                    cluster_means = torch.concat(
                        [cluster_means, mean_cidx], dim=0)

                query_node_embeddings = pred_node_embeddings[query_object_indices]

                # Pairwise similarity matrix between query nodes and cluster means.
                query_mean_distances = torch.matmul(
                    query_node_embeddings, cluster_means.t()
                )

                # Choose cluster with maximum similarity.
                query_cluster_asgns = torch.argmax(
                    query_mean_distances, dim=1).cpu()
                query_cluster_asgns = query_cluster_asgns.apply_(
                    lambda x: all_cluster_indices[x]
                )

                # tuple of (key, scene).
                json_key, json_scene = scene_json_list[scene_num]
                rule = json_scene["rule"]

                save_key = f"{json_key}"

                scene_pred = deepcopy(json_scene["initial"])
                scene_table_objects = scene_pred["scene"]["table"]

                # move objects on the table to their respective boxes.
                for i, oidx in enumerate(query_object_indices):
                    object_name = graph_objects[oidx]
                    new_cidx = int(query_cluster_asgns[i])

                    assert object_name in scene_table_objects

                    scene_pred["scene"]["table"].remove(object_name)
                    scene_pred["scene"][str(new_cidx)].append(object_name)

                ged, ged_norm = calculate_ged_lsa(
                    scene_pred, json_scene["goal"])

                result = {
                    "data_key": save_key,
                    "objects": json_scene["objects"],
                    "rule": json_scene["rule"],
                    "edit_distance": json_scene["edit_distance"],
                    "ged": ged,
                    "ged_normalized": ged_norm,
                }

                result_array.append(result)

                start += node_len

        return result_array

    def configure_optimizers(self):
        """Configures the pytorch lightning optimizer."""

        return Adam(
            self.parameters(),
            lr=self.optimizer_params["lrate"],
            weight_decay=self.optimizer_params["wtdecay"],
        )
