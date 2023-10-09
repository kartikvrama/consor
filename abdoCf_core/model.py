"""Pytorch Model of the Rearrangement Baseline by Abdo et al. 2016."""

from math import sqrt

import torch
from torch import nn


class OrganizeMyShelves(nn.Module):
    """Pytorch Implementation of the Collaborative Filtering approach by Abdo 2016.

    Source: https://journals.sagepub.com/doi/10.1177/0278364916649248
    """

    def __init__(self, hidden_dimension, num_pairs, num_schemas, lambda_reg=1e-5):
        """Initialize model parameters.

        Args:
            hidden_dimension: Dimension of the learned preference vector.
            num_pairs: Number of all object-object pairs in the data.
            num_schemas: Number of organizational schemas in the data.
            lambda_reg: Weight regularization constant. Defaults to 1e-5.
        """

        super().__init__()

        # learnable variables
        biases_obj_pair = torch.randn(num_pairs, 1, requires_grad=True)  # b_i
        biases_schema = torch.randn(1, num_schemas, requires_grad=True)  # b_j
        obj_preference_matrix = torch.randn(
            hidden_dimension, num_pairs, requires_grad=True
        )  # s_i
        schema_preference_matrix = torch.randn(
            hidden_dimension, num_schemas, requires_grad=True
        )  # t_j

        # model dimensions
        self.num_pairs = num_pairs
        self.num_schemas = num_schemas
        self.hidden_dimension = hidden_dimension
        self.lambda_reg = lambda_reg

        # make torch parameters
        self.biases_obj_pair = nn.Parameter(biases_obj_pair)
        self.biases_schema = nn.Parameter(biases_schema)
        self.obj_preference_matrix = nn.Parameter(obj_preference_matrix)
        self.schema_preference_matrix = nn.Parameter(schema_preference_matrix)

        # random initialization
        nn.init.kaiming_uniform_(self.biases_obj_pair, a=sqrt(5))
        nn.init.kaiming_uniform_(self.biases_schema, a=sqrt(5))
        nn.init.kaiming_uniform_(self.obj_preference_matrix, a=sqrt(5))
        nn.init.kaiming_uniform_(self.schema_preference_matrix, a=sqrt(5))

    def forward(self):
        """Returns predicted ratings for all object-object pairs per schema."""

        r_pred = (
            self.biases_obj_pair.repeat(1, self.num_schemas)
            + self.biases_schema.repeat(self.num_pairs, 1)
            + torch.matmul(self.obj_preference_matrix.T, self.schema_preference_matrix)
        )

        assert r_pred.size() == (
            self.num_pairs,
            self.num_schemas,
        ), "Predicted ranking matrix has the wrong dimensions."

        return r_pred.flatten()

    def calculate_loss(self, r_pred, r_actual, nonneg_indices_ravel, nonneg_indices_xy):
        """Calculates MSE loss for predictions of known rating values.

        Args:
            r_pred: Predicted ratings from the forward function.
            r_actual: Actual ratings for all object-object pairs per schema.
            nonneg_indices_ravel: Indices of known ratings in flattened r_pred.
            nonneg_indices_xy: Row and Col indices of known ratings in r_pred.
        """

        assert len(r_pred) == len(r_actual)

        mse_loss = nn.MSELoss(reduction="sum")(
            r_pred[nonneg_indices_ravel], r_actual[nonneg_indices_ravel]
        )

        nonneg_indices_x, nonneg_indices_y = nonneg_indices_xy

        regularization_loss = (
            torch.sum(self.biases_obj_pair[nonneg_indices_x, 0] ** 2)
            + torch.sum(self.biases_schema[0, nonneg_indices_y] ** 2)
            + torch.sum(self.obj_preference_matrix[:, nonneg_indices_x] ** 2)
            + torch.sum(self.schema_preference_matrix[:, nonneg_indices_y] ** 2)
        )

        return mse_loss, mse_loss + 0.5 * self.lambda_reg * regularization_loss
