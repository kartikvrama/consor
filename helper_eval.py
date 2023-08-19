"""Helper functions for evaluating rearrangement methods."""

import numpy as np
from scipy.optimize import linear_sum_assignment


def calculate_evaluation_metrics(results_list):
    """Calculates success rate and non-zero SED from evaluation results.
    
    Args:
    results_list: List of evaluation results (refer to test method in
        consore_core.model).
    """

    success_rate = 0.0
    non_zero_sed_array = []

    for result_dict in results_list:
        if result_dict['sed'] == 0:
            success_rate += 1
        else:
            non_zero_sed_array.append(result_dict['sed'])

    success_rate /= len(results_list)
    non_zero_sed_mean = np.mean(non_zero_sed_array)
    non_zero_sed_std = np.std(non_zero_sed_array)

    return success_rate, non_zero_sed_mean, non_zero_sed_std

def _typecase_json_keys(scene_dict):
    """Sets the keys of a scene dictionary as string or integer.
    """

    result = {}
    for key, value in scene_dict.items():
        if key == 'table':
            result[str(key)] = value
        else:
            result[int(key)] = value

    return result


class GraphStruct:
    """Common data structure for rearrangement scene graphs.
    
    Attributes:
        scene_dict: Dictionary representation of the scene, with keys as
            receptacle labels and values as lists of objects.
    """

    def __init__(self, graph):

        self.immovable_objects = ['table', 'box']

        assert isinstance(graph, dict), \
            f'Input graph type {type(graph)} is not a dictionary'

        # graph in json format provided
        if set(graph.keys()) == set(['objects', 'edges', 'dynamic_edge_mask', 'rule']):

            self.convert_json_scene_to_dict(graph)

        # receptacle:objects dict provided
        elif all([x in graph.keys() for x in ['objects', 'scene']]):

            # Assert there are no instance ids for object names
            assert all(['-' not in obj for obj in graph['objects']])

            self.scene_dict = _typecase_json_keys(graph['scene'])

        else:  # cluster provided, assume no objects on the table

            raise NotImplementedError('Format of the graph provided is wrong')

    def __len__(self):

        return len(list(self.scene_dict.keys()))  # number of boxes + 1 table

    def convert_json_scene_to_dict(self, json_scene):
        """Converts a JSON scene to a dictionary saved in memory.
        
        Args:
            json_scene: Rearrangement scene loaded from JSON data file.
        """

        num_immovable_objs = 1 + max([i for i, obj in enumerate(json_scene['objects'])
                                      if obj.split('-')[0] in self.immovable_objects])

        movable_edge_ids = [i for i, eh in enumerate(json_scene['edges'][0])
                            if eh >= num_immovable_objs]  # all movable edge heads

        # Initialize final graph (Note: clusters numbered from 1)
        self.scene_dict = dict({1 + int(obj.split('-')[1]): [] for obj in json_scene['objects']
                                if obj.split('-')[0] == 'box'})
        self.scene_dict.update({'table': []})

        for edge_id in movable_edge_ids:

            edge_hd, edge_tl = json_scene['edges'][0][edge_id], json_scene['edges'][1][edge_id]

            object_name = json_scene['objects'][edge_hd].split('-')[0]

            if edge_tl == 0:  # object on table (edge id 0)

                self.scene_dict['table'].append(object_name)

            else:

                box_id = edge_tl  # edge id of box_i is i

                if box_id in self.scene_dict.keys():

                    self.scene_dict[box_id].append(object_name)

                else:

                    raise ValueError(
                        'Issue with json_scene_to_cluster [utils_eval]')

        for key in self.scene_dict:  # sort objects in each box

            self.scene_dict[key] = sorted(self.scene_dict[key])

    def return_scene_objects(self):
        """Returns a sorted list of objects in scene_dict."""

        assert all([obj not in self.immovable_objects
                    for value in self.scene_dict.values() for obj in value])

        object_list = [obj for value in self.scene_dict.values()
                       for obj in value]

        return sorted(object_list)

    def to_ordered_list(self):
        """Converts scene_dict into an ordered list of object lists, excluding
            table objects."""

        return [list(self.scene_dict[k]) for k in range(1, len(self.scene_dict.keys()))]


def calculate_scene_edit_distance_lsa(scene_a, scene_b):
    """Calculates the edit distance between two rearrangement scenes.
    
    This function computes the linear sum assignment between the cluster
    representations of two scenes, and returns the minimum number of objects
    required to move to transform one scene into the other.
    
    Args:
        scene_a: A dictionary representation of the predicted scene.
        scene_b: A dictionary representation of the goal scene.

    Returns:
        Absolute and relative edit distance between input scenes.    
    """

    # Convert mixed modality inputs into clusters with semantic concepts
    scene_struct_a = GraphStruct(scene_a)
    scene_struct_b = GraphStruct(scene_b)

    # cannot calculate edit distance between scenes with unequal number of objects or clusters
    if len(scene_struct_b) != len(scene_struct_a) or \
            scene_struct_a.return_scene_objects() != scene_struct_b.return_scene_objects():

        return np.inf, 1.0

    # number of movable objs, for ged normalization
    num_movable_objects = len(scene_struct_a.return_scene_objects())

    # Calculate linear sum assignment from pairwise cluster similarity
    lsa_coords = find_lsa(scene_struct_a.to_ordered_list(),
                          scene_struct_b.to_ordered_list())

    # Objects on the table are always mapped together
    lsa_coords.append(('table', 'table'))

    # Initialize ged metric and misplaced objects dictionary
    sed = 0
    misplaced_objs = dict({})

    for i, asgn_xy in enumerate(lsa_coords):

        asgn_id_pred, asgn_id_goal = asgn_xy

        _, pred_goal_xy_unmatched = return_list_intersection(
            scene_struct_a.scene_dict[asgn_id_pred],
            scene_struct_b.scene_dict[asgn_id_goal]
        )

        misplaced_objs[i] = pred_goal_xy_unmatched

        sed += len(misplaced_objs[i])

    # constant prevents division by zero.
    sed_norm = min(1, float(sed)/(num_movable_objects + 1e-6))

    return sed, sed_norm  # prevent division by zero and cap at 1


def find_lsa(cluster_a, cluster_b):
    """Wraps around the linear sum assignment function to compute LSA for scene
        clusters.

    Args:
        cluster_a: Dictionary representation of scene A.
        cluster_b: Dictinoary representation of scene B.
    
    """

    cost_matrix = np.zeros(shape=(len(cluster_a), len(cluster_b)))

    for row_id, cluster_a_group in enumerate(cluster_a):

        for column_id, cluster_b_group in enumerate(cluster_b):

            pred_goal_xy_matching, _ = return_list_intersection(cluster_a_group,
                                                                cluster_b_group)

            # negative to convert similarity to distance
            cost_matrix[row_id, column_id] = \
                -len(pred_goal_xy_matching)

    lsa_tuple = linear_sum_assignment(cost_matrix)

    return [(x+1, y+1) for x, y in zip(lsa_tuple[0], lsa_tuple[1])]


def return_list_intersection(list_query, list_value):
    """Returns the intersection of two lists, and unmatched elements in the
        query list.

    Args:
        list_query: List of items to be matched.
        list_value: List of items to be matched against.
    """

    matched_elems = []

    unmatched_elems_query = list_query.copy()
    unmatched_elems_value = list_value.copy()

    for obj in list_query:

        if obj in unmatched_elems_value:

            # Add obj to matched list
            matched_elems.append(obj)

            # Remove obj from original lists
            unmatched_elems_query.remove(obj)
            unmatched_elems_value.remove(obj)

    return matched_elems, unmatched_elems_query
