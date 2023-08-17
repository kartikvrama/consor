'''Helper functions for performing evaluation.'''

from copy import deepcopy
from matplotlib import rcParams
import matplotlib.pyplot as plt

import numpy as np

from scipy.optimize import linear_sum_assignment


def typecase_json_keys(json_elem):

    result = dict()

    for key, value in json_elem.items():
        if key == 'table':
            result[str(key)] = value
        else:
            result[int(key)] = value

    return result


class GraphStruct:
    """Common data structure for rearrangement scene graphs."""

    def __init__(self, graph):

        self.immovable_objects = ['table', 'box']

        assert type(graph) == type(dict()), \
            f'Input graph type {type(graph)} is not a dictionary'

        # graph in json format provided
        if set(graph.keys()) == set(['objects', 'edges', 'dynamic_edge_mask', 'rule']):

            self.json_graph_to_cluster(graph)

        # receptacle:objects dict provided
        elif all([x in graph.keys() for x in ['objects', 'scene']]):

            # Assert there are no instance ids for object names
            assert all(['-' not in obj for obj in graph['objects']])

            self.graph_dict = typecase_json_keys(graph['scene'])

        else:  # cluster provided, assume no objects on the table

            raise NotImplementedError('Format of the graph provided is wrong')

    def __len__(self):

        return len(list(self.graph_dict.keys()))  # number of boxes + 1 table

    def json_graph_to_cluster(self, json_graph):
        ''' Convert json graph to cluster '''

        num_immovable_objs = 1 + max([i for i, obj in enumerate(json_graph['objects'])
                                      if obj.split('-')[0] in self.immovable_objects])

        movable_edge_ids = [i for i, eh in enumerate(json_graph['edges'][0])
                            if eh >= num_immovable_objs]  # all movable edge heads

        # Initialize final graph (Note: clusters numbered from 1)
        self.graph_dict = dict({1 + int(obj.split('-')[1]): [] for obj in json_graph['objects']
                                if obj.split('-')[0] == 'box'})
        self.graph_dict.update({'table': []})

        for edge_id in movable_edge_ids:

            edge_hd, edge_tl = json_graph['edges'][0][edge_id], json_graph['edges'][1][edge_id]

            object_name = json_graph['objects'][edge_hd].split('-')[0]

            if edge_tl == 0:  # object on table (edge id 0)

                self.graph_dict['table'].append(object_name)

            else:

                box_id = edge_tl  # edge id of box_i is i

                if box_id in self.graph_dict.keys():

                    self.graph_dict[box_id].append(object_name)

                else:

                    raise ValueError(
                        'Issue with json_graph_to_cluster [utils_eval]')

        for key in self.graph_dict:  # sort objects in each box

            self.graph_dict[key] = sorted(self.graph_dict[key])

    def movable_objects(self):
        ''' Returns a sorted list of all movable objects in the scene '''

        assert all([obj not in self.immovable_objects
                    for value in self.graph_dict.values() for obj in value])

        object_list = [obj for value in self.graph_dict.values()
                       for obj in value]

        return sorted(object_list)

    def to_set(self):
        ''' Return a partition of objects grouped by their box receptacle, 
                ordering boxes by number and EXCLUDING objects on the table '''

        return [list(self.graph_dict[k]) for k in range(1, len(self.graph_dict.keys()))]

    def display(self):

        return display_cluster(self.graph_dict)


def calculate_ged_lsa(graph_dict_pred, graph_dict_goal, debug=False):

    # Convert mixed modality inputs into clusters with semantic concepts
    graph_struct_pred = GraphStruct(graph_dict_pred)
    graph_struct_goal = GraphStruct(graph_dict_goal)

    # cannot calculate edit distance between scenes with unequal number of objects or clusters
    if len(graph_struct_goal) != len(graph_struct_pred) or \
            graph_struct_pred.movable_objects() != graph_struct_goal.movable_objects():

        return np.inf, 1.0

    # number of movable objs, for ged normalization
    num_movable_objects = len(graph_struct_pred.movable_objects())

    # Calculate linear sum assignment from pairwise cluster similarity
    lsa_coords = find_lsa(graph_struct_pred.to_set(),
                          graph_struct_goal.to_set())

    # Objects on the table are always mapped together
    # TODO: extend to other receptacle categories
    lsa_coords.append(('table', 'table'))

    # Initialize ged metric and misplaced objects dictionary
    ged = 0
    misplaced_objs = dict({})

    for i, asgn_xy in enumerate(lsa_coords):

        asgn_id_pred, asgn_id_goal = asgn_xy

        _, pred_goal_xy_unmatched = return_list_intersection(
            graph_struct_pred.graph_dict[asgn_id_pred],
            graph_struct_goal.graph_dict[asgn_id_goal]
        )

        misplaced_objs[i] = pred_goal_xy_unmatched

        ged += len(misplaced_objs[i])

    if debug:

        print('Input 1 (default=predicted)')
        display_cluster(graph_struct_pred.graph_dict)

        print('Input 2 (default=goal)')
        display_cluster(graph_struct_goal.graph_dict)

        print('Number of table objects in pred-{} and goal-{}'
              .format(len(graph_struct_pred.graph_dict['table']),
                      len(graph_struct_goal.graph_dict['table'])))

        print('Misplaced objects:')

        for asgn_id, key in enumerate(misplaced_objs.keys()):

            asgn_id_pred, asgn_id_goal = lsa_coords[asgn_id]
            print(
                f'Mapping:{asgn_id_pred}-{asgn_id_goal} | {key}:{misplaced_objs[key]}')

        input('utils_eval line 152')

    ged_norm = min(1, float(ged)/(num_movable_objects+1e-6))

    return ged, ged_norm  # prevent division by zero and cap at 1


def find_lsa(cluster_pred, cluster_goal):
    '''
        Find the linear sum assignment between two clusters (list of lists) 
            that maximizes the semantic similarity overlap
    '''

    # TODO: if clusters have semantic labels, how to perform mixed lsa (some clusters have hard mappings to each other, others dont)

    cost_matrix = np.zeros(shape=(len(cluster_pred),
                                  len(cluster_goal)))

    for row_id, cluster_pred_group in enumerate(cluster_pred):

        for column_id, cluster_goal_group in enumerate(cluster_goal):

            pred_goal_xy_matching, _ = return_list_intersection(cluster_pred_group,
                                                                cluster_goal_group)

            # negative to convert similarity to distance
            cost_matrix[row_id, column_id] = \
                -len(pred_goal_xy_matching)

    # #DEBUG
    # print(f'cost matrix:\n{cost_matrix}\n---')

    lsa_tuple = linear_sum_assignment(cost_matrix)

    return [(x+1, y+1) for x, y in zip(lsa_tuple[0], lsa_tuple[1])]


def return_list_intersection(list_query, list_value):
    '''
        Returns matched and unmatched elements in query
    '''

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
