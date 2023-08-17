"""Rules for generating object clusters for each schema."""

import semantic2thor
from sklearn.cluster import AgglomerativeClustering
from nltk.corpus import wordnet as wn
import numpy as np

import nltk
nltk.download('wordnet')
nltk.download('omw-1.4')


def return_clustering_func(rule):
    """Returns the corresponding clustering function for each schema from a string label.

    Args:
        rule: Schema name.
    """

    clustering_func_dict = {'class': clusterby_class,
                            'ooe': clusterby_OOE,
                            'utility': clusterby_utility,
                            'affordance': clusterby_affordance}

    if rule not in clustering_func_dict:
        return NotImplementedError(f'{rule} is an unrecognized organizational schema.')

    return clustering_func_dict[rule]


def shuffle_cluster_keys(object_grouping_dict):
    """Shuffles the indices of receptalces in a scene, with indices starting from 1 

    Args:
        object_grouping_dict: Dictionary representing where objects are placed in the scene.
    """

    if object_grouping_dict is None:
        return None

    # New cluster indices starting indexing from 1.
    # excluding table.
    grouping_indices = np.arange(
        1, len(object_grouping_dict.keys()), dtype=np.int32)
    np.random.shuffle(grouping_indices)

    object_grouping_dict_modified = dict({})

    count = 0
    for key, value in object_grouping_dict.items():

        if key == 'table':
            object_grouping_dict_modified['table'] = value

        else:  # changing [key] to grouping_indices[count].
            object_grouping_dict_modified[grouping_indices[count]] = value
            count += 1

    return object_grouping_dict_modified


def clusterby_class(object_names, debug=False):
    """Clusters objects based on wu-palmer similarity calculated from the WordNet ontology.  
    """

    object_synsets = [wn.synset(semantic2thor.load(
        obj)['Wordnet Name']) for obj in object_names]

    num_objects = len(object_names)

    # Create distance matrix with (1 - wup) as distance metric.
    def distance_metric(x, y): return 1.0 - x.wup_similarity(y)
    pairwise_distance_matrix = np.array([distance_metric(obj_i, obj_k)
                                         for obj_i in object_synsets
                                         for obj_k in object_synsets])\
        .reshape(num_objects, num_objects)

    if debug:
        print('class')
        print(object_names)
        print(pairwise_distance_matrix)

    clusters = AgglomerativeClustering(n_clusters=None, metric="precomputed", linkage="single",
                                       distance_threshold=0.3).fit_predict(pairwise_distance_matrix)

    num_clusters = max(clusters) + 1

    if num_clusters == 1:
        return None  # deny examples with single cluster.

    object_grouping_dict = dict({i: [] for i in range(num_clusters)})
    object_grouping_dict.update({'table': []})  # add table to the scene.

    for c, o in zip(clusters, object_names):
        object_grouping_dict[c].append(o)

    for key in object_grouping_dict.keys():  # sort elements withing group.
        object_grouping_dict[key] = sorted(object_grouping_dict[key])

    return shuffle_cluster_keys(object_grouping_dict)


def return_utility_similarity(path1, path2):
    """Calculates the wu-palmer similarity between two objects using the Walmart ontology.

    Objects in the Walmart ontology have multiple parents of varying depths from the root. Therefore, there are multiple similarity measures between any two nodes in the ontology. This function ensures that the deepest common parent is used to calculate wu-palmer similarity.  

    Args:
        path1: List of possible paths from the root node to child node for object 1.
        path2: List of possible paths from the root node to child node for object 2.
    """

    def path2depth(x): return [dict(
        {parent: i+1 for i, parent in enumerate(xchild)}) for xchild in x]

    path_dict_1 = path2depth(path1)
    path_dict_2 = path2depth(path2)

    max_similarity = -np.inf

    for elem_path1 in path_dict_1:
        for elem_path2 in path_dict_2:

            if elem_path1 == elem_path2:

                wup_similarity = 1.0

            else:

                object1_depth = max(elem_path1.values()) + 1
                object2_depth = max(elem_path2.values()) + 1

                common_parents = [
                    key for key in elem_path1.keys() if key in elem_path2.keys()]

                def temp_func(x): return elem_path1[x]

                # choose lcs with most depth.
                lcs_name = max(common_parents, key=temp_func)
                lcs_depth = temp_func(lcs_name)

                wup_similarity = 2*lcs_depth/(object1_depth + object2_depth)

            if wup_similarity > max_similarity:

                max_similarity = wup_similarity

    return max_similarity


def clusterby_utility(object_names, debug=False):
    """Clusters objects based on the wu-palmer similarity calculated from the object's location in Walmart.  
    """

    num_objects = len(object_names)

    object_utility_paths = [semantic2thor.walmart(obj) for obj in object_names]

    # Create distance matrix with (1 - wup) as distance metric.
    def distance_metric(x, y): return 1.0 - return_utility_similarity(x, y)
    pairwise_distance_matrix = np.array([distance_metric(path_i, path_k)
                                         for path_i in object_utility_paths
                                         for path_k in object_utility_paths])\
        .reshape(num_objects, num_objects)

    if debug:
        print('utility')
        print(object_names)
        print(pairwise_distance_matrix)

    clusters = AgglomerativeClustering(n_clusters=None, metric="precomputed", linkage="single",
                                       distance_threshold=0.59).fit_predict(pairwise_distance_matrix)

    num_clusters = max(clusters) + 1

    if num_clusters == 1:
        return None  # deny examples with single cluster.

    object_grouping_dict = dict({i: [] for i in range(num_clusters)})
    object_grouping_dict.update({'table': []})  # add table to the scene.

    for c, o in zip(clusters, object_names):
        object_grouping_dict[c].append(o)

    for key in object_grouping_dict.keys():  # sort elements withing group.
        object_grouping_dict[key] = sorted(object_grouping_dict[key])

    return shuffle_cluster_keys(object_grouping_dict)


def clusterby_affordance(object_names, debug=False):
    """Clusters objects based on distances between vector representations of affordance labels.

    Each object (with one or more affordance labels) is represented using a binary vector representation, and a threshold on the jaccard distance between vectors is applied to cluster objects. 
    """

    num_objects = len(object_names)

    def _distance_metric(obj_i, obj_k):

        pca_obj_i = semantic2thor.affordances(obj_i)[1]
        pca_obj_k = semantic2thor.affordances(obj_k)[1]

        if all(pca_obj_i == pca_obj_k):

            return 0

        assert all([e == 0 or e == 1 for e in pca_obj_i]) and \
            all([e == 0 or e == 1 for e in pca_obj_k]
                ), 'Affordance vectors are not binary!'

        dot_prod = np.dot(pca_obj_i, pca_obj_k)

        # jaccard distance = 1 - jaccard similarity = 1 - (a n b)/(a U b).
        return 1 - dot_prod/(sum(pca_obj_i**2) + sum(pca_obj_k**2) - dot_prod)

    pairwise_distance_matrix = np.array([_distance_metric(obj_i, obj_k)
                                         for obj_i in object_names
                                         for obj_k in object_names])\
        .reshape(num_objects, num_objects)

    if debug:
        for obj in set(object_names):
            obj_affs = semantic2thor.affordances(obj)[0]
            obj_affs = [key for key, value in obj_affs.items() if value]
            print('{}:{}'.format(obj, obj_affs))
        print(object_names)
        print(np.round(pairwise_distance_matrix, 2))

    clusters = AgglomerativeClustering(n_clusters=None, metric="precomputed", linkage="single",
                                       distance_threshold=0.4).fit_predict(pairwise_distance_matrix)

    num_clusters = max(clusters) + 1

    if num_clusters == 1:
        return None  # deny examples with single cluster..

    object_grouping_dict = dict({i: [] for i in range(num_clusters)})
    object_grouping_dict.update({'table': []})  # add table to the scene.

    for c, o in zip(clusters, object_names):
        object_grouping_dict[c].append(o)

    for key in object_grouping_dict.keys():  # sort elements withing group.
        object_grouping_dict[key] = sorted(object_grouping_dict[key])

    return shuffle_cluster_keys(object_grouping_dict)


def clusterby_OOE(object_names, debug=False):
    """Clusters objects such that each cluster has no more than one object type.
    """

    if debug:
        for obj in set(object_names):
            print(obj)

    # List of unique objects.
    object_names_set = set(object_names)

    # Each object's count.
    object_count = {u: 0 for u in object_names_set}

    for u in object_names:
        object_count[u] += 1

    # OOE: clusters are defined by number of instances.
    num_clusters = max(list(object_count.values()))

    object_grouping_dict = {i: [] for i in range(num_clusters)}
    object_grouping_dict.update({'table': []})  # add table to the scene.

    # Add each object type to the max number of clusters.
    for u in object_names_set:

        for c in range(object_count[u]):

            object_grouping_dict[c].append(u)

    for key in object_grouping_dict:  # sort elements withing group.
        object_grouping_dict[key] = sorted(object_grouping_dict[key])

    return shuffle_cluster_keys(object_grouping_dict)
