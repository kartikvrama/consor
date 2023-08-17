"""Helper functions for loading and visualizing semantic rearrangement data."""

import logging
import platform

import numpy as np
import matplotlib.pyplot as plt

from nltk.corpus import wordnet as wn


class ObjectStruct: # struct of an object

    def __init__(self, object_dict):

        self.name = object_dict['conceptnet_name']

        self.wn_str = object_dict['wordnet_name']
        self.wn_syn = wn.synset(object_dict['wordnet_name'])

        self.activity_label = object_dict['activity']
        self.aff_label = object_dict['affordance']

        self.flag_stackable = True if object_dict['stackable'] == 'y' else False

    def __eq__(self, other):
        ''' Checks if two objects have the same conceptnet and wordnet name '''

        # print("__eq__ called: %r == %r" % (self, other))

        if isinstance(other, ObjectStruct):
            return self.name == other.name and self.wn_str == other.wn_str

        return False


def return_conceptnet_path():

    if platform.system() == 'Linux':
        conceptnet_path = '/home/kartik/Documents/datasets/numberbatch-en.txt' 

    elif platform.system() == 'Windows':
        conceptnet_path = 'C:/Users/karti/Documents/datasets/numberbatch-en.txt' 

    else:
        print(f'Unfamiliar OS named {platform.system()}, please add to the list')
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

    plt.setp(
        ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")

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
    plt.savefig('{}.png'.format(filename))


def display_cluster(cluster_dict):

    for cluster_id, cluster_stuff in cluster_dict.items():

        print(f'{cluster_id}:{cluster_stuff}')

    print('----')


def display_data(json_data):
    objects = json_data['objects']
    edges = json_data['edges']
    relations = json_data['relations']

    immovable_objects = ['table', 'box'] #TODO: get from dyn edge mask

    print('---')

    cluster = dict({obj:[] for obj in objects if obj.split('-')[0] in immovable_objects})

    for i in range(len(edges[0])):

        # print('{} {} {}'.format(objects[edges[0][i]], relations[i], 
        #                         objects[edges[1][i]]))

        head, tail = objects[edges[0][i]], objects[edges[1][i]]

        if head.split('-')[0] != 'box':

            if tail in cluster:

                cluster[tail].append(head.split('-')[0])

            else:

                cluster[tail] = [head.split('-')[0]]

    display_cluster(cluster)