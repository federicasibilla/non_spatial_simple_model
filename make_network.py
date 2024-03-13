"""
File for creating the network, consumer preference matrix and metabolic matrix

Contains:
    - make_net_mat: function to create the graph and its visualization, also traduces it into consumer preferences matrix, toxins matrix, combined matrix, and creates random metabolic matrix


"""

import networkx as nx
import matplotlib.pyplot as plt
from networkx.drawing.nx_agraph import to_agraph
from IPython.display import Image, display
from PIL import Image
import numpy as np
import pandas as pd
import itertools
import io
import os

# make_net_mat: function to create the graph and its visualization, also traduces it into consumer preferences matrix, toxins matrix, combined matrix, and creates random metabolic matrix
# takes       : species_dict, dictionary with all species and their name/function; nutrients_dict, dictionary with all the nutrients and the edges they form between species, toxins_dict, dictionary with all the toxins
# returns     : other than making an image of the network, returns the consumer preference matrix, the metabolic matrix, the inhibition matrix and the combined consumer and inhibition matrix

def make_net_mat(species_dict, nutrients_dict, toxins_dict):

    # change dictionaries in feeding graph
    edges = []
    labels = []
    for nutrient, relationships in nutrients_dict.items():
        nutrient_attr = nutrient
        connections = relationships[0].split(',')
        cij = relationships[1:]
        for idx, connection in enumerate(connections):
            source = connection[0]
            targets = connection[1:]
            weight = cij[idx]
            for target in targets:
                edge = (source, target, weight)
                attr = nutrient_attr
                edges.append(edge)
                labels.append(attr)
    # create directed graph with weighted edges
    G = nx.DiGraph()
    G.add_nodes_from(species_dict)
    for i, (source, target, _) in enumerate(edges):
        G.add_edge(source, target, weight=edges[i][2], label=labels[i])
    # visualize graph
    agraph = to_agraph(G)
    for u, v, data in G.edges(data=True):
        agraph.add_edge(u, v, label=data['label'], penwidth=data['weight'] * 3)
    agraph.layout(prog='neato')
    img = agraph.draw(format='png')
    with open('network_image.png', 'wb') as f:
        f.write(img)

    # repeat for toxins network
    edges_t = []
    labels_t = []
    for toxin, relationships in toxins_dict.items():
        toxin_attr = toxin
        connections = relationships[0].split(',')
        cij = relationships[1:]
        for idx, connection in enumerate(connections):
            source = connection[0]
            targets = connection[1:]
            weight = cij[idx]
            for target in targets:
                edge = (source, target, weight)
                attr = toxin_attr
                edges_t.append(edge)
                labels_t.append(attr)
    # create directed graph with weighted edges
    G_T = nx.DiGraph()
    G_T.add_nodes_from(species_dict)
    for i, (source, target, _) in enumerate(edges_t):
        G_T.add_edge(source, target, weight=edges[i][2], label=labels_t[i])
    # visualize graph
    agraph = to_agraph(G_T)
    for u, v, data in G_T.edges(data=True):
        agraph.add_edge(u, v, label=data['label'], penwidth=data['weight'] * 3)
    agraph.layout(prog='neato')
    img = agraph.draw(format='png')
    with open('inh_network_image.png', 'wb') as f:
        f.write(img)
    
    # create consumer preference matrix from feeding graph
    species = [node for node in G.nodes()]
    nutrients = sorted(set(edge for _, _, edge in G.edges(data='label')))

    consumer_preference_matrix = pd.DataFrame(0, index=species, columns=nutrients)
    
    for source, target, attr in G.edges(data=True):
        if attr['label'] in nutrients:
            consumer_preference_matrix.loc[target, attr['label']] = attr['weight']

    # save consumer preferences matrix matrix
    consumer_preference_matrix.to_csv('/Users/federicasibilla/Documenti/Tesi/Codice/non_spatial_simple_model/consumer_preference_matrix.csv', index=True, mode='w')

    # create inhibition matrix from toxins graph
    species = [node for node in G_T.nodes()]
    toxins = sorted(set(edge for _, _, edge in G_T.edges(data='label')))

    inhibition_matrix = pd.DataFrame(0, index=species, columns=toxins)
    
    for source, target, attr in G_T.edges(data=True):
        if attr['label'] in toxins:
            inhibition_matrix.loc[target, attr['label']] = attr['weight']

    # save inhibition matrix
    inhibition_matrix.to_csv('/Users/federicasibilla/Documenti/Tesi/Codice/non_spatial_simple_model/inhibition_matrix.csv', index=True, mode='w')   

    # unite nutrients and toxins
    chemicals = nutrients+toxins 
    G_combined = nx.DiGraph()
    G_combined.add_nodes_from(species_dict)
    G_combined.add_edges_from(G.edges(data=True))
    G_combined.add_edges_from(G_T.edges(data=True))

    # create combined consumer and inhibition matrix
    species = [node for node in G_combined.nodes()]

    combined_matrix = pd.DataFrame(0, index=species, columns=chemicals)
    
    for source, target, attr in G_combined.edges(data=True):
        if attr['label'] in chemicals:
            combined_matrix.loc[target, attr['label']] = attr['weight']

    # save combined matrix
    combined_matrix.to_csv('/Users/federicasibilla/Documenti/Tesi/Codice/non_spatial_simple_model/total_matrix.csv', index=True, mode='w')

    # create metabolic matrix (production rates are random and non zero when there is a connection in the graph)
    metabolic_matrix = pd.DataFrame(0, index=chemicals, columns=chemicals)

    #add 1 if there exists and element that eats nutrient column and secretes nutrient row
    for edge in G_combined.edges(data=True):
        out_substance = edge[2]['label']
        source = edge[0]
        in_substance = [ed[2]['label'] for ed in G_combined.in_edges(source,data=True)]
        for in_nutrient in in_substance:
            metabolic_matrix.at[in_nutrient, out_substance]=1

    rn = np.random.rand(len(chemicals), len(chemicals))
    rn_metabolic_matrix = metabolic_matrix*rn
 
    nor_metabolic_matrix = rn_metabolic_matrix.T/(np.sum(rn_metabolic_matrix, axis=1)+1E-8)

    # save metabolic matrix
    nor_metabolic_matrix.to_csv('/Users/federicasibilla/Documenti/Tesi/Codice/non_spatial_simple_model/metabolic_matrix.csv', index=True, mode='w')

    return consumer_preference_matrix, inhibition_matrix, nor_metabolic_matrix, combined_matrix



