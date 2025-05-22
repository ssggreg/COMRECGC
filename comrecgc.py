import os
import torch
import torch_geometric.utils as torch_utils
import networkx as nx
import random
import pickle
from torch.masked import masked_tensor, as_masked_tensor
from tqdm import tqdm
import argparse
import os.path
import torch
import numpy as np
import torch_geometric as tg
from torch_geometric.data import Data, Batch, DataLoader
from torch_geometric.transforms import RemoveIsolatedNodes, ToUndirected
from torch_geometric.utils.convert import to_networkx, from_networkx
import copy

import util
import distance
from data import load_dataset # loads datasets
from gnn import load_trained_gnn, load_trained_prediction


##Global variables
    
MAX_COUNTERFACTUAL_SIZE = 0

graph_map = {}  # graph_hash -> {edge_index, x}
graph_index_map = {}  # graph hash -> index in counterfactual_graphs
counterfactual_candidates = []  # [{frequency: int, graph_hash: str, importance_parts: tuple, input_graphs_covering_indexes: set}]
input_graphs_covered = []  # [int] with of number of input graphs
covering_graphs = set()  # dictionary graph hash which is in first #number input graph counterfactual list (i.e., contributing input_graph_covered)
transitions = {}  # graph_hash -> {transitions ([hashes], [actions], [importance_parts], tensor(input_graph_covering_for_all_neighbours))}
start = {} #graph from which the random walk starts
is_sample = ''
starting_step = 1
traversed_hashes = []  # list of traversed graph hashes
sample_size = 0

# Hardware and hyperparameters setup

def get_args():
    parser = argparse.ArgumentParser(description='Graph Global Counterfactual Summary')
    parser.add_argument('--dataset', type=str, default='mutagenicity', choices=['mutagenicity', 'aids', 'nci1', 'proteins'])
    parser.add_argument('--theta', type=float, default=0.05, help='distance threshold value during training.')
    parser.add_argument('--teleport', type=float, default=0.1, help='teleport probability to input graphs')
    parser.add_argument('--steps', type=int, default=50000, help='random walk step size')
    parser.add_argument('--heads', type=int, default=5, help='number of heads')
    parser.add_argument('--k', type=int, default=100000, help='number of graphs will be selected from counterfactuals')
    parser.add_argument('--device1', type=str, help='Cuda device or cpu for gnn model', default='0')
    parser.add_argument('--device2', type=str, help='Cuda device or cpu for neurosed model', default='0')
    parser.add_argument('--sample_size', type=int, help='Sample count for neighbour graphs', default=10000)
    parser.add_argument('--sample', action='store_true')
    return parser.parse_args()

def prepare_devices(device1, device2):
    device1 = 'cuda:' + device1 if torch.cuda.is_available() and device1 in ['0', '1', '2', '3'] else 'cpu'
    device2 = 'cuda:' + device2 if torch.cuda.is_available() and device2 in ['0', '1', '2', '3'] else 'cpu'

    return device1, device2


### Graphs operations


def calculate_hash(graph_embedding):
    if isinstance(graph_embedding, (np.ndarray,)):
        return hash(graph_embedding.tobytes())
    else:
        raise Exception('graph_embedding should be ndarray')


def node_label_change(graph):
    neighbor_graphs_actions = []
    neighbor_graphs = []
    for i in range(graph.x.shape[0]):
        for j in range(graph.x.shape[1]):
            # if graph['node_labels'][i] != j:
            if graph.x[i, j] != 1:
                neighbor_graph_action = ('NLC', i, j)
                neighbor_graphs_actions.append(neighbor_graph_action)
                neighbor_graphs.append(neighbor_graph_access(graph, neighbor_graph_action))
    return neighbor_graphs_actions, neighbor_graphs


def node_addition(graph):
    neighbor_graphs_actions = []
    neighbor_graphs = []
    for i in range(graph.x.shape[0]):
        for j in range(graph.x.shape[1]):  # Add a new node with label j connected with node i.
            neighbor_graph_action = ('NA', i, j)
            neighbor_graphs_actions.append(neighbor_graph_action)
            neighbor_graphs.append(neighbor_graph_access(graph, neighbor_graph_action))
    return neighbor_graphs_actions, neighbor_graphs


def isolated_node_addition(graph):
    neighbor_graphs_actions = []
    neighbor_graphs = []
    for j in range(graph.x.shape[1]):  # Add a new isolated node with label j
        neighbor_graph_action = ('INA', j, j)
        neighbor_graphs_actions.append(neighbor_graph_action)
        neighbor_graphs.append(neighbor_graph_access(graph, neighbor_graph_action))
    return neighbor_graphs_actions, neighbor_graphs


def node_removal(graph):
    degree = torch_utils.degree(graph.edge_index[0], num_nodes=graph.num_nodes)
    neighbor_graphs_actions = []
    neighbor_graphs = []
    for i in range(len(degree)):
        if degree[i] == 1:  # Remove nodes with exactly one edge only.
            neighbor_graph_action = ('NR', i, i)
            neighbor_graphs_actions.append(neighbor_graph_action)
            neighbor_graphs.append(neighbor_graph_access(graph, neighbor_graph_action))
    return neighbor_graphs_actions, neighbor_graphs


def isolated_node_removal(graph):
    degree = torch_utils.degree(graph.edge_index[0], num_nodes=graph.num_nodes)
    neighbor_graphs_actions = []
    neighbor_graphs = []
    for i in range(len(degree)):
        if degree[i] == 0:  # Remove isolated nodes only.
            neighbor_graph_action = ('INR', i, i)
            neighbor_graphs_actions.append(neighbor_graph_action)
            neighbor_graphs.append(neighbor_graph_access(graph, neighbor_graph_action))
    return neighbor_graphs_actions, neighbor_graphs


def edge_change(graph, keep_bridge=True, only_removal=False):
    nxg = torch_utils.to_networkx(graph, to_undirected=True)  # 157 µs ± 71.9 µs per loop
    bridges = set(nx.bridges(nxg)) if keep_bridge else set()  # 556 µs ± 31.2 µs per loop
    num_nodes = graph.x.shape[0]
    neighbor_graphs_actions = []
    neighbor_graphs = []
    for i in range(num_nodes):
        for j in range(i + 1, num_nodes):
            if nxg.has_edge(i, j):
                if keep_bridge and (i, j) not in bridges:  # edge exist and its removal does not disconnect the graph
                    neighbor_graph_action = ('ER', i, j)
                else:  # remove edge regardlessly
                    neighbor_graph_action = ('ERR', i, j)
                neighbor_graphs_actions.append(neighbor_graph_action)
                neighbor_graphs.append(neighbor_graph_access(graph, neighbor_graph_action))
            elif not nxg.has_edge(i, j) and not only_removal:  # add edges
                neighbor_graph_action = ('EA', i, j)
                neighbor_graphs_actions.append(neighbor_graph_action)
                neighbor_graphs.append(neighbor_graph_access(graph, neighbor_graph_action))
    return neighbor_graphs_actions, neighbor_graphs


def neighbor_graph_access(graph, neighbor_graph_action):
    neighbor_graph = graph.clone()
    action = neighbor_graph_action[0]
    if action == 'NOTHING':
        neighbor_graph = neighbor_graph
    elif action == 'NLC':  # node label change
        _, i, j = neighbor_graph_action
        neighbor_graph.x[i] = 0  # 6.93 µs ± 301 ns per loop
        neighbor_graph.x[i][j] = 1  # 7.9 µs ± 420 ns per loop
    elif action == 'NA':  # node addition
        _, i, j = neighbor_graph_action
        neighbor_graph.num_nodes += 1
        neighbor_graph.edge_index = torch.hstack([graph.edge_index, torch.tensor([[i, graph.num_nodes], [graph.num_nodes, i]])])  # 14.1 µs ± 57.3 ns per loop, 3 times faster than padder.
        neighbor_graph.x = torch.vstack([graph.x, torch.nn.functional.one_hot(torch.tensor(j), graph.x.shape[1])])  # 36.8 µs ± 340 ns per loop, similar to padder.
    elif action == 'INA':  # isolated node addition.
        _, i, j = neighbor_graph_action
        neighbor_graph.num_nodes += 1
        neighbor_graph.x = torch.vstack([graph.x, torch.nn.functional.one_hot(torch.tensor(j), graph.x.shape[1])])  # 36.8 µs ± 340 ns per loop, similar to padder.
    elif action in ('NR', 'INR'):  # (isolated) node removal
        _, i, j = neighbor_graph_action
        indices = torch.LongTensor(list(range(i)) + list(range(i + 1, graph.num_nodes)))  # 4.93 µs ± 244 ns per loop
        neighbor_graph.num_nodes -= 1
        neighbor_graph.edge_index = torch_utils.subgraph(indices, graph.edge_index, relabel_nodes=True, num_nodes=graph.num_nodes)[0]  # 80.5 µs ± 992 ns per loop
        neighbor_graph.x = neighbor_graph.x[indices]  # 7.44 µs ± 149 ns per loop
    elif action in ('ER', 'ERR'):  # edge removal (regardlessly)
        _, i, j = neighbor_graph_action
        neighbor_graph.edge_index = graph.edge_index[:, ~((graph.edge_index[0] == i) & (graph.edge_index[1] == j) | (graph.edge_index[0] == j) & (graph.edge_index[1] == i))]  # 78.9 µs ± 1.87 µs per loop
    elif action == 'EA':  # edge addition
        _, i, j = neighbor_graph_action
        neighbor_graph.edge_index = torch.hstack([graph.edge_index, torch.tensor([[i, j], [j, i]])])  # 14 µs ± 262 ns per loop
    else:
        raise NotImplementedError(f'Neighbor edit action {action} not supported. ')
    return neighbor_graph


### Maintaining the set of graph explored
    


def is_counterfactual_array_full():
    return len(counterfactual_candidates) >= MAX_COUNTERFACTUAL_SIZE


def get_minimum_frequency():
    return counterfactual_candidates[-1]['frequency']


def is_graph_counterfactual(graph_hash):
    return counterfactual_candidates[graph_index_map[graph_hash]]['importance_parts'][0] >= 0.5


def reorder_counterfactual_candidates(start_idx):
    swap_idx = start_idx - 1
    while swap_idx >= 0 and counterfactual_candidates[start_idx]['frequency'] > counterfactual_candidates[swap_idx]['frequency']:
        swap_idx -= 1
    swap_idx += 1
    if swap_idx < start_idx:
        graph_index_map[counterfactual_candidates[start_idx]['graph_hash']] = swap_idx
        graph_index_map[counterfactual_candidates[swap_idx]['graph_hash']] = start_idx
        counterfactual_candidates[start_idx], counterfactual_candidates[swap_idx] = counterfactual_candidates[swap_idx], counterfactual_candidates[start_idx]
    return swap_idx


def update_input_graphs_covered(add_graph_covering_list=None, remove_graph_covering_list=None):
    global input_graphs_covered
    if add_graph_covering_list is not None:
        input_graphs_covered += add_graph_covering_list
    if remove_graph_covering_list is not None:
        input_graphs_covered -= remove_graph_covering_list


def check_reinforcement_condition(graph_hash):
    return is_graph_counterfactual(graph_hash)


def populate_counterfactual_candidates(graph_hash, importance_parts, input_graphs_covering_list = None, bypass_size = False):
    is_new_graph = False
    if graph_hash in graph_index_map:
        graph_idx = graph_index_map[graph_hash]
        condition = check_reinforcement_condition(graph_hash)
        if condition:
            counterfactual_candidates[graph_idx]['frequency'] += 1
            swap_idx = reorder_counterfactual_candidates(graph_idx)
        else:
            swap_idx = graph_idx
    else:
        is_new_graph = True
        if is_counterfactual_array_full() and not bypass_size:
            deleting_graph_hash = counterfactual_candidates[-1]['graph_hash']
            del graph_index_map[deleting_graph_hash]
            del graph_map[deleting_graph_hash]
            if deleting_graph_hash in transitions:
                del transitions[deleting_graph_hash]
            counterfactual_candidates[-1] = {
                "frequency": get_minimum_frequency() + 1,
                "graph_hash": graph_hash,
                "importance_parts": importance_parts,
                "input_graphs_covering_list": input_graphs_covering_list
            }
        else:
            counterfactual_candidates.append({
                'frequency': 2,
                'graph_hash': graph_hash,
                "importance_parts": importance_parts,
                "input_graphs_covering_list": input_graphs_covering_list
            })
        graph_idx = len(counterfactual_candidates) - 1
        graph_index_map[graph_hash] = graph_idx
        swap_idx = reorder_counterfactual_candidates(graph_idx)

    # updating input_graphs_covered entries
    if swap_idx == graph_idx:  # no swap
        if is_new_graph and graph_idx < len(input_graphs_covered) and is_graph_counterfactual(graph_hash):
            update_input_graphs_covered(add_graph_covering_list=input_graphs_covering_list)
            covering_graphs.add(graph_hash)
    else:  # swapped graph_idx position has swapped graph now
        swapped_graph = counterfactual_candidates[graph_idx]
        if is_graph_counterfactual(swapped_graph['graph_hash']) and graph_idx >= len(input_graphs_covered) > swap_idx:
            update_input_graphs_covered(remove_graph_covering_list=swapped_graph['input_graphs_covering_list'])
            covering_graphs.remove(swapped_graph['graph_hash'])
        if is_new_graph:
            if is_graph_counterfactual(graph_hash) and swap_idx < len(input_graphs_covered):
                update_input_graphs_covered(add_graph_covering_list=input_graphs_covering_list)
                covering_graphs.add(graph_hash)
        else:
            if is_graph_counterfactual(graph_hash) and swap_idx < len(input_graphs_covered) <= graph_idx:
                update_input_graphs_covered(add_graph_covering_list=input_graphs_covering_list)
                covering_graphs.add(graph_hash)


### Core random walk part

def move_from_known_graph(hashes, importances, importance_args): 
    probabilities = []
    importance_values = [importances[j][0] for j in range(len(hashes))] # importances contains the probabilities of being counterfactuals

    for i, hash_i in enumerate(hashes):
        importance_value = importance_values[i]

        if hash_i in graph_index_map:  # and is_graph_counterfactual(hash_i):  # reinforcing only seen counterfactuals
            frequency = counterfactual_candidates[graph_index_map[hash_i]]['frequency']
        else:
            frequency = get_minimum_frequency() if is_counterfactual_array_full() else 1
        probabilities.append(importance_value * frequency)

    if sum(probabilities) == 0:  # if probability values are all 0, we assign equal probs to all transitions
        probabilities = np.ones(len(probabilities)) / len(probabilities)
    else:
        probabilities = np.array(probabilities) / sum(probabilities)
    selected_hash_idx = random.choices(range(len(hashes)), weights=probabilities)[0]
    return selected_hash_idx


def move_to_next_graph(graphs_hash, start_graphs_hash, importance_args, teleport_probability): #### Picks a lead head then moves along
    not_teleport = False
    if random.uniform(0, 1) < teleport_probability:  # teleport to start
        return None, not not_teleport, None, None, None
    else:
        start_map_data = [graph_map[hash] for hash in start_graphs_hash]
        for graph_hash in graphs_hash:
          graph, _, _ = graph_map[graph_hash]
          if graph_hash not in transitions:
            neighbor_graphs_actions_edge_change, neighbor_graphs_edge_change = edge_change(graph, keep_bridge=True, only_removal=False)  # still n nodes
            neighbor_graphs_actions_node_label_change, neighbor_graphs_node_label_change = node_label_change(graph)  # still n nodes
            neighbor_graphs_actions_node_addition, neighbor_graphs_node_addition = node_addition(graph)  # n+1 nodes
            neighbor_graphs_actions_node_removal, neighbor_graphs_node_removal = node_removal(graph)  # n-1 nodes

            neighbor_graphs_actions = neighbor_graphs_actions_edge_change + neighbor_graphs_actions_node_label_change + neighbor_graphs_actions_node_addition + neighbor_graphs_actions_node_removal
            all_graph_set = neighbor_graphs_edge_change + neighbor_graphs_node_label_change + neighbor_graphs_node_addition + neighbor_graphs_node_removal

            if sample_size < len(neighbor_graphs_actions) and is_sample:
                samples = random.sample(range(len(neighbor_graphs_actions)), sample_size)
                neighbor_graphs_actions = [neighbor_graphs_actions[sample] for sample in samples]
                all_graph_set = [all_graph_set[sample] for sample in samples]

            # neighbor_graphs_importance_parts, neighbor_graphs_embeddings, neighbor_graphs_coverage_matrix = call(all_graph_set, importance_args)
            neighbor_graphs_importance_parts, neighbor_graphs_embeddings = call(all_graph_set, importance_args)

            target_graphs_set = set()
            target_graphs_hashes = []
            target_graphs_actions = []
            target_graphs_importance_parts = []
            needed_i = []
            target_graphs_embedding = []

            target_graphs = []

            for i in range(len(neighbor_graphs_embeddings)):
                graph_neighbour_hash = calculate_hash(neighbor_graphs_embeddings[i])
                graph_neighbour_embedding = neighbor_graphs_embeddings[i]
                if graph_neighbour_hash not in target_graphs_set:
                    needed_i.append(i)
                    target_graphs_embedding.append(neighbor_graphs_embeddings[i])
                    target_graphs_importance_parts.append(neighbor_graphs_importance_parts[i])
                    target_graphs_hashes.append(graph_neighbour_hash)
                    target_graphs_set.add(graph_neighbour_hash)
                    target_graphs_actions.append(neighbor_graphs_actions[i])

                    target_graphs.append(all_graph_set[i])

            assert len(graph_map) == len(graph_index_map)
            transitions[graph_hash] = (target_graphs_hashes, target_graphs, target_graphs_importance_parts, target_graphs_embedding)

        select = random.choices(range(len(graphs_hash)))[0] # Picks the lead head, then go towards a counterfactual
        graph_hash = graphs_hash[select]

        target_graphs_hashes, target_graphs, target_graphs_importance_parts, target_graphs_embedding = transitions[graph_hash]

        selected_hash_idx = move_from_known_graph(target_graphs_hashes, target_graphs_importance_parts, importance_args)

        selected_hash = target_graphs_hashes[selected_hash_idx]
        selected_importance_parts = target_graphs_importance_parts[selected_hash_idx]

        selected_graph = target_graphs[selected_hash_idx]

        selected_embedding = target_graphs_embedding[selected_hash_idx]
        selected_elements = util.graph_element_counts([selected_graph])

        if selected_hash not in graph_map:
            graph_map[selected_hash] = [selected_graph, selected_embedding, selected_elements]  # next graph addition to memory

        _, start_embedding, start_elements = graph_map[start_graphs_hash[select]]
        populate_counterfactual_candidates(selected_hash, selected_importance_parts)
        recourse = np.array((selected_embedding - start_embedding) / (selected_elements + start_elements))

        next_hash = []
        next_importance = []
        diff = []
        graph_map_data = []
        s_graph = graph_map[selected_hash]

        for k,i in enumerate(graphs_hash): # Non lead heads follow.
          if k != select:
            _, start_embedding, start_elements = start_map_data[k]

            target_graphs_hashes, target_graphs, target_graphs_importance_parts, target_graphs_embedding = transitions[i]

            selected_elements = util.graph_element_counts(target_graphs)

            matching_recourses = (np.array(target_graphs_embedding)- np.array(start_embedding)) / np.array(selected_elements + start_elements)[:,np.newaxis]

            difference = np.linalg.norm(matching_recourses-recourse, axis=-1)
            match_r = np.argmin(difference)
            select_hash = target_graphs_hashes[match_r]
            next_hash.append(select_hash)
            next_importance.append(target_graphs_importance_parts[match_r])
            diff.append(np.min(difference))
            graph_map_data.append([target_graphs[match_r], target_graphs_embedding[match_r], selected_elements[match_r]])

            populate_counterfactual_candidates(target_graphs_hashes[match_r], target_graphs_importance_parts[match_r]) #bypasses any coverage
          else:
            next_hash.append(selected_hash)
            next_importance.append(selected_importance_parts)
            diff.append(0)
            graph_map_data.append(s_graph)

        for i, hash in enumerate(next_hash):
          if hash not in graph_map:
            graph_map[hash] = graph_map_data[i]
            populate_counterfactual_candidates(hash, next_importance[i], bypass_size = True)

        for i, hash in enumerate(start_graphs_hash):
          if hash not in graph_map:
            graph_map[hash] = start_map_data[i]
            populate_counterfactual_candidates(hash, next_importance[i]*0, bypass_size = True)

        return next_hash, not_teleport, recourse, next_importance, diff


def dynamic_teleportation_probabilities():
    input_graphs_covered_exp = np.exp(input_graphs_covered)
    return (1 / input_graphs_covered_exp) / (1 / input_graphs_covered_exp).sum()


def restart_randomwalk(input_graphs, recourse_size, importance_args): #outputs a list of size recourse_size
    dynamic_probs = dynamic_teleportation_probabilities()
    idx = random.choices(range(dynamic_probs.shape[0]), weights=dynamic_probs, k = recourse_size)
    graphs = input_graphs[idx]
    importance_parts, graph_embeddings = call(graphs, importance_args) 

    input_graphs_covering_list = idx
    graphs_hash = []
    elements = util.graph_element_counts(graphs)
    for k,i in enumerate(graph_embeddings):
      graph_hash = calculate_hash(i)
      graphs_hash.append(graph_hash)
      if graph_hash not in graph_map:
          graph_map[graph_hash] = [graphs[k], i, elements[k]]
      populate_counterfactual_candidates(graph_hash, np.array([0, 1]))
      input_graphs_covered[input_graphs_covering_list[k]] += 1
    for i in idx:
      if i in start:
        start[i] += 1
      else:
        start[i] = 1
    return graphs_hash, idx


# GNN utils


def prepare_and_get(graphs, gnn_model, original_graph_indices, device1, device2, dataset_name):
    original_graphs = graphs[original_graph_indices.tolist()]
    neurosed_folder = f'data/{dataset_name}/neurosed'
    if not os.path.exists(neurosed_folder):
        os.makedirs(neurosed_folder)

    neurosed_model_path = os.path.join(neurosed_folder, 'best_model.pt')
    neurosed_model = distance.load_neurosed(original_graphs, neurosed_model_path=neurosed_model_path, device=device2)

    original_graphs_elements_counts = util.graph_element_counts(original_graphs)

    return {
        'gnn_model': gnn_model,
        'neurosed_model': neurosed_model,
        'original_graphs': original_graphs,
        'original_graphs_element_counts': original_graphs_elements_counts,
        'gnn_device': device1,
        'neurosed_device': device2
    }


def call(graphs, wargs): #from restart random walk, need GREED model somewhere in wargs: neurosed model

    try:
        preds, _ = prediction(wargs['gnn_model'], Batch.from_data_list(graphs).to(wargs['gnn_device']))
        graph_embeddings = neurosed_embedding(wargs['neurosed_model'], Batch.from_data_list(graphs).to(wargs['neurosed_device']))
        preds = preds.cpu().numpy()
        graph_embeddings = graph_embeddings.cpu().numpy()
    except RuntimeError as re:
        loader = DataLoader(graphs, batch_size=128)
        preds, graph_embeddings = [], []
        for batch in loader:
            pred, _ = prediction(wargs['gnn_model'], batch.to(wargs['gnn_device']))
            graph_embedding = neurosed_embedding(wargs['neurosed_model'], Batch.from_data_list(graphs).to(wargs['neurosed_device']))
            preds.append(pred)
            graph_embeddings.append(graph_embedding)
        preds = torch.cat(preds).cpu().numpy()
        graph_embeddings = torch.cat(graph_embeddings).cpu().numpy()

    torch.cuda.set_device(wargs['gnn_device'])
    torch.cuda.empty_cache()
    torch.cuda.set_device(wargs['neurosed_device'])
    torch.cuda.empty_cache()

    neurosed_setting(wargs['neurosed_model'], graphs)
    coverage = np.ones(shape=preds.shape)
    torch.cuda.set_device(wargs['neurosed_model'].device)
    torch.cuda.empty_cache()

    return np.stack([preds, coverage]).T, graph_embeddings



@torch.no_grad()
def prediction(model, graphs): # from call
    node_embeddings, graph_embeddings, preds = model(graphs)
    preds = torch.exp(preds)
    return preds[:, [1]].sum(axis=1), graph_embeddings

@torch.no_grad()
def neurosed_embedding(model, graphs):
    graph_embeddings = model.embed_model(graphs)
    return graph_embeddings


@torch.no_grad()
def neurosed_setting(model, graphs):
    model.embed_targets(graphs)


### Random walk core function


def counterfactual_summary_with_randomwalk(dataset_name, input_graphs, importance_args, teleport_probability, max_steps, heads = 5):
    start_graphs_hash, indx = restart_randomwalk(input_graphs, heads, importance_args) #lists of size recourse_size
    cur_graph_hash = copy.deepcopy(start_graphs_hash)
    recourse_num = 0

    for step in tqdm(range(starting_step, max_steps + 1)):
        traversed_hashes.append(cur_graph_hash)
        next_graph_hash, is_teleported, recourse, next_importance, diff = move_to_next_graph(graphs_hash=cur_graph_hash,
                                                            start_graphs_hash = start_graphs_hash,
                                                            importance_args=importance_args,
                                                            teleport_probability=teleport_probability)

        if is_teleported:
          start_graphs_hash, indx = restart_randomwalk(input_graphs, heads, importance_args)
          cur_graph_hash = copy.deepcopy(start_graphs_hash)

        assert len(graph_map) == len(graph_index_map) # memory checks
        assert set(graph_index_map.keys()) == set(graph_map.keys())

    save_item = {
        'graph_map': graph_map,
        'graph_index_map': graph_index_map,
        'counterfactual_candidates': counterfactual_candidates,
        'MAX_COUNTERFACTUAL_SIZE': MAX_COUNTERFACTUAL_SIZE,
        'traversed_hashes': traversed_hashes,
        'input_graphs_covered': input_graphs_covered,
    }
    if not os.path.exists(f'results/{dataset_name}/counterfactuals/'):
        os.makedirs(f'results/{dataset_name}/counterfactuals/')
    torch.save(save_item, f'results/{dataset_name}/counterfactuals/comrecgc_k_{heads}.pt')




def main():
    args = get_args()
    
    dataset_name = args.dataset

    # Hyperparameters

    # global MAX_COUNTERFACTUAL_SIZE
    global MAX_COUNTERFACTUAL_SIZE
    MAX_COUNTERFACTUAL_SIZE = args.k
    
    teleport_probability = args.teleport # Tau in the paper
    max_steps = args.steps # M in the paper
    number_heads = args.heads # k in the paper
    global sample_size
    sample_size = args.sample_size
    global is_sample
    is_sample = args.sample
    
    np.random.seed(0) 
    random.seed(0)
    torch.manual_seed(0)
    
    device1, device2 = prepare_devices(args.device1, args.device2) #device 1 for GNN, device 2 for neurosed model to estimate GED
    
    graphs = load_dataset(dataset_name) # loads dataset
    gnn_model = load_trained_gnn(dataset_name, device=device1) # load the gnn we want to explain
    gnn_model.eval()
    
    # Load prediction based on model
    preds = load_trained_prediction(dataset_name, device=device1)
    preds = preds.cpu().numpy()
    input_graph_indices = np.array(range(len(preds)))[preds == 0] #reject graphs
    input_graphs = graphs[input_graph_indices.tolist()]
    
    # setting covered graph numbers to 0
    global input_graphs_covered
    input_graphs_covered = torch.zeros(len(input_graphs), dtype=torch.float)
    
    importance_args = prepare_and_get(graphs, gnn_model, input_graph_indices, device1=device1, device2=device2, dataset_name=dataset_name)
    counterfactual_summary_with_randomwalk(dataset_name = dataset_name,
                                       input_graphs=input_graphs,
                                       importance_args=importance_args,
                                       teleport_probability=teleport_probability,
                                       max_steps=max_steps,
                                       heads = number_heads)

if __name__ == "__main__":
    main()