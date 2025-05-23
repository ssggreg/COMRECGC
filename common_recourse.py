import torch
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
import util

import copy
from sklearn import metrics
from sklearn.cluster import DBSCAN, HDBSCAN
from pathlib import Path

import distance
from data import load_dataset # loads datasets
from gnn import load_trained_gnn, load_trained_prediction



def get_args():
    parser = argparse.ArgumentParser(description='Graph Global Counterfactual Summary')
    parser.add_argument('--dataset', type=str, default='mutagenicity', choices=['mutagenicity', 'aids', 'nci1', 'proteins'])
    parser.add_argument('--theta', type=float, default=0.1, help='distance threshold value for maximum length recourse.')
    parser.add_argument('--delta', type=float, default=0.02, help='distance threshold value to be considered common')
    parser.add_argument('--recourse_size', type=int, default=100, help='number of recourse for the summary')
    parser.add_argument('--cf_size', type=int, default=100000, help='maximum number of counterfactuals for building the recourse')
    parser.add_argument('--cluster_size', type=int, default=3, help='minimum number of samples per cluster')
    return parser.parse_args()

    

def greedy_counterfactual_summary_from_covering_sets(counterfactual_covering, graphs_covered_by, k):
    """
    :param counterfactual_covering: Counterfactual -> Original graphs covered.
    :param graphs_covered_by: Original graphs -> counterfactuals that cover it.
    :param k: Number of counterfactuals in the summary.

    :return: List of indices of selected counterfactuals as summary, and the set of indices of the covered graphs.
    """

    # Greedily add the counterfactuals with maximum coverage in the remaining graphs.
    coverings = {}
    covered = set()

    # while len(indices) < k:
    for i in tqdm(range(1, k + 1)):
        counterfactual_index, covered_indices = max(counterfactual_covering.items(), key=lambda pair: len(pair[1]))
        covered.update(covered_indices)
        counterfactual_covering.pop(counterfactual_index)
        for covered_index in covered_indices:  # Update the mapping.
            for other_counterfactual_index in graphs_covered_by[covered_index] - {counterfactual_index}:
                if other_counterfactual_index in counterfactual_covering:
                    counterfactual_covering[other_counterfactual_index].remove(covered_index)

        coverings[i] = (counterfactual_index, len(covered))

    return coverings


def coverage_summary(db_2, rec, idxs, radius, threshold_theta, recourse_size):
    """
    Generate a counterfactual summary that efficiently covers original graphs using cluster-based recourse selection.

    Args:
        db_2: A clustering object with a `.labels_` attribute (e.g., from KMeans) indicating cluster assignments.
        rec (torch.Tensor): The recourse embeddings (shape: [num_counterfactuals, embedding_dim]).
        idxs (List[Tuple[int, int]]): List of (original_graph_index, counterfactual_index) pairs where similarity <= threshold.
        radius (float): Maximum distance from a cluster centroid to consider a counterfactual part of the cluster.
        threshold_theta (float): Maximum centroid norm to include the cluster in the final summary.
        recourse_size (int): Desired number of counterfactuals in the summary.

    Returns:
        covering (List[int]): Indices of selected counterfactuals forming the summary.
        cost (List[float]): Cumulative centroid norms after each selection.
        size (List[int]): Cumulative number of unique original graphs covered after each selection.
    """
    common_recourse = {}
    centroid_norms = {}
    graph_coverage_map = {}

    for cluster_label in range(max(db_2.labels_) + 1):
        covered_graphs = set()
        covered_hashes = set()

        # Get points and indices for this cluster
        cluster_mask = db_2.labels_ == cluster_label
        cluster_points = rec[cluster_mask]
        cluster_indices = [i for i, is_in_cluster in enumerate(cluster_mask) if is_in_cluster]

        # Compute centroid and distances
        centroid = torch.mean(cluster_points, dim=0)
        distances = torch.norm(cluster_points - centroid, dim=-1)

        for i, dist in enumerate(distances):
            if dist < radius:
                original_idx, cf_idx = idxs[cluster_indices[i]]
                if original_idx not in covered_graphs:
                    covered_graphs.add(original_idx)
                    covered_hashes.add(cf_idx)

        common_recourse[cluster_label] = covered_graphs
        centroid_norms[cluster_label] = torch.norm(centroid).item()
        graph_coverage_map[cluster_label] = covered_hashes

    # Filter clusters based on centroid norm
    filtered_covering = {}
    graphs_covered_by = {}

    for label, graphs in common_recourse.items():
        if centroid_norms[label] < threshold_theta:
            filtered_covering[label] = graphs
            for g in graphs:
                graphs_covered_by.setdefault(g, set()).add(label)

    # Select counterfactuals greedily
    selected = greedy_counterfactual_summary_from_covering_sets(
        counterfactual_covering=filtered_covering,
        graphs_covered_by=graphs_covered_by,
        k=min(recourse_size, len(filtered_covering))
    )

    covering, cost, size = [], [], []
    cumulative_cost = 0
    covered_hashes = set()

    for label in selected:
        cf_index = selected[label][1]  # counterfactual ID
        rec_label = selected[label][0]  # cluster label

        covering.append(cf_index)
        covered_hashes.update(graph_coverage_map[rec_label])
        cumulative_cost += centroid_norms[rec_label]
        cost.append(cumulative_cost)
        size.append(len(covered_hashes))

    return covering, cost, size

        

    
    
def main():
    """
    Orchestrate the generation of cluster-based counterfactual summaries.

    Steps:
      1. Parse arguments.
      2. Load dataset, GNN, and saved predictions.
      3. Split graphs into positively and negatively predicted sets.
      4. For each counterfactual method:
         a. Load its candidate counterfactuals.
         b. Filter and augment the counterfactual pool.
         c. Compute pairwise normalized distances S.
         d. Extract (graph, cf) pairs with S <= θ.
         e. Compute cluster embeddings with Δ-normalized differences.
         f. Run DBSCAN to group recourses.
         g. Summarize clusters via a greedy covering algorithm.
         h. Serialize results to disk.

    Returns:
        None  # results are written as pickle files under ./results/{dataset}/common_recourse/
    """
    args = get_args()

    # Device setup
    device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
    np.random.seed(0)
    torch.manual_seed(0)

    # Load data and model
    graphs = load_dataset(args.dataset)
    gnn_model = load_trained_gnn(args.dataset, device).eval()
    preds = load_trained_prediction(args.dataset, device).cpu()

    # Split by prediction
    idx_main = torch.where(preds == 0)[0]
    idx_reverse = torch.where(preds != 0)[0]
    originals = graphs[idx_main.tolist()]
    counterfactual_pool = graphs[idx_reverse.tolist()]

    # Prepare output directory
    out_dir = Path(f'./results/{args.dataset}/common_recourse')
    out_dir.mkdir(parents=True, exist_ok=True)

    # Process each method’s results
    for file in Path(f'./results/{args.dataset}/counterfactuals').glob('*.pt'):
        method_name = file.stem
        print(f"\nProcessing method: {method_name}")

        # Load candidates
        data = torch.load(file)
        candidates = data['counterfactual_candidates']
        graph_map = data['graph_map']

        # Build initial list of counterfactual graphs
        cfs = []
        for c in candidates:
            if c['importance_parts'][0] >= 0.5:
                cfs.extend(graph_map[c['graph_hash']])
        # If too few, add all reverse-predicted graphs
        if len(cfs) < len(originals):
            for g in counterfactual_pool:
                g.edge_weight = getattr(g, 'edge_attr', torch.ones(g.edge_index.size(1)))
                g.pred = getattr(g, 'y', torch.tensor(0))
                for attr in ('edge_attr', 'y'):
                    if hasattr(g, attr):
                        delattr(g, attr)
                cfs.append(g)

        # Compute neurosed distances S_normalized (originals × cfs)
        neurosed = distance.load_neurosed(
            originals, neurosed_model_path=f'data/{args.dataset}/neurosed/best_model.pt',
            device=device
        ).to(device).eval()

        with torch.no_grad():
            batch_cfs = tg.data.Batch.from_data_list(cfs).to(device)
            batch_orig = tg.data.Batch.from_data_list(originals).to(device)
            S = neurosed.predict_outer_with_queries(cfs, batch_size=128).cpu()

        # Normalize by element counts
        cnt_orig = util.graph_element_counts(originals)
        cnt_cfs  = util.graph_element_counts(cfs)
        scale = (
            torch.cartesian_prod(cnt_cfs, cnt_orig)
                 .sum(dim=1)
                 .view(len(cfs), len(originals))
        )
        S = (S / scale).T

        # Extract all (orig_idx, cf_idx) with S <= θ
        idxs = torch.where(S <= args.theta)
        idxs = [(i.item(), j.item()) for i, j in zip(*idxs)]

        # Build Δ-embeddings: (cf_emb – orig_emb) / scale
        with torch.no_grad():
            emb_cfs = neurosed.embed_model(batch_cfs).cpu()
            emb_org = neurosed.embed_model(batch_orig).cpu()

        diffs = (emb_cfs.unsqueeze(1) - emb_org) / scale.unsqueeze(2)
        flat_diffs = diffs.view(-1, emb_org.size(1))

        # Select only the valid pairs
        linear_idx = [orig + cf * len(originals) for orig, cf in idxs]
        rec = flat_diffs[linear_idx]

        # Cluster with DBSCAN
        db = DBSCAN(eps=args.delta, min_samples=args.cluster_size)
        db.fit(rec)

        # Summarize via greedy cover
        result = coverage_summary(
            db_2=db,
            rec=rec,
            idxs=idxs,
            radius=args.delta,
            threshold_theta=args.theta,
            recourse_size=args.recourse_size
        )

        # Persist
        with open(out_dir / f"{method_name}.pkl", 'wb') as f:
            pickle.dump(result, f)

        # Clean up
        del neurosed, diffs, flat_diffs, rec, S
        torch.cuda.empty_cache()


if __name__ == "__main__":
    main()