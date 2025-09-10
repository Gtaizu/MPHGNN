import os
import dgl
import torch
import random
import pandas as pd
import numpy as np
from torch import nn
from torch_geometric.utils import to_undirected
from tqdm import tqdm
from torch.utils.data import DataLoader


def load_test_data(args):
    k = args.k
    data = torch.load("MiDrug_data_end.pth", weights_only=False)

    miRNA_drug = data['miRNA', 'MiDrug', 'drug'].edge_label_index
    miRNA_drug_label = data['miRNA', 'MiDrug', 'drug'].edge_label

    gene_drug_link = data['gene', 'Genedrug', 'drug'].edge_index
    gene_drug_link = gene_drug_link.cpu().T.numpy()

    miRNA_gene_link = data['miRNA', 'Migene', 'gene'].edge_index
    miRNA_gene_link = miRNA_gene_link.cpu().T.numpy()

    gene_gene_link = data['gene', 'GeGe', 'gene'].edge_index
    gene_gene_link = to_undirected(gene_gene_link)
    gene_gene_link = gene_gene_link.cpu().T.numpy()

    num_mirnas = data['miRNA'].num_nodes
    num_drugs = data['drug'].num_nodes
    num_gene = data['gene'].num_nodes

    positive_mask = miRNA_drug_label == 1
    positive_edges = miRNA_drug[:, positive_mask]
    negative_mask = miRNA_drug_label == 0
    negative_edges = miRNA_drug[:, negative_mask]

    adj_matrix = np.zeros((num_mirnas, num_drugs), dtype=np.float32)

    src, dst = positive_edges.cpu().numpy()
    adj_matrix[src, dst] = 1 
    drug_miRNA_link = torch.stack([
        positive_edges[1], 
        positive_edges[0]  
    ])
    drug_miRNA_link = drug_miRNA_link.cpu().T.numpy()
    miRNA_drug_link = positive_edges.cpu().T.numpy()

    links = {'gene-drug': gene_drug_link, 'miRNA-drug': miRNA_drug_link,
             'miRNA-gene': miRNA_gene_link, 'gene-gene': gene_gene_link}

    graph_data = {('gene', 'gene-drug', 'drug'): (torch.tensor(gene_drug_link[:, 0]),
                                                  torch.tensor(gene_drug_link[:, 1])),
                  ('miRNA', 'miRNA-drug', 'drug'): (torch.tensor(miRNA_drug_link[:, 0]),
                                                    torch.tensor(miRNA_drug_link[:, 1])),
                  ('drug', 'drug-miRNA', 'miRNA'): (torch.tensor(drug_miRNA_link[:, 0]),
                                                    torch.tensor(drug_miRNA_link[:, 1])),
                  ('miRNA', 'miRNA-gene', 'gene'): (torch.tensor(miRNA_gene_link[:, 0]),
                                                    torch.tensor(miRNA_gene_link[:, 1]))}
    g = dgl.heterograph(
        graph_data,
        num_nodes_dict={'miRNA': 605, 'drug': 216,
                        'gene': num_gene}
    )
    drug_feature = data['drug'].x.cpu().numpy()
    miRNA_feature = data['miRNA'].x.cpu().numpy()
    gene_feature = data['gene'].x.cpu().numpy()

    g.nodes['drug'].data['h'] = torch.from_numpy(drug_feature).to(torch.float32)
    g.nodes['miRNA'].data['h'] = torch.from_numpy(miRNA_feature).to(torch.float32)
    g.nodes['gene'].data['h'] = torch.from_numpy(gene_feature).to(torch.float32)

    data, label = [], []

    pos_pairs_np = miRNA_drug_link
    neg_pairs_np = negative_edges.cpu().numpy().T

    num_positives = len(pos_pairs_np)

    print('Generating Meta-Path Instances(It takes time)...')
    with tqdm(total=num_positives + len(neg_pairs_np)) as pbar:
        pbar.set_description('Processing')
        for miRNA_id, drug_id in pos_pairs_np:
            mpi = meta_path_instance(args, miRNA_id, drug_id, links, k)
            data.append(mpi)
            label.append(1)
            pbar.update()

        for miRNA_id, drug_id in neg_pairs_np:
            mpi = meta_path_instance(args, miRNA_id, drug_id, links, k)
            data.append(mpi)
            label.append(0)
            pbar.update()

        print('Preparing dataset...')
        data = np.array(data)
        label = np.array(label)
    print('Data prepared !')
    return g, data, label, miRNA_drug, miRNA_drug_label


def load_data(args):
    dataset = args.dataset
    k = args.k

    data = torch.load("MiDrug_data_end.pth", weights_only=False)

    gene_drug_link = data['gene', 'Genedrug', 'drug'].edge_index
    gene_drug_link = gene_drug_link.cpu().T.numpy()

    miRNA_gene_link = data['miRNA', 'Migene', 'gene'].edge_index
    miRNA_gene_link = miRNA_gene_link.cpu().T.numpy()

    gene_gene_link = data['gene', 'GeGe', 'gene'].edge_index
    gene_gene_link = to_undirected(gene_gene_link)
    gene_gene_link = gene_gene_link.cpu().T.numpy()

    miRNA_drug = data['miRNA', 'MiDrug', 'drug'].edge_index
    num_mirnas = data['miRNA'].num_nodes
    num_drugs = data['drug'].num_nodes
    num_genes = data['gene'].num_nodes
    adj_matrix = np.zeros((num_mirnas, num_drugs), dtype=np.float32)

    src, dst = miRNA_drug.cpu().numpy()
    adj_matrix[src, dst] = 1

    drug_miRNA_link = torch.stack([
        miRNA_drug[1],
        miRNA_drug[0]
    ])

    drug_miRNA_link = drug_miRNA_link.cpu().T.numpy()
    miRNA_drug_link = miRNA_drug.cpu().T.numpy()

    links = {'gene-drug': gene_drug_link, 'miRNA-drug': miRNA_drug_link,
             'miRNA-gene': miRNA_gene_link, 'gene-gene': gene_gene_link}

    graph_data = {('gene', 'gene-drug', 'drug'): (torch.tensor(gene_drug_link[:, 0]),
                                                  torch.tensor(gene_drug_link[:, 1])),
                  ('miRNA', 'miRNA-drug', 'drug'): (torch.tensor(miRNA_drug_link[:, 0]),
                                                    torch.tensor(miRNA_drug_link[:, 1])),
                  ('drug', 'drug-miRNA', 'miRNA'): (torch.tensor(drug_miRNA_link[:, 0]),
                                                    torch.tensor(drug_miRNA_link[:, 1])),
                  ('miRNA', 'miRNA-gene', 'gene'): (torch.tensor(miRNA_gene_link[:, 0]),
                                                    torch.tensor(miRNA_gene_link[:, 1]))}

    g = dgl.heterograph(graph_data)


    drug_feature = data['drug'].x.cpu().numpy()
    miRNA_feature = data['miRNA'].x.cpu().numpy()
    gene_feature = data['gene'].x.cpu().numpy()


    g.nodes['drug'].data['h'] = torch.from_numpy(drug_feature).to(torch.float32)
    g.nodes['miRNA'].data['h'] = torch.from_numpy(miRNA_feature).to(torch.float32)
    g.nodes['gene'].data['h'] = torch.from_numpy(gene_feature).to(torch.float32)

    data, label = [], []


    positive_pairs = np.argwhere(adj_matrix == 1).tolist()  
    num_positives = len(positive_pairs)


    negative_pairs = np.argwhere(adj_matrix == 0).tolist() 

    random.seed(args.seed)
    sampled_negative_pairs = random.sample(negative_pairs, num_positives)

    print('Generating Meta-Path Instances(It takes time)...')
    with tqdm(total=num_positives + len(sampled_negative_pairs)) as pbar:
        pbar.set_description('Processing')
  
        for miRNA_id, drug_id in positive_pairs:
            mpi = meta_path_instance(args, miRNA_id, drug_id, links, k)
            data.append(mpi)
            label.append(1)  
            pbar.update()

        for miRNA_id, drug_id in sampled_negative_pairs:
            mpi = meta_path_instance(args, miRNA_id, drug_id, links, k)
            data.append(mpi)
            label.append(0)  
            pbar.update()

        print('Preparing dataset...')
        data = np.array(data)
        label = np.array(label)
        np.save('{}_temp_{}k/data.npy'.format(args.dataset, args.k), data)
        np.save('{}_temp_{}k/label.npy'.format(args.dataset, args.k), label)
    print('Data prepared !')
    return g, data, label, adj_matrix

def topk_filtering(d_d: np.array, k: int):
    """Convert the Topk similarities to 1 and generate the Topk interactions."""
    for i in range(len(d_d)):
        sorted_idx = np.argpartition(d_d[i], -k - 1)
        d_d[i, sorted_idx[-k - 1:-1]] = 1
    return np.array(np.where(d_d == 1)).T


def meta_path_instance(args, miRNA_id: int, drug_id: int, links: dict, k: int):
    """Generate the pseudo meta-path instances.
    """
    mpi = []
    mpi.extend([[miRNA_id, miRNA, drug, drug_id]
                for miRNA in links['miRNA-gene'][links['miRNA-gene'][:, 0] == miRNA_id][:, 1]
                for drug in links['gene-drug'][links['gene-drug'][:, 1] == drug_id][:, 0]])
    mpi.extend([
        [miRNA_id, gene1, gene2, drug_id]
        for gene1 in links['miRNA-gene'][links['miRNA-gene'][:, 0] == miRNA_id][:, 1]
        for gene2 in links['gene-gene'][links['gene-gene'][:, 0] == gene1][:, 1]
    ])
    mpi.extend([
        [miRNA_id, gene1, gene2, drug_id]
        for gene2 in links['gene-drug'][links['gene-drug'][:, 1] == drug_id][:, 0]
        for gene1 in links['gene-gene'][links['gene-gene'][:, 0] == gene2][:, 1]
    ])

    if not mpi:

        all_genes = np.unique(links['gene-drug'][0])
        if len(all_genes) > 0:
            random_gene = np.random.choice(all_genes)
            mpi.append([miRNA_id, random_gene, random_gene, drug_id])
        else:
            all_genes = np.unique(links['miRNA-gene'][1])
            random_gene = np.random.choice(all_genes)
            mpi.append([miRNA_id, random_gene, random_gene, drug_id])

    if len(mpi) < k * (k + 2) + 1:
        for i in range(k * (k + 2) + 1 - len(mpi)):
            random.seed(args.seed)
            mpi.append(random.choice(mpi))
    elif len(mpi) > k * (k + 2) + 1:
        mpi = mpi[:k * (k + 2) + 1]
    return mpi


def remove_graph(g, test_miRNA_id, test_drug_id):
    etype = ('miRNA', 'miRNA-drug', 'drug')
    edges_id = g.edge_ids(torch.tensor(test_miRNA_id),
                          torch.tensor(test_drug_id),
                          etype=etype)
    g = dgl.remove_edges(g, edges_id, etype=etype)
    etype = ('drug', 'drug-miRNA', 'miRNA')
    edges_id = g.edge_ids(torch.tensor(test_drug_id),
                          torch.tensor(test_miRNA_id),
                          etype=etype)
    g = dgl.remove_edges(g, edges_id, etype=etype)
    return g


def get_data_loaders(data, batch_size, shuffle, drop=False):
    """Build data loader for train data and test data.
    """
    return DataLoader(data, batch_size=batch_size, shuffle=shuffle, drop_last=drop)

