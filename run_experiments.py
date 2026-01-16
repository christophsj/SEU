#!/usr/bin/env python3
# _*_ coding:utf-8 _*_

"""
Automated experiment runner for SEU Entity Alignment
Runs experiments 3 times for each dataset and saves results
"""

import os
import json
import time
import numpy as np
import tensorflow as tf
from tqdm import tqdm
from scipy import optimize
from utils import load_triples, load_aligned_pair, test
from datetime import datetime
import sys

# Configuration
SEED = 12345


# Auto-detect available datasets by matching translated entity names with KG folders
def get_available_datasets():
    """Find datasets that have both KG data and translated entity names"""
    kg_dirs = [d for d in os.listdir("KGs") if os.path.isdir(f"KGs/{d}")]
    translated_files = [
        f.replace(".json", "")
        for f in os.listdir("translated_ent_name")
        if f.endswith(".json")
    ]

    # Map KG folders to translated entity name files
    dataset_mapping = {
        "fr_en": "dbp_fr_en",
        "ja_en": "dbp_ja_en",
        "zh_en": "dbp_zh_en",
        "EN_DE_15K_V2": "en_de_15k_v2",
        "EN_FR_15K_V2": "en_fr_15k_v2",
        "D_W_15K_V2": "d_w_15k_v2",
        "D_Y_15K_V2": "d_y_15k_v2",
    }

    datasets = []
    for kg_dir in kg_dirs:
        trans_name = dataset_mapping.get(kg_dir)
        if trans_name and trans_name in translated_files:
            datasets.append((kg_dir, trans_name))

    return datasets


DATASETS = get_available_datasets()
FEATURE_MODES = ["word-level", "char-level", "hybrid-level"]
GRAPH_DEPTH = 2
NUM_RUNS = 3
USE_CPU = True

# Set random seed
np.random.seed(SEED)

# Set device
if USE_CPU:
    os.environ["CUDA_VISIBLE_DEVICES"] = "-1"


def load_word_embeddings(embedding_path="./glove.6B.300d.txt"):
    """Load pre-trained word embeddings"""
    print(f"Loading word embeddings from {embedding_path}...")
    word_vecs = {}
    with open(embedding_path, encoding="UTF-8") as f:
        for line in tqdm(f.readlines(), desc="Loading embeddings"):
            line = line.split()
            word_vecs[line[0]] = np.array([float(x) for x in line[1:]])
    return word_vecs


def generate_bigram_dict(ent_names):
    """Generate bigram dictionary from entity names"""
    d = {}
    count = 0
    for _, name in ent_names:
        for word in name:
            word = word.lower()
            for idx in range(len(word) - 1):
                if word[idx : idx + 2] not in d:
                    d[word[idx : idx + 2]] = count
                    count += 1
    return d


def generate_features(ent_names, word_vecs, node_size, bigram_dict):
    """Generate word-level and character-level features"""
    ent_vec = np.zeros((node_size, 300))
    char_vec = np.zeros((node_size, len(bigram_dict)))

    for i, name in ent_names:
        k = 0
        for word in name:
            word = word.lower()
            if word in word_vecs:
                ent_vec[i] += word_vecs[word]
                k += 1
            for idx in range(len(word) - 1):
                char_vec[i, bigram_dict[word[idx : idx + 2]]] += 1

        if k:
            ent_vec[i] /= k
        else:
            ent_vec[i] = np.random.random(300) - 0.5

        if np.sum(char_vec[i]) == 0:
            char_vec[i] = np.random.random(len(bigram_dict)) - 0.5

        ent_vec[i] = ent_vec[i] / np.linalg.norm(ent_vec[i])
        char_vec[i] = char_vec[i] / np.linalg.norm(char_vec[i])

    return ent_vec, char_vec


def build_sparse_rel_matrix(all_triples, node_size):
    """Build relational adjacency matrix"""
    dr = {}
    for x, r, y in all_triples:
        if r not in dr:
            dr[r] = 0
        dr[r] += 1

    sparse_rel_matrix = []
    for i in range(node_size):
        sparse_rel_matrix.append([i, i, np.log(len(all_triples) / node_size)])

    for h, r, t in all_triples:
        sparse_rel_matrix.append([h, t, np.log(len(all_triples) / dr[r])])

    sparse_rel_matrix = np.array(sorted(sparse_rel_matrix, key=lambda x: x[0]))
    sparse_rel_matrix = tf.SparseTensor(
        indices=sparse_rel_matrix[:, :2],
        values=sparse_rel_matrix[:, 2],
        dense_shape=(node_size, node_size),
    )
    return sparse_rel_matrix


def cal_sims(test_pair, feature):
    """Calculate similarity matrix"""
    feature_a = tf.gather(indices=test_pair[:, 0], params=feature)
    feature_b = tf.gather(indices=test_pair[:, 1], params=feature)
    return tf.matmul(feature_a, tf.transpose(feature_b, [1, 0]))


def propagate_features(test_pair, feature, sparse_rel_matrix, depth):
    """Propagate features through graph and compute similarities"""
    sims = cal_sims(test_pair, feature)
    for i in range(depth):
        feature = tf.sparse.sparse_dense_matmul(sparse_rel_matrix, feature)
        feature = tf.nn.l2_normalize(feature, axis=-1)
        sims += cal_sims(test_pair, feature)
    sims /= depth + 1
    return sims


def run_hungarian(sims):
    """Run Hungarian algorithm"""
    result = optimize.linear_sum_assignment(sims, maximize=True)

    # Calculate metrics
    c = 0
    for i, j in enumerate(result[1]):
        if i == j:
            c += 1
    hits1 = 100 * c / len(result[0])

    return {"hits@1": hits1}


def run_sinkhorn(sims, batch_size=1024):
    """Run Sinkhorn operation"""
    sims = tf.exp(sims * 50)
    for k in range(10):
        sims = sims / tf.reduce_sum(sims, axis=1, keepdims=True)
        sims = sims / tf.reduce_sum(sims, axis=0, keepdims=True)

    # Calculate metrics
    results = []
    for epoch in range(len(sims) // batch_size + 1):
        sim = sims[epoch * batch_size : (epoch + 1) * batch_size]
        rank = tf.argsort(-sim, axis=-1)
        ans_rank = np.array(
            [
                i
                for i in range(
                    epoch * batch_size, min((epoch + 1) * batch_size, len(sims))
                )
            ]
        )
        results.append(
            tf.where(
                tf.equal(
                    tf.cast(rank, ans_rank.dtype),
                    tf.tile(np.expand_dims(ans_rank, axis=1), [1, len(sims)]),
                )
            ).numpy()
        )
    results = np.concatenate(results, axis=0)

    hits1, hits10, mrr = 0, 0, 0
    for x in results[:, 1]:
        if x < 1:
            hits1 += 1
        if x < 10:
            hits10 += 1
        mrr += 1 / (x + 1)

    return {
        "hits@1": hits1 / len(sims) * 100,
        "hits@10": hits10 / len(sims) * 100,
        "MRR": mrr / len(sims) * 100,
    }


def run_single_experiment(
    dataset_kg_path, dataset_trans_name, feature_mode, word_vecs, run_num
):
    """Run a single experiment for a dataset"""
    print(f"\n{'='*80}")
    print(
        f"Dataset: {dataset_kg_path} | Mode: {feature_mode} | Run: {run_num + 1}/{NUM_RUNS}"
    )
    print(f"{'='*80}")

    results = {
        "dataset": dataset_kg_path,
        "translated_name": dataset_trans_name,
        "feature_mode": feature_mode,
        "run": run_num + 1,
        "timestamp": datetime.now().isoformat(),
    }

    try:
        # Load entity names
        ent_names_path = f"translated_ent_name/{dataset_trans_name}.json"
        ent_names = json.load(open(ent_names_path, "r"))

        # Load KGs and test set
        file_path = f"KGs/{dataset_kg_path}/"
        all_triples, node_size, rel_size = load_triples(file_path, True)
        train_pair, test_pair = load_aligned_pair(file_path, ratio=0)

        print(f"Node size: {node_size}, Relation size: {rel_size}")
        print(f"Test pairs: {len(test_pair)}")

        # Generate bigram dictionary
        bigram_dict = generate_bigram_dict(ent_names)
        print(f"Bigram dictionary size: {len(bigram_dict)}")

        # Generate features
        ent_vec, char_vec = generate_features(
            ent_names, word_vecs, node_size, bigram_dict
        )

        # Select feature mode
        if feature_mode == "word-level":
            feature = ent_vec
        elif feature_mode == "char-level":
            feature = char_vec
        else:  # hybrid-level
            feature = np.concatenate([ent_vec, char_vec], -1)

        feature = tf.nn.l2_normalize(feature, axis=-1)

        # Build sparse relational matrix
        sparse_rel_matrix = build_sparse_rel_matrix(all_triples, node_size)

        # Feature propagation
        print("\nPropagating features through graph...")
        start_time = time.time()
        sims = propagate_features(test_pair, feature, sparse_rel_matrix, GRAPH_DEPTH)
        sims_numpy = sims.numpy()
        propagation_time = time.time() - start_time
        results["propagation_time"] = propagation_time
        print(f"Propagation completed in {propagation_time:.2f} seconds")

        # Hungarian algorithm
        print("\nRunning Hungarian algorithm...")
        start_time = time.time()
        hungarian_results = run_hungarian(sims_numpy)
        hungarian_time = time.time() - start_time
        hungarian_results["time"] = hungarian_time
        results["hungarian"] = hungarian_results
        print(
            f"Hungarian - Hits@1: {hungarian_results['hits@1']:.2f}% (Time: {hungarian_time:.2f}s)"
        )

        # Sinkhorn operation
        print("\nRunning Sinkhorn operation...")
        start_time = time.time()
        sinkhorn_results = run_sinkhorn(sims)
        sinkhorn_time = time.time() - start_time
        sinkhorn_results["time"] = sinkhorn_time
        results["sinkhorn"] = sinkhorn_results
        print(
            f"Sinkhorn - Hits@1: {sinkhorn_results['hits@1']:.2f}%, "
            f"Hits@10: {sinkhorn_results['hits@10']:.2f}%, "
            f"MRR: {sinkhorn_results['MRR']:.2f}% (Time: {sinkhorn_time:.2f}s)"
        )

        results["status"] = "success"

    except Exception as e:
        print(f"ERROR: {str(e)}")
        results["status"] = "failed"
        results["error"] = str(e)

    return results


def run_all_experiments(feature_mode="hybrid-level"):
    """Run experiments for all datasets"""
    # Load word embeddings once
    word_vecs = load_word_embeddings()

    all_results = []

    for dataset_kg_path, dataset_trans_name in DATASETS:
        for run_num in range(NUM_RUNS):
            result = run_single_experiment(
                dataset_kg_path, dataset_trans_name, feature_mode, word_vecs, run_num
            )
            all_results.append(result)

            # Save intermediate results
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_file = f"results/intermediate_results_{timestamp}.json"
            os.makedirs("results", exist_ok=True)
            with open(output_file, "w") as f:
                json.dump(all_results, f, indent=2)

    return all_results


def save_results(results):
    """Save final results with summary statistics"""
    os.makedirs("results", exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    # Save raw results
    output_file = f"results/all_experiments_{timestamp}.json"
    with open(output_file, "w") as f:
        json.dump(results, f, indent=2)

    # Calculate and save summary statistics
    summary = {}
    for dataset_kg_path, dataset_trans_name in DATASETS:
        dataset_results = [
            r
            for r in results
            if r["dataset"] == dataset_kg_path and r["status"] == "success"
        ]
        if dataset_results:
            # Hungarian stats
            hungarian_hits1 = [r["hungarian"]["hits@1"] for r in dataset_results]
            # Sinkhorn stats
            sinkhorn_hits1 = [r["sinkhorn"]["hits@1"] for r in dataset_results]
            sinkhorn_hits10 = [r["sinkhorn"]["hits@10"] for r in dataset_results]
            sinkhorn_mrr = [r["sinkhorn"]["MRR"] for r in dataset_results]

            summary[dataset_kg_path] = {
                "runs": len(dataset_results),
                "hungarian": {
                    "hits@1_mean": np.mean(hungarian_hits1),
                    "hits@1_std": np.std(hungarian_hits1),
                    "hits@1_min": np.min(hungarian_hits1),
                    "hits@1_max": np.max(hungarian_hits1),
                },
                "sinkhorn": {
                    "hits@1_mean": np.mean(sinkhorn_hits1),
                    "hits@1_std": np.std(sinkhorn_hits1),
                    "hits@10_mean": np.mean(sinkhorn_hits10),
                    "hits@10_std": np.std(sinkhorn_hits10),
                    "MRR_mean": np.mean(sinkhorn_mrr),
                    "MRR_std": np.std(sinkhorn_mrr),
                },
            }

    summary_file = f"results/summary_{timestamp}.json"
    with open(summary_file, "w") as f:
        json.dump(summary, f, indent=2)

    # Print summary
    print("\n" + "=" * 80)
    print("EXPERIMENT SUMMARY")
    print("=" * 80)
    for dataset, stats in summary.items():
        print(f"\n{dataset}:")
        print(
            f"  Hungarian - Hits@1: {stats['hungarian']['hits@1_mean']:.2f}% ± {stats['hungarian']['hits@1_std']:.2f}%"
        )
        print(
            f"  Sinkhorn  - Hits@1: {stats['sinkhorn']['hits@1_mean']:.2f}% ± {stats['sinkhorn']['hits@1_std']:.2f}%"
        )
        print(
            f"              Hits@10: {stats['sinkhorn']['hits@10_mean']:.2f}% ± {stats['sinkhorn']['hits@10_std']:.2f}%"
        )
        print(
            f"              MRR: {stats['sinkhorn']['MRR_mean']:.2f}% ± {stats['sinkhorn']['MRR_std']:.2f}%"
        )

    print(f"\nResults saved to:")
    print(f"  - {output_file}")
    print(f"  - {summary_file}")


if __name__ == "__main__":
    print("=" * 80)
    print("SEU Entity Alignment - Automated Experiment Runner")
    print("=" * 80)
    print(f"Detected datasets: {len(DATASETS)}")
    for kg_path, trans_name in DATASETS:
        print(f"  - {kg_path} (using {trans_name} translations)")
    print(f"Feature mode: hybrid-level")
    print(f"Runs per dataset: {NUM_RUNS}")
    print(f"Graph depth: {GRAPH_DEPTH}")
    print("=" * 80)

    start_time = time.time()
    results = run_all_experiments()
    total_time = time.time() - start_time

    save_results(results)

    print(f"\nTotal execution time: {total_time/60:.2f} minutes")
    print("All experiments completed!")
