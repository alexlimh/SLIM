import os, sys
import glob
import jsonlines
import pickle
import torch
import scipy
import numpy as np
from scipy.sparse import csr_matrix, vstack
from tqdm import tqdm

def torch2scipy(entry):
    i, sparse_vec = entry
    dense_vec = sparse_vec.to_dense().numpy()
    sparse_vec = csr_matrix(dense_vec)
    return (i, sparse_vec)

def main(input_paths, output_dir, threshold):
    os.makedirs(output_dir, exist_ok=True)
    input_paths = sorted(glob.glob(input_paths))
    total_sparse_vecs = []
    sparse_ranges = []
    start = 0
    for i, input_path in tqdm(list(enumerate(input_paths))):
        data = torch.load(input_path)
        for sparse_vec in data:
            end = start + sparse_vec.shape[0]
            sparse_ranges.append((start, end))
            start = end
        sparse_vecs = torch.cat(data, 0).coalesce()
        indices = sparse_vecs.indices().numpy()
        values = sparse_vecs.values().numpy()
        pos = np.where(values >= threshold)[0]
        values = values[pos]
        indices = (indices[0][pos], indices[1][pos])
        sparse_vecs = csr_matrix((values, indices), shape=sparse_vecs.shape)
        total_sparse_vecs.append(sparse_vecs)
    total_sparse_vecs = vstack(total_sparse_vecs)

    output_path = f'{output_dir}/sparse_vec.npz'
    print(f"Writing tensor to {output_path}")
    scipy.sparse.save_npz(output_path, total_sparse_vecs)
    
    output_path = f'{output_dir}/sparse_range.pkl'
    print(f"Writing tensor to {output_path}")
    with open(output_path, "wb") as f:
        pickle.dump(sparse_ranges, f)
    
if __name__ == '__main__':
    input_paths = sys.argv[1]
    output_dir = sys.argv[2]
    threshold = sys.argv[3]
    main(input_paths, output_dir, float(threshold))
