import os, sys
import glob
import jsonlines
import pickle
from tqdm import tqdm

def main(input_paths, output_dir, vocab_file, threshold=0):
    os.makedirs(output_dir, exist_ok=True)
    with open(vocab_file) as f:
        lines = f.readlines()
    vocab = []
    for line in lines:
        vocab.append(line.strip())

    input_paths = glob.glob(input_paths)
    for i, input_path in tqdm(enumerate(list(input_paths))):
        results = []
        with jsonlines.open(input_path) as f:
            for entry in tqdm(f):
                vector = {}
                for term, weight in entry["vector"].items():
                    if weight > int(float(threshold) * 100):
                        vector[term] = weight
                entry["vector"] = vector
                entry["contents"] = ""
                if len(vector) > 0:
                    results.append(entry)
        with jsonlines.open(f'{output_dir}/split_{i:03d}.jsonl', 'w') as writer:
            writer.write_all(results)

if __name__ == '__main__':
    input_paths = sys.argv[1]
    output_dir = sys.argv[2]
    vocab_file = sys.argv[3]
    threshold = sys.argv[4]
    main(input_paths, output_dir, vocab_file, threshold)
