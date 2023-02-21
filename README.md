# SLIM: Sparsified Late Interaction for Multi-Vector Retrieval with Inverted Indexes

This page describes how to implement [SLIM](https://arxiv.org/abs/2302.06587). The source code is based on [this branch](https://github.com/facebookresearch/dpr-scale/tree/citadel) of the dpr-scale repo.
```
@misc{https://doi.org/10.48550/arxiv.2302.06587,
doi = {10.48550/ARXIV.2302.06587},
url = {https://arxiv.org/abs/2302.06587},
author = {Li, Minghan and Lin, Sheng-Chieh and Ma, Xueguang and Lin, Jimmy},
keywords = {Information Retrieval (cs.IR), FOS: Computer and information sciences, FOS: Computer and information sciences},
title = {SLIM: Sparsified Late Interaction for Multi-Vector Retrieval with Inverted Indexes},
publisher = {arXiv},
year = {2023},
copyright = {Creative Commons Attribution 4.0 International}
}

```
In the following, we describe how to train, encode, rerank, and retrieve with SLIM on MS MARCO passage-v1 and TREC DeepLearning 2019/2020.
## Dependencies
First, make sure you have [Anaconda3](https://docs.anaconda.com/anaconda/install/index.html) installed.
Then use conda to create a new environment and activate it:
```
conda create -n dpr-scale python=3.8
conda activate dpr-scale
```
Now let's install the packages. First, follow the instructions [here](https://pytorch.org/get-started/locally/) to install PyTorch on your machine.
Finally install the packages in `requirement.txt`. Remember to comment out the packages in the .txt file that you've already installed to avoid conflicts.
```
pip install -r requirement.txt
```

To do retrieval using Pyserini, it is necessary to create another virtual environment due to package conflicts. A detailed instruction about Pyserini could be found [here](https://github.com/castorini/pyserini/blob/master/docs/installation.md).

## MS MARCO Passage-v1
### Data Prep
First, download the data from the [MS MARCO](https://microsoft.github.io/msmarco/) official website. Make sure to download and decompress the Collection, Qrels Train, Qrels Dev, and Queries.

Then, download and decompress the training data `train.jsonl.gz` from [Tevatron](https://huggingface.co/datasets/Tevatron/msmarco-passage/tree/main). We then split the training data into train and dev:
```
PYTHONPATH=. python dpr_scale/utils/prep_msmarco_exp.py --doc_path <train file path> --output_dir_path <output dir path>
```
By default we use 1\% training data as the validation set.

### Pre-trained Model Checkpoints

#### Checkpoints on Huggingface Hub
- [SLIM](https://huggingface.co/castorini/slim-msmarco-passage/tree/main)

- [SLIM++](https://huggingface.co/castorini/slim-pp-msmarco-passage)

#### Indexes
Please follow the instructions [here](https://github.com/castorini/pyserini/blob/master/docs/experiments-slim.md) to use the Lucene indexes from Pyserini.


### Training
To train the model, run:

```
PYTHONPATH=.:$PYTHONPATH python dpr_scale/main.py -m \
--config-name msmarco_aws.yaml \
task=multiterm task/model=mtsplade_model \
task.model.sparse_mode=True \
task.in_batch_eval=True datamodule.num_test_negative=10 trainer.max_epochs=6 \
task.shared_model=True +task.cross_batch=False +task.in_batch=True \
+task.query_topk=20 +task.context_topk=20 \
+task.teacher_coef=0 +task.tau=1 \
+task.query_router_marg_load_loss_coef=0 +task.context_router_marg_load_loss_coef=0 \
+task.query_expert_load_loss_coef=1e-5 +task.context_expert_load_loss_coef=1e-5 \
datamodule.batch_size=8 datamodule.num_negative=7 \
trainer=gpu_1_host trainer.num_nodes=4 trainer.gpus=8
```
where mtsplade is a deprecated name of SLIM. 

### Reranking
To quickly examine the quality of our trained model without the hassle of indexing, we could use the model to rerank the retrieved top-1000 candidates of BM25 and evaluate the results:
```
PATH_TO_OUTPUT_DIR=your_path_to_output_dir
CHECKPOINT_PATH=your_path_to_ckpt
DATA_PATH=/data_path/msmarco_passage/msmarco_corpus.tsv
PATH_TO_QUERIES_TSV=/data_path/msmarco_passage/dev_small.tsv
PATH_TO_TREC_TSV=/data_path/msmarco_passage/bm25.trec

PYTHONPATH=.:$PYTHONPATH python dpr_scale/citadel_scripts/run_reranking.py -m \
--config-name msmarco_aws.yaml \
task=multiterm_rerank task/model=mtsplade_model \
task.shared_model=True \
+task.query_topk=20 +task.context_topk=20 \
+task.output_dir=$PATH_TO_OUTPUT_DIR \
+task.checkpoint_path=$CHECKPOINT_PATH \
datamodule=generate_query_emb \
datamodule.test_path=$PATH_TO_TREC_TSV \
+datamodule.test_question_path=$PATH_TO_QUERIES_TSV \
+datamodule.query_trec=True \
+datamodule.test_passage_path=$DATA_PATH \
+topk=1000 +cross_encoder=False \
+qrel_path=None \
+create_train_dataset=False \
+dataset=msmarco_passage
```
To get the `bm25.trec` file, please see the details [here](https://github.com/castorini/pyserini/blob/master/docs/experiments-msmarco-passage.md).

### Generate embeddings
If you are dealing with large corpus with million of documents, shard the corpus first before encoding.
Run the command with different shards in parallel:
```
CHECKPOINT_PATH=your_path_to_ckpt
for i in {0..5}
do 
    CTX_EMBEDDINGS_DIR=your_path_to_shard00${i}_embeddings
    DATA_PATH=/data_path/msmarco_passage/msmarco_corpus.00${i}.tsv
    PYTHONPATH=.:$PYTHONPATH python dpr_scale/citadel_scripts/generate_multiterm_embeddings.py -m \
    --config-name msmarco_aws.yaml \
    datamodule=generate \
    task.shared_model=True \
    task=multiterm task/model=mtsplade_model \
    +task.query_topk=20 +task.context_topk=20 \
    datamodule.test_path=$DATA_PATH \
    +task.ctx_embeddings_dir=$CTX_EMBEDDINGS_DIR \
    +task.checkpoint_path=$CHECKPOINT_PATH \
    +task.vocab_file=$VOCAB_FILE \
    +task.add_context_id=False > nohup${i}.log 2>&1&
done
```
The last argument `add_context_id` is for analysis if set `True`. 

### Prune embeddings
To reduce the index size, we only keep the embeddings with weights larger than some threshold:
```
pruning_weight=0.5 # default
PYTHONPATH=.:$PYTHONPATH python prune_doc.py \
"$CTX_EMBEDDINGS_DIR/*/shard*/doc/*" \
$OUPUT_DIR \
$VOCAB_FILE \
$pruning_weight 
```

### Compress sparse token vectors
We need to compress the sparse token vectors into `.npz` format using Scipy to save storage space:
```
THRESHOLD=0.0
PYTHONPATH=.:$PYTHONPATH python compress_tok.py \
"$CTX_EMBEDDINGS_DIR/*/shard*/tok/*" \
$OUPUT_DIR \
$THRESHOLD
```
If you want to further decrease the storage for token vectors, you could increase the threshold which basically does the same thing as weight pruning in the section.

### Retrieval
We use Pyserini to do indexing and retrieval. Create an virtual environment for Pyserini and refer to [here](https://github.com/castorini/pyserini/blob/master/docs/experiments-slim.md) for detailed instructions.

### Get evaluation metrics for MSMARCO
This python script uses pytrec_eval in background:
```
python dpr_scale/citadel_scripts/msmarco_eval.py /data_path/data/msmarco_passage/qrels.dev.small.tsv PATH_TO_OUTPUT_TREC_FILE
```

### Get evaluation metrics for TREC DeepLearning 2019 and 2020
We use [Pyserini](https://github.com/castorini/pyserini) to evaluate on trec dl. Feel free to use pytrec_eval as well. The reason is that we need to deal with qrels with different relevance levels in TREC DL. If you plan to use pyserini, please install it in a different environment to avoid package conflicts with dpr-scale.
```
# Recall
python -m pyserini.eval.trec_eval -c -mrecall.1000 -l 2 /data_path/trec_dl/2019qrels-pass.txt PATH_TO_OUTPUT_TREC_FILE

# nDCG@10
python -m pyserini.eval.trec_eval -c -mndcg_cut.10 /data_path/trec_dl/2019qrels-pass.txt PATH_TO_OUTPUT_TREC_FILE
```
For BEIR evaluation, please refer to [CITADEL](https://github.com/facebookresearch/dpr-scale/tree/citadel) for detailed description.

## License
The majority of SLIM is licensed under CC-BY-NC which inherits from [CITADEL](https://github.com/facebookresearch/dpr-scale/tree/citadel).
