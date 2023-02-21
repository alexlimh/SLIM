#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved

import os
import pathlib
import pickle
import torch
import scipy
import numpy as np
import jsonlines
from dpr_scale.utils.utils import PathManager
from pytorch_lightning.utilities.cloud_io import load as pl_load

import collections
from tqdm import tqdm
import concurrent.futures
from dpr_scale.task.mtsplade_task import MultiTermRetrieverTask

class GenerateMultiTermEmbeddingsTask(MultiTermRetrieverTask):
    def __init__(self, ctx_embeddings_dir, checkpoint_path,
        add_context_id, vocab_file=None, weight_threshold=0., rank1=False, **kwargs):
        super().__init__(**kwargs)
        self.ctx_embeddings_dir = ctx_embeddings_dir
        self.checkpoint_path = checkpoint_path
        self.add_context_id = add_context_id # for token/expert distribution analysis
        self.weight_threshold = weight_threshold # for on-the-fly pruning
        self.vocab_file = vocab_file
        self.rank1 = rank1
        pathlib.Path(ctx_embeddings_dir).mkdir(parents=True, exist_ok=True)

    def setup(self, stage: str):
        super().setup("train")
        print(f"Loading checkpoint from {self.checkpoint_path}")
        checkpoint = pl_load(
            self.checkpoint_path, map_location=lambda storage, loc: storage)
        self.load_state_dict(checkpoint['state_dict'])
        with open(self.vocab_file) as f:
            lines = f.readlines()
        self.vocab = []
        for line in lines:
            self.vocab.append(line.strip())

    def forward(self, contexts_ids):
        # encode contexts
        contexts_repr = self.encode_contexts(contexts_ids)  # ctx_cnt x d
        return contexts_repr
    
    def encode_contexts(self, contexts_ids):
        contexts_repr = self._encode_sequence(
            contexts_ids, self.context_encoder,**self.context_kwargs
        )  # ctx_cnt x d
        return contexts_repr

    def _eval_step(self, batch, batch_idx):
        contexts_ids = batch["contexts_ids"]  # bs x ctx_cnt x ctx_len
        contexts_repr = self(contexts_ids)
        batch_sparse_vecs = []
        lengths = contexts_repr["attention_mask"].sum(1)
        for i, length in enumerate(lengths):
            batch_sparse_vecs.append(contexts_repr["sparse_weights"][i][:length, :].to_sparse().detach().cpu())

        contexts_repr = {k:v.detach().cpu() for k, v in contexts_repr.items() if k != "sparse_weights"}
        batch_results = []
        for batch_id, corpus_id in enumerate(batch["corpus_ids"]):
            results = {"id":str(corpus_id), "contents":"", "vector":{}} 
            for position, (expert_topk_ids, expert_topk_weights, attention_score, context_id) in enumerate(zip(contexts_repr["expert_ids"][batch_id],
                                                                                                contexts_repr["expert_weights"][batch_id],
                                                                                                contexts_repr["attention_mask"][batch_id],
                                                                                                contexts_ids["input_ids"].cpu()[batch_id][1:])):
                if attention_score > 0:
                    for expert_id, expert_weight in zip(expert_topk_ids, expert_topk_weights):
                        if expert_weight > self.weight_threshold:
                            term = self.vocab[expert_id.item()]
                            tf = int(expert_weight.item() * 100)
                            results["vector"][term] = max(tf, results["vector"].get(term, 0))    
            batch_results.append(results)
        return batch_results, batch_sparse_vecs

    def test_step(self, batch, batch_idx):
        return self._eval_step(batch, batch_idx)

    def test_epoch_end(self, contexts_reprs):
        results = []
        sparse_vecs = []
        for context_repr, sparse_vec in contexts_reprs:
            results.extend(context_repr)
            sparse_vecs.extend(sparse_vec)

        if not self.ctx_embeddings_dir:
            self.ctx_embeddings_dir = self.trainer.weights_save_path

        doc_dir = os.path.join(self.ctx_embeddings_dir, "doc")
        tok_dir = os.path.join(self.ctx_embeddings_dir, "tok")
        os.makedirs(doc_dir, exist_ok=True)
        os.makedirs(tok_dir, exist_ok=True)
        embedding_path = os.path.join(
            doc_dir, f"split_{self.global_rank:04}.jsonl")
        sparse_embedding_path = os.path.join(
            tok_dir, f"split_{self.global_rank:04}.pt")

        print(f"\nWriting tensors to {embedding_path}")
        with jsonlines.open(embedding_path, 'w') as writer:
            writer.write_all(results)
        
        print(f"\nWriting tensors to {sparse_embedding_path}")
        torch.save(sparse_vecs, sparse_embedding_path)
        torch.distributed.barrier()  # make sure rank 0 waits for all to complete


class GenerateMultiTermQueryEmbeddingsTask(GenerateMultiTermEmbeddingsTask):
    def __init__(
        self,
        hnsw_index=False,
        output_path="/tmp/results.jsonl",
        query_emb_output_dir=None,
        passages="",
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.hnsw_index = hnsw_index
        self.output_path = output_path
        self.query_emb_output_dir = query_emb_output_dir

    def forward(self, query_ids):
        # encode questions
        query_repr = self.encode_queries(query_ids)  # q_cnt x d
        return query_repr

    def encode_queries(self, query_ids):
        query_repr = self._encode_sequence(query_ids, self.query_encoder, **self.query_kwargs)  # bs x d
        return query_repr

    def _eval_step(self, batch, batch_idx):
        query_ids = batch["query_ids"]  # bs x ctx_cnt x ctx_len
        topic_ids = batch["topic_ids"] # add question topic id
        queries_repr = self(query_ids)
        batch_sparse_vecs = []
        batch_sparse_lens = []
        lengths = queries_repr["attention_mask"].sum(1)
        for i, length in enumerate(lengths):
            batch_sparse_lens.append(length.item())
            batch_sparse_vecs.append(queries_repr["sparse_weights"][i][:length, :].detach().cpu().numpy())
        batch_sparse_vecs = np.concatenate(batch_sparse_vecs, 0)
        batch_sparse_vecs = scipy.sparse.csr_matrix(batch_sparse_vecs)

        queries_repr = {k:v.detach().cpu() for k, v in queries_repr.items()}
        batch_embeddings = []
        for batch_id, topic_id in enumerate(batch["topic_ids"]):
            for position, (expert_topk_ids, expert_topk_weights, attention_score) in enumerate(zip(queries_repr["expert_ids"][batch_id],
                                                                            queries_repr["expert_weights"][batch_id],
                                                                            queries_repr["attention_mask"][batch_id])):
                if attention_score > 0:
                    results = {"id":f"{topic_id}_{position}", "vector":{}}
                    for expert_id, expert_weight in zip(expert_topk_ids, expert_topk_weights):
                        if expert_weight > 0:
                            results["vector"][self.vocab[expert_id.item()]] = expert_weight.item()
                    if len(results["vector"]) > 0:
                        batch_embeddings.append(results)
        return batch_embeddings, batch_sparse_vecs, batch_sparse_lens

    def test_step(self, batch, batch_idx):
        return self._eval_step(batch, batch_idx)

    def test_epoch_end(self, queries_reprs):
        embeddings = []
        sparse_vecs = []
        sparse_lens = []
        for batch_queries_repr, batch_sparse_vecs, batch_sparse_lens in tqdm(queries_reprs):
            embeddings.extend(batch_queries_repr)
            sparse_vecs.extend(batch_sparse_vecs)
            sparse_lens.extend(batch_sparse_lens)
        sparse_vecs = scipy.sparse.vstack(sparse_vecs)
        doc_output_dir = os.path.join(self.query_emb_output_dir, "doc")
        tok_output_dir = os.path.join(self.query_emb_output_dir, "tok")
        os.makedirs(doc_output_dir, exist_ok=True)
        os.makedirs(tok_output_dir, exist_ok=True)
        embedding_out_file = os.path.join(doc_output_dir, "query_repr.jsonl")
        sparse_vec_out_file = os.path.join(tok_output_dir, "sparse_vec.npz")
        sparse_range_out_file = os.path.join(tok_output_dir, "sparse_range.pkl")

        pathlib.Path(embedding_out_file).parent.mkdir(parents=True, exist_ok=True)
        print(f"\nWriting tensors to {embedding_out_file}")
        with jsonlines.open(embedding_out_file, 'w') as writer:
            writer.write_all(embeddings)
        
        print(f"\nWriting tensors to {sparse_vec_out_file}")
        scipy.sparse.save_npz(sparse_vec_out_file, sparse_vecs)

        print(f"\nWriting tensors to {sparse_range_out_file}")
        start = 0
        sparse_ranges = []
        for length in sparse_lens:
            end = start + length
            sparse_ranges.append((start, end))
            start = end
        with open(sparse_range_out_file, "wb") as f:
            pickle.dump(sparse_ranges, f)

class RerankMultiTermRetrieverTask(MultiTermRetrieverTask):
    def __init__(
        self,
        checkpoint_path,
        output_dir,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.checkpoint_path = checkpoint_path
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)
    
    def setup(self, stage: str):
        super().setup("train")
        print(f"Loading checkpoint from {self.checkpoint_path}")
        checkpoint = pl_load(
            self.checkpoint_path, map_location=lambda storage, loc: storage)
        self.load_state_dict(checkpoint['state_dict'])
    
    def expert_sim_score(self, query_repr, context_repr):
        bz = query_repr["sparse_weights"].shape[0]
        lq = query_repr["sparse_weights"].shape[1]
        ld = context_repr["sparse_weights"].shape[1]
        query_expert_weights = query_repr["sparse_weights"].view(-1, query_repr["sparse_weights"].shape[-1])
        context_expert_weights = context_repr["sparse_weights"].view(-1, context_repr["sparse_weights"].shape[-1])
        scores = torch.sparse.mm(query_expert_weights.to_sparse(), context_expert_weights.t().to_sparse()).to_dense()
        scores = scores.view(bz, lq, bz, ld).permute(0,2,1,3)
        indices = torch.LongTensor([i for i in range(context_repr["sparse_weights"].shape[0])])
        scores = scores[indices, indices]
        scores = scores.max(-1).values.sum(1)
        return scores

    def encode_queries(self, query_ids):
        query_repr = self._encode_sequence(query_ids, self.query_encoder, **self.query_kwargs)  # bs x d
        return query_repr

    def encode_contexts(self, contexts_ids):
        contexts_repr = self._encode_sequence(
            contexts_ids, self.context_encoder, **self.context_kwargs
        )  # ctx_cnt x d
        return contexts_repr

    def _eval_step(self, batch, batch_idx):
        q_ids = batch["query_ids"]  # bs x q_cnt x q_len
        ctx_ids = batch["contexts_ids"]
        q_repr, ctx_repr = self(q_ids, ctx_ids)
        scores = self.expert_sim_score(q_repr, ctx_repr)
        return [batch["qid"], batch["ctx_id"], scores.cpu()]

    def test_step(self, batch, batch_idx):
        return self._eval_step(batch, batch_idx)
    
    def test_epoch_end(self, test_outputs):
        qids, ctx_ids, scores = [], [], []
        
        for entry in test_outputs:
            b_qids, b_ctx_ids, b_scores = entry
            qids.extend(b_qids)
            ctx_ids.extend(b_ctx_ids)
            scores.append(b_scores)

        scores = torch.cat(scores, dim=0)
        out_file = os.path.join(
            self.output_dir, f"scores_{self.global_rank:04}.pkl")
        print(f"\nWriting scores to {out_file}")
        with PathManager.open(out_file, mode="wb") as f:
            pickle.dump(scores, f, protocol=4)

        out_file = os.path.join(
            self.output_dir, f"qids_{self.global_rank:04}.pkl")
        with PathManager.open(out_file, mode="wb") as f:
            pickle.dump(qids, f, protocol=4)

        out_file = os.path.join(
            self.output_dir, f"ctx_ids_{self.global_rank:04}.pkl")
        with PathManager.open(out_file, mode="wb") as f:
            pickle.dump(ctx_ids, f, protocol=4)
        torch.distributed.barrier()  # make sure rank 0 waits for all to complete