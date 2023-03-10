# (c) Facebook, Inc. and its affiliates. Confidential and proprietary.
"""
This module computes evaluation metrics for MSMARCO dataset on the ranking task.
Command line:
python msmarco_eval_ranking.py <path_to_reference_file> <path_to_candidate_file>

Creation Date : 06/12/2018
Last Modified : 10/26/2022
Authors : Daniel Campos <dacamp@microsoft.com>, Rutger van Haasteren <ruvanh@microsoft.com>
"""
import json
import csv
import sys
import pytrec_eval
import collections
from collections import Counter
import numpy as np
MaxMRRRank = 10


def load_reference_from_stream(f):
    """Load Reference reference relevant passages
    Args:f (stream): stream to load.
    Returns:qids_to_relevant_passageids (dict): dictionary mapping from query_id (int) to relevant passages (list of ints). 
    """
    qids_to_relevant_passageids = {}
    for line in f:
        try:
            line = line.strip().split('\t')
            qid = int(line[0])
            if qid in qids_to_relevant_passageids:
                pass
            else:
                qids_to_relevant_passageids[qid] = []
            qids_to_relevant_passageids[qid].append(int(line[2]))
        except IOError as error:
            print(error)
            print(f'{line} is not valid format')
        except IndexError as error:
            print(error)
            print(f'{line} is not valid format')
    return qids_to_relevant_passageids


def load_reference(path_to_reference):
    """Load Reference reference relevant passages
    Args:path_to_reference (str): path to a file to load.
    Returns:qids_to_relevant_passageids (dict): dictionary mapping from query_id (int) to relevant passages (list of ints). 
    """
    with open(path_to_reference,'r') as f:
        qids_to_relevant_passageids = load_reference_from_stream(f)
    return qids_to_relevant_passageids


def load_candidate_from_stream(f):
    """Load candidate data from a stream.
    Args:f (stream): stream to load.
    Returns:qid_to_ranked_candidate_passages (dict): dictionary mapping from query_id (int) to a list of 1000 passage ids(int) ranked by relevance and importance
    """
    qid_to_ranked_candidate_passages = {}
    for line in f:
        try:
            line = line.strip().split('\t')
            qid = int(line[0])
            pid = int(line[1])
            rank = int(line[2])
            if qid in qid_to_ranked_candidate_passages:
                pass    
            else:
                # By default, all PIDs in the list of 1000 are 0. Only override those that are given
                tmp = [0] * 1000
                qid_to_ranked_candidate_passages[qid] = tmp
            qid_to_ranked_candidate_passages[qid][rank-1]=pid
        except IOError as error:
            print(error)
            print(f'{line} is not valid format')
        except IndexError as error:
            print(error)
            print(f'{line} is not valid format')
    return qid_to_ranked_candidate_passages


def load_candidate_from_stream_trec(f):
    """Load candidate data from a stream.
    Args:f (stream): stream to load.
    Returns:qid_to_ranked_candidate_passages (dict): dictionary mapping from query_id (int) to a list of 1000 passage ids(int) ranked by relevance and importance
    """
    qid_to_ranked_candidate_passages = {}
    for line in f:
        try:
            line = line.strip().split(' ')
            qid = int(line[0])
            pid = int(line[2])
            rank = int(line[3])
            if qid not in qid_to_ranked_candidate_passages:
                # By default, all PIDs in the list of 1000 are 0. Only override those that are given
                tmp = [0] * 1000
                qid_to_ranked_candidate_passages[qid] = tmp
            qid_to_ranked_candidate_passages[qid][rank-1] = pid
        except IOError as error:
            print(error)
            print(f'{line} is not valid format')
        except IndexError as error:
            print(error)
            print(f'{line} is not valid format')
    return qid_to_ranked_candidate_passages


def load_candidate_from_stream_json(f):
    data = json.load(f)
    return {int(row["id"]): [int(ctx["id"]) for ctx in row["ctxs"]] for row in data}

def load_candidate(path_to_candidate):
    """Load candidate data from a file.
    Args:path_to_candidate (str): path to file to load.
    Returns:qid_to_ranked_candidate_passages (dict): dictionary mapping from query_id (int) to a list of 1000 passage ids(int) ranked by relevance and importance
    """
    with open(path_to_candidate,'r') as f:
        if ".json" in path_to_candidate:
            qid_to_ranked_candidate_passages = load_candidate_from_stream_json(f)
        elif ".trec" in path_to_candidate:
            qid_to_ranked_candidate_passages = load_candidate_from_stream_trec(f)
        else:
            qid_to_ranked_candidate_passages = load_candidate_from_stream(f)
    return qid_to_ranked_candidate_passages


def load_reference_for_trec_eval(path_to_reference):
    ref = {}
    with open(path_to_reference) as inf:
        reader = csv.reader(inf, delimiter='\t')
        for row in reader:
            qid, _, pid, rel = row
            if qid not in ref:
                ref[qid] = {}
            ref[qid][pid] = int(rel)
    return ref


def load_candidate_for_trec_eval(path_to_candidate):
    if ".json" in path_to_candidate:
        with open(path_to_candidate, 'r') as inf:
            data = json.load(inf)
        return {
            row["id"]: {
                ctx["id"]: ctx["score"] for ctx in row["ctxs"]
            } for row in data
        }
    elif ".trec" in path_to_candidate:
        with open(path_to_candidate, 'r') as inf:
            lines = inf.readlines()
        results = collections.defaultdict(dict)
        for line in lines:
            qid, _, doc_id, _, score, _ = line.strip().split(" ")
            results[qid][doc_id] = float(score)
        return results
    else:
        with open(path_to_candidate, 'r') as inf:
            lines = inf.readlines()
        results = collections.defaultdict(dict)
        for line in lines:
            qid, doc_id, _, score = line.strip().split("\t")
            results[qid][doc_id] = float(score)
        return results


def quality_checks_qids(qids_to_relevant_passageids, qids_to_ranked_candidate_passages):
    """Perform quality checks on the dictionaries

    Args:
    p_qids_to_relevant_passageids (dict): dictionary of query-passage mapping
        Dict as read in with load_reference or load_reference_from_stream
    p_qids_to_ranked_candidate_passages (dict): dictionary of query-passage candidates
    Returns:
        bool,str: Boolean whether allowed, message to be shown in case of a problem
    """
    message = ''
    allowed = True

    # Check that we do not have multiple passages per query
    for qid in qids_to_ranked_candidate_passages:
        # Remove all zeros from the candidates
        duplicate_pids = {item for item, count in Counter(qids_to_ranked_candidate_passages[qid]).items() if count > 1}

        if len(duplicate_pids - {0}) > 0:
            message = "Cannot rank a passage multiple times for a single query. QID={qid}, PID={pid}".format(
                    qid=qid, pid=list(duplicate_pids)[0])
            allowed = False

    return allowed, message


def compute_metrics(qids_to_relevant_passageids, qids_to_ranked_candidate_passages):
    """Compute MRR metric
    Args:    
    p_qids_to_relevant_passageids (dict): dictionary of query-passage mapping
        Dict as read in with load_reference or load_reference_from_stream
    p_qids_to_ranked_candidate_passages (dict): dictionary of query-passage candidates
    Returns:
        dict: dictionary of metrics {'MRR': <MRR Score>}
    """
    all_scores = {}
    MRR = []
    ranking = []
    for qid in qids_to_ranked_candidate_passages:
        if qid in qids_to_relevant_passageids:
            ranking.append(0)
            target_pid = qids_to_relevant_passageids[qid]
            candidate_pid = qids_to_ranked_candidate_passages[qid]
            for i in range(0, MaxMRRRank):
                if candidate_pid[i] in target_pid:
                    MRR.append(1/(i + 1))
                    ranking.pop()
                    ranking.append(i+1)
                    break
    if len(ranking) == 0:
        raise IOError("No matching QIDs found. Are you sure you are scoring the evaluation set?")
    
    all_scores['MRR @10'] = np.sum(MRR)/len(qids_to_relevant_passageids)
    all_scores['MRR @10 std'] = np.std(MRR, ddof=1)
    all_scores['QueriesRanked'] = len(qids_to_relevant_passageids)
    return all_scores
                

def compute_metrics_from_files(path_to_reference, path_to_candidate, perform_checks=True):
    """Compute MRR metric
    Args:    
    p_path_to_reference_file (str): path to reference file.
        Reference file should contain lines in the following format:
            QUERYID\tPASSAGEID
            Where PASSAGEID is a relevant passage for a query. Note QUERYID can repeat on different lines with different PASSAGEIDs
    p_path_to_candidate_file (str): path to candidate file.
        Candidate file sould contain lines in the following format:
            QUERYID\tPASSAGEID1\tRank
            If a user wishes to use the TREC format please run the script with a -t flag at the end. If this flag is used the expected format is 
            QUERYID\tITER\tDOCNO\tRANK\tSIM\tRUNID 
            Where the values are separated by tabs and ranked in order of relevance 
    Returns:
        dict: dictionary of metrics {'MRR': <MRR Score>}
    """
    
    qids_to_relevant_passageids = load_reference(path_to_reference)
    qids_to_ranked_candidate_passages = load_candidate(path_to_candidate)
    if perform_checks:
        allowed, message = quality_checks_qids(qids_to_relevant_passageids, qids_to_ranked_candidate_passages)
        if message != '': print(message)

    return compute_metrics(qids_to_relevant_passageids, qids_to_ranked_candidate_passages)


def main():
    """Command line:
    python msmarco_eval_ranking.py <path_to_reference_file> <path_to_candidate_file>
    """

    if len(sys.argv) == 3:
        path_to_reference = sys.argv[1]
        path_to_candidate = sys.argv[2]
        metrics = compute_metrics_from_files(path_to_reference, path_to_candidate)
        print('#####################')
        for metric in sorted(metrics):
            print('{}: {}'.format(metric, metrics[metric]))
        print('#####################')
        print("pytrec eval")
        evaluator = pytrec_eval.RelevanceEvaluator(
            load_reference_for_trec_eval(path_to_reference),
            {'map_cut', 'ndcg_cut', 'recip_rank', 'recall_20', 'recall_50', 'recall_100', 'recall_1000'},
        )
        result = evaluator.evaluate(load_candidate_for_trec_eval(path_to_candidate))
        eval_query_cnt = 0
        ndcg = []
        Map = []
        mrr = []
        recalls = Counter()
        for k in result.keys():
            eval_query_cnt += 1
            ndcg.append(result[k]["ndcg_cut_10"])
            Map.append(result[k]["map_cut_10"])
            mrr.append(result[k]["recip_rank"])
            for topk in [20, 50, 100, 1000]:
                recalls[topk] += result[k][f"recall_{topk}"]
        final_ndcg = sum(ndcg) / eval_query_cnt
        final_Map = sum(Map) / eval_query_cnt
        final_mrr = sum(mrr) / eval_query_cnt
        final_recalls = {}
        for topk in [20, 50, 100, 1000]:
            final_recalls[topk] = recalls[topk] / eval_query_cnt
        print(f"NDCG@10:{final_ndcg}, std={np.std(ndcg, ddof=1)}" )
        print(f"map@10:{final_Map}, std={np.std(Map, ddof=1)}")
        print(f"pytrec_mrr:{final_mrr}, std={np.std(mrr, ddof=1)}")
        for topk in [20, 50, 100, 1000]:
            print(f"recall@{topk}"+":" + str(final_recalls[topk]))
    else:
        print('Usage: msmarco_eval_ranking.py <reference ranking> <candidate ranking>')
        exit()
    

if __name__ == '__main__':
    main()
