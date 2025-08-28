import inflection
import re
from string import punctuation, digits
import os
import pandas as pd
from tqdm import tqdm
import pickle
import torch
import torch.nn as nn
import torch.nn.functional as F
try:
    from sklearn.externals import joblib
except:
    import joblib

# English stop words
stop_words = {'i', 'me', 'my', 'myself', 'we', 'our', 'ours', 'ourselves', 'you', 'your', 'yours', 'yourself',
              'yourselves', 'he', 'him', 'his', 'himself', 'she', 'her', 'hers', 'herself', 'it', 'its', 'itself',
              'they', 'them', 'their', 'theirs', 'themselves', 'what', 'which', 'who', 'whom', 'this', 'that', 'these',
              'those', 'am', 'is', 'are', 'was', 'were', 'be', 'been', 'being', 'have', 'has', 'had', 'having', 'do',
              'does', 'did', 'doing', 'a', 'an', 'the', 'and', 'but', 'if', 'or', 'because', 'as', 'until', 'while',
              'of', 'at', 'by', 'for', 'with', 'about', 'against', 'between', 'into', 'through', 'during', 'before',
              'after', 'above', 'below', 'to', 'from', 'up', 'down', 'in', 'out', 'on', 'off', 'over', 'under', 'again',
              'further', 'then', 'once', 'here', 'there', 'when', 'where', 'why', 'how', 'all', 'any', 'both', 'each',
              'few', 'more', 'most', 'other', 'some', 'such', 'no', 'nor', 'not', 'only', 'own', 'same', 'so', 'than',
              'too', 'very', 's', 't', 'can', 'will', 'just', 'don', 'should', 'now', 'd', 'll', 'm', 'o', 're', 've',
              'y', 'ain', 'aren', 'couldn', 'didn', 'doesn', 'hadn', 'hasn', 'haven', 'isn', 'ma', 'mightn', 'mustn',
              'needn', 'shan', 'shouldn', 'wasn', 'weren', 'won', 'wouldn', 'b', 'c', 'e', 'f', 'g', 'h', 'j', 'k', 'l',
              'n', 'p', 'q', 'u', 'v', 'w', 'x', 'z', 'us', 'always', 'already', 'would', 'however', 'perhaps', 'done',
              'cannot', 'can', 'sure', 'without', 'hi', 'could', 'doesn', 'must', 'able', 'much', 'everyone', 'anyone',
              'whatever', 'anyhow', 'yet', 'hence'}

# Java language keywords
java_keywords = {'abstract', 'assert', 'boolean', 'break', 'byte', 'case', 'catch', 'char', 'class', 'const',
                 'continue', 'default', 'do', 'double', 'else', 'enum', 'extends', 'false', 'final', 'finally', 'float',
                 'for', 'goto', 'if', 'implements', 'import', 'instanceof', 'int', 'interface', 'long', 'native', 'new',
                 'null', 'package', 'private', 'protected', 'public', 'return', 'short', 'static', 'strictfp', 'super',
                 'switch', 'synchronized', 'this', 'throw', 'throws', 'transient', 'true', 'try', 'void', 'volatile',
                 'while', 'blc'}
filter_words_set = stop_words.union(java_keywords)

def normalize(tokens):
    normalized_tokens = [tok.lower() for tok in tokens]
    return normalized_tokens


def filter_words(tokens):
    tokens = [tok for tok in tokens if tok not in filter_words_set and len(tok) > 1]
    return tokens


def tokenize(sent):
    filter_sent = re.sub('[^a-zA-Z]', ' ', sent)
    tokens = filter_sent.split()
    return tokens

def split_camelcase(tokens, retain_camelcase=True):
    """
    :param tokens: [str]
    :param retain_camelcase: if True, the corpus will retain camel words after splitting them.
    :return:
    """

    def split_by_punc(token):
        new_tokens = []
        split_toks = re.split(fr'[{punctuation}]+', token)
        if len(split_toks) > 1:
            return_tokens.remove(token)
            for st in split_toks:
                if not st:  # st may be '', e.g. tok = '*' then split_toks = ['', '']
                    continue
                return_tokens.append(st)
                new_tokens.append(st)
        return new_tokens

    def split_by_camel(token):
        camel_split = inflection.underscore(token).split('_')
        if len(camel_split) > 1:
            if any([len(cs) > 2 for cs in camel_split]):
                return_tokens.extend(camel_split)
                camel_word_split_record[token] = camel_split
                if not retain_camelcase:
                    return_tokens.remove(token)

    camel_word_split_record = {}  # record camel words and their generation e.g. CheckBuff: [check, buff]
    # return_tokens = tokens[:]
    return_tokens = []
    for tok in tokens:
        return_tokens.append(tok)
        if not bool(re.search(r'[a-zA-Z]', tok)):
            continue
        new_tokens = split_by_punc(tok)
        new_tokens = new_tokens if new_tokens else [tok]
        for nt in new_tokens:
            split_by_camel(nt)
    return return_tokens, camel_word_split_record

def record_camel_word_split(camel_word_split_record):
    processed_camel_word_split_record = {}
    for camel_word, split_camel in camel_word_split_record.items():
        camel_word = normalize([camel_word])
        camel_word = filter_words(camel_word)

        split_camel = normalize(split_camel)
        split_camel = filter_words(split_camel)
        if camel_word and split_camel:
            processed_camel_word_split_record[camel_word[0]] = split_camel
    return processed_camel_word_split_record

def preprocess(sentence):
    tokens = tokenize(sentence)
    split_tokens, camel_word_split_record = split_camelcase(tokens)
    normalized_tokens = normalize(split_tokens)
    filter_tokens = filter_words(normalized_tokens)

    processed_camel_word_split_record = record_camel_word_split(camel_word_split_record)

    return filter_tokens, camel_word_split_record, processed_camel_word_split_record

def load_file(path):
    assert path[-4:] == '.pkl'
    try:
        with open(path, 'rb') as f:
            file = pickle.load(f)
    except FileNotFoundError:
        with open(f'{path[:-4]}.joblib', 'rb') as f:
            file = joblib.load(f)
    return file


def save_file(path, file):
    assert path[-4:] == '.pkl'
    try:
        with open(path, 'wb') as f:
            pickle.dump(file, f)
    except OverflowError:
        os.remove(path)
        with open(f'{path[:-4]}.joblib', 'wb') as f:
            joblib.dump(file, f)

def load_and_split_reports(report_corpus, commit2paths, split_ratio='8:1:1'):

    def filter_buggy_path(commit2paths, buggy_file_paths, commit):
        """filter out buggy files that are 'ADD'"""
        valid_paths = [path for path in buggy_file_paths if path in commit2paths[commit]]
        return valid_paths

    """
    sort reports by 'report_timestamp', then split.
    """
    sorted_report_corpus = report_corpus.sort_values('report_timestamp')
    n_train = int(int(split_ratio.split(':')[0]) / 10 * len(sorted_report_corpus))
    n_val = int(int(split_ratio.split(':')[1]) / 10 * len(sorted_report_corpus))
    train_reports = sorted_report_corpus.iloc[:n_train]
    val_reports = sorted_report_corpus.iloc[n_train: n_train + n_val]
    test_reports = sorted_report_corpus.iloc[n_train + n_val:]

    print(f'split_ratio = {split_ratio}')
    print(f'train reports: {len(train_reports)}')
    print(f'val   reports: {len(val_reports)}')
    print(f'test  reports: {len(test_reports)}')


    for reports, tag in zip((train_reports, val_reports, test_reports), ('train', 'val', 'test')):
        valid_buggy_paths = []
        for _, r in tqdm(reports.iterrows(), ncols=80):
            commit = r['commit']
            buggy_file_paths = r['buggy_paths'].split('\n')
            # 有效的buggy files，即在当前版本中存在的文件名才是有效的
            buggy_file_paths = filter_buggy_path(commit2paths, buggy_file_paths, commit)
            # tomcat train中有6个没有buggy files的['83fe69a','7963a16','d9330a9','3560f39','c0f1bb9','1aced29']
            # val中有1个没有buggy files的: '7fd7279'
            # test中有2个没有buggy files的: ['5344de0',''c0c5017]
            if not buggy_file_paths:
                if tag == 'train':
                    train_reports = train_reports[train_reports['commit'] != commit]
                elif tag == 'val':
                    val_reports = val_reports[val_reports['commit'] != commit]
                elif tag == 'test':
                    test_reports = test_reports[test_reports['commit'] != commit]
            else:
                valid_buggy_paths.append('\n'.join(buggy_file_paths))

        if tag == 'train':
            assert len(valid_buggy_paths) == len(train_reports)
            train_reports['valid_buggy_paths'] = valid_buggy_paths
        elif tag == 'val':
            assert len(valid_buggy_paths) == len(val_reports)
            val_reports['valid_buggy_paths'] = valid_buggy_paths
        elif tag == 'test':
            assert len(valid_buggy_paths) == len(test_reports)
            test_reports['valid_buggy_paths'] = valid_buggy_paths

    return train_reports, val_reports, test_reports


def prepare_labels(labels_dict, num_files):

    all_labels_dict = {}
    positive_indices_dict = {}
    for tag in ('train', 'val', 'test'):
        labels = []
        positive_indices_list = []
        for report_id, pos_file_ids in labels_dict[tag][1].items():
            l = torch.zeros(num_files)
            l[pos_file_ids] = 1
            labels.append(l)
            positive_indices_list.append(torch.tensor(pos_file_ids, dtype=torch.long))
        labels_matrix_tensor = torch.stack(labels)

        all_labels_dict[tag] = labels_matrix_tensor
        positive_indices_dict[tag] = positive_indices_list

    return all_labels_dict['train'], all_labels_dict['val'], all_labels_dict['test'], positive_indices_dict['train'], positive_indices_dict['val'], positive_indices_dict['test']


def prepare_labels1(label_dict, num_files, candidate_file_ID_dict):
    # label_dict: {issue_id: [positive_developer_id1, positive_developer_id2, ...]}
    # num_developers: total number of developers

    # 假设输入的issues是按照某个顺序排列的，并且label_dict的key与这个顺序相对应
    positive_indices_list = []
    for report_id, pos_files in label_dict[1].items():
        pos_idxs = [candidate_file_ID_dict[file_id] for file_id in pos_files]
        # 确保索引在范围内
        pos_idxs = [min(file_id, num_files - 1) for file_id in pos_idxs]
        positive_indices_list.append(torch.tensor(pos_idxs, dtype=torch.long))

    return positive_indices_list

class MultiLabelRankingLoss(nn.Module):
    def __init__(self, margin=1.0):
        super(MultiLabelRankingLoss, self).__init__()
        self.margin = margin

    def forward(self, scores, positive_indices, num_developers, train_candidate_files_dict, mode):
        # scores: [batch_size, num_developers]
        # positive_indices: 一个包含每个issue的positive developer索引的列表（每个列表对应一个issue）
        # num_developers: developers的总数

        # 确保scores和positive_indices的batch_size一致
        assert scores.size(0) == len(positive_indices)

        if mode == 'train':
            loss = 0.0
            for i, pos_idxs in enumerate(positive_indices):
                # 训练集里的负样本是选取的300个
                candidates = train_candidate_files_dict[i]
                candidates_dict = {value: index for index, value in enumerate(candidates)}

                neg_idxs = torch.tensor([candidates_dict[j] for j in candidates if j not in pos_idxs], dtype=torch.long, device=scores.device)
                new_pos_idxs = torch.tensor([candidates_dict[j.item()] for j in pos_idxs], dtype=torch.long, device=scores.device)

                # 获取当前issue的正样本得分和负样本得分
                pos_scores = scores[i, new_pos_idxs]
                neg_scores = scores[i, neg_idxs]

                # 对每个正样本，计算其相对于所有负样本的合页损失
                hinge_losses = F.relu(neg_scores.unsqueeze(1) - pos_scores.unsqueeze(0) + self.margin)
                hinge_losses = hinge_losses.view(-1)  # 展平为一维
                loss += hinge_losses.sum()

            # 计算平均损失，使用所有正样本-负样本对作为分母
            num_pos_neg_pairs = sum(len(pos_idxs) * (num_developers - len(pos_idxs)) for pos_idxs in positive_indices)
            if num_pos_neg_pairs == 0:  # 避免除以零
                return torch.tensor(0.0, device=scores.device)
            loss /= num_pos_neg_pairs
            return loss
        else:
            loss = 0.0
            for i, pos_idxs in enumerate(positive_indices):
                neg_idxs = torch.tensor([j for j in range(num_developers) if j not in pos_idxs], dtype=torch.long,
                                        device=scores.device)

                # 获取当前issue的正样本得分和负样本得分
                pos_scores = scores[i, pos_idxs]
                neg_scores = scores[i, neg_idxs]

                # 对每个正样本，计算其相对于所有负样本的合页损失
                hinge_losses = F.relu(neg_scores.unsqueeze(1) - pos_scores.unsqueeze(0) + self.margin)
                hinge_losses = hinge_losses.view(-1)  # 展平为一维
                loss += hinge_losses.sum()

            # 计算平均损失，使用所有正样本-负样本对作为分母
            num_pos_neg_pairs = sum(len(pos_idxs) * (num_developers - len(pos_idxs)) for pos_idxs in positive_indices)
            if num_pos_neg_pairs == 0:  # 避免除以零
                return torch.tensor(0.0, device=scores.device)
            loss /= num_pos_neg_pairs
            return loss

def accuracy_at_one_multi_label(scores, labels, K=10):
    # 获取每个issue的最高得分索引
    _, predicted_indices = torch.max(scores, dim=1)

    # 初始化准确率计数器
    correct_predictions = 0
    # 遍历每个样本的预测索引和正样本索引列表
    for predicted_idx, positive_indices in zip(predicted_indices, labels):
        # 检查预测的索引是否在正样本索引列表中
        if predicted_idx.item() in positive_indices:
            correct_predictions += 1

    accuracy = correct_predictions / scores.shape[0]

    # 初始化 MRR 的总和
    reciprocal_rank = []
    avg_p = []
    at_k = [0] * K  # 计算前top10的结果
    num_samples = scores.shape[0]
    # 遍历每个样本计算MRR
    valid_report_num = 0
    for i in range(num_samples):
        candidate_score = {dev_id: score for dev_id, score in enumerate(scores[i])}
        sorted_candidate_score = dict(sorted(candidate_score.items(), key=lambda item: item[1], reverse=True))
        ranked_candidates = list(sorted_candidate_score.keys())
        ranked_candidates_index = {dev_id: rank_index for rank_index, dev_id in enumerate(ranked_candidates)}
        # 找到第一个相关开发者的索引位置（即第一个True的位置）
        relevant_indices = labels[i].tolist()
        if len(relevant_indices) > 0:  # 确保存在相关开发者  numel()是统计数量的吧
            valid_report_num += 1
            # 找到第一个相关开发者的排名（索引位置加1，因为排名通常从1开始）
            top_rank = min([ranked_candidates_index[file_id] for file_id in relevant_indices])
            reciprocal_rank.append(1 / (top_rank+1))
            if (top_rank + 1) <= K:
                at_k[top_rank] += 1

            buggy_code_ranks = list(sorted([ranked_candidates_index[file_id] for file_id in relevant_indices]))
            precision_k = [(i + 1) / (rank + 1) for i, rank in enumerate(buggy_code_ranks)]
            avg_p.append(sum(precision_k) / len(buggy_code_ranks))

    mean_avg_p = sum(avg_p) / valid_report_num
    mrr = sum(reciprocal_rank) / valid_report_num
    hit_k = [sum(at_k[:i + 1]) / valid_report_num for i in range(K)]

    return mean_avg_p, mrr, hit_k


def evaluate_train(scores, labels, train_candidate_files_dict, K=10):
    # 获取每个issue的最高得分索引
    _, predicted_indices = torch.max(scores, dim=1)

    # 初始化 MRR 的总和
    reciprocal_rank = []
    avg_p = []
    at_k = [0] * K  # 计算前top10的结果
    num_samples = scores.shape[0]
    valid_report_num = 0
    for ns in range(num_samples):
        # 训练集里的负样本是选取的300个
        candidates = train_candidate_files_dict[ns]
        candidates_dict = {value: index for index, value in enumerate(candidates)}
        relevant_indices = labels[ns].tolist()
        new_relevant_indices = [candidates_dict[j] for j in relevant_indices]

        candidate_score = {dev_id: score for dev_id, score in enumerate(scores[ns])}
        sorted_candidate_score = dict(sorted(candidate_score.items(), key=lambda item: item[1], reverse=True))
        ranked_candidates = list(sorted_candidate_score.keys())
        ranked_candidates_index = {dev_id: rank_index for rank_index, dev_id in enumerate(ranked_candidates)}
        # 找到第一个相关开发者的索引位置（即第一个True的位置）

        if len(new_relevant_indices) > 0:  # 确保存在相关开发者  numel()是统计数量的吧
            valid_report_num += 1
            # 找到第一个相关开发者的排名（索引位置加1，因为排名通常从1开始）
            top_rank = min([ranked_candidates_index[file_id] for file_id in new_relevant_indices])
            reciprocal_rank.append(1 / (top_rank+1))
            if (top_rank + 1) <= K:
                at_k[top_rank] += 1

            buggy_code_ranks = list(sorted([ranked_candidates_index[file_id] for file_id in new_relevant_indices]))
            precision_k = [(i + 1) / (rank + 1) for i, rank in enumerate(buggy_code_ranks)]
            avg_p.append(sum(precision_k) / len(buggy_code_ranks))

    mean_avg_p = sum(avg_p) / valid_report_num
    mrr = sum(reciprocal_rank) / valid_report_num
    hit_k = [sum(at_k[:i + 1]) / valid_report_num for i in range(K)]

    return mean_avg_p, mrr, hit_k

def prepare_labels_new(labels_dict, num_files):

    all_labels_dict = {}
    positive_indices_dict = {}
    for tag in ('train', 'val', 'test'):
        labels = []
        positive_indices_list = []
        for report_id, pos_file_ids in labels_dict[tag][1].items():
            l = torch.zeros(num_files)
            l[pos_file_ids] = 1
            labels.append(l)
            positive_indices_list.append(torch.tensor(pos_file_ids, dtype=torch.long))
        labels_matrix_tensor = torch.stack(labels)

        all_labels_dict[tag] = labels_matrix_tensor
        positive_indices_dict[tag] = positive_indices_list

    return all_labels_dict['train'], all_labels_dict['val'], all_labels_dict['test'], positive_indices_dict['train'], positive_indices_dict['val'], positive_indices_dict['test']

def metrics(scores, labels, K=10):
    # 获取每个issue的最高得分索引
    _, predicted_indices = torch.max(scores, dim=1)

    # 初始化 MRR 的总和
    reciprocal_rank = []
    avg_p = []
    at_k = [0] * K  # 计算前top10的结果
    num_samples = scores.shape[0]
    # 遍历每个样本计算MRR
    valid_report_num = 0
    for i in range(num_samples):
        candidate_score = {dev_id: score for dev_id, score in enumerate(scores[i])}
        sorted_candidate_score = dict(sorted(candidate_score.items(), key=lambda item: item[1], reverse=True))
        ranked_candidates = list(sorted_candidate_score.keys())
        ranked_candidates_index = {dev_id: rank_index for rank_index, dev_id in enumerate(ranked_candidates)}
        # 找到第一个相关开发者的索引位置（即第一个True的位置）
        relevant_indices = labels[i].tolist()
        if len(relevant_indices) > 0:  # 确保存在相关开发者  numel()是统计数量的吧
            valid_report_num += 1
            # 找到第一个相关开发者的排名（索引位置加1，因为排名通常从1开始）
            top_rank = min([ranked_candidates_index[file_id] for file_id in relevant_indices])
            reciprocal_rank.append(1 / (top_rank+1))
            if (top_rank + 1) <= K:
                at_k[top_rank] += 1

            buggy_code_ranks = list(sorted([ranked_candidates_index[file_id] for file_id in relevant_indices]))
            precision_k = [(i + 1) / (rank + 1) for i, rank in enumerate(buggy_code_ranks)]
            avg_p.append(sum(precision_k) / len(buggy_code_ranks))

    mean_avg_p = sum(avg_p) / valid_report_num
    mrr = sum(reciprocal_rank) / valid_report_num
    hit_k = [sum(at_k[:i + 1]) / valid_report_num for i in range(K)]

    return mean_avg_p, mrr, hit_k