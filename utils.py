import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

class EarlyStopping:
    """Early stops the training if validation mAP doesn't improve after a given patience."""

    def __init__(self, patience=7, verbose=False, delta=0, path='checkpoint.pt', trace_func=print):
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_loss_min = np.Inf
        self.delta = delta
        self.path = path
        self.trace_func = trace_func

    def __call__(self, val_loss, model):

        score = -val_loss

        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(val_loss, model)
        elif score < self.best_score + self.delta:
            self.counter += 1
            self.trace_func(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.save_checkpoint(val_loss, model)
            self.counter = 0

    def save_checkpoint(self, val_loss, model):
        '''Saves model when validation mAP increases.'''
        if self.verbose:
            self.trace_func(f'Validation loss decreased ({self.val_loss_min:.6f} --> {val_loss:.6f}).  Saving model ...')
        torch.save(model.state_dict(), self.path)
        self.val_loss_min = val_loss

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


class MultiLabelRankingLoss(nn.Module):
    def __init__(self, margin=1.0):
        super(MultiLabelRankingLoss, self).__init__()
        self.margin = margin

    def forward(self, scores, positive_indices, num_developers, train_candidate_files_dict, mode):
        # scores: [batch_size, num_developers]
        # positive_indices
        # num_developers

        assert scores.size(0) == len(positive_indices)

        if mode == 'train':
            loss = 0.0
            for i, pos_idxs in enumerate(positive_indices):
                candidates = train_candidate_files_dict[i]
                candidates_dict = {value: index for index, value in enumerate(candidates)}

                neg_idxs = torch.tensor([candidates_dict[j] for j in candidates if j not in pos_idxs], dtype=torch.long, device=scores.device)
                new_pos_idxs = torch.tensor([candidates_dict[j.item()] for j in pos_idxs], dtype=torch.long, device=scores.device)

                pos_scores = scores[i, new_pos_idxs]
                neg_scores = scores[i, neg_idxs]

                hinge_losses = F.relu(neg_scores.unsqueeze(1) - pos_scores.unsqueeze(0) + self.margin)
                hinge_losses = hinge_losses.view(-1)  
                loss += hinge_losses.sum()

            num_pos_neg_pairs = sum(len(pos_idxs) * (num_developers - len(pos_idxs)) for pos_idxs in positive_indices)
            if num_pos_neg_pairs == 0:  
                return torch.tensor(0.0, device=scores.device)
            loss /= num_pos_neg_pairs
            return loss
        else:
            loss = 0.0
            for i, pos_idxs in enumerate(positive_indices):
                neg_idxs = torch.tensor([j for j in range(num_developers) if j not in pos_idxs], dtype=torch.long,
                                        device=scores.device)

                pos_scores = scores[i, pos_idxs]
                neg_scores = scores[i, neg_idxs]

                hinge_losses = F.relu(neg_scores.unsqueeze(1) - pos_scores.unsqueeze(0) + self.margin)
                hinge_losses = hinge_losses.view(-1)  
                loss += hinge_losses.sum()

            num_pos_neg_pairs = sum(len(pos_idxs) * (num_developers - len(pos_idxs)) for pos_idxs in positive_indices)
            if num_pos_neg_pairs == 0:  
                return torch.tensor(0.0, device=scores.device)
            loss /= num_pos_neg_pairs
            return loss

def accuracy_at_one_multi_label(scores, labels, K=10):
    _, predicted_indices = torch.max(scores, dim=1)

    correct_predictions = 0
    for predicted_idx, positive_indices in zip(predicted_indices, labels):
        if predicted_idx.item() in positive_indices:
            correct_predictions += 1

    accuracy = correct_predictions / scores.shape[0]

    reciprocal_rank = []
    avg_p = []
    at_k = [0] * K  
    num_samples = scores.shape[0]
    valid_report_num = 0
    for i in range(num_samples):
        candidate_score = {dev_id: score for dev_id, score in enumerate(scores[i])}
        sorted_candidate_score = dict(sorted(candidate_score.items(), key=lambda item: item[1], reverse=True))
        ranked_candidates = list(sorted_candidate_score.keys())
        ranked_candidates_index = {dev_id: rank_index for rank_index, dev_id in enumerate(ranked_candidates)}
        relevant_indices = labels[i].tolist()
        if len(relevant_indices) > 0: 
            valid_report_num += 1
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


def metrics(scores, labels, K=10):
    _, predicted_indices = torch.max(scores, dim=1)

    reciprocal_rank = []
    avg_p = []
    at_k = [0] * K  
    num_samples = scores.shape[0]
    valid_report_num = 0
    for i in range(num_samples):
        candidate_score = {dev_id: score for dev_id, score in enumerate(scores[i])}
        sorted_candidate_score = dict(sorted(candidate_score.items(), key=lambda item: item[1], reverse=True))
        ranked_candidates = list(sorted_candidate_score.keys())
        ranked_candidates_index = {dev_id: rank_index for rank_index, dev_id in enumerate(ranked_candidates)}
        relevant_indices = labels[i].tolist()
        if len(relevant_indices) > 0: 
            valid_report_num += 1
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