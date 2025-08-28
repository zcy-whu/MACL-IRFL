# 1.先过滤掉那些location超过十个以上的bug report（很有可能就是语法问题之类的）filter_reports()
# 2.生成report-report-similar关系，用codebert计算 generate_report_feat() calculate_RR_relations()
# 3.统计有多少个file genrate_file_nodes()
# 4.生成code之间的共现关系和依赖关系 calculate_CC_relations() generate_file_relations()
# 5.生成report-code-similar关系，用codebert计算 generate_file_feat() generate_RC_relations()
# 6.生成标签
# 7.生成dgl图
import pandas as pd
import pickle
from tqdm import tqdm
from process.utils import *
from collections import Counter, defaultdict
import torch
import torch.nn.functional as F
from transformers import RobertaTokenizer, RobertaConfig, RobertaModel
from torch.utils.data import Dataset, DataLoader
import dgl
from dgl.data.utils import save_graphs, load_graphs

# aspectj(554), birt(4009), eclipse(6261), jdt(6108), swt(4086), tomcat(1020)
# org.aspectj, birt, eclipse.platform.ui, eclipse.jdt.ui, eclipse.platform.swt, tomcat
project_name = 'eclipse'
baseURL = 'F:\\实验\\GNNBL\\processed_data\\'
code_dir = 'F:\\source_code\\eclipse.platform.ui\\'

def filter_reports(baseURL, project_name):
    bug_reports_df = pd.read_csv(f'{baseURL}{project_name}/preprocessed_reports/report_corpus.csv')
    bug_reports_df.fillna('', inplace=True)
    with open(f'{baseURL}{project_name}/collected_codes/commit2code_paths.pkl', 'rb') as f:
        commit2code_paths = pickle.load(f)

    def preprocessReport(reports):
        # 使用del删除列'processed_text'
        # del reports['processed_text']
        report_tokens = []
        report_record = []
        for _, r in tqdm(reports.iterrows(), ncols=80):
            title = r['summary']
            body = r['description']
            r_content = title + ' ' + body
            r_tokens, _, issue_camel_info = preprocess(r_content)

            report_tokens.append(r_tokens[1:])  # remove first token 'bug'
            report_record.append(issue_camel_info)
        reports['report_tokens'] = report_tokens
        reports['report_keywords'] = report_record

        return reports

    def filter_buggy_path(commit2paths, buggy_file_paths, commit):
        """filter out buggy files that are 'ADD'"""
        valid_paths = [path for path in buggy_file_paths if path in commit2paths[commit]]
        return valid_paths

    # sort reports by 'report_timestamp', then split.
    # 统计每个report对应的file的数量，然后统计中位数是多少，删除超过这个数值的issue
    report_locations_num = []
    valid_reports_ids = []
    sorted_report_corpus = bug_reports_df.sort_values('report_timestamp')
    processed_report_corpus = preprocessReport(sorted_report_corpus)
    for report_index, (_, report) in enumerate(processed_report_corpus.iterrows()):
        commit = report['commit']
        buggy_file_paths = report['buggy_paths'].split('\n')
        valid_buggy_paths = filter_buggy_path(commit2code_paths, buggy_file_paths, commit)
        report_locations_num.append(len(valid_buggy_paths))
        if not valid_buggy_paths:
            print(report_index, report['bug_id'], 'has no locations')
        elif len(valid_buggy_paths) >= 10:
            print(report_index, report['bug_id'], 'locations num is', len(valid_buggy_paths))
            print('bug summary:', report['summary'], 'bug description:', report['description'])
            print(valid_buggy_paths)
            print("##################################################################################################")
        else:
            valid_reports_ids.append(report['bug_id'])

    # 使用Counter来统计每个元素的出现次数
    # count_dict = Counter(report_locations_num)
    # for element, count in count_dict.items():
    #     print(f"元素 {element} 出现了 {count} 次")

    filtered_report_corpus = processed_report_corpus.loc[processed_report_corpus['bug_id'].isin(valid_reports_ids)]  # 1056个report过滤之后只剩下1020个
    with open(f'{baseURL}{project_name}/preprocessed_reports/processed_reports', 'wb') as f:
        pickle.dump(filtered_report_corpus, f)

    train_reports, val_reports, test_reports = load_and_split_reports(filtered_report_corpus, commit2code_paths)
    with open(f'{baseURL}{project_name}/preprocessed_reports/train_reports', 'wb') as f:
        pickle.dump(train_reports, f)
    with open(f'{baseURL}{project_name}/preprocessed_reports/val_reports', 'wb') as f:
        pickle.dump(val_reports, f)
    with open(f'{baseURL}{project_name}/preprocessed_reports/test_reports', 'wb') as f:
        pickle.dump(test_reports, f)
    # 为bug report生成索引
    report_corpus = pd.concat([train_reports, val_reports, test_reports], axis=0)
    bugid2idx = {value: index for index, value in enumerate(report_corpus['bug_id'].tolist())}
    with open(f'{baseURL}{project_name}/preprocessed_reports/bugid2idx.pkl', 'wb') as f:
        pickle.dump(bugid2idx, f)

    return filtered_report_corpus


class ReportDataset(Dataset):
    def __init__(self, report_df, tokenizer):
        self.report_df = report_df
        self.tokenizer = tokenizer

    def __len__(self):
        return len(self.report_df)

    def __getitem__(self, idx):
        report = self.report_df.iloc[idx]
        report_id = report['bug_id']
        report_tokens = report['report_tokens']
        report_camel_info = report['report_keywords']
        report_keywords = list(report_camel_info.keys())

        # 合并关键词和 tokens
        text = ' '.join(report_keywords + report_tokens)

        # 使用 tokenizer 进行编码
        inputs = self.tokenizer(text, return_tensors='pt', truncation=True, max_length=512, padding='max_length')
        attention_mask = inputs['attention_mask']  # 这里添加了attention_mask
        # 返回需要的张量（例如 input_ids）和 issue_id
        return inputs['input_ids'], attention_mask, report_id


def generate_report_feat(baseURL, project_name):
    with open(f'{baseURL}{project_name}/preprocessed_reports/processed_reports', 'rb') as f:
        filtered_report_corpus = pickle.load(f)
    # 加载codebert模型
    codebert_url = 'F:/实验/CodeBERT-master/Siamese-model/demo/codebert-base/'
    config = RobertaConfig.from_pretrained(codebert_url)
    tok = RobertaTokenizer.from_pretrained(codebert_url)
    model = RobertaModel.from_pretrained(codebert_url, config=config)
    model.eval()

    # 创建 Dataset 实例
    dataset = ReportDataset(filtered_report_corpus, tok)
    # 创建 DataLoader 实例
    batch_size = 128  # 你可以设置合适的批处理大小
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False)

    report_feat = []
    # 遍历 DataLoader 来获取批处理数据
    with torch.no_grad():
        for batch in tqdm(dataloader):
            input_ids, attention_masks, report_ids = batch
            # 注意：这里我们假设 model 的 forward 方法返回了第二个元素作为特征（这通常不是标准的 RobertaModel 的行为）
            # 你可能需要根据你的模型输出进行调整
            outputs = model(input_ids.squeeze(dim=1), attention_mask=attention_masks.squeeze(dim=1))
            # 假设 outputs[1] 是你想要的特征（这只是一个示例，你需要根据你的模型来调整）
            # 你可能需要处理 outputs 来获取你想要的特定特征
            features = outputs[1][:, :128]

            # 将特征和对应的 issue_id 添加到字典中
            # 注意：你可能需要确保特征和 issue_id 的批处理是对应的
            # 这里只是一个简化的示例，实际中你可能需要更复杂的逻辑来处理
            report_feat.append((report_ids, features))
    with open(f'{baseURL}{project_name}/preprocessed_reports/reports_feat', 'wb') as f:
        pickle.dump(report_feat, f)

def generate_same_location_reports_historyInfo(reports):
    # 只能统计训练集的report之间可以相互计算相似度
    simi_reports_dict = {}
    for _, ri in tqdm(reports.iterrows()):
        simi_reports_dict[ri['bug_id']] = []
        ri_buggy_path = ri.valid_buggy_paths.split('\n')
        ri_timestamp = ri['report_timestamp']
        for _, rj in reports.iterrows():
            if ri['bug_id'] != rj['bug_id']:
                rj_timestamp = rj['report_timestamp']
                if ri_timestamp > rj_timestamp:
                    rj_buggy_path = rj.valid_buggy_paths.split('\n')
                    # 检查两个集合是否有交集
                    if set(ri_buggy_path).intersection(set(rj_buggy_path)):
                        simi_reports_dict[ri['bug_id']].append(rj['bug_id'])
    return simi_reports_dict


def calculate_RR_relations(baseURL, project_name):
    with open(f'{baseURL}{project_name}/preprocessed_reports/reports_feat', 'rb') as f:
        reports_feat = pickle.load(f)
    with open(f'{baseURL}{project_name}/preprocessed_reports/bugid2idx.pkl', 'rb') as f:
        bugid2idx = pickle.load(f)
    # 训练集的report之间可以相互计算相似度
    with open(f'{baseURL}{project_name}/preprocessed_reports/train_reports', 'rb') as f:
        train_reports = pickle.load(f)
    # 验证集和测试集中的report只能与训练集中的report计算相似度
    with open(f'{baseURL}{project_name}/preprocessed_reports/val_reports', 'rb') as f:
        val_reports = pickle.load(f)
    with open(f'{baseURL}{project_name}/preprocessed_reports/test_reports', 'rb') as f:
        test_reports = pickle.load(f)
    report_corpus = pd.concat([train_reports, val_reports, test_reports], axis=0)

    report_feat_dict = {}
    for report_ids, feats in reports_feat:
        for report_id, feat in zip(report_ids, feats):
            report_feat_dict[report_id.item()] = feat
    # 每个report只能与之前的report计算相似度
    simi_reports_dict = generate_same_location_reports_historyInfo(report_corpus)
    similar_relations_dict = {}
    train_RR_relation = []
    for _, r in report_corpus.iterrows():
        report_id = r['bug_id']
        simi_reports = simi_reports_dict[report_id]
        if len(simi_reports) != 0:
            report_vec = report_feat_dict[report_id]
            for simi_report_id in simi_reports:
                simi_report_vec = report_feat_dict[simi_report_id]
                cos_sim = F.cosine_similarity(report_vec.unsqueeze(0), simi_report_vec.unsqueeze(0), dim=1)
                train_RR_relation.append([bugid2idx[report_id], bugid2idx[simi_report_id], cos_sim.item()])
    similar_relations_dict['all'] = train_RR_relation

    with open(f'{baseURL}{project_name}/relations/RR_relations', 'wb') as f:
        pickle.dump(similar_relations_dict, f)


def genrate_file_nodes(baseURL, project_name):
    # file节点是不是也是只有train里有，因为val和test都是与train的节点构建关系
    with open(f'{baseURL}{project_name}/collected_codes/commit2code_paths.pkl', 'rb') as f:
        commit2code_paths = pickle.load(f)
    with open(f'{baseURL}{project_name}/preprocessed_reports/train_reports', 'rb') as f:
        train_reports = pickle.load(f)

    all_code_paths = set()
    for _, r in train_reports.iterrows():
        code_paths = commit2code_paths[r['commit']]
        all_code_paths = all_code_paths.union(set(code_paths))
    all_code_paths_list = sorted(list(all_code_paths))
    file_nodes_dict = {value: index for index, value in enumerate(all_code_paths_list)}

    with open(f'{baseURL}{project_name}/collected_codes/file_nodes_dict', 'wb') as f:
        pickle.dump(file_nodes_dict, f)

    return file_nodes_dict


class FileDataset(Dataset):
    def __init__(self, codes_list, tokenizer):
        self.codes_list = codes_list
        self.tokenizer = tokenizer

    def __len__(self):
        return len(self.codes_list)

    def __getitem__(self, idx):
        file_path, file_info = self.codes_list[idx]
        path_tokens = file_info['path_tokens']
        code_keywords = file_info['code_keywords']

        file_keywords_text = ' '.join(path_tokens + list(code_keywords))

        # 使用 tokenizer 进行编码
        keywords_inputs = self.tokenizer(file_keywords_text, return_tensors='pt', truncation=True, max_length=512, padding='max_length')
        keywords_attention_mask = keywords_inputs['attention_mask']  # 这里添加了attention_mask

        return keywords_inputs['input_ids'], keywords_attention_mask, file_path


def generate_file_feat(baseURL, project_name):
    # 只统计训练集中最后一个版本的file文件
    # with open(f'{baseURL}{project_name}\\DGL\\collected_codes\\preprocessed_codes', 'rb') as f:
    #     preprocessed_codes = pickle.load(f)
    with open(f'{baseURL}{project_name}/collected_codes/file_nodes_dict', 'rb') as f:
        file_nodes_dict = pickle.load(f)
    # 加载codebert模型
    codebert_url = 'F:/实验/CodeBERT-master/Siamese-model/demo/codebert-base/'
    config = RobertaConfig.from_pretrained(codebert_url)
    tok = RobertaTokenizer.from_pretrained(codebert_url)
    model = RobertaModel.from_pretrained(codebert_url, config=config)
    model.eval()

    codes_info = {}
    for chunk_index in range(5):
        with open(f'{baseURL}{project_name}/collected_codes/preprocessed_codes_{chunk_index}', 'rb') as f:
            preprocessed_codes_i = pickle.load(f)

        for _, file_info in preprocessed_codes_i.iterrows():
            if file_info['path'] not in file_nodes_dict.keys():
                continue
            if file_info['path'] not in codes_info.keys():
                codes_info[file_info['path']] = {}
                codes_info[file_info['path']]['code_keywords'] = set(file_info['code_keywords'].keys())
                codes_info[file_info['path']]['path_tokens'] = file_info['path_tokens']
            else:
                codes_info[file_info['path']]['code_keywords'] = codes_info[file_info['path']]['code_keywords'].union(
                    set(file_info['code_keywords'].keys()))

    codes_list = []
    for code_index, (code_path, tokens_info) in enumerate(codes_info.items()):
        codes_list.append((code_path, tokens_info))


    # 创建 Dataset 实例
    dataset = FileDataset(codes_list, tok)
    # 创建 DataLoader 实例
    batch_size = 128  # 你可以设置合适的批处理大小
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False)

    file_feat = []
    with torch.no_grad():
        for batch in tqdm(dataloader):
            keywords_input_ids, keywords_attention_masks, file_paths = batch
            # content_input_ids, content_attention_masks, keywords_input_ids, keywords_attention_masks, file_paths, file_commits = batch
            # content_outputs = model(content_input_ids.squeeze(dim=1), attention_mask=content_attention_masks.squeeze(dim=1))
            # content_features = content_outputs[1][:, :128]
            keywords_outputs = model(keywords_input_ids.squeeze(dim=1),
                                     attention_mask=keywords_attention_masks.squeeze(dim=1))
            keywords_features = keywords_outputs[1][:, :128]
            # file_feat.append((file_commits, file_paths, content_features, keywords_features))
            file_feat.append((file_paths, keywords_features))

    with open(f'{baseURL}{project_name}/collected_codes/files_feat', 'wb') as f:
        pickle.dump(file_feat, f)


def calculate_CC_relations(baseURL, project_name):
    with open(f'{baseURL}{project_name}/collected_codes/file_nodes_dict', 'rb') as f:
        file_nodes_dict = pickle.load(f)
    with open(f'{baseURL}{project_name}/preprocessed_reports/train_reports', 'rb') as f:
        train_reports = pickle.load(f)

    co_citation_relations = []
    co_citation_counts = defaultdict(int)  # 用于跟踪每对pi和pj出现的次数

    for report in tqdm(train_reports.itertuples(), total=train_reports.shape[0], ncols=80):
        buggy_path = report.valid_buggy_paths.split('\n')
        if len(buggy_path) > 1:
            for i in range(len(buggy_path)):
                pi = buggy_path[i]
                for j in range(i + 1, len(buggy_path)):  # 只考虑不重复的pi和pj对
                    pj = buggy_path[j]
                    # 更新pi和pj这对元素的出现次数
                    pair_key = (file_nodes_dict[pi], file_nodes_dict[pj])
                    co_citation_counts[pair_key] += 1

    # 如果需要包含pi和pj的反向关系也带有权重，可以添加以下循环
    for pair_key, count in co_citation_counts.items():
        pi, pj = pair_key
        co_citation_relations.append([pi, pj, count])  # 注意这里的顺序是pj, pi
        co_citation_relations.append([pj, pi, count])

    with open(f'{baseURL}{project_name}/relations/CC_citation_relations', 'wb') as f:
        pickle.dump(co_citation_relations, f)


def generate_file_relations():
    with open(f'{baseURL}{project_name}\\DGL\\dataset\\file_nodes_dict', 'rb') as f:
        file_nodes_dict = pickle.load(f)
    relations_file = pd.read_csv(f'{baseURL}{project_name}\\DGL\\relations\\FileDependencies.csv')

    # 要先把file的名称换成id
    CC_depend_relations = []
    for _, r in tqdm(relations_file.iterrows()):
        from_file = r['From File'].split(code_dir)[1].replace("\\", "/")
        to_file = r['To File'].split(code_dir)[1].replace("\\", "/")
        CC_depend_relations.append([file_nodes_dict[from_file], file_nodes_dict[to_file]])

    with open(f'{baseURL}{project_name}\\DGL\\relations\\CC_depend_relations', 'wb') as f:
        pickle.dump(CC_depend_relations, f)


def generate_RC_relations(baseURL, project_name):
    with open(f'{baseURL}{project_name}/preprocessed_reports/reports_feat', 'rb') as f:
        reports_feat = pickle.load(f)
    with open(f'{baseURL}{project_name}/collected_codes/files_feat', 'rb') as f:
        file_feat = pickle.load(f)
    with open(f'{baseURL}{project_name}/preprocessed_reports/bugid2idx.pkl', 'rb') as f:
        bugid2idx = pickle.load(f)
    with open(f'{baseURL}{project_name}/collected_codes/file_nodes_dict', 'rb') as f:
        file_nodes_dict = pickle.load(f)
    with open(f'{baseURL}{project_name}/collected_codes/train_final_version_commit_codes.pkl', 'rb') as f:
        train_final_version_commit_codes = pickle.load(f)
    # 训练集的report之间可以相互计算相似度
    with open(f'{baseURL}{project_name}/preprocessed_reports/train_reports', 'rb') as f:
        train_reports = pickle.load(f)
    # 验证集和测试集中的report只能与训练集中的report计算相似度
    with open(f'{baseURL}{project_name}/preprocessed_reports/val_reports', 'rb') as f:
        val_reports = pickle.load(f)
    with open(f'{baseURL}{project_name}/preprocessed_reports/test_reports', 'rb') as f:
        test_reports = pickle.load(f)

    # 是不是还要过滤掉那些location不在训练集中的report？
    invalid_reports_in_val_test = {}
    for reports, tag in zip((val_reports, test_reports), ('val', 'test')):
        count = 0
        invalid_reports = []
        for _, r in reports.iterrows():
            locations = r.valid_buggy_paths.split('\n')
            for l in locations:
                if l not in train_final_version_commit_codes['path'].tolist():
                    count += 1
                    invalid_reports.append((r['bug_id'], bugid2idx[r['bug_id']]))
        print(tag, '中没有location的report数：', count)  # val中有14个，test中有2个
        invalid_reports_in_val_test[tag] = invalid_reports

    report_feat_dict = {}
    for report_ids, feats in reports_feat:
        for report_id, feat in zip(report_ids, feats):
            report_feat_dict[report_id.item()] = feat

    file_feat_dict = {}
    for file_paths, feats in file_feat:
        for path, feat in zip(file_paths, feats):
            file_feat_dict[path] = feat

    # 训练集的report-code计算ground truth之间的相似度
    RC_relations_dict = {}
    for reports, tag in zip((train_reports, val_reports, test_reports), ('train', 'val', 'test')):
        RC_relations = []
        for _, r in reports.iterrows():
            report_id = r['bug_id']
            report_vec = report_feat_dict[report_id]
            locations = r.valid_buggy_paths.split('\n')
            for l in locations:
                if l not in train_final_version_commit_codes['path'].tolist():
                    continue
                file_vec = file_feat_dict[l]
                cos_sim = F.cosine_similarity(report_vec.unsqueeze(0), file_vec.unsqueeze(0), dim=1)
                RC_relations.append([bugid2idx[report_id], file_nodes_dict[l], cos_sim.item()])
        RC_relations_dict[tag] = RC_relations

    with open(f'{baseURL}{project_name}/relations/RC_relations', 'wb') as f:
        pickle.dump(RC_relations_dict, f)


def generate_labels(baseURL, project_name):
    with open(f'{baseURL}{project_name}/preprocessed_reports/bugid2idx.pkl', 'rb') as f:
        bugid2idx = pickle.load(f)
    with open(f'{baseURL}{project_name}/collected_codes/file_nodes_dict', 'rb') as f:
        file_nodes_dict = pickle.load(f)
    # 训练集的report之间可以相互计算相似度
    with open(f'{baseURL}{project_name}/preprocessed_reports/train_reports', 'rb') as f:
        train_reports = pickle.load(f)
    # 验证集和测试集中的report只能与训练集中的report计算相似度
    with open(f'{baseURL}{project_name}/preprocessed_reports/val_reports', 'rb') as f:
        val_reports = pickle.load(f)
    with open(f'{baseURL}{project_name}/preprocessed_reports/test_reports', 'rb') as f:
        test_reports = pickle.load(f)
    with open(f'{baseURL}{project_name}/collected_codes/train_final_version_commit_codes.pkl', 'rb') as f:
        train_final_version_commit_codes = pickle.load(f)

    all_labels_dict = {}
    train_labels = {}
    train_labels1 = {}
    for _, r in train_reports.iterrows():
        report_id = r['bug_id']
        locations = r.valid_buggy_paths.split('\n')
        train_labels[report_id] = []
        train_labels1[bugid2idx[report_id]] = []
        for l in locations:
            if l not in file_nodes_dict.keys():
                continue
            train_labels[report_id].append(l)
            train_labels1[bugid2idx[report_id]].append(file_nodes_dict[l])
    all_labels_dict['train'] = (train_labels, train_labels1)

    for reports, tag in zip((val_reports, test_reports), ('val', 'test')):
        test_labels = {}
        test_labels1 = {}
        for _, r in reports.iterrows():
            report_id = r['bug_id']
            locations = r.valid_buggy_paths.split('\n')
            test_labels[report_id] = []
            test_labels1[bugid2idx[report_id]] = []
            for l in locations:
                if l not in train_final_version_commit_codes['path'].tolist():
                    continue
                test_labels[report_id].append(l)
                test_labels1[bugid2idx[report_id]].append(file_nodes_dict[l])
        all_labels_dict[tag] = (test_labels, test_labels1)

    with open(f'{baseURL}{project_name}/graph/labels', 'wb') as f:
        pickle.dump(all_labels_dict, f)

def generate_dgl(baseURL, project_name):
    with open(f'{baseURL}{project_name}/relations/RC_relations', 'rb') as f:
        RC_relations_dict = pickle.load(f)
    with open(f'{baseURL}{project_name}/relations/CC_citation_relations', 'rb') as f:
        CC_citation_relations = pickle.load(f)
    with open(f'{baseURL}{project_name}/relations/RR_relations', 'rb') as f:
        all_RR_relations = pickle.load(f)

    with open(f'{baseURL}{project_name}/collected_codes/file_nodes_dict', 'rb') as f:
        file_nodes_dict = pickle.load(f)
    # 训练集的report之间可以相互计算相似度
    with open(f'{baseURL}{project_name}/preprocessed_reports/train_reports', 'rb') as f:
        train_reports = pickle.load(f)
    # 验证集和测试集中的report只能与训练集中的report计算相似度
    with open(f'{baseURL}{project_name}/preprocessed_reports/val_reports', 'rb') as f:
        val_reports = pickle.load(f)
    with open(f'{baseURL}{project_name}/preprocessed_reports/test_reports', 'rb') as f:
        test_reports = pickle.load(f)

    train_RC_relations = pd.DataFrame(data=RC_relations_dict['train'], columns=['report', 'file', 'weight'])
    # 下面这些是真实的关系
    val_RC_relations = pd.DataFrame(data=RC_relations_dict['val'], columns=['report', 'file', 'weight'])
    test_RC_relations = pd.DataFrame(data=RC_relations_dict['test'], columns=['report', 'file', 'weight'])
    RR_relations = pd.DataFrame(data=all_RR_relations['all'], columns=['reporti', 'reportj', 'weight'])

    train_CC_citation_relations = pd.DataFrame(data=CC_citation_relations, columns=['filei', 'filej', 'weight'])

    relation_dict = dict()
    RC_relations_report = torch.tensor(
        train_RC_relations['report'].tolist() + val_RC_relations['report'].tolist() + test_RC_relations['report'].tolist())
    RC_relations_file = torch.tensor(
        train_RC_relations['file'].tolist() + val_RC_relations['file'].tolist() + test_RC_relations['file'].tolist())
    RC_relations_weight = torch.tensor(
        train_RC_relations['weight'].tolist() + val_RC_relations['weight'].tolist() + test_RC_relations['weight'].tolist())
    relation_dict[('report', 'located', 'file')] = (RC_relations_report, RC_relations_file)
    relation_dict[('file', 'located_by', 'report')] = (RC_relations_file, RC_relations_report)

    RR_relations_reporti = torch.tensor(RR_relations['reporti'].tolist())
    RR_relations_reportj = torch.tensor(RR_relations['reportj'].tolist())
    RR_relations_weight = torch.tensor(RR_relations['weight'].tolist())
    relation_dict[('report', 'similar', 'report')] = (RR_relations_reporti, RR_relations_reportj)
    relation_dict[('report', 'similar_by', 'report')] = (RR_relations_reportj, RR_relations_reporti)

    CC_citation_relations_filei = torch.tensor(train_CC_citation_relations['filei'].tolist())
    CC_citation_relations_filej = torch.tensor(train_CC_citation_relations['filej'].tolist())
    CC_citation_relations_weight = torch.tensor(train_CC_citation_relations['weight'].tolist())
    relation_dict[('file', 'co-cited', 'file')] = (CC_citation_relations_filei, CC_citation_relations_filej)

    g = dgl.heterograph(relation_dict,
                        num_nodes_dict={'report': len(train_reports) + len(val_reports) + len(test_reports),
                                        'file': len(file_nodes_dict)})
    # 添加关系权重
    g['report', 'located', 'file'].edata['w'] = RC_relations_weight
    g['file', 'located_by', 'report'].edata['w'] = RC_relations_weight
    g['report', 'similar', 'report'].edata['w'] = RR_relations_weight
    g['report', 'similar_by', 'report'].edata['w'] = RR_relations_weight
    g['file', 'co-cited', 'file'].edata['w'] = CC_citation_relations_weight

    g.nodes['report'].data['_ID'] = torch.arange(0, len(train_reports) + len(val_reports) + len(test_reports))
    g.nodes['file'].data['_ID'] = torch.arange(0, len(file_nodes_dict))

    # 要不要添加特征？
    with open(f'{baseURL}{project_name}/preprocessed_reports/reports_feat', 'rb') as f:
        reports_feat = pickle.load(f)
    with open(f'{baseURL}{project_name}/collected_codes/files_feat', 'rb') as f:
        file_feat = pickle.load(f)
    with open(f'{baseURL}{project_name}/preprocessed_reports/bugid2idx.pkl', 'rb') as f:
        bugid2idx = pickle.load(f)
    swapped_report_nodes_dict = {v: k for k, v in bugid2idx.items()}
    with open(f'{baseURL}{project_name}/collected_codes/file_nodes_dict', 'rb') as f:
        file_nodes_dict = pickle.load(f)
    swapped_file_nodes_dict = {v: k for k, v in file_nodes_dict.items()}

    report_feat_dict = {}
    for report_ids, feats in reports_feat:
        for report_id, feat in zip(report_ids, feats):
            report_feat_dict[report_id.item()] = feat

    file_feat_dict = {}
    for file_paths, feats in file_feat:
        for path, feat in zip(file_paths, feats):
            file_feat_dict[path] = feat

    train_report_feats = []
    for report_id in g.nodes['report'].data['_ID']:
        train_report_feats.append(report_feat_dict[swapped_report_nodes_dict[report_id.item()]])
    train_report_feats_tensor = torch.stack(train_report_feats)
    g.nodes['report'].data['feat'] = train_report_feats_tensor

    train_file_feats = []
    for file_id in g.nodes['file'].data['_ID']:
        train_file_feats.append(file_feat_dict[swapped_file_nodes_dict[file_id.item()]])
    train_file_feats_tensor = torch.stack(train_file_feats)
    g.nodes['file'].data['feat'] = train_file_feats_tensor
    print(g)
    save_graphs(f'{baseURL}{project_name}/{project_name}.bin', g, {})


if __name__ == "__main__":
    # aspectj(554), birt(4009), eclipse(6261), jdt(6108), swt(4086), tomcat(1020)
    # org.aspectj, birt, eclipse.platform.ui, eclipse.jdt.ui, eclipse.platform.swt, tomcat
    # project_name = 'eclipse'
    baseURL = '../processed_data/'
    # code_dir = 'F:\\source_code\\eclipse.platform.ui\\'

    projects = ['birt', 'eclipse', 'jdt', 'swt', 'tomcat']
    code_dirs = ['F:\\source_code\\birt\\', 'F:\\source_code\\eclipse.platform.ui\\',
                 'F:\\source_code\\eclipse.jdt.ui\\', 'F:\\source_code\\eclipse.platform.swt\\',
                 'F:\\source_code\\tomcat\\']
    for project_name, code_dir in zip(projects, code_dirs):
        if project_name != 'tomcat':
            continue
        # filter_reports(baseURL, project_name)
        # generate_report_feat(baseURL, project_name)
        # calculate_RR_relations(baseURL, project_name)
        # genrate_file_nodes(baseURL, project_name)
        # generate_file_feat(baseURL, project_name)  # 执行这个函数之前需要用collected_codes文件处理codes，生成preprocessed_codes文件
        # calculate_CC_relations(baseURL, project_name)
        # # generate_file_relations()  # 先用understand生成fileDependency文件 暂时不用depend关系了
        # generate_RC_relations(baseURL, project_name)
        # generate_labels(baseURL, project_name)
        generate_dgl(baseURL, project_name)
        print()