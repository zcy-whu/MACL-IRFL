import pickle
from git import Repo
from glob import glob
import os
import math
import pandas as pd
from tqdm import tqdm
from process.utils import *
from process.preprocessor import Preprocessor

preprocessor = Preprocessor()

code_dir = 'F:\\source_code\\eclipse.platform.ui\\'
project_name = 'eclipse'
datafolder = 'F:\\实验\\GNNBL\\processed_data\\'

repo = Repo(code_dir)
assert repo.bare is False
repo = repo.git

def collect_codes(paths):
    codes = []
    for p in paths:
        with open(p, 'r', errors='ignore') as f:
            content = f.read()
        codes.append(content)
    return codes

def generate_collected_codes():
    with open(f'{datafolder}{project_name}\\DGL\\dataset\\train_reports', 'rb') as f:
        train_reports = pickle.load(f)
    commit_version = train_reports.sort_values('commit_timestamp').iloc[-1]['commit']

    collected_codes = []
    repo.checkout(commit_version + '~1', '-f')  # 切换到指定的分支或提交
    code_paths = glob(f'{code_dir}/**/*.java', recursive=True)
    print(commit_version, '分支下的code files数量为：', len(code_paths))  # aspectj:6084, birt:8798, eclipse:6173, SWT:2716, JDT:10431
    commit2code_paths_issue_commit = [os.path.relpath(p, start=code_dir).replace('\\', '/') for p in code_paths]

    add_mod_codes = collect_codes(code_paths)
    assert len(add_mod_codes) == len(commit2code_paths_issue_commit)
    collected_codes += list(zip([commit_version] * len(add_mod_codes), commit2code_paths_issue_commit, add_mod_codes))

    columns = ['commit', 'path', 'content']
    df = pd.DataFrame(data=collected_codes, columns=columns)
    with open(f'{datafolder}{project_name}\\DGL\\collected_codes\\train_final_version_commit_codes.pkl', 'wb') as f:
        pickle.dump(df, f)

class CodeCorpus(object):
    def __init__(self, save_dir):
        self.collected_codes_dir = save_dir

    def generate_corpus(self):
        print('load collected codes...')
        # eclipse到675703报错，memory error
        codes_df = pd.read_csv(f'{datafolder}{project_name}\\collected_codes\\collected_codes.csv')  # aspectj:20283, birt:69408, eclipse:1080042, SWT:51777,JDT:89740
        codes_df.fillna('', inplace=True)
        del codes_df['processed_code']

        # chunk_size = math.ceil(len(codes_df) / 5)  # 使用整除来获取每个部分的行数
        # # 使用列表推导式来创建五个部分
        # chunks = [codes_df.iloc[i:i + chunk_size] for i in range(0, len(codes_df), chunk_size)]
        # for chunk_index, chunk in enumerate(chunks):
        #     code_contents = chunk['content'].tolist()
        #     code_paths = chunk['path'].tolist()
        #     code_tokens, code_keywords, path_tokens, path_keywords, camel_words_record = self.preprocess(code_contents,
        #                                                                                                  code_paths)
        #     # 这是把原始的代码也存进去了？
        #     corpus = chunk[['commit', 'path']]
        #     corpus['code_tokens'] = code_tokens
        #     corpus['code_keywords'] = code_keywords
        #     corpus['path_tokens'] = path_tokens
        #     corpus['path_keywords'] = path_keywords
        #
        #     with open(f'{self.collected_codes_dir}preprocessed_codes_{chunk_index}', 'wb') as f:
        #         pickle.dump(corpus, f)

        code_contents = codes_df['content'].tolist()
        code_paths = codes_df['path'].tolist()
        code_tokens, code_keywords, path_tokens, path_keywords, camel_words_record = self.preprocess(code_contents, code_paths)
        # 这是把原始的代码也存进去了？
        corpus = codes_df[['commit', 'path']]
        corpus['code_tokens'] = code_tokens
        corpus['code_keywords'] = code_keywords
        corpus['path_tokens'] = path_tokens
        corpus['path_keywords'] = path_keywords

        # with open(f'{self.collected_codes_dir}train_final_version_commit_preprocessed_codes', 'wb') as f:
        #     pickle.dump(corpus, f)
        with open(f'{self.collected_codes_dir}preprocessed_codes', 'wb') as f:
            pickle.dump(corpus, f)


    @staticmethod
    def preprocess(code_contents, code_paths):
        camel_word_record = {}
        code_tokens = []
        code_keywords = []
        path_tokens = []
        path_keywords = []
        code_contents = list(zip(code_contents, code_paths, range(len(code_contents))))
        # logging.info(f'multiprocessing: NUM_CPU = {NUM_CPU}')
        #
        # with multiprocessing.Pool(NUM_CPU) as p:
        #     results = list(tqdm(
        #         p.imap(preprocessor.preprocess_code_with_multiprocess, code_contents),
        #         total=len(code_contents),
        #         ncols=80,
        #         desc=f'preprocessing code'
        #     ))
        results = []
        # index=-1
        for r in tqdm(code_contents, ncols=80, desc=f'preprocessing code'):
            results.append(preprocessor.preprocess_code_with_multiprocess(r))

        results = list(sorted(results, key=lambda x: x[4]))

        for each_tokens, each_record, each_path_tokens, each_path_record, _ in results:
            code_tokens.append(each_tokens)
            code_keywords.append(each_record)
            path_tokens.append(each_path_tokens)
            path_keywords.append(each_path_record)
            camel_word_record.update(each_record)
            camel_word_record.update(each_path_record)
        return code_tokens, code_keywords, path_tokens, path_keywords, camel_word_record


# 先对收集的代码进行预处理分词清洗
generate_collected_codes()
code_corpus = CodeCorpus(f'{datafolder}{project_name}\\collected_codes\\')
code_corpus.generate_corpus()

print()