import pickle
import json
from collections import Counter, defaultdict

train_filename = 'output2.json'
target_filename = 'PY_non_terminal.pickle'

type_dict = dict()
num_id = set()
type_list = list()
num_type = 0
dic_id = dict()
empty = set()

def process(filename):
    corpus_n = []
    corpus_parent = []

    with open(filename, encoding='utf-8') as lines:
        for line in lines:
            data = json.loads(line)
            line_n = []
            has_sibling = Counter()
            parent_counter = defaultdict(int)
            parent_list = []

            for i, dic in enumerate(data):
                global num_type
                type_name = dic['type']
                base_id = type_dict.setdefault(type_name, num_type)
                if base_id == num_type:
                    type_list.append(type_name)
                    num_type += 1

                if 'children' in dic.keys():
                    empty.add(i)
                    id_val = base_id * 4 + (3 if has_sibling[i] else 2)

                    children = dic['children']
                    for j in children:
                        parent_counter[j] = j - i

                    if len(children) > 1:
                        has_sibling.update(children)
                else:
                    id_val = base_id * 4 + (1 if has_sibling[i] else 0)

                line_n.append(id_val)
                parent_list.append(parent_counter[i])
                num_id.add(id_val)

            corpus_n.append(line_n)
            corpus_parent.append(parent_list)

    return corpus_n, corpus_parent

def map_dense_id(data):
    return [[dic_id[i] if i in dic_id else dic_id.setdefault(i, len(dic_id)) for i in line_id] for line_id in data]

def save(filename, type_dict, num_type, dic_id, vocab_size, train_data, train_parent, empty_set):
    with open(filename, 'wb') as f:
        save_data = {
            'vocab_size': vocab_size,
            'train_data': train_data,
            'train_parent': train_parent,
        }
        pickle.dump(save_data, f)

if __name__ == '__main__':
    train_data, train_parent = process(train_filename)
    train_data = map_dense_id(train_data)
    vocab_size = len(num_id)

    print('Set with empty as value:', len(empty), empty)
    print('The vocabulary:', num_id)
    print('Types', type_dict)

    save(target_filename, type_dict, num_type, dic_id, vocab_size, train_data, train_parent, empty)