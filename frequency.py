import pickle
import json
from collections import Counter

train_filename = 'output2.json'
target_filename = 'freq_dict_PY.pickle'

freq_dict = Counter()
terminal_num = set()
terminal_num.add('EMPTY')


def process(filename):
    with open(filename, encoding='utf-8') as lines:
        for line in lines:
            data = json.loads(line)
            for dic in data:
                value = dic.get('value', 'EMPTY')
                terminal_num.add(value)
                freq_dict[value] += 1


def save(filename):
    with open(filename, 'wb') as f:
        sv = {'freq_dict': freq_dict, 'terminal_num': terminal_num}
        pickle.dump(sv, f)


if __name__ == '__main__':
    process(train_filename)
    save(target_filename)
    print(freq_dict['EMPTY'])
    print(freq_dict.most_common(10))
    print(terminal_num)