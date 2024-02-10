# Function: get the frequency of each terminal(including empty values) and the dictionary of terminal and its id

import pickle
import json
from collections import Counter

train_filename = '../output2.json'
freq_dict_filename = 'freq_dict_PY.pickle'
terminal_dict_filename = '../terminal_dict_PY.pickle'

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

def get_terminal_dict(vocab_size, freq_dict, total_length, verbose=False):
    terminal_dict = dict()
    sorted_freq_dict = sorted(freq_dict.items(), key=lambda x: x[1], reverse=True)

    if vocab_size > len(sorted_freq_dict):
        vocab_size = len(sorted_freq_dict)

    if verbose:
        for i in range(vocab_size):
            print('the %d frequent terminal: %s, its frequency: %.1f' % (
                i, sorted_freq_dict[i][0], float(sorted_freq_dict[i][1]) / total_length))

    new_freq_dict = sorted_freq_dict[:vocab_size]

    for i, (terminal, frequent) in enumerate(new_freq_dict):
        terminal_dict[terminal] = i

    return terminal_dict, sorted_freq_dict

def save(filename, freq_dict, terminal_num):
    with open(filename, 'wb') as f:
        sv = {'freq_dict': freq_dict, 'terminal_num': terminal_num}
        pickle.dump(sv, f)

def save1(filename, terminal_dict, terminal_num, sorted_freq_dict):
  with open(filename, 'wb') as f:
    sv = {'terminal_dict': terminal_dict,'terminal_num': terminal_num, 'vocab_size': vocab_size, 'sorted_freq_dict': sorted_freq_dict,}
    pickle.dump(sv, f)


if __name__ == '__main__':
    process(train_filename)
    save(freq_dict_filename, freq_dict, terminal_num)

    #determine how many terminals(unique) are in the vocabulary
    vocab_size = 10
    total_length = sum(freq_dict.values())

    terminal_dict, sorted_freq_dict = get_terminal_dict(vocab_size, freq_dict, total_length, True)
    save1(terminal_dict_filename, terminal_dict, terminal_num, sorted_freq_dict)
    print(freq_dict)
    print(terminal_dict)
    print(terminal_num)
    print(sorted_freq_dict)