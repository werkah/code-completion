# Function: get info about unk percentage, attn percentage and the terminal corpus (0 for the ones with children, unique if for every new value)

import pickle
import json
from collections import deque

terminal_dict_filename = 'terminal_dict_PY.pickle'
train_filename = 'output2.json'
target_filename = 'PY_terminal_whole.pickle'


def restore_term_dict(filename):
    with open(filename, 'rb') as f:
        save = pickle.load(f)
        return save['terminal_dict'], save['terminal_num'], save['vocab_size']



def process(filename, terminal_dict, unk_id, attn_size):
    with open(filename, encoding='utf-8') as lines:
        terminal_corpus = []
        unk_id_dict = {}
        attn_que = deque(maxlen=attn_size)
        attn_success_total = attn_fail_total = length_total = 0
        for line in lines:
            data = json.loads(line)
            terminal_line = []
            attn_que.clear()
            attn_success_cnt = attn_fail_cnt = 0
            for i, dic in enumerate(data):
                dic_value = dic.get('value', 'EMPTY')
                if dic_value in terminal_dict.keys():
                    terminal_line.append(terminal_dict[dic_value])
                    attn_que.append('Normal')
                #if there is an unk, we give it unique id, if it appears again in attention window, we give it the same id
                else:
                    if dic_value in attn_que:
                        terminal_line.append(unk_id_dict[dic_value])
                        attn_success_cnt += 1
                    else:
                        unk_id += 1
                        unk_id_dict[dic_value] = unk_id
                        terminal_line.append(unk_id)
                        attn_fail_cnt += 1
                    attn_que.append(dic_value)
            terminal_corpus.append(terminal_line)
            attn_success_total += attn_success_cnt
            attn_fail_total += attn_fail_cnt
            length_total += len(data)
        with open('output.txt', 'a') as f:
            f.write(
                'Statistics: attn_success_total: %d, attn_fail_total: %d, length_total: %d, attn_success percentage: %.4f, total unk percentage: %.4f\n' %
                (attn_success_total, attn_fail_total, length_total,
                 float(attn_success_total) / length_total, float(attn_success_total + attn_fail_total) / length_total))

        return terminal_corpus


def save(filename, terminal_dict, terminal_num, vocab_size, attn_size, train_data):
    with open(filename, 'wb') as f:
        save = {'terminal_dict': terminal_dict,
                'terminal_num': terminal_num,
                'vocab_size': vocab_size,
                'attn_size': attn_size,
                'train_data': train_data,
                }
        pickle.dump(save, f)


if __name__ == '__main__':
    #determine attention window size
    attn_size = 10
    terminal_dict, terminal_num, vocab_size = restore_term_dict(terminal_dict_filename)
    train_data = process(train_filename, terminal_dict, vocab_size, attn_size=attn_size)
    save(target_filename, terminal_dict, terminal_num, vocab_size, attn_size, train_data)
    print(train_data)