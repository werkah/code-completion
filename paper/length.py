import json


def get_total_length(filename):
    total_length = 0
    with open(filename, 'r', encoding='utf-8') as file:
        for line in file:
            data = json.loads(line)
            total_length += len(data)

    return total_length


name = 'output2.json'
length = get_total_length(name)
print('Total Length:', length)
