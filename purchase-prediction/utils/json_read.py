import json_lines


def get_jsonl_data(path, file_name):
    data = []
    with open(path + file_name) as f:
        for item in json_lines.reader(f):
            data.append(item)

    return data
