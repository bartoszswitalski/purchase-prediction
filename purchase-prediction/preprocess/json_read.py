"""
    Name:       json_ready.py
    Purpose:    JSON data preprosessing.

    @author Bartosz Świtalski, Piotr Frątczak

    Warsaw University of Technology
    Faculty of Electronics and Information Technology
"""
import json_lines


def get_jsonl_data(file_name):
    data = []
    with open('data/' + file_name) as f:
        for item in json_lines.reader(f):
            data.append(item)

    return data
