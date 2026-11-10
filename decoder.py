import numpy as np
import json

with open(r"C:\Папки\GitHub\Numbers_auto_AI\data\REC_data\test\data.json", 'r') as file:
    data = json.load(file)


def m(str1):
    alphabet = '0123456789ABEKMHOPCTYX'
    matrix = np.zeros((9, 22), dtype=int)
    for row_idx, char in enumerate(str1):
        if char in alphabet:
            col_idx = alphabet.index(char)
            matrix[row_idx, col_idx] = 1
    return matrix


for key in data:
    data[key] = m(data[key])
