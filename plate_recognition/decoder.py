"""Кодирование и декодирование автомобильных номеров."""
import numpy as np

ALPHABET = "0123456789ABEKMHOPCTYX"
SEQUENCE_LENGTH = 9


def encode(text: str) -> np.ndarray:
    """Преобразует строку номера в матрицу one-hot размера (9, len(ALPHABET))."""
    matrix = np.zeros((SEQUENCE_LENGTH, len(ALPHABET)), dtype=np.float32)

    for row_index, character in enumerate(text[:SEQUENCE_LENGTH]):
        if character in ALPHABET:
            matrix[row_index, ALPHABET.index(character)] = 1

    return matrix


def decode(matrix: np.ndarray) -> str:
    """Преобразует one-hot матрицу или вероятности обратно в строку."""
    result = []

    for row in matrix:
        if np.all(row == 0):
            continue
        result.append(ALPHABET[int(np.argmax(row))])

    return "".join(result)
