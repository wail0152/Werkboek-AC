import numpy as np
import json


def matrix_product(M: np.ndarray, v: np.ndarray) -> np.ndarray:
    if len(M[0]) != len(v):
        #raise DimensionError("Vectors niet hetzelfde formaat")
        print("nope")

    width = 1 if v.ndim == 1 else len(v[0])
    result = np.zeros((len(M), width), dtype=float)

    for dim in range(width):
        for row in range(len(M)):
            for matrix_index in range(len(M[row])):
                if v.ndim > 1:
                    result[row][dim] += M[row][matrix_index] * v[matrix_index][dim]
                else:
                    result[row][dim] += M[row][matrix_index] * v[matrix_index]
    return result


def read_network(filename: str) -> np.ndarray: # TODO
    with open(filename) as f:
        data = json.load(f)

    results = []
    for layer_index in range(1, len(data) + 1):
        current_layer = data["layer" + str(layer_index)]
        current_weights = current_layer["weights"]
        results.append(np.zeros((int(current_layer["size_out"]), int(current_layer["size_in"])), dtype=float))

        for weight_index in range(len(current_weights)):
            for weight in current_weights[str(weight_index + 1)]:
                results[layer_index - 1][int(weight) - 1][weight_index] += float(current_weights[str(weight_index + 1)][weight])

        if layer_index > 1:
            result = matrix_product(results[layer_index - 1], result)
        else:
            result = results[0]

    return result


print(read_network("example-2layer.json"))

#[[ 0.38  0.7   0.25  0.17 -0.88]
# [ 0.01  0.43  0.08 -0.42  0.26]]

#[[0.5, 0.2, -0.1, 0.9],
# [0.2, -0.5, 0.3, 0.1]]

#[[0.5, 0.2, 0, 0, -0.2],
# [0.2, -0.5, -0.1, 0.9, -0.8],
# [0, 0.2, 0, 0.1, -0.1],
# [0.1, 0.8, 0.3, 0, -0.7]]