import numpy as np


def random_uniform(rows, cols):
    return (np.random.rand(rows, cols) - 0.5) * 0.001


def prepare_data(data, first, last_not_included):
    # data are of shape [len, ...]
    if first < 0:
        flat_size = np.prod(data.shape) / data.shape[0]
        temp = np.zeros((last_not_included - first, flat_size), dtype=np.float32)
        if last_not_included <= 0:
            return temp
        temp[(-first):(last_not_included - first), :] = np.reshape(data[0:last_not_included, ...], (last_not_included, flat_size))
        return temp
    else:
        return np.reshape(data[first:last_not_included, ...], (last_not_included - first, -1))


if __name__ == '__main__':
    data = np.random.uniform(size=(10, 2, 3))
    print(data)
    print(prepare_data(data, -2, 3))
