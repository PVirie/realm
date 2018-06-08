import numpy as np
import util
import quadprog


def get_statistics(y, x, W, lamb):
    yxt = np.matmul(y, np.transpose(x))
    xxt = np.matmul(x, np.transpose(x))

    a = quadprog_solve_qp(y, W, x, b, lamb)
    xa = x * a

    axat = np.matmul(a, np.transpose(xa))
    xaxat = np.matmul(xa, np.transpose(xa))

    return yxt, xxt, axat, xaxat


def is_pos_def(x):
    return np.all(np.linalg.eigvals(x) > 0)


# solve \min_a Tr[(y - (W.(x*a) + b))(y - (W.(x*a) + b))^\intercal + lamb.a.a^\intercal] s.t. \sum(a) = 1, a_i > 0 \forall i
def quadprog_solve_qp(y, W, x, b, lamb):
    dim = x.shape[0]
    Wx = np.matmul(W, np.diag(x))

    if y is None:
        qp_G = (np.matmul(np.transpose(Wx), Wx) + np.identity(dim) - Wx - np.transpose(Wx)) * 2
        qp_a = np.zeros([dim])
    else:
        qp_G = (np.matmul(np.transpose(Wx), Wx) + np.identity(dim) * lamb) * 2
        qp_a = np.matmul(np.transpose(Wx), y - b)

    qp_C = np.concatenate([np.ones([dim, 1]), np.identity(dim)], axis=1)
    qp_b = np.zeros([1 + dim], dtype=np.float64)
    qp_b[0] = 1

    return np.maximum(quadprog.solve_qp(qp_G, qp_a, qp_C, qp_b, meq=1)[0], 0.0)


class Network:
    def __init__(self, input_shape, output_shape, learning_rate=0.01):
        self.W = util.random_uniform(output_shape, input_shape + 1)
        self.A = util.random_uniform(input_shape, input_shape + 1)

    def learn(self, x, y):

        print("learn")

    def classify(self, x):
        a = quadprog_solve_qp(None, self.A[:, 0:-1], x, self.A[:, -2:-1], 0.0)
        return np.matmul(self.W[:, 0:-1], a * x) + self.W[:, -2:-1]

    def save(self):
        # self.saver.save(self.sess, "./artifacts/" + "weights")
        print("save")

    def load(self):
        # self.saver.restore(self.sess, "./artifacts/" + "weights")
        print("load")


if __name__ == '__main__':

    y = np.random.randn(10).astype(np.float64)
    W = np.random.randn(10, 10).astype(np.float64)
    x = np.random.randn(10).astype(np.float64)
    b = np.random.randn(10).astype(np.float64)

    a = quadprog_solve_qp(y, W, x, b, 0.001)
    print(a)
