import numpy as np
import quadprog
import os

dir_path = os.path.dirname(os.path.realpath(__file__))


def random_uniform(rows, cols):
    return (np.random.rand(rows, cols) - 0.5) * 0.001


def cross_cov(y, x):
    return np.matmul(np.expand_dims(y, 1), np.transpose(np.expand_dims(x, 1)))


def is_pos_def(x):
    return np.all(np.linalg.eigvals(x) > 0)


# solve \min_a Tr[(y - (W.(x*a) + b))(y - (W.(x*a) + b))^\intercal + lamb.a.a^\intercal] s.t. \sum(a) = 1, a_i > 0 \forall i
def quadprog_solve_qp(y, W, x, b, sum_active, lamb):
    dim = x.shape[0]
    Wx = np.matmul(W, np.diag(x))

    if y is None:
        qp_G = (np.matmul(np.transpose(Wx), Wx) + np.identity(dim) - Wx - np.transpose(Wx)) + np.identity(dim) * lamb
        qp_a = (b - np.matmul(np.transpose(Wx), b))
    else:
        qp_G = (np.matmul(np.transpose(Wx), Wx)) + np.identity(dim) * lamb
        qp_a = np.matmul(np.transpose(Wx), y - b)

    qp_C = np.concatenate([np.ones([dim, 1]), np.identity(dim), np.identity(dim) * -1.0], axis=1)
    qp_b = np.zeros([1 + dim + dim], dtype=np.float64)
    qp_b[0] = sum_active
    qp_b[(1 + dim):] = -2.0

    # print(qp_G.shape, qp_a.shape, qp_C.shape, qp_b.shape)

    return np.maximum(quadprog.solve_qp(qp_G, qp_a, qp_C, qp_b, meq=1)[0], 0)


class Network:
    def __init__(self, input_shape, output_shape, learning_rate=0.01):
        self.W = random_uniform(output_shape, input_shape + 1)
        self.yxt_ = random_uniform(output_shape, input_shape + 1)
        self.xxt_ = np.identity(input_shape + 1) * 0.001

        self.A = np.concatenate([np.identity(input_shape), np.zeros([input_shape, 1])], axis=1)
        self.axat_ = random_uniform(input_shape, input_shape + 1)
        self.xaxat_ = np.identity(input_shape + 1) * 0.001

        self.learning_rate_ = learning_rate

    def learn(self, x, y, lamb=0.0001):

        alpha = self.learning_rate_
        _alpha = 1 - self.learning_rate_

        dim = x.shape[0]

        x1 = np.append(x, [1], axis=0)
        self.yxt_ = self.yxt_ * _alpha + cross_cov(y, x1) * alpha
        self.xxt_ = self.xxt_ * _alpha + cross_cov(x1, x1) * alpha
        self.W = np.matmul(self.yxt_, np.transpose(np.linalg.pinv(np.transpose(self.xxt_))))

        a = quadprog_solve_qp(y, self.W[:, 0:-1], x, self.W[:, -1], float(dim), 0.00001)

        xa = x * a
        xa1 = np.append(xa, [1], axis=0)
        self.axat_ = self.axat_ * _alpha + cross_cov(a, xa1) * alpha
        self.xaxat_ = self.xaxat_ * _alpha + cross_cov(xa1, xa1) * alpha
        self.A = np.matmul(self.axat_, np.transpose(np.linalg.pinv(np.transpose(self.xaxat_ + lamb * np.identity(dim + 1)))))

        return a

    def project(self, x):
        W_ = self.W[:, 0:-1]
        x_ = np.matmul(np.transpose(W_), np.matmul(W_, x))
        return x_

    def classify(self, x):
        dim = x.shape[0]
        a = quadprog_solve_qp(None, self.A[:, 0:-1], x, self.A[:, -1], float(dim), 0.00001)
        return np.matmul(self.W[:, 0:-1], x * a) + self.W[:, -1], a

    def classify_no_focus(self, x):
        return np.matmul(self.W[:, 0:-1], x) + self.W[:, -1]

    def save(self, session):
        path = os.path.join(dir_path, "..", "artifacts", session)
        if not os.path.exists(path):
            os.makedirs(path)
        np.save(os.path.join(path, "yxt.npy"), self.yxt_)
        np.save(os.path.join(path, "xxt.npy"), self.xxt_)
        np.save(os.path.join(path, "axat.npy"), self.axat_)
        np.save(os.path.join(path, "xaxat.npy"), self.xaxat_)

    def load(self, session):
        path = os.path.join(dir_path, "..", "artifacts", session)
        if not os.path.exists(path):
            return False
        self.yxt_ = np.load(os.path.join(path, "yxt.npy"))
        self.xxt_ = np.load(os.path.join(path, "xxt.npy"))
        self.axat_ = np.load(os.path.join(path, "axat.npy"))
        self.xaxat_ = np.load(os.path.join(path, "xaxat.npy"))

        self.W = np.matmul(self.yxt_, np.linalg.inv(self.xxt_))
        self.A = np.matmul(self.axat_, np.linalg.inv(self.xaxat_))

        return True


if __name__ == '__main__':

    y = np.random.randn(10)
    W = np.random.randn(10, 10)
    x = np.random.randn(10)
    b = np.random.randn(10)

    a = quadprog_solve_qp(None, W, x, b, 10.0, 0.0)
    print(a)

    network = Network(10, 10, 0.1)
    # network.load("test")
    x = np.random.randn(10)
    y = np.random.randn(10)
    for i in range(20):
        a = network.learn(x, y)
        y2, a2 = network.classify(x)
        y1 = network.classify_no_focus(x * a)
        y0 = network.classify_no_focus(x)
        print(a, a2)
        diff2 = y2 - y
        diff1 = y1 - y
        diff0 = y0 - y
        print(np.sum(diff2 * diff2), np.sum(diff1 * diff1), np.sum(diff0 * diff0))

    # network.save("test")
