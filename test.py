from tensorflow.examples.tutorials.mnist import input_data
import numpy as np
import tensorflow as tf
import quadprog


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


def apply_gradients(gradients, delta, rate=0.001, name="adam"):
    training_op = tf.train.AdamOptimizer(rate, name=name).apply_gradients(gradients)
    if delta is not None:
        return {"op": training_op, "cost": delta}
    else:
        return {"op": training_op}


def random_uniform(rows, cols):
    return (np.random.rand(rows, cols) - 0.5) * 0.001


class AUC:
    def __init__(self, shape):
        self.shape = shape
        self.x = tf.placeholder(dtype=tf.float32, shape=[shape[0], None])
        self.learning_rate = tf.placeholder(dtype=tf.float32, shape=None)
        self.w = tf.Variable(random_uniform(shape[1], shape[0]), dtype=tf.float32)

        self.y, self.learn_op = self.create_forward_graph(self.x)

    def feedup(self, session, x):
        if len(x.shape) < 2:
            x = np.expand_dims(x, axis=1)
        out = session.run(self.y, feed_dict={self.x: x})
        return out

    def learn(self, session, x, learning_rate=0.01, iterations=20):
        for i in range(iterations):
            __ = session.run(self.learn_op, feed_dict={self.x: x, self.learning_rate: learning_rate})
            print(__, i)
        return __

    def create_forward_graph(self, x):
        y = tf.nn.elu(tf.matmul(self.w, x))
        x_ = tf.nn.elu(tf.matmul(tf.transpose(self.w), y))

        self.objective = tf.trace(tf.matmul(x - x_, tf.transpose(x - x_))) / tf.cast(tf.reduce_prod(tf.shape(x)), tf.float32)
        grad = tf.gradients(self.objective, [self.w])
        learn_op = apply_gradients(zip(grad, [self.w]), self.objective, self.learning_rate)

        return y, learn_op


if __name__ == '__main__':

    mnist = input_data.read_data_sets("data/", one_hot=True)
    # data = np.concatenate([mnist.train.images, mnist.test.images, mnist.validation.images], axis=0)
    # labels = np.concatenate([mnist.train.labels, mnist.test.labels, mnist.validation.labels], axis=0)

    y = np.transpose(1 - mnist.train.labels)
    xt = mnist.train.images
    x = np.transpose(xt)

    sess = tf.Session()

    # auc = AUC([28 * 28, 16])
    # sess.run(tf.global_variables_initializer())

    # error = auc.learn(sess, x, 0.1, 100)
    # h = auc.feedup(sess, x)
    # h_ = auc.feedup(sess, np.transpose(mnist.test.images))

    h = x
    h_ = np.transpose(mnist.test.images)

    # ----------------------- test

    h1 = np.concatenate([h, np.ones([1, h.shape[1]])], axis=0)
    h1t = np.transpose(h1)

    yht = np.matmul(y, h1t)
    hht = np.matmul(h1, h1t)

    W = np.matmul(yht, np.transpose(np.linalg.pinv(np.transpose(hht))))

    h1_ = np.concatenate([h_, np.ones([1, h_.shape[1]])], axis=0)

    r = np.matmul(W, h1_)
    r = np.argmin(r, axis=0)
    g = np.argmax(mnist.test.labels, axis=1)
    print(np.sum(r == g) * 100 / h1_.shape[1])

    # ----------------------- with attention

    a = np.zeros(h.shape)
    for i in range(h.shape[1]):
        a[:, i] = quadprog_solve_qp(y[:, i], W[:, 0:-1], h[:, i], W[:, -1], float(h.shape[0]), 0.00001)

    aht = np.matmul(a, h1t)
    A = np.matmul(aht, np.transpose(np.linalg.pinv(np.transpose(hht))))

    ra = np.matmul(A, h1_)

    ha1_ = np.concatenate([h_ * ra, np.ones([1, h_.shape[1]])], axis=0)

    r2 = np.matmul(W, ha1_)
    r2 = np.argmin(r2, axis=0)
    g = np.argmax(mnist.test.labels, axis=1)
    print(np.sum(r2 == g) * 100 / h_.shape[1])

    # ---------------------- with second attention

    a2 = np.zeros(h.shape)
    for i in range(h.shape[1]):
        a2[:, i] = quadprog_solve_qp(a[:, i], A[:, 0:-1], h[:, i], A[:, -1], float(h.shape[0]), 0.00001)

    aht2 = np.matmul(a2, h1t)
    A2 = np.matmul(aht2, np.transpose(np.linalg.pinv(np.transpose(hht))))

    ra2 = np.matmul(A2, h1_)

    ha1_ = np.concatenate([h_ * ra2, np.ones([1, h_.shape[1]])], axis=0)

    ra3 = np.matmul(A, ha1_)

    ha2_ = np.concatenate([h_ * ra3, np.ones([1, h_.shape[1]])], axis=0)

    r3 = np.matmul(W, ha2_)
    r3 = np.argmin(r3, axis=0)
    g = np.argmax(mnist.test.labels, axis=1)
    print(np.sum(r3 == g) * 100 / h_.shape[1])

    sess.close()
