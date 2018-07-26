from tensorflow.examples.tutorials.mnist import input_data
import numpy as np
import tensorflow as tf
import quadprog


def quadprog_solve_qp(Wx, b, sum_active, lamb):
    dim = b.shape[0]

    qp_G = (np.matmul(np.transpose(Wx), Wx) + np.identity(dim) - Wx - np.transpose(Wx)) + np.identity(dim) * lamb
    qp_a = (b - np.matmul(np.transpose(Wx), b))

    qp_C = np.identity(dim)
    qp_b = np.zeros([dim], dtype=np.float64)
    # qp_b[0] = sum_active
    # qp_b[(1 + dim):] = -2.0

    # print(qp_G.shape, qp_a.shape, qp_C.shape, qp_b.shape)

    return np.maximum(quadprog.solve_qp(qp_G, qp_a, qp_C, qp_b, meq=0)[0], 0)


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
        h = tf.matmul(self.w, x)
        hs = tf.shape(h)
        y = tf.where(tf.random_uniform(hs) - h < 0, tf.ones(hs), tf.zeros(hs))
        # y = tf.nn.sigmoid(h)
        x_ = tf.nn.sigmoid(tf.matmul(tf.transpose(self.w), y))

        self.objective = tf.trace(tf.matmul(x - x_, tf.transpose(x - x_))) / tf.cast(tf.reduce_prod(tf.shape(x)), tf.float32)
        grad = tf.gradients(self.objective, [self.w])
        learn_op = apply_gradients(zip(grad, [self.w]), self.objective, self.learning_rate)

        return y, learn_op


if __name__ == '__main__':

    mnist = input_data.read_data_sets("data/", one_hot=True)
    # data = np.concatenate([mnist.train.images, mnist.test.images, mnist.validation.images], axis=0)
    # labels = np.concatenate([mnist.train.labels, mnist.test.labels, mnist.validation.labels], axis=0)

    y = np.transpose(1 - mnist.test.labels)
    xt = mnist.test.images
    x = np.transpose(xt)

    sess = tf.Session()
    auc = AUC([28 * 28, 256])
    sess.run(tf.global_variables_initializer())

    error = auc.learn(sess, x, 0.01, 400)
    h = auc.feedup(sess, x)
    h_ = auc.feedup(sess, np.transpose(mnist.test.images))

    ht = np.transpose(h)

    # ----------------------- test

    yht = np.matmul(y, ht)
    hht = np.matmul(h, ht)

    W = np.matmul(yht, np.transpose(np.linalg.pinv(np.transpose(hht))))

    r = np.matmul(W, h_)
    r = np.argmin(r, axis=0)
    g = np.argmax(mnist.test.labels, axis=1)
    print(np.sum(r == g) * 100 / h_.shape[1])

    # ----------------------- fixed point

    # xa1 = np.concatenate([y * h, np.ones([1, y.shape[1]])], axis=0)
    # axat = np.matmul(y, np.transpose(xa1))
    # xaxat = np.matmul(xa1, np.transpose(xa1))
    # A = np.matmul(axat, np.transpose(np.linalg.pinv(np.transpose(xaxat + 0.0001 * np.identity(xa1.shape[0])))))

    # count_correct = 0
    # for i in range(h_.shape[1]):
    #     Ax = np.matmul(A[:, 0:-1], np.diag(h_[:, i]))
    #     b = A[:, -1]
    #     r2 = quadprog_solve_qp(Ax, b, 9.0, 0.00001)

    #     if np.argmin(r2) == np.argmax(mnist.test.labels[i, :]):
    #         count_correct = count_correct + 1

    # print(count_correct * 100 / h_.shape[1])

    sess.close()
