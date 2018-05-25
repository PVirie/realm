import numpy as np
import util


def sigmoid(x):
    return 1.0 / (1 + np.exp(-x))


def get_best_input(y, W, Wb):
    h = np.matmul(np.linalg.pinv(W), y - Wb)
    diff = (y - np.matmul(W, h) - Wb)**2
    return h, diff


def get_best_attention(h, x):
    a = sigmoid(np.divide(h, x))
    diff = (h - a * x)**2
    return a, diff


class Network:
    def __init__(self, input_shape, output_shape, learning_rate=0.01):

        self.W = util.random_uniform(output_shape, input_shape)
        self.Wb = util.random_uniform(output_shape, 1)
        self.A = util.random_uniform(input_shape, input_shape)
        self.Ab = util.random_uniform(input_shape, 1)
        self.C = util.random_uniform(1, input_shape)
        self.Cb = util.random_uniform(1, 1)

    def learn(self, x, y):

        # h, diff = get_best_input(y, self.W, self.Wb)
        # # has W embedded y?
        # if diff > 0.2:
        #     # incrementally learn Wx+b -> y
        # else:
        #     # has

    def classify(self, x):

        h = x
        for i in range(10):
            c = np.matmul(self.C, h) + self.Cb
            if c[0, 0] > 0.8:
                break
            a = sigmoid(np.matmul(self.A, h) + self.Ab)
            h = a * x

        return np.matmul(self.W, h) + self.Wb

    def save(self):
        # self.saver.save(self.sess, "./artifacts/" + "weights")

    def load(self):
        # self.saver.restore(self.sess, "./artifacts/" + "weights")


if __name__ == '__main__':

    sess = tf.Session()

    input = tf.placeholder(tf.float32, [None, None])
    output = mirroring_relus(input)
    input_, residue = inverse_mirroring_relus(output)
    error = tf.reduce_sum(tf.squared_difference(input_, input))
    cpu_input = (np.random.rand(1, 10) - 0.5)
    print sess.run((error, input_, input), feed_dict={input: cpu_input})
