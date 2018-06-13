import tensorflow as tf
import numpy as np
import os


dir_path = os.path.dirname(os.path.realpath(__file__))


def mirroring_relus(input):
    return tf.stack([tf.nn.relu(input), tf.nn.relu(-input)], axis=2)


def inverse_mirroring_relus(input):
    bases = tf.unstack(input, axis=2)
    conditions = bases[0] > bases[1]
    out = tf.where(conditions, bases[0], -bases[1])
    residue = tf.where(conditions, tf.square(bases[1]), tf.square(bases[0]))
    return out, residue


def random_uniform(rows, cols):
    return (np.random.rand(rows, cols) - 0.5) * 0.001


def apply_gradients(gradients, delta, rate=0.001, name="adam"):
    training_op = tf.train.AdamOptimizer(rate, name=name).apply_gradients(gradients)
    if delta is not None:
        return {"op": training_op, "cost": delta}
    else:
        return {"op": training_op}


class Layer:
    def __init__(self, shape, learning_rate=0.01, learning_coeff=0.001, name="Layer"):
        self.name = name
        self.shape = shape
        self.learning_rate = learning_rate
        self.x = tf.placeholder(dtype=tf.float32, shape=[shape[0], 1])
        self.z_ = tf.placeholder(dtype=tf.float32, shape=[shape[1], 1, 2])
        self.w = tf.Variable(random_uniform(shape[1], shape[0] + 1), dtype=tf.float32)
        self.C = tf.Variable(np.zeros((shape[0] + 1, shape[0] + 1)), dtype=tf.float32)

        self.y, self.learn_ops, self.update_ops = self.create_forward_graph(self.x, learning_coeff)
        self.z = mirroring_relus(self.y)
        self.y_, residue = inverse_mirroring_relus(self.z_)
        self.x_ = self.create_backward_graph(self.y_)

        self.saver = tf.train.Saver(var_list=[self.C], keep_checkpoint_every_n_hours=1)

    def non_linear_feedup(self, session, x):
        if len(x.shape) < 2:
            x = np.expand_dims(x, axis=1)
        out = session.run(self.z, feed_dict={self.x: x})
        return np.reshape(out, [-1])

    def feedup(self, session, x):
        if len(x.shape) < 2:
            x = np.expand_dims(x, axis=1)
        out = session.run(self.y, feed_dict={self.x: x})
        return out[:, 0]

    def project(self, session, z):
        z_ = np.reshape(z, [-1, 1, 2])
        out = session.run(self.x_, feed_dict={self.z_: z_})
        return out[:, 0]

    def collect(self, session, x):
        if len(x.shape) < 2:
            x = np.expand_dims(x, axis=1)
        session.run(self.learn_ops, feed_dict={self.x: x})

    def update(self, session):
        for i in range(20):
            __ = session.run((self.update_ops, self.objective))
        return __[1]

    def learn(self, session, x):
        self.collect(session, x)
        __ = self.update(session)
        return __

    def create_forward_graph(self, input, learning_coeff):
        expanded = tf.concat([input, [[1]]], axis=0)
        y = tf.matmul(self.w, expanded)

        gather_C = tf.assign(self.C, tf.matmul(expanded, expanded, transpose_b=True) * (learning_coeff) + self.C * (1 - learning_coeff)).op

        wtw = tf.matmul(self.w, self.w, transpose_a=True)
        self.objective = tf.trace(self.C - tf.matmul(self.C, wtw) - tf.matmul(wtw, self.C) + tf.matmul(tf.matmul(wtw, self.C), wtw))
        grad_C = tf.gradients(self.objective, [self.w])
        learn_op_w = apply_gradients(zip(grad_C, [self.w]), self.objective, self.learning_rate)

        return y, gather_C, learn_op_w

    def create_backward_graph(self, inner):
        input = tf.matmul(self.w, inner, transpose_a=True)
        return input[:-1, :]

    def save(self, session, session_name):
        path = os.path.join(dir_path, "..", "artifacts", session_name)
        if not os.path.exists(path):
            os.makedirs(path)
        self.saver.save(session, os.path.join(path, self.name))

    def load(self, session, session_name):
        path = os.path.join(dir_path, "..", "artifacts", session_name)
        if not os.path.exists(path):
            return False
        self.saver.restore(session, os.path.join(path, self.name))
        return True


if __name__ == '__main__':
    # test here
    sess = tf.Session()
    layer = Layer([20, 4], 0.01, 0.001)
    sess.run(tf.global_variables_initializer())

    error = layer.learn(sess, np.random.rand(20))
    y = layer.non_linear_feedup(sess, np.random.rand(20))
    print(error, y.shape)

    sess.close()
