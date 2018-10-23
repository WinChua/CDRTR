from .context import CDRTR
from CDRTR.core.DeepModel import cnn_text, factorization_machine
from CDRTR.core.DeepModel import embedding_lookup, embedding_layer
from CDRTR.config import Config
import unittest
import tensorflow as tf
import numpy as np


class CNNTestSuite(unittest.TestCase):
    def setUp(self):
        self.gpuConfig = Config.getGPUConfig()
        tf.logging.set_verbosity(tf.logging.ERROR)
        pass

    def test_all(self):
        x = np.random.randint(0, 9, size=(2, 10))
        sl, vs, es, fs, nf = 10, 10, 5, [3], 1
        icx = tf.placeholder(
                dtype=tf.int32,
                shape=[None, sl],
                name="inputx")

        W = embedding_layer(vs, es)
        x_feature = embedding_lookup(icx, W)
        ocx, drop = cnn_text(sl, fs, nf, x_feature)
        cnn_out = tf.expand_dims(ocx, 1)
        score_pred = factorization_machine(cnn_out)
        session = tf.Session(config=self.gpuConfig)
        init = tf.global_variables_initializer()
        session.run(init)
        print "Hello, World"
        print session.run(score_pred, feed_dict={icx: x, drop: 0.5})
        session.close()

    def tearDown(self):
        pass


if __name__ == "__main__":
    unittest.main()
