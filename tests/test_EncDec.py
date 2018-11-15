from .context import CDRTR
from CDRTR.core.DeepModel import EncDec, Encoder, Decoder, AutoEncDec
from CDRTR.config import Config

import unittest
import tensorflow as tf
import numpy as np


class EncDecTestSuite(unittest.TestCase):
    def setUp(self):
        self.gpuConfig = Config.getGPUConfig()
        tf.logging.set_verbosity(tf.logging.ERROR)

    def test_all(self):
        ipt_x = np.random.randint(0, 9, size=(2, 10))
        input_T = tf.placeholder(shape=[None, 10], dtype=tf.float32)
        encdec = AutoEncDec(input_T, [6, 3], [3, 6])
        optimizer = tf.train.AdamOptimizer(0.001)
        train_op = optimizer.minimize(encdec.loss)
        init = tf.global_variables_initializer()
        sess = tf.Session(config=self.gpuConfig)
        sess.run(init)
        for i in range(10):
            _, loss = sess.run([train_op, encdec.loss], feed_dict={encdec.input: ipt_x})
            print "The", i, "loss of first run is", loss
        sess.close()

    def test_Enc(self):
        ipt_x = np.random.randint(0, 9, size=(2, 10))
        input_T = tf.placeholder(shape=[None, 10], dtype=tf.float32)
        enc = Encoder(input_T, [6, 3])
        init = tf.global_variables_initializer()
        sess = tf.Session(config=self.gpuConfig)
        sess.run(init)
        opt = sess.run(enc.output, feed_dict={enc.input: ipt_x})
        print opt

    def tearDown(self):
        pass


if __name__ == "__main__":
    unittest.main()



