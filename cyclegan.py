from models import *
import tensorflow as tf
import time
import numpy as np
from utils import *

np.random.seed(25)


class Cyclegan(object):

    def __init__(self, sess):

        self.sess = sess
        self.batch_size = 1
        self.image_size = 128  # reduced in utils

        self.L1_lambda = 10
        # self.data_dir = "datasets/vangogh2photo/" for the reduced dataset
        self.data_dir = "/content/drive/My Drive/vangogh2photobig"

        self.discriminatorA = Discriminator(sess)
        self.discriminatorB = Discriminator(sess)

        # Build generator networks
        self.generatorAToB = Generator(sess)
        self.generatorBToA = Generator(sess)

        self.d_optim = None
        self.g_optim = None
        self.d_loss = None
        self.g_loss = None
        self.d_vars = None
        self.g_vars = None
        self.real_A = None
        self.real_B = None
        self.fake_A = None
        self.fake_A_ = None
        self.fake_B = None
        self.fake_B_ = None
        self.fake_A_sample = None
        self.fake_B_sample = None
        self.test_A = None
        self.test_B = None
        self.testB = None
        self.testA = None

        self.build_model()

    def build_model(self):

        self.real_A = tf.placeholder(tf.float32,
                                     [None, self.image_size, self.image_size, 3],
                                     name='real_A')
        self.real_B = tf.placeholder(tf.float32,
                                     [None, self.image_size, self.image_size, 3],
                                     name='real_B')

        self.fake_B = self.generatorAToB.predict(self.real_A)
        self.fake_A = self.generatorBToA.predict(self.real_B)
        self.fake_A_ = self.generatorBToA.predict(self.fake_B)
        self.fake_B_ = self.generatorAToB.predict(self.fake_A)

        DA_fake = self.discriminatorA.predict(self.fake_A)
        DB_fake = self.discriminatorB.predict(self.fake_B)

        g_loss_a2b = self.mse_criterion(DB_fake, tf.ones_like(DB_fake)) \
                     + self.L1_lambda * self.abs_criterion(self.real_A, self.fake_A_) \
                     + self.L1_lambda * self.abs_criterion(self.real_B, self.fake_B_)
        g_loss_b2a = self.mse_criterion(DA_fake, tf.ones_like(DA_fake)) \
                     + self.L1_lambda * self.abs_criterion(self.real_A, self.fake_A_) \
                     + self.L1_lambda * self.abs_criterion(self.real_B, self.fake_B_)
        self.g_loss = self.mse_criterion(DA_fake, tf.ones_like(DA_fake)) \
                      + self.mse_criterion(DB_fake, tf.ones_like(DB_fake)) \
                      + self.L1_lambda * self.abs_criterion(self.real_A, self.fake_A_) \
                      + self.L1_lambda * self.abs_criterion(self.real_B, self.fake_B_)

        self.fake_A_sample = tf.placeholder(tf.float32,
                                            [None, self.image_size, self.image_size, 3], name='fake_A_sample')
        self.fake_B_sample = tf.placeholder(tf.float32,
                                            [None, self.image_size, self.image_size, 3], name='fake_B_sample')

        DB_real = self.discriminatorB.predict(self.real_B)
        DA_real = self.discriminatorA.predict(self.real_A)
        DB_fake_sample = self.discriminatorB.predict(self.fake_B_sample)
        DA_fake_sample = self.discriminatorA.predict(self.fake_A_sample)

        db_loss_real = self.mse_criterion(DB_real, tf.ones_like(DB_real))
        db_loss_fake = self.mse_criterion(DB_fake_sample, tf.zeros_like(DB_fake_sample))
        db_loss = (db_loss_real + db_loss_fake) / 2
        da_loss_real = self.mse_criterion(DA_real, tf.ones_like(DA_real))
        da_loss_fake = self.mse_criterion(DA_fake_sample, tf.zeros_like(DA_fake_sample))
        da_loss = (da_loss_real + da_loss_fake) / 2
        self.d_loss = da_loss + db_loss

        """
        self.g_loss_a2b_sum = tf.summary.scalar("g_loss_a2b", g_loss_a2b)
        self.g_loss_b2a_sum = tf.summary.scalar("g_loss_b2a", g_loss_b2a)
        self.g_loss_sum = tf.summary.scalar("g_loss", self.g_loss)
        self.g_sum = tf.summary.merge([g_loss_a2b_sum, self.g_loss_b2a_sum, self.g_loss_sum])
        self.db_loss_sum = tf.summary.scalar("db_loss", db_loss)
        self.da_loss_sum = tf.summary.scalar("da_loss", da_loss)
        self.d_loss_sum = tf.summary.scalar("d_loss", self.d_loss)
        self.db_loss_real_sum = tf.summary.scalar("db_loss_real", db_loss_real)
        self.db_loss_fake_sum = tf.summary.scalar("db_loss_fake", db_loss_fake)
        self.da_loss_real_sum = tf.summary.scalar("da_loss_real", da_loss_real)
        self.da_loss_fake_sum = tf.summary.scalar("da_loss_fake", da_loss_fake)
        self.d_sum = tf.summary.merge(
            [self.da_loss_sum, self.da_loss_real_sum, self.da_loss_fake_sum,
             self.db_loss_sum, self.db_loss_real_sum, self.db_loss_fake_sum,
             self.d_loss_sum]
        )
        """

        self.test_A = tf.placeholder(tf.float32,
                                     [None, self.image_size, self.image_size, 3], name='test_A')
        self.test_B = tf.placeholder(tf.float32,
                                     [None, self.image_size, self.image_size, 3], name='test_B')
        self.testB = self.generatorAToB.predict(self.test_A)
        self.testA = self.generatorBToA.predict(self.test_B)

        t_vars = tf.trainable_variables()
        self.d_vars = [var for var in t_vars if 'discriminator' in str(var.name)]
        self.g_vars = [var for var in t_vars if 'generator' in str(var.name)]
        # for var in t_vars: print(var.name)

    def train(self):

        """Train cyclegan"""
        self.d_optim = tf.train.AdamOptimizer(0.0002, beta1=0.5).minimize(self.d_loss, var_list=self.d_vars)
        self.g_optim = tf.train.AdamOptimizer(0.0002, beta1=0.5).minimize(self.g_loss, var_list=self.g_vars)

        init_op = tf.global_variables_initializer()
        self.sess.run(init_op)
        # self.writer = tf.summary.FileWriter("./logs", self.sess.graph)

        counter = 1
        epochs = 500
        start_time = time.time()

        for epoch in range(epochs):
            dataA, dataB = load_images(self.data_dir)
            np.random.shuffle(dataA)
            np.random.shuffle(dataB)
            batch_idxs = min(len(dataA), len(dataB)) // self.batch_size

            for idx in range(0, batch_idxs):
                batchA = dataA[idx * self.batch_size:(idx + 1) * self.batch_size]
                batchB = dataB[idx * self.batch_size:(idx + 1) * self.batch_size]
                # batch_images = np.array(batch_images).astype(np.float32)

                # Update G network and record fake outputs
                fake_A, fake_B, _ = self.sess.run([self.fake_A, self.fake_B, self.g_optim],
                                                  feed_dict={self.real_A: batchA,
                                                             self.real_B: batchB})

                # Update D network
                _ = self.sess.run([self.d_optim], feed_dict={self.real_A: batchA,
                                                             self.real_B: batchB,
                                                             self.fake_A_sample: fake_A,
                                                             self.fake_B_sample: fake_B})
                # self.writer.add_summary(summary_str, counter)

                counter += 1

            if epoch % 10 == 0:
                # Get a batch of test data
                batchA, batchB = load_test_batch(data_dir=self.data_dir, batch_size=1)

                generatedA, generatedB, reconsA, reconsB = self.sess.run(
                    [self.fake_A, self.fake_B, self.fake_A_, self.fake_B_],
                    feed_dict={self.real_A: batchA,
                               self.real_B: batchB})

                # Save original, generated and reconstructed images
                for i in range(len(generatedA)):
                    save_images(originalA=batchA[i], generatedB=generatedB[i], recosntructedA=reconsA[i],
                                originalB=batchB[i], generatedA=generatedA[i], reconstructedB=reconsB[i],
                                path="results/gen_{}_{}".format(epoch, i))

    @staticmethod
    def mse_criterion(x, target):
        return tf.reduce_mean((x - target) ** 2)

    @staticmethod
    def abs_criterion(x, target):
        return tf.reduce_mean(tf.abs(x - target))
