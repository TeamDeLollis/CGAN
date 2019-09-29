import argparse
import tensorflow as tf
tf.compat.v1.set_random_seed(2)
from ops import *
from utils import *
from cyclegan import *
import time
import numpy as np


parser = argparse.ArgumentParser(description='')
parser.add_argument('--mode', type=str, default='train')
args = parser.parse_args()


def main(_):
    tfconfig = tf.compat.v1.ConfigProto(allow_soft_placement=True)
    tfconfig.gpu_options.allow_growth = True

    """
    with tf.compat.v1.Session(config=tfconfig) as sess:
        sess.run(tf.global_variables_initializer())
        data_dir = "datasets/vangogh2photo/"
        batch_size = 1
        epochs = 500
        # mode = 'train'

        if args.mode == 'train':
            
            Load dataset
            
            imagesA, imagesB = load_images(data_dir=data_dir)

            # Build and compile generator networks
            discriminatorA = Discriminator(sess)
            discriminatorB = Discriminator(sess)

            # Build generator networks
            generatorAToB = Generator(sess)
            generatorBToA = Generator(sess)

            
            Create an adversarial network
            
            inputA = Input(shape=(128, 128, 3))
            inputB = Input(shape=(128, 128, 3))

            # Generated images using both of the generator networks
            # generatedB = generatorAToB(inputA)
            # generatedA = generatorBToA(inputB)

            # Reconstruct images back to original images
            # reconstructedA = generatorBToA(generatedB)
            # reconstructedB = generatorAToB(generatedA)

            # generatedAId = generatorBToA(inputA)  # this must be an identity --> should return the same
            # generatedBId = generatorAToB(inputB)  # this must be an identity --> should return the same

            # Make both of the discriminator networks non-trainable
            # discriminatorA.trainable = False
            # discriminatorB.trainable = False

            # probsA = discriminatorA(generatedA)  # Dx(F(y))
            # probsB = discriminatorB(generatedB)  # Dy(G(x))

            # adversarial_model = Model(inputs=[inputA, inputB],
            #                          outputs=[probsA, probsB, reconstructedA, reconstructedB,
            #                                   generatedAId, generatedBId])
            # adversarial_model.compile(loss=['mse', 'mse', 'mae', 'mae', 'mae', 'mae'],
            #                          loss_weights=[1, 1, 10.0, 10.0, 1.0, 1.0],
            #                          optimizer=common_optimizer)

            tensorboard = TensorBoard(log_dir="logs/{}".format(time.time()), write_images=True, write_grads=True,
                                      write_graph=True)
            tensorboard.set_model(generatorAToB)
            tensorboard.set_model(generatorBToA)
            tensorboard.set_model(discriminatorA)
            tensorboard.set_model(discriminatorB)

            real_labels = np.ones((batch_size, 7, 7, 1))
            fake_labels = np.zeros((batch_size, 7, 7, 1))

            for epoch in range(epochs):
                print("Epoch:{}".format(epoch))

                dis_losses = []
                gen_losses = []

                num_batches = int(min(imagesA.shape[0], imagesB.shape[0]) / batch_size)
                # print("Number of batches:{}".format(num_batches))

                for index in range(num_batches):
                    # print("Batch:{}".format(index))

                    # Sample images
                    batchA = imagesA[index * batch_size:(index + 1) * batch_size]
                    batchB = imagesB[index * batch_size:(index + 1) * batch_size]

                    # Translate images to opposite domain
                    generatedB = generatorAToB.predict(batchA)
                    generatedA = generatorBToA.predict(batchB)

                    # Train the discriminator A on real and fake images
                    dALoss1 = discriminatorA.train_on_batch(batchA, real_labels)
                    dALoss2 = discriminatorA.train_on_batch(generatedA, fake_labels)

                    # Train the discriminator B on ral and fake images
                    dBLoss1 = discriminatorB.train_on_batch(batchB, real_labels)
                    dbLoss2 = discriminatorB.train_on_batch(generatedB, fake_labels)

                    # Calculate the total discriminator loss
                    d_loss = 0.5 * np.add(0.5 * np.add(dALoss1, dALoss2), 0.5 * np.add(dBLoss1, dbLoss2))

                    if index % 10 == 0:
                        print("d_loss:{}".format(d_loss))

                    Train the generator networks
                    g_loss = adversarial_model.train_on_batch([batchA, batchB],
                                                              [real_labels, real_labels, batchA, batchB, batchA,
                                                               batchB])
                    if index % 10 == 0:
                        print("g_loss:{}".format(g_loss))

                    dis_losses.append(d_loss)
                    gen_losses.append(g_loss)

                Save losses to Tensorboard after each epoch


                # Sample and save images after every 10 epochs
                if epoch % 10 == 0:
                    # Get a batch of test data
                    batchA, batchB = load_test_batch(data_dir=data_dir, batch_size=2)

                    # Generate images
                    generatedB = generatorAToB.predict(batchA)
                    generatedA = generatorBToA.predict(batchB)

                    # Get reconstructed images
                    reconsA = generatorBToA.predict(generatedB)
                    reconsB = generatorAToB.predict(generatedA)

                    # Save original, generated and reconstructed images
                    for i in range(len(generatedA)):
                        save_images(originalA=batchA[i], generatedB=generatedB[i], recosntructedA=reconsA[i],
                                    originalB=batchB[i], generatedA=generatedA[i], reconstructedB=reconsB[i],
                                    path="results/gen_{}_{}".format(epoch, i))


        elif args.mode == 'predict':
            # Build generator networks
            generatorAToB = build_generator()
            generatorBToA = build_generator()

            generatorAToB.load_weights("generatorAToB.h5")
            generatorBToA.load_weights("generatorBToA.h5")

            # Get a batch of test data
            batchA, batchB = load_test_batch(data_dir=data_dir, batch_size=2)

            # Save images
            generatedB = generatorAToB.predict(batchA)
            generatedA = generatorBToA.predict(batchB)

            reconsA = generatorBToA.predict(generatedB)
            reconsB = generatorAToB.predict(generatedA)

            for i in range(len(generatedA)):
                save_images(originalA=batchA[i], generatedB=generatedB[i], recosntructedA=reconsA[i],
                            originalB=batchB[i], generatedA=generatedA[i], reconstructedB=reconsB[i],
                            path="results/test_{}".format(i))

    """

    with tf.Session(config=tfconfig) as sess:
        model = cyclegan(sess)
        if args.mode == 'train':
            model.train()
        else:
            model.test()

if __name__ == '__main__':
    tf.compat.v1.app.run()
