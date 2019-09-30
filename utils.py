from glob import glob
import matplotlib.pyplot as plt
import numpy as np
from scipy.misc import imread, imresize


def load_images(data_dir):
    imagesA = glob(data_dir + '/TrainA/*.*')
    imagesB = glob(data_dir + '/TrainB/*.*')

    min_l = len(imagesB)
    print(len(imagesA))
    print(len(imagesB))
    allImagesA = []
    allImagesB = []

    for index, filename in enumerate(imagesA):
        if index > 300:
            break
        imgA = imread(filename, mode='RGB')
        if index < min_l - 1:
            imgB = imread(imagesB[index], mode='RGB')
            imgB = imresize(imgB, (128, 128))

        imgA = imresize(imgA, (128, 128))

        if np.random.random() > 0.5:
            imgA = np.fliplr(imgA)
            if index < min_l - 1:
                imgB = np.fliplr(imgB)

        allImagesA.append(imgA)

        if index < min_l - 1:
            allImagesB.append(imgB)

    # Normalize images
    allImagesA = np.array(allImagesA) / 127.5 - 1.
    allImagesB = np.array(allImagesB) / 127.5 - 1.

    return allImagesA, allImagesB


def load_test_batch(data_dir, batch_size):
    imagesA = glob(data_dir + '/TestA/*.*')
    imagesB = glob(data_dir + '/TestB/*.*')

    imagesA = np.random.choice(imagesA, batch_size)
    imagesB = np.random.choice(imagesB, batch_size)

    allA = []
    allB = []

    for i in range(len(imagesA)):
        # Load images and resize images
        imgA = imresize(imread(imagesA[i], mode='RGB').astype(np.float32), (128, 128))
        imgB = imresize(imread(imagesB[i], mode='RGB').astype(np.float32), (128, 128))

        allA.append(imgA)
        allB.append(imgB)

    return np.array(allA) / 127.5 - 1.0, np.array(allB) / 127.5 - 1.0


def save_images(originalA, generatedB, recosntructedA, originalB, generatedA, reconstructedB, path):
    """
    Save images
    """
    fig = plt.figure()
    ax = fig.add_subplot(2, 3, 1)
    ax.imshow(originalA)
    ax.axis("off")
    ax.set_title("Original")

    ax = fig.add_subplot(2, 3, 2)
    ax.imshow(generatedB)
    ax.axis("off")
    ax.set_title("Generated")

    ax = fig.add_subplot(2, 3, 3)
    ax.imshow(recosntructedA)
    ax.axis("off")
    ax.set_title("Reconstructed")

    ax = fig.add_subplot(2, 3, 4)
    ax.imshow(originalB)
    ax.axis("off")
    ax.set_title("Original")

    ax = fig.add_subplot(2, 3, 5)
    ax.imshow(generatedA)
    ax.axis("off")
    ax.set_title("Generated")

    ax = fig.add_subplot(2, 3, 6)
    ax.imshow(reconstructedB)
    ax.axis("off")
    ax.set_title("Reconstructed")

    plt.savefig(path)
