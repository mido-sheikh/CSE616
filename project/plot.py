import random
from preprocess import *
import matplotlib.pyplot as plt


def list_images(dataset, dataset_y, signs, ylabel="", cmap=None):
    """
    Display a list of images in a single figure with matplotlib.
        Parameters:
            images: An np.array compatible with plt.imshow.
            lanel (Default = No label): A string to be used as a label for each image.
            cmap (Default = None): Used to display gray images.
    """
    plt.figure(figsize=(15, 16))
    for i in range(6):
        plt.subplot(1, 6, i+1)
        indx = random.randint(0, len(dataset))
        cmap = 'gray' if len(dataset[indx].shape) == 2 else cmap
        plt.imshow(dataset[indx], cmap = cmap)
        plt.xlabel(signs[dataset_y[indx]])
        plt.ylabel(ylabel)
        plt.xticks([])
        plt.yticks([])
    plt.tight_layout(pad=0, h_pad=0, w_pad=0)
    plt.show()


def histogram_plot(dataset, label, n_classes):
    """
    Plots a histogram of the input data.
        Parameters:
            dataset: Input data to be plotted as a histogram.
            lanel: A string to be used as a label for the histogram.
    """
    hist, bins = np.histogram(dataset, bins=n_classes)
    width = 0.7 * (bins[1] - bins[0])
    center = (bins[:-1] + bins[1:]) / 2
    plt.bar(center, hist, align='center', width=width)
    plt.xlabel(label)
    plt.ylabel("Image count")
    plt.show()


def plot_figures(X_train, y_train, X_valid, y_valid, X_test, y_test, signs, n_classes):
    # Plotting sample examples
    list_images(X_train, y_train, signs, "Training example")
    list_images(X_test, y_test, signs, "Testing example")
    list_images(X_valid, y_valid, signs, "Validation example")

    # Plotting histograms of the count of each sign
    histogram_plot(y_train, "Training examples", n_classes)
    histogram_plot(y_test, "Testing examples", n_classes)
    histogram_plot(y_valid, "Validation examples", n_classes)

    # Sample images after greyscaling
    gray_images = list(map(gray_scale, X_train))
    list_images(gray_images, y_train, signs, "Gray Scale image", "gray")

    # Sample images after Local Histogram Equalization
    equalized_images = list(map(local_histo_equalize, gray_images))
    list_images(equalized_images, y_train, signs, "Equalized Image", "gray")

    # Sample images after normalization
    n_training = X_train.shape
    normalized_images = np.zeros((n_training[0], n_training[1], n_training[2]))
    for i, img in enumerate(equalized_images):
        normalized_images[i] = image_normalize(img)
    list_images(normalized_images, y_train, signs, "Normalized Image", "gray")
    normalized_images = normalized_images[..., None]
    return normalized_images
