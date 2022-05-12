import pickle
from sklearn.utils import shuffle
import csv
import os
from sklearn.metrics import confusion_matrix
from VGGnet import *
from config import *
from plot import *


#   The pickled data is a dictionary with 4 key/value pairs: ('features', 'labels', 'sizes', 'coords')
def load_data():
    training_file = "./data/train.p"
    validation_file = "./data/valid.p"
    testing_file = "./data/test.p"

    with open(training_file, mode='rb') as f:
        train = pickle.load(f)
    with open(validation_file, mode='rb') as f:
        valid = pickle.load(f)
    with open(testing_file, mode='rb') as f:
        test = pickle.load(f)

    X_train, y_train = train['features'], train['labels']
    X_valid, y_valid = valid['features'], valid['labels']
    X_test, y_test = test['features'], test['labels']
    return X_train, y_train, X_valid, y_valid, X_test, y_test


#   Mapping ClassID to traffic sign names
def load_names():
    signs = []
    with open('signnames.csv', 'r') as csvfile:
        signnames = csv.reader(csvfile, delimiter=',')
        next(signnames, None)
        for row in signnames:
            signs.append(row[1])
        csvfile.close()
    return signs


def train_model(normalized_images, X_valid, y_train, n_classes):
    # Validation set preprocessing
    X_valid_preprocessed = preprocess(X_valid)
    VGGNet_Model = VGGnet(n_out=n_classes)
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        num_examples = len(y_train)
        print("Training...")
        for i in range(EPOCHS):
            normalized_images, y_train = shuffle(normalized_images, y_train)
            for offset in range(0, num_examples, BATCH_SIZE):
                end = offset + BATCH_SIZE
                batch_x, batch_y = normalized_images[offset:end], y_train[offset:end]
                sess.run(VGGNet_Model.training_operation, feed_dict={x: batch_x, y: batch_y, keep_prob: 0.5, keep_prob_conv: 0.7})

            validation_accuracy = VGGNet_Model.evaluate(X_valid_preprocessed, y_valid)
            print("EPOCH {} : Validation Accuracy = {:.3f}%".format(i+1, (validation_accuracy*100)))
        VGGNet_Model.saver.save(sess, os.path.join(DIR, 'VGGnet'))
        print("Model saved")


def test_model(VGGNet_Model, X_test, y_test):
    # Test set preprocessing
    X_test_preprocessed = preprocess(X_test)
    with tf.Session() as sess:
        VGGNet_Model.saver.restore(sess, os.path.join(DIR, "VGGNet"))
        y_pred = VGGNet_Model.y_predict(X_test_preprocessed)
        test_accuracy = sum(y_test == y_pred) / len(y_test)
        print("Test Accuracy = {:.1f}%".format(test_accuracy * 100))
        return X_test, y_pred


def confusion_matrix(y_test, y_pred):
    cm = confusion_matrix(y_test, y_pred)
    cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    cm = np.log(.0001 + cm)
    plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
    plt.title('Log of normalized Confusion Matrix')
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.show()


if __name__ == '__main__':
    signs = load_names()
    X_train, y_train, X_valid, y_valid, X_test, y_test = load_data()
    n_classes = len(np.unique(y_train))
    normalized_images = plot_figures(X_train, y_train, X_valid, y_valid, X_test, y_test, signs, n_classes)
    X_train, y_train = shuffle(X_train, y_train)
    train_model(normalized_images, X_valid, y_train, n_classes)
    VGGNet_Model = VGGnet()
    X_test, y_pred = test_model(VGGNet_Model, X_test, y_test)
    confusion_matrix(X_test, y_pred)


