# Load pickled data
import pickle
from sklearn.utils import shuffle

training_file = 'traffic-signs-data/train.p'
testing_file = 'traffic-signs-data/test.p'

with open(training_file, mode='rb') as f:
    train = pickle.load(f)
with open(testing_file, mode='rb') as f:
    test = pickle.load(f)

X_train, y_train = train['features'], train['labels']
X_test, y_test = test['features'], test['labels']

### Replace each question mark with the appropriate value.

# TODO: Number of training examples
n_train = len(X_train)

# TODO: Number of testing examples.
n_test = len(X_test)

# TODO: What's the shape of an traffic sign image?
image_shape = X_train[0].shape

# TODO: How many unique classes/labels there are in the dataset.
n_classes = len(set(y_train))

print("Number of training examples =", n_train)
print("Number of testing examples =", n_test)
print("Image data shape =", image_shape)
print("Number of classes =", n_classes)

### Data exploration visualization goes here.
### Feel free to use as many code cells as needed.
import matplotlib.pyplot as plt
import numpy as np

# Visualizations will be shown in the notebook.
#%matplotlib inline

"""
def rgbToGray(image):
    grey = np.zeros((image.shape[0], image.shape[1])) # init 2D numpy array
    # get row number
    for rownum in range(len(image)):
        for colnum in range(len(image[rownum])):
            grey[rownum][colnum] = np.average(image[rownum][colnum])
    return grey
"""

def rgbToGray(rgb):
    return np.dot(rgb[..., :3], [0.299, 0.587, 0.114])

def add_noise(img):
    row, col, channels = img.shape
    mean = 0
    gauss = np.random.normal(mean, 1, (row, col, channels))
    gauss = gauss.reshape(row, col, channels)
    noisy = img + gauss
    return noisy


def make_gray(colored_ones):
    "Convert images to gray"
    gray_ones = np.zeros((colored_ones.shape[0], colored_ones.shape[1], colored_ones.shape[2], 1)) # init 2D numpy array
    for imgnum in range(len(colored_ones)):
        gray_ones[imgnum] = rgbToGray(colored_ones[imgnum]).reshape(32,32,1)
    return gray_ones

def normalize(img):
    img *= 255.0/img.max()
    return img

def pre_process(X_train, y_train):
    "Pre process data converting to gray and normalize"
    gray_ones = make_gray(X_train)
    preprocessed = np.zeros((gray_ones.shape[0], gray_ones.shape[1], gray_ones.shape[2], 1))
    # get row number
    index = 0
    for imgnum in range(len(gray_ones)):
        preprocessed[imgnum] = gray_ones[imgnum] #add_noise(normalize(gray_ones[imgnum]))
        index = index + 1
        if index % 1000 == 0:
            print(str(index) + "/" + str(len(X_train)))
    return preprocessed, y_train

print("Convert images..")
X_train, y_train = pre_process(X_train, y_train)
X_test = make_gray(X_test)

X_validation = X_test[int(len(X_test)/2):]
y_validation = y_test[int(len(X_test)/2):]
y_test = y_test[0:int(len(X_test/2))]
X_test = X_test[0:int(len(X_test/2))]

print(X_train.shape)

def show_images(X_train, y_train):
    "Display image examples and number of images"
    sign_classes = set(y_train)
    for class_index in sign_classes:
        data_for_class = [index for index in range(len(y_train)) if y_train[index] == class_index]
        index = data_for_class[0]
        img = X_train[index].reshape((32, 32))
        print(img.shape)
        plt.imshow(img, cmap='gray')
        plt.title("Traffic sign class =" + str(y_train[index]))
        plt.xlabel("Count of examples in test data: " + str(len(data_for_class)))
        plt.show()

#show_images(X_train, y_train)

from tensorflow.contrib.layers import flatten
import tensorflow as tf

EPOCHS = 50
BATCH_SIZE = 128

def create_fc_layer_connect(mu, sigma, layer0, keep_prop, in_size, out_size):
    fc_W = tf.Variable(tf.truncated_normal(shape=(in_size, out_size), mean=mu, stddev=sigma))
    fc_b = tf.Variable(tf.zeros(out_size))
    fc = tf.matmul(layer0, fc_W) + fc_b

    fc = tf.nn.relu(fc)

    fc_drop = tf.nn.dropout(fc, keep_prob)
    return fc_drop

def create_conv_layer(mu, sigma, input_layer, conv_shape, pool_size=None):
    conv_W = tf.Variable(tf.truncated_normal(shape=(5, 5, 1, 6), mean=mu, stddev=sigma))
    conv_b = tf.Variable(tf.zeros(6))
    conv = tf.nn.conv2d(input_layer, conv_W, strides=[1, 1, 1, 1], padding='VALID') + conv_b

    # SOLUTION: Activation.
    conv = tf.nn.relu(conv)

    # SOLUTION: Pooling. Input = 28x28x6. Output = 14x14x6.
    if pool_size != None:
        conv = tf.nn.max_pool(conv, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='VALID')
    return conv


def LeNet(x, keep_prob):
    # Hyperparameters
    mu = 0.0
    sigma = 0.1

    # SOLUTION: Layer 1: Convolutional. Input = 32x32x1. Output = 28x28x6.
    conv1_W = tf.Variable(tf.truncated_normal(shape=(5, 5, 1, 6), mean=mu, stddev=sigma))
    conv1_b = tf.Variable(tf.zeros(6))
    conv1 = tf.nn.conv2d(x, conv1_W, strides=[1, 1, 1, 1], padding='VALID') + conv1_b

    # SOLUTION: Activation.
    conv1 = tf.nn.relu(conv1)

    # SOLUTION: Pooling. Input = 28x28x6. Output = 14x14x6.
    conv1 = tf.nn.max_pool(conv1, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='VALID')

    # SOLUTION: Layer 2: Convolutional. Output = 10x10x16.
    conv2_W = tf.Variable(tf.truncated_normal(shape=(5, 5, 6, 16), mean = mu, stddev = sigma))
    conv2_b = tf.Variable(tf.zeros(16))
    conv2 = tf.nn.conv2d(conv1, conv2_W, strides=[1, 1, 1, 1], padding='VALID') + conv2_b

    # SOLUTION: Activation.
    conv2 = tf.nn.relu(conv2)

    # SOLUTION: Pooling. Input = 10x10x16. Output = 5x5x16.
    conv2 = tf.nn.max_pool(conv2, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='VALID')

    # SOLUTION: Flatten. Input = 5x5x16. Output = 400.
    fc0   = flatten(conv2)

    fc0_drop = tf.nn.dropout(fc0, keep_prob)

    # SOLUTION: Layer 3: Fully Connected. Input = 400. Output = 120.
    fc1 = create_fc_layer_connect(mu, sigma, fc0_drop, keep_prob, 400, 100)
    fc2 = create_fc_layer_connect(mu, sigma, fc1, keep_prob, 100, 100)
    fc3 = create_fc_layer_connect(mu, sigma, fc2, keep_prob, 100, 84)
    fc4 = create_fc_layer_connect(mu, sigma, fc3, keep_prob, 84, 43)

    """
    fc1_W = tf.Variable(tf.truncated_normal(shape=(400, 120), mean=mu, stddev=sigma))
    fc1_b = tf.Variable(tf.zeros(120))
    fc1 = tf.matmul(fc0_drop, fc1_W) + fc1_b

    # SOLUTION: Activation.
    fc1 = tf.nn.relu(fc1)

    fc1_drop = tf.nn.dropout(fc1, keep_prob)

    # SOLUTION: Layer 4: Fully Connected. Input = 120. Output = 84.
    fc2_W = tf.Variable(tf.truncated_normal(shape=(120, 84), mean=mu, stddev=sigma))
    fc2_b = tf.Variable(tf.zeros(84))
    fc2 = tf.matmul(fc1_drop, fc2_W) + fc2_b

    # SOLUTION: Activation.
    fc2 = tf.nn.relu(fc2)

    # SOLUTION: Layer 5: Fully Connected. Input = 84. Output = 43.
    fc3_W = tf.Variable(tf.truncated_normal(shape=(84, 43), mean=mu, stddev=sigma))
    fc3_b = tf.Variable(tf.zeros(43))

    return tf.matmul(fc2, fc3_W) + fc3_b
    """
    return fc4

x = tf.placeholder(tf.float32, (None, 32, 32, 1))
y = tf.placeholder(tf.int32, (None))
keep_prob = tf.placeholder(tf.float32)

one_hot_y = tf.one_hot(y, 43)

#Training pipeline
rate = 0.001

logits = LeNet(x, keep_prob)
cross_entropy = tf.nn.softmax_cross_entropy_with_logits(logits, one_hot_y)
loss_operation = tf.reduce_mean(cross_entropy)
optimizer = tf.train.AdamOptimizer(learning_rate=rate)
training_operation = optimizer.minimize(loss_operation)

#Model evaluation
correct_prediction = tf.equal(tf.argmax(logits, 1), tf.argmax(one_hot_y, 1))
accuracy_operation = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
saver = tf.train.Saver()

def evaluate(X_data, y_data):
    num_examples = len(X_data)
    total_accuracy = 0
    sess = tf.get_default_session()
    for offset in range(0, num_examples, BATCH_SIZE):
        batch_x, batch_y = X_data[offset:offset+BATCH_SIZE], y_data[offset: offset+BATCH_SIZE]
        accuracy = sess.run(accuracy_operation, feed_dict={x: batch_x, y: batch_y, keep_prob: 1.0})
        total_accuracy = total_accuracy + (accuracy*len(batch_x))
    return total_accuracy / num_examples

#Train the model

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    num_examples = len(X_train)
    print("Training")
    print()
    for i in range(EPOCHS):
        X_train, y_train = shuffle(X_train, y_train)
        for offset in range(0, num_examples, BATCH_SIZE):
            end = offset + BATCH_SIZE
            batch_x, batch_y = X_train[offset:end], y_train[offset:end]
            sess.run(training_operation, feed_dict={x: batch_x, y: batch_y, keep_prob: 0.7})
        validation_accuracy = evaluate(X_validation, y_validation)
        print("EPOCH {} ...".format(i+1))
        print("Validation accuracy = {:.3f}".format(validation_accuracy))
        print()
    saver.save(sess, 'lenet')
    print("Model saved")
