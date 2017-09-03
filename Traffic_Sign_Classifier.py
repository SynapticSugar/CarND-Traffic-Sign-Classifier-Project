# Load pickled data
import pickle
import csv
# TODO: Fill this in based on where you saved the training and testing data

text_labels = []
with open('signnames.csv', 'r') as csvfile:
    reader = csv.reader(csvfile, delimiter=',')
    for row in reader:
        text_labels += [row[1]]

training_file = 'traffic-signs-data/train.p'
validation_file='traffic-signs-data/valid.p'
testing_file = 'traffic-signs-data/test.p'

with open(training_file, mode='rb') as f:
    train = pickle.load(f)
with open(validation_file, mode='rb') as f:
    valid = pickle.load(f)
with open(testing_file, mode='rb') as f:
    test = pickle.load(f)
    
X_train, y_train = train['features'], train['labels']
X_valid, y_valid = valid['features'], valid['labels']
X_test, y_test = test['features'], test['labels']

### Replace each question mark with the appropriate value. 
### Use python, pandas or numpy methods rather than hard coding the results

import numpy as np

# TODO: Number of training examples
n_train = len(X_train)

# TODO: Number of validation examples
n_valid = len(X_valid)

# TODO: Number of testing examples.
n_test = len(X_test)

# TODO: What's the shape of an traffic sign image?
image_shape = np.shape(X_train[0])

# TODO: How many unique classes/labels there are in the dataset.
n_classes = len(np.unique(y_train))

print("Number of training examples =", n_train)
print("Number of testing examples =", n_test)
print("Number of validation examples =", n_valid)
print("Image data shape =", image_shape)
print("Number of classes =", n_classes)

### Data exploration visualization code goes here.
### Feel free to use as many code cells as needed.
import matplotlib.pyplot as plt
#%matplotlib inline
if 0:
    # Visualizations will be shown in the notebook.

    labels_valid, sizes_valid = np.unique(y_valid, return_counts=True)
    labels_train, sizes_train = np.unique(y_train, return_counts=True)
    labels_test, sizes_test = np.unique(y_test, return_counts=True)
    ind = labels_valid
    p1 = plt.barh(ind, sizes_train, .35, color='blue')
    p2 = plt.barh(ind, sizes_valid, .35, left = sizes_train, color='red')
    p3 = plt.barh(ind, sizes_test, .35, left = sizes_valid + sizes_train, color='green')
    plt.yticks(ind, text_labels)
    plt.legend((p1[0], p2[0], p3[0]), ('Train', 'Validate', 'Test'))
    plt.title('Distribution of Traffic Signs')
    plt.show()

    import math
    plt.figure(figsize=(15, 30))
    n_unique, unique_index = np.unique(y_train, return_index=True)
    for i in range(0,len(n_unique)):
        plt.subplot(math.ceil(len(n_unique)/4), 4, i+1)
        plt.imshow(X_train[unique_index[i]])
        plt.title(text_labels[y_train[unique_index[i]]])
        plt.axis('off')
    plt.suptitle('Traffic Sign Lables')
    plt.show()

### Preprocess the data here. It is required to normalize the data. Other preprocessing steps could include 
### converting to grayscale, etc.
### Feel free to use as many code cells as needed.


# Add synthesized data
from skimage import transform
print("Generating Synthetic Data", end='', flush=True)
for i in range (0,0,1):
    print(".", end='', flush=True)
    index = random.randint(0, len(X_train))
    img = np.copy(X_train[index])
    rot = (random.random()*.26)
    scal = 1. + (random.random()-0.5) * 0.1
    x = np.random.randint(-3, 3)
    y = np.random.randint(-3, 3)
    tform = transform.SimilarityTransform(rotation=rot, scale=scal, translation=(x, y))
    img = transform.warp(img, tform, preserve_range=True)
    y_train = np.concatenate((y_train, np.reshape(y_train[index],(1,))), axis=0)
    img = np.reshape(img,(1,32,32,3))
    X_train = np.concatenate((X_train, np.array(img)), axis=0)
    n_train += 1
print("done. ", n_train, " samples.");

# Shuffle training set
print("Shuffle training set.")
from sklearn.utils import shuffle
X_train, y_train = shuffle(X_train, y_train)

# Quick Normalize
def normalize(data):
    return data/255 - 0.5

import cv2
def preprocess_data(data):
    i, x, y, d = np.shape(data)
    result = []
    cnt = 0
    for n in data:
        img = cv2.cvtColor(n, cv2.COLOR_BGR2GRAY)
        img = cv2.equalizeHist(img)
        result.append(img)
    result = np.reshape(np.array(result), (i,x,y,1))
    return normalize(result)

if 0:
    print("Normalizing the input")
    X_train = normalize(X_train) 
    X_test = normalize(X_test) 
    X_valid = normalize(X_valid)
    n_depth = 3
if 1:
    print("Preprocessing images.")
    X_train = preprocess_data(X_train) 
    X_test = preprocess_data(X_test) 
    X_valid = preprocess_data(X_valid)
    n_depth = 1

# Convert to YUV and Normalize
if 0:
    # Global Normalization of Y Channel
    from sklearn.preprocessing import normalize
    from skimage import img_as_float, img_as_int, color, exposure
    #Histogram equalization
    #print("Histogram equalization.")
    #X_train = exposure.equalize_hist(X_train)
    #print(np.shape(X_train), X_train[1,1,1])
    print("RGB to HSV.")
    X_train = color.rgb2hsv(X_train)
    X_train = (X_train - .5) * 2.
    print(np.shape(X_train), X_train[1,1,1])
    X_train = np.reshape(X_train,(n_train,32,32,1))
    X_valid = exposure.equalize_hist(X_valid)
    X_valid = color.rgb2hsv(X_valid)
    X_valid = (X_valid - .5) * 2.
    X_valid = np.reshape(X_valid,(n_valid,32,32,1))
    X_test = exposure.equalize_hist(X_test)
    X_test = color.rgb2hsv(X_test)
    X_test = (X_test - .5) * 2.
    X_test = np.reshape(X_test,(n_test,32,32,1))
    n_depth = 1
    #print(X_train[1,1,1])

#import sys
#sys.exit(0)

### Define your architecture here.
### Feel free to use as many code cells as needed.
import tensorflow as tf

#X_train = tf.image.convert_image_dtype(X_train, tf.float32)
#X_train = tf.image.rgb_to_hsv(X_train)

EPOCHS = 2
BATCH_SIZE = 128

from tensorflow.contrib.layers import flatten

def LeNet(x, depth, keep_prob=1.0):    
    # Arguments used for tf.truncated_normal, randomly defines variables for the weights and biases for each layer
    mu = 0
    sigma = 0.1
    
    # TODO: Layer 1: Convolutional. Input = 32x32x3. Output = 28x28x6.
    c1_w = tf.Variable(tf.truncated_normal(shape=(5,5,depth,32), mean = mu, stddev = sigma))
    c1_b = tf.Variable(tf.zeros(32))
    conv_1 = tf.nn.conv2d(x, c1_w, strides=[1,1,1,1], padding='VALID')
    conv_1 = tf.nn.bias_add(conv_1, c1_b)
    
    # TODO: Activation.
    conv_1 = tf.nn.relu(conv_1)

    # TODO: Pooling. Input = 28x28x6. Output = 14x14x6.
    pool_1 = tf.nn.max_pool(conv_1, ksize=[1,2,2,1], strides=[1,2,2,1], padding='VALID')

    # TODO: Layer 2: Convolutional. Output = 10x10x16.
    c2_w = tf.Variable(tf.truncated_normal(shape=(5,5,32,64), mean = mu, stddev = sigma))
    c2_b = tf.Variable(tf.zeros(64))
    conv_2 = tf.nn.conv2d(pool_1, c2_w, strides=[1,1,1,1], padding='VALID')
    conv_2 = tf.nn.bias_add(conv_2, c2_b)
    
    # TODO: Activation.
    conv_2 = tf.nn.relu(conv_2)
    
    # Layer 3: Convolutional. Output = 10x10x16.
    c3_w = tf.Variable(tf.truncated_normal(shape=(5,5,64,128), mean = mu, stddev = sigma))
    c3_b = tf.Variable(tf.zeros(128))
    conv_3 = tf.nn.conv2d(conv_2, c3_w, strides=[1,1,1,1], padding='VALID')
    conv_3 = tf.nn.bias_add(conv_3, c3_b)
    # Activation.
    conv_3 = tf.nn.relu(conv_3)

    # TODO: Pooling. Input = 10x10x16. Output = 5x5x16.
    pool_2 = tf.nn.max_pool(conv_3, ksize=[1,2,2,1], strides=[1,2,2,1], padding='VALID')
    
    # TODO: Flatten. Input = 5x5x16. Output = 400.
    pool_2 = tf.contrib.layers.flatten(pool_2)

    
    # TODO: Layer 4: Fully Connected. Input = 400. Output = 120.
    fc1_w = tf.Variable(tf.truncated_normal(shape=(1152,120), mean = mu, stddev = sigma))
    fc1_b = tf.Variable(tf.zeros(120))
    #fc_1 = tf.add(tf.matmul(pool_2, fc1_w), fc1_b)
    fc_1 = tf.matmul(pool_2, fc1_w) + fc1_b
    
    # TODO: Activation.
    fc_1 = tf.nn.relu(fc_1)

    # Dropout
    fc_1 = tf.nn.dropout(fc_1, keep_prob)

    # TODO: Layer 5: Fully Connected. Input = 120. Output = 84.
    fc2_w = tf.Variable(tf.truncated_normal(shape=(120,84), mean = mu, stddev = sigma))
    fc2_b = tf.Variable(tf.zeros(84))
    fc_2 = tf.add(tf.matmul(fc_1, fc2_w), fc2_b)
    
    # TODO: Activation.
    fc_2 = tf.nn.relu(fc_2)

    # Dropout
    fc_2 = tf.nn.dropout(fc_2, keep_prob)

    # TODO: Layer 6: Fully Connected. Input = 84. Output = 43.
    fc3_w = tf.Variable(tf.truncated_normal(shape=(84,n_classes), mean = mu, stddev = sigma))
    fc3_b = tf.Variable(tf.zeros(n_classes))
    logits = tf.add(tf.matmul(fc_2, fc3_w), fc3_b)    
    
    return logits

### Train your model here.
### Calculate and report the accuracy on the training and validation set.
### Once a final model architecture is selected, 
### the accuracy on the test set should be calculated and reported as well.
### Feel free to use as many code cells as needed.

x = tf.placeholder(tf.float32, (None, 32, 32, n_depth))
y = tf.placeholder(tf.int32, (None))
keep_prob = tf.placeholder(tf.float32)
learning_rate = tf.placeholder(tf.float32)
one_hot_y = tf.one_hot(y, n_classes)

#Training Pipeline
rate = 0.001
decay = 1.0

logits = LeNet(x, n_depth)
cross_entropy = tf.nn.softmax_cross_entropy_with_logits(labels=one_hot_y, logits=logits)
loss_operation = tf.reduce_mean(cross_entropy)
optimizer = tf.train.AdamOptimizer(learning_rate = rate)
#optimizer = tf.train.GradientDescentOptimizer(learning_rate = learning_rate)
training_operation = optimizer.minimize(loss_operation)

#Model Evaluation
correct_prediction = tf.equal(tf.argmax(logits, 1), tf.argmax(one_hot_y, 1))
accuracy_operation = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
saver = tf.train.Saver()

def evaluate(X_data, y_data):
    num_examples = len(X_data)
    total_accuracy = 0
    total_loss = 0
    sess = tf.get_default_session()
    for offset in range(0, num_examples, BATCH_SIZE):
        batch_x, batch_y = X_data[offset:offset+BATCH_SIZE], y_data[offset:offset+BATCH_SIZE]
        loss, accuracy = sess.run([loss_operation, accuracy_operation], feed_dict={x: batch_x, y: batch_y, keep_prob: 1.0})
        total_accuracy += (accuracy * len(batch_x))
        total_loss += (loss * len(batch_x))
    return total_loss / num_examples, total_accuracy / num_examples

train_loss = []
train_acc = []
valid_loss = []
valid_acc = []

#Train Model
with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    num_examples = len(X_train)
    
    print("Training...")
    print()
    for i in range(EPOCHS):
        X_train, y_train = shuffle(X_train, y_train)
        for offset in range(0, num_examples, BATCH_SIZE):
            end = offset + BATCH_SIZE
            batch_x, batch_y = X_train[offset:end], y_train[offset:end]
            sess.run(training_operation, feed_dict={x: batch_x, y: batch_y, learning_rate: rate, keep_prob: 0.7})

        training_loss, training_accuracy = evaluate(X_train, y_train)
        train_loss.append(training_loss)
        train_acc.append(training_accuracy*100.0)
        validation_loss, validation_accuracy = evaluate(X_valid, y_valid)
        valid_loss.append(validation_loss)
        valid_acc.append(validation_accuracy*100.0)
        print("EPOCH {} ...".format(i+1))        
        print("Training Accuracy = {:.3f}".format(training_accuracy))
        print("Training Loss = {:.3f}".format(training_loss))        
        print("Validation Accuracy = {:.3f}".format(validation_accuracy))
        print("Validation Loss = {:.3f}".format(validation_loss))
        print("Learning Rate = {:.6f}".format(rate))
        print()
        rate = rate * decay

    saver.save(sess, './lenet')
    print("Model saved")

#Evaluate Model
with tf.Session() as sess:
    saver.restore(sess, tf.train.latest_checkpoint('.'))

    test_loss, test_accuracy = evaluate(X_test, y_test)
    print("Test Accuracy = {:.3f}".format(test_accuracy))

#Plot Training Stats
fig, ax1 = plt.subplots()
plt.title("Training Results [Test Accuracy = {:.3f}]".format(test_accuracy))
ax1.plot(range(1,EPOCHS+1), train_loss, 'b', label='Training Loss')
ax1.plot(range(1,EPOCHS+1), valid_loss, 'g', label='Validation Loss')
ax1.set_ylabel('Loss')
ax1.set_xlabel('Epoch')
ax2 = ax1.twinx()
ax2.plot(range(1,EPOCHS+1), valid_acc, 'r', label='Validation Accuracy')
ax2.set_ylabel('Accuracy (%)')
fig.tight_layout()
ax2.set_ylim([0,100])
lines, labels = ax1.get_legend_handles_labels()
lines2, labels2 = ax2.get_legend_handles_labels()
ax2.legend(lines + lines2, labels + labels2, bbox_to_anchor=(0., -.2, 1., -.2), mode="expand", borderaxespad=0., ncol=3, loc=3)
plt.show()
