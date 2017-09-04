# Load pickled data
import pickle
import csv
# TODO: Fill this in based on where you saved the training and testing data


text_labels = []
with open('signnames.csv', 'r') as csvfile:
    reader = csv.reader(csvfile, delimiter=',')
    n = 0;
    for row in reader:
        if n > 0: #skip first row
            text_labels += [row[1]]
        n += 1

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
print(y_test)

(X_train1, y_train1) = pickle.load(open('augmented.p', "rb"))
print("aug set: ", len(X_train1));
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
from skimage.transform import warp, SimilarityTransform
from skimage import img_as_ubyte
from skimage.exposure import adjust_gamma
import math
import numpy as np
from numpy import random
import warnings

def jitter(img):
    ''' Jitter the image as described in the paper referenced here:
        http://yann.lecun.com/exdb/publis/pdf/sermanet-ijcnn-11.pdf'''
    
    # knobs
    max_rot = 15 * math.pi/180.0
    max_scale = 0.1
    max_delta = 2
    max_gamma = 0.7

    # randomize
    rot = random.uniform(-1,1) * max_rot
    scl = random.uniform(-1,1) * max_scale + 1.0
    xd = random.randint(-max_delta, max_delta)
    yd = random.randint(-max_delta, max_delta)
    gamma = random.uniform(-1,1) * max_gamma + 1.0

    # scale, roation, and translation
    tform = SimilarityTransform(rotation=rot, scale=scl, translation=(xd, yd))
    offx, offy = np.array(img.shape[:2]) / 2
    recenter = SimilarityTransform(translation=(offx, offy))
    recenter_inv = SimilarityTransform(translation=(-offx, -offy))
    img = warp(img, (recenter_inv + (tform + recenter)).inverse, mode='edge')
    
    # gamma
    img = adjust_gamma(img, gamma)
    
    # convert back to RGB [0-255] and ignore the silly precision warning
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return img_as_ubyte(img)

def jitter_data(data):
    result = np.empty_like(data)
    for i in range(data.shape[0]):
        result[i] = jitter(data[i])
    return result

print("Adding synthetic data.")
import os.path
if (os.path.isfile('traffic-signs-data/synthetic.p') == False):
    print("Generating Synthetic Data.", end='', flush=True)
    for i in range(0,5,1):
        if (i == 0):
            X_synth = jitter_data(X_train)
            y_synth = y_train
        else:            
            X_synth = np.concatenate((X_synth, jitter_data(X_train)))
            y_synth = np.concatenate((y_synth, y_train))
        print(".", end='', flush=True)
    n_synth = len(X_synth)
    print("done. ", n_train + n_synth, " samples.");
    pickle.dump((X_synth, y_synth), open('traffic-signs-data/synthetic.p', "wb"))
else:
    print("Loading Synthetic data.")
    (X_synth, y_synth) = pickle.load(open('traffic-signs-data/synthetic.p', "rb"))
print(np.shape(X_synth[0]))

# Display an example
i = random.randint(0,len(X_train))
plt.subplot(1,2,1)
plt.title("Original")
plt.imshow(X_train[i])
plt.subplot(1,2,2)
plt.title("Synthesized")
plt.imshow(X_synth[i])
plt.show()

X_train = np.concatenate((X_train, X_synth))
y_train = np.concatenate((y_train, y_synth))
n_train = len(X_train)
print("Total training samples = ", n_train);

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
        img = cv2.cvtColor(n, cv2.COLOR_RGB2GRAY)
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

#import sys
#sys.exit(0)

### Define your architecture here.
### Feel free to use as many code cells as needed.
import tensorflow as tf

#X_train = tf.image.convert_image_dtype(X_train, tf.float32)
#X_train = tf.image.rgb_to_hsv(X_train)

from tensorflow.contrib.layers import flatten

conv_1 = None
conv_2 = None
conv_3 = None

def LeNet(x, depth, keep_prob=1.0):
    global conv_1, conv_2, conv_3
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
k_pred = tf.placeholder(tf.int32)

#Training Pipeline
EPOCHS = 20
BATCH_SIZE = 128
rate = 0.001
decay = 1.0
drop = 0.5

logits = LeNet(x, n_depth)
cross_entropy = tf.nn.softmax_cross_entropy_with_logits(labels=one_hot_y, logits=logits)
loss_operation = tf.reduce_mean(cross_entropy)
optimizer = tf.train.AdamOptimizer(learning_rate = rate)
#optimizer = tf.train.GradientDescentOptimizer(learning_rate = learning_rate)

#Model Evaluation
training_operation = optimizer.minimize(loss_operation)
predict_operation = tf.argmax(logits, 1)
softmax_operation = tf.nn.softmax(logits)
topk_operation = tf.nn.top_k(softmax_operation, k_pred)
correct_prediction = tf.equal(predict_operation, tf.argmax(one_hot_y, 1))
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
            sess.run(training_operation, feed_dict={x: batch_x, y: batch_y, learning_rate: rate, keep_prob: drop})

        training_loss, training_accuracy = evaluate(X_train, y_train)
        train_loss.append(training_loss)
        train_acc.append(training_accuracy*100.0)
        validation_loss, validation_accuracy = evaluate(X_valid, y_valid)
        valid_loss.append(validation_loss)
        valid_acc.append(validation_accuracy*100.0)
        print("EPOCH {} ...".format(i+1))        
        #print("Training Accuracy = {:.3f}".format(training_accuracy))
        #print("Training Loss = {:.3f}".format(training_loss))        
        print("Validation Accuracy = {:.3f}".format(validation_accuracy))
        #print("Validation Loss = {:.3f}".format(validation_loss))
        #print("Learning Rate = {:.6f}".format(rate))
        #print()
        rate = rate * decay

    saver.save(sess, './lenet')
    print("Model saved")

#Evaluate Model
with tf.Session() as sess:
    saver.restore(sess, tf.train.latest_checkpoint('.'))

    test_loss, test_accuracy = evaluate(X_test, y_test)
    print("Test Accuracy = {:.3f}".format(test_accuracy))

#Plot Training Stats
if 0:
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


### Load the images and plot them here.
### Feel free to use as many code cells as needed.

X_wild = []
y_wild = []
for i in range(1,6,1):
    img = cv2.imread('traffic-signs-data/wild_{}.png'.format(i), cv2.IMREAD_COLOR)
    RGB_img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    X_wild.append(RGB_img)
with open('traffic-signs-data/wild.csv', 'r') as csvfile:
    for row in csv.reader(csvfile, delimiter=','):
        y_wild.append(int(row[0]))

n_wild = len(y_wild)
if 0:
    plt.figure(figsize=(5, 4))
    for i in range(0,n_wild):
        plt.subplot(1, n_wild, i+1)
        plt.imshow(X_wild[i])
        plt.title(text_labels[y_wild[i]])
        plt.axis('off')
    plt.suptitle('Wild Traffic Signs')
    plt.show()


### Run the predictions here and use the model to output the prediction for each image.
### Make sure to pre-process the images with the same pre-processing pipeline used earlier.
### Feel free to use as many code cells as needed.

X_wild = preprocess_data(X_wild)

#Evaluate Model
def evaluate_topk(X_data, k):
    sess = tf.get_default_session()
    X_data = np.expand_dims(X_data, axis=0)
    topk = sess.run(topk_operation, feed_dict={x: X_data, keep_prob: 1.0, k_pred: k})
    return topk

def evaluate_prediction(X, y):
    sess = tf.get_default_session()
    X = np.expand_dims(X, axis=0)
    p = sess.run(predict_operation, feed_dict={x: X, keep_prob: 1.0})
    c = (p == y)
    return p,c

if 1:
    with tf.Session() as sess:
        saver.restore(sess, tf.train.latest_checkpoint('.'))
        for i in range(0,5,1):
            p,c = evaluate_prediction(X_wild[i], y_wild[i])
            print(p,c)


### Calculate the accuracy for these 5 new images. 
### For example, if the model predicted 1 out of 5 signs correctly, it's 20% accurate on these new images.
with tf.Session() as sess:
    saver.restore(sess, tf.train.latest_checkpoint('.'))
    test_loss, test_accuracy = evaluate(X_wild, y_wild)
    print("Wild Test Accuracy = {:.3f}".format(test_accuracy))


### Print out the top five softmax probabilities for the predictions on the German traffic sign images found on the web. 
### Feel free to use as many code cells as needed.
if 0:
    with tf.Session() as sess:
        saver.restore(sess, tf.train.latest_checkpoint('.'))
        for i in range(0,5,1):
            top5 = evaluate_topk(X_wild[i], 5)
            plt.bar(top5.indices[0], top5.values[0])
            plt.xlabel("Class")
            plt.ylabel('Softmax probability')
            plt.show()


### Visualize your network's feature maps here.
### Feel free to use as many code cells as needed.

# image_input: the test image being fed into the network to produce the feature maps
# tf_activation: should be a tf variable name used during your training procedure that represents the calculated state of a specific weight layer
# activation_min/max: can be used to view the activation contrast in more detail, by default matplot sets min and max to the actual min and max values of the output
# plt_num: used to plot out multiple different weight feature map sets on the same block, just extend the plt number for each new feature map entry

def outputFeatureMap(image_input, tf_activation, title, activation_min=-1, activation_max=-1 ,plt_num=1):
    # Here make sure to preprocess your image_input in a way your network expects
    # with size, normalization, ect if needed
    # image_input =
    # Note: x should be the same name as your network's tensorflow data placeholder variable
    # If you get an error tf_activation is not defined it may be having trouble accessing the variable from inside a function
    activation = tf_activation.eval(session=sess,feed_dict={x : image_input})
    featuremaps = activation.shape[3]
    cols = (featuremaps+1) / 4 + 1
    rows = (featuremaps+1) / cols + 1
    plt.figure(plt_num, figsize=(15,rows))
    for featuremap in range(featuremaps):
        plt.subplot(rows, cols, featuremap+1) # sets the number of feature maps to show on each row and column
        #plt.title('FeatureMap ' + str(featuremap)) # displays the feature map number
        if activation_min != -1 & activation_max != -1:
            plt.imshow(activation[0,:,:, featuremap], interpolation="nearest", vmin =activation_min, vmax=activation_max, cmap="gray")
        elif activation_max != -1:
            plt.imshow(activation[0,:,:, featuremap], interpolation="nearest", vmax=activation_max, cmap="gray")
        elif activation_min !=-1:
            plt.imshow(activation[0,:,:, featuremap], interpolation="nearest", vmin=activation_min, cmap="gray")
        else:
            plt.imshow(activation[0,:,:, featuremap], interpolation="nearest", cmap="gray")
    plt.suptitle(title)
    plt.show()

with tf.Session() as sess:
    saver.restore(sess, tf.train.latest_checkpoint('.'))
    img = np.expand_dims(X_wild[0], axis=0)
    outputFeatureMap(img, conv_1, "Layer 1")
    outputFeatureMap(img, conv_2, "Layer 2")
    outputFeatureMap(img, conv_3, "Layer 3")
