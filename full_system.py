import math
import numpy as np
import cv2
from scipy.ndimage import label
import os



np.set_printoptions(threshold=np.nan)

txtfile = open('output_images.txt', 'w')
dirname = '../output_images'

#os.mkdir(dirname)

letter_count=0
input_file_name = '../test.PNG'
input_image = cv2.imread(input_file_name, 0)
image_copy = cv2.imread(input_file_name, 0)
input_image[input_image<128]=0#Black
input_image[input_image>=128]=255#White

[nrows, ncols]=input_image.shape

letter_sequence = [] #sequence of letters, spaces and lines.

line_started=0;
lines_list=[];

letter_dict = {1:'0',2:'1',3:'2',4:'3',5:'4',6:'5',7:'6',8:'7',9:'8',10:'9',11:'A',12:'B',13:'C',14:'D',15:'E',16:'F',17:'G',18:'H',19:'I',20:'J',21:'K',22:'L',23:'M',24:'N',25:'O',26:'P',27:'Q',28:'R',29:'S',30:'T',31:'U',32:'V',33:'W',34:'X',35:'Y',36:'Z',37:'a',38:'b',39:'c',40:'d',41:'e',42:'f',43:'g',44:'h',45:'i',46:'j',47:'k',48:'l',49:'m',50:'n',51:'o',52:'p',53:'q',54:'r',55:'s',56:'t',57:'u',58:'v',59:'w',60:'x',61:'y',62:'z'}

import os, datetime
import numpy as np
import tensorflow as tf
import sys
import datetime
from tensorflow.contrib.layers.python.layers import batch_norm
from TrainDataLoader import *

#Restore parameters

restore_path = "run2/alexnet_bn-15000"
print(restore_path)
# Dataset Parameters
batch_size = 1
load_size = 224 
fine_size = 224
c = 1
data_mean = np.asarray([1])


for i in range(nrows):
    cont_in_row=0
    for j in range(ncols):
        if line_started==0 and input_image[i][j]==0:
            line_started=1
            line_top=i
            cont_in_row=1
        elif line_started==1 and input_image[i][j]==0:
            cont_in_row=1
    if cont_in_row==0 and line_started==1:
        line_bottom=i-1
        lines_list.append([line_top, line_bottom])
        line_started=0

for line in lines_list:
    print("line")
    letter_sequence.append("line")
    input_image_seg=input_image[line[0]:line[1]+1,:]
    [nrows, ncols]=input_image_seg.shape
    letter_started=0;
    letters_list=[];
    for i in range(ncols):
        cont_in_col=0
        for j in range(nrows):
            if letter_started==0 and input_image_seg[j][i]==0:
                letter_started=1
                letter_left=i
                letter_top=line[0]+j
                letter_bottom=line[0]+j
                cont_in_col=1
            elif letter_started==1 and input_image_seg[j][i]==0:
                letter_top=min(letter_top, line[0]+j)
                letter_bottom=max(letter_bottom, line[0]+j)
                cont_in_col=1
        if cont_in_col==0 and letter_started==1:
            letter_right=i-1
            image_copy= cv2.rectangle(image_copy, (letter_left-2,letter_top-2), (letter_right+2,letter_bottom+2), (0,0,255), 1)
            letters_list.append([letter_left,letter_top,letter_right,letter_bottom])
            letter_started=0
            
    max_width=0
    for i in range(len(letters_list)):
        width_current=letters_list[i][2]-letters_list[i][0]
        max_width=max(width_current,max_width)
        
               
    for i in range(len(letters_list)):
        letter=letters_list[i]
        if i!=0:
            prev_letter=letters_list[i-1]
            space_between=letter[0]-prev_letter[2]
            if space_between>0.4*max_width:
                print("space")
                letter_sequence.append("space")
        width=letter[2]-letter[0]+4
        height=letter[3]-letter[1]+10
        diff=math.floor(abs(width-height)/2)
        letter_image=input_image[letter[1]-5:letter[3]+5,letter[0]-2:letter[2]+2]
        if height>width:
            padding=255*np.ones((height,diff))            
            letter_image=np.concatenate((padding,letter_image,padding),axis=1)
        elif height<width:
            padding=255*np.ones((diff,width))
            letter_image=np.concatenate((padding,letter_image,padding),axis=0)
        if (np.average(input_image[letter[1]:letter[3],letter[0]:letter[2]])<.15*255) and .65< abs(letter[2]-letter[0])/abs(letter[3]-letter[1]) <1.2 :
           print("period")
           letter_sequence.append("period")
        else:   
            print("letter")
            letter_sequence.append("letter")
            letter_count+=1
            name="letter"+str(letter_count)+".jpg"
            savename = os.path.join(dirname, name)
            cv2.imwrite( savename, letter_image);
            txtfile.write(savename + '\n')
            #cv2.imshow('image',letter_image);
            #cv2.waitKey(0)

txtfile.close()

cv2.imwrite( "full_image.jpg", image_copy);
cv2.imshow('image_copy',image_copy);
cv2.waitKey(0)


# Training Parameters
learning_rate = 0.0005
dropout = 0.5 # Dropout, probability to keep units
training_iters = 50000
step_display = 1
step_save = 5000
path_save = './alexnet_bn'
endtime = datetime.datetime.now()+datetime.timedelta(hours=17)
start_from = ''

def batch_norm_layer(x, train_phase, scope_bn):
    return batch_norm(x, decay=0.9, center=True, scale=True,
    updates_collections=None,
    is_training=train_phase,
    reuse=None,
    trainable=True,
    scope=scope_bn)
    
def alexnet(x, keep_dropout, train_phase):
    weights = {
        'wc1': tf.Variable(tf.random_normal([11, 11, 1, 96], stddev=np.sqrt(2./(11*11*3)))),
        'wc2': tf.Variable(tf.random_normal([5, 5, 96, 256], stddev=np.sqrt(2./(5*5*96)))),
        'wc3': tf.Variable(tf.random_normal([3, 3, 256, 384], stddev=np.sqrt(2./(3*3*256)))),
        'wc4': tf.Variable(tf.random_normal([3, 3, 384, 256], stddev=np.sqrt(2./(3*3*384)))),
        'wc5': tf.Variable(tf.random_normal([3, 3, 256, 256], stddev=np.sqrt(2./(3*3*256)))),

        'wf6': tf.Variable(tf.random_normal([7*7*256, 4096], stddev=np.sqrt(2./(7*7*256)))),
        'wf7': tf.Variable(tf.random_normal([4096, 4096], stddev=np.sqrt(2./4096))),
        'wo': tf.Variable(tf.random_normal([4096, 63], stddev=np.sqrt(2./4096)))
    }

    biases = {
        'bo': tf.Variable(tf.ones(63))
    }

    # Conv + ReLU + Pool, 224->55->27
    conv1 = tf.nn.conv2d(x, weights['wc1'], strides=[1, 4, 4, 1], padding='SAME')
    conv1 = batch_norm_layer(conv1, train_phase, 'bn1')
    conv1 = tf.nn.relu(conv1)
    pool1 = tf.nn.max_pool(conv1, ksize=[1, 3, 3, 1], strides=[1, 2, 2, 1], padding='SAME')

    # Conv + ReLU  + Pool, 27-> 13
    conv2 = tf.nn.conv2d(pool1, weights['wc2'], strides=[1, 1, 1, 1], padding='SAME')
    conv2 = batch_norm_layer(conv2, train_phase, 'bn2')
    conv2 = tf.nn.relu(conv2)
    pool2 = tf.nn.max_pool(conv2, ksize=[1, 3, 3, 1], strides=[1, 2, 2, 1], padding='SAME')

    # Conv + ReLU, 13-> 13
    conv3 = tf.nn.conv2d(pool2, weights['wc3'], strides=[1, 1, 1, 1], padding='SAME')
    conv3 = batch_norm_layer(conv3, train_phase, 'bn3')
    conv3 = tf.nn.relu(conv3)

    # Conv + ReLU, 13-> 13
    conv4 = tf.nn.conv2d(conv3, weights['wc4'], strides=[1, 1, 1, 1], padding='SAME')
    conv4 = batch_norm_layer(conv4, train_phase, 'bn4')
    conv4 = tf.nn.relu(conv4)

    # Conv + ReLU + Pool, 13->6
    conv5 = tf.nn.conv2d(conv4, weights['wc5'], strides=[1, 1, 1, 1], padding='SAME')
    conv5 = batch_norm_layer(conv5, train_phase, 'bn5')
    conv5 = tf.nn.relu(conv5)
    pool5 = tf.nn.max_pool(conv5, ksize=[1, 3, 3, 1], strides=[1, 2, 2, 1], padding='SAME')

    # FC + ReLU + Dropout
    fc6 = tf.reshape(pool5, [-1, weights['wf6'].get_shape().as_list()[0]])
    fc6 = tf.matmul(fc6, weights['wf6'])
    fc6 = batch_norm_layer(fc6, train_phase, 'bn6')
    fc6 = tf.nn.relu(fc6)
    fc6 = tf.nn.dropout(fc6, keep_dropout)
    
    # FC + ReLU + Dropout
    fc7 = tf.matmul(fc6, weights['wf7'])
    fc7 = batch_norm_layer(fc7, train_phase, 'bn7')
    fc7 = tf.nn.relu(fc7)
    fc7 = tf.nn.dropout(fc7, keep_dropout)

    # Output FC
    out = tf.add(tf.matmul(fc7, weights['wo']), biases['bo'])
    
    return out

# Construct dataloader

opt_data_run = {
    #'data_h5': 'miniplaces_256_val.h5',
    'data_root': '../output_images',   # MODIFY PATH ACCORDINGLY
    'data_list': './output_images.txt',   # MODIFY PATH ACCORDINGLY
    'load_size': load_size,
    'fine_size': fine_size,
    'data_mean': data_mean,
    'randomize': False
    }

#loader_train = DataLoaderDisk(**opt_data_train)
#loader_val = DataLoaderDisk(**opt_data_val)
loader_test = DataLoaderDisk(**opt_data_run)
#loader_train = DataLoaderH5(**opt_data_train)
#loader_val = DataLoaderH5(**opt_data_val)

# tf Graph input
x = tf.placeholder(tf.float32, [None, fine_size, fine_size])
#y = tf.placeholder(tf.int64, None)
keep_dropout = tf.placeholder(tf.float32)
train_phase = tf.placeholder(tf.bool)

# Construct model
logits = alexnet(tf.reshape(x, [-1, fine_size, fine_size, 1]), keep_dropout, train_phase)

probabilities = tf.nn.softmax(logits)

# Define loss and optimizer
#loss = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(labels=y, logits=logits))

# define summary writer
#writer = tf.train.SummaryWriter('.', graph=tf.get_default_graph())

# Launch the graph
lettercode_sequence = []
with tf.Session() as sess:
    # Initialization
    saver=tf.train.Saver()
    saver.restore(sess,restore_path) 
    # Evaluate on the whole validation set
    print('Test on the whole test set...')
    num_batch = loader_test.size()//batch_size
    acc1_total = 0.
    acc5_total = 0.
    loader_test.reset()
    for i in range(num_batch):
        images_batch, name_batch = loader_test.next_batch(batch_size)    
        output = sess.run(probabilities, feed_dict={x: images_batch, train_phase: False, keep_dropout: 1.})#, dropout: 1.})
        output = output[0, 0:]
        sorted_inds = [i[0] for i in sorted(enumerate(-output), key=lambda x:x[1])]
        print( str(name_batch[:]) + " ", end='')
        for j in range(5):
            index = sorted_inds[j]
            print(str(index) + ' ', end='')
        print("")
        lettercode_sequence.append(sorted_inds[0])
        sys.stdout.flush()

    print('***END***')

    sys.stdout.flush()


outputfile = open("Output_text.txt",'w')
counter = 0

for item in letter_sequence:
    if item == "letter":
        lettercode = lettercode_sequence[counter]
        letter = letter_dict[lettercode]
        outputfile.write(letter)
        counter += 1
    if item == "space":
        outputfile.write(" ")
    if item == "line":
        outputfile.write("\n")
    if item == "period":
        outputfile.write(".")
            
outputfile.close()