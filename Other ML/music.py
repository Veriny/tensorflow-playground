import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
import numpy as np
import glob
import pandas as pd
from tqdm import tqdm
import midi
import midi_manipulation
from tensorflow.python.ops import control_flow_ops

##Music generator with Restricted Boltzmann Machine

lowest_note = midi_manipulation.lowerBound
highest_notes = midi_manipulation.upperBound
note_range = highest_notes - lowest_note

###########
#Helper Functions - Yoinked from Siraj Raval.. well actually i made this with the help of his tutorial
def sample(probs):
    return tf.floor(probs + tf.random_uniform(tf.shape(probs), 0, 1))

                    
                    
def gibbs_sample(k):
    #Runs a k-step gibbs chain to sample from the probability distribution of the RBM defined by W, bh, bv
    def gibbs_step(count, k, xk):
        #Runs a single gibbs step. The visible values are initialized to xk
        hk = sample(tf.sigmoid(tf.matmul(xk, W) + bh)) #Propagate the visible values to sample the hidden values
        xk = sample(tf.sigmoid(tf.matmul(hk, tf.transpose(W)) + bv)) #Propagate the hidden values to sample the visible values
        return count+1, k, xk
    #Run gibbs steps for k iterations
    ct = tf.constant(0) #counter
    [_, _, x_sample] = control_flow_ops.while_loop(lambda count, num_iter, *args: count < num_iter, gibbs_step, [ct, tf.constant(k), x])
    #This is not strictly necessary in this implementation, but if you want to adapt this code to use one of TensorFlow's
    #optimizers, you need this in order to stop tensorflow from propagating gradients back through the gibbs step
    x_sample = tf.stop_gradient(x_sample)
    return x_sample

def get_songs(path):
    files = glob.glob('{}/*.mid*'.format(path))
    songs = []
    for f in tqdm(files):
        try:
            song = np.array(midi_manipulation.midiToNoteStateMatrix(f))
            if np.array(song).shape[0] > 50:
                songs.append(song)
        except Exception as e:
            raise e
    return songs
                    
songs = get_songs('Pop_Music_Midi') #These songs have already been converted from midi to msgpack
print ("{} songs processed".format(len(songs)))
###########

#Timesteps created at a time
num_timesteps = 15
#Visible layers
num_visible = 2
#Hidden layers
num_hidden = 50
#blah blah
num_epochs = 500
learning_rate = tf.constant(0.005, tf.float32)
batch_size = 250

##variables, variables
x = tf.placeholder(tf.float32, [None, num_visible], name  = "x")
W = tf.Variable(tf.random_normal([num_visible, num_hidden], 0.01), name = "W")
bh = tf.Variable(tf.zeros([1, num_hidden]), name = "bh")
bv = tf.Variable(tf.zeros([1, num_visible]), name = "bv")


x_sample = gibbs_sample(1)
#sample of hidden nodes, from original input
h = sample(tf.sigmoid(tf.matmul(x, W) + bh))
#sample of hidden nodes from sample input
h_sample = sample(tf.sigmoid(tf.matmul(x_sample, W) + bh))

#Update values of w, bh, and bv
size_bt = tf.cast(tf.shape(x)[0], tf.float32)
W_adder  = tf.multiply(learning_rate/size_bt, tf.subtract(tf.matmul(tf.transpose(x), h), tf.matmul(tf.transpose(x_sample), h_sample)))
bv_adder = tf.multiply(learning_rate/size_bt, tf.reduce_sum(tf.subtract(x, x_sample), 0, True))
bh_adder = tf.multiply(learning_rate/size_bt, tf.reduce_sum(tf.subtract(h, h_sample), 0, True))

updater = [W.assign_add(W_adder), bh.assign_add(bh_adder), bv.assign_add(bv_adder)]


with tf.Session() as sess:
    init = tf.initialize_all_variables()
    sess.run(init)
    #Running through the desired number of epochs...
    for epoch in tqdm(range(num_epochs)):
        for song in songs:
            song = np.array(song)
            song = song[:int(np.floor(song.shape[0]/num_timesteps)*num_timesteps)]
            song = np.reshape(song, [song.shape[0]/num_timesteps, song.shape[1]*num_timesteps])
            #Train the RBM
            for i in range(1, len(song), batch_size):
                tr_x = song[i:i+batch_size]
                sess.run(updater, feed_dict = {x: tr_x})
    sample = gibbs_sample(1).eval(session = sess, feed_dict = {x: np.zeros((50, n_visible))})
    for i in range(sample.shape(0)):
        if not any(sample[i,:]):
            continue
        #Reshape vector
        S = np.reshape(sample[i,:], (num_timesteps, 2*note_range))
        midi_manipulation.noteStateMatrixToMidi(S, "generated_chord_{}".format(i))

                    
                    

