import theano
import numpy as np
import theano.tensor as T
import UtilRNN
import RNN




RNN_EPOCH = 300
RNN_WIDTH = 1024#512
RNN_DEPTH = 1#2
RNN_X_DIMENSION = 48
RNN_Y_DIMENSION = 48
RNN_LEARNING_RATE = 0.000001
RNN_BATCH_SIZE = 2
RNN_MOMENTUM = 0.9
RNN_DECAY = 0.99994
RNN_ALPHA = 0.99
RNN_GRAD_BOUND = 0.1
RNN_OUTPUT_FILE = 'result_best.lab'


rnn = RNN.RNN( RNN_X_DIMENSION, RNN_Y_DIMENSION, RNN_DEPTH, RNN_WIDTH, RNN_LEARNING_RATE, RNN_BATCH_SIZE, RNN_GRAD_BOUND, RNN_MOMENTUM, RNN_DECAY, RNN_ALPHA )
data = UtilRNN.LoadTrainRNN( 'data/train_light.post','data/train.lab' )
UtilRNN.TrainRNN( data ,rnn, RNN_BATCH_SIZE, RNN_EPOCH )



