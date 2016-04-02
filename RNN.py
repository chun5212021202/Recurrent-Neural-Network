import theano
import theano.tensor as T
import numpy as np
from itertools import izip






#########################################################
#														#
#						RNN STRUCTURE					#
#														#
#########################################################

class RNN(object):
	def __init__(self, x_dimension, y_dimension, depth, width, learning_rate, batch_size, grad_bound, momentum, decay, alpha):

		self.depth = depth
		self.learning_rate = theano.shared(np.array(learning_rate, dtype = theano.config.floatX))
		self.MOMENTUM = momentum
		self.DECAY = decay

		self.RMSProp_init = True
		self.ALPHA = alpha
		self.grad_bound = grad_bound

		y_hat_seq = T.tensor3()
		x_seq = T.tensor3()								#	(batch_size, frame_number, x_dim)
		x_seq_T = x_seq.swapaxes(0,2).swapaxes(0,1)		#	(frame_number, x_dim, batch_size)




 ######### PARAMETERS #########
 		#	hidden layer
		self.w = []	
		self.b = []


		a_init = theano.shared(np.array( np.zeros((width, batch_size)), dtype = theano.config.floatX))
		y_init = theano.shared(np.array( np.zeros((y_dimension, batch_size)), dtype = theano.config.floatX))

		for i in range(depth):	# i layer

			if i == 0 :
				self.w.append(theano.shared(np.array(np.random.uniform(low = -0.1, high = 0.1, size =(width, x_dimension)),dtype = theano.config.floatX)))
				self.b.append(theano.shared(np.array(np.random.uniform(low = -0.1, high = 0.1, size =(width)),dtype = theano.config.floatX)))
				
			else :
				self.w.append(theano.shared(np.array(np.random.uniform(low = -0.1, high = 0.1, size =(width, width)),dtype = theano.config.floatX)))
				self.b.append(theano.shared(np.array(np.random.uniform(low = -0.1, high = 0.1, size =(width)),dtype = theano.config.floatX)))


		#	recurrent
		self.w_h = theano.shared(np.array(np.random.uniform(low = -0.1, high = 0.1, size =(width, width)),dtype = theano.config.floatX))
		self.b_h = theano.shared(np.array(np.random.uniform(low = -0.1, high = 0.1, size =(width)),dtype = theano.config.floatX))


		#	output layer
		self.w_output = theano.shared(np.array(np.random.uniform(low = -0.1, high = 0.1, size =(y_dimension, width)),dtype = theano.config.floatX))	# w for each neuron in last layer (y_dimension neurons per layer, each neurons 'width' w)
		self.b_output = theano.shared(np.array(np.random.uniform(low = -0.1, high = 0.1, size =(y_dimension)),dtype = theano.config.floatX))	# b for each neuron in last layer ('y_dimension' neurons per layer)


 ######### OUTPUT #########

		def step(x_t, a_tm1, y_tm1) :
			a = []
			z = []
			print x_t,a_tm1[0]
			for i in range(self.depth):
				if i == 0 :
					z.append(T.dot(self.w[i], x_t) + T.dot(self.w_h, a_tm1) + self.b[i].dimshuffle(0,'x') + self.b_h.dimshuffle(0,'x'))

				else :
					z.append(T.dot(self.w[i], a[i-1]) + self.b[i].dimshuffle(0,'x'))

				a.append(self.activation(z[i]))

			z_output = T.dot(self.w_output, a[-1]) + self.b_output.dimshuffle(0,'x')
			y = self.softmax(z_output)

			return a[-1], y



		[a_seq, y_seq], _ = theano.scan(	#	y_seq (frame_number, y_dim, batch_size)
			step,							#	a_seq (frame_number, width, batch_size)
			sequences = x_seq_T,			#	x_seq_T(frame_number, x_dim, batch_size)
			outputs_info = [a_init, y_init]

			)
		y_seq_T = ( y_seq.swapaxes(0,1).swapaxes(0,2) )


 ######### CALCULATE ERROR #########

		#	cost function
		cost = T.sum( (y_hat_seq * -T.log(y_seq_T)) ) / batch_size

		#	gradients
		dw = T.grad(cost, self.w)
		db = T.grad(cost, self.b)
		dw_output = T.grad(cost, self.w_output)
		db_output = T.grad(cost, self.b_output)
		dw_h = T.grad(cost, self.w_h)
		db_h = T.grad(cost, self.b_h)



 ######### UPDATE #########

		#	update list
		parameters = []
		gradients = []
		for i in range(depth):
			parameters.append(self.w[i])
			parameters.append(self.b[i])
			gradients.append(T.clip(dw[i], -grad_bound, grad_bound))
			gradients.append(T.clip(db[i], -grad_bound, grad_bound))
		parameters.append(self.w_output)
		parameters.append(self.b_output)
		parameters.append(self.w_h)
		parameters.append(self.b_h)
		gradients.append(T.clip(dw_output, -grad_bound, grad_bound))
		gradients.append(T.clip(db_output, -grad_bound, grad_bound))
		gradients.append(T.clip(dw_h, -grad_bound, grad_bound))
		gradients.append(T.clip(db_h, -grad_bound, grad_bound))

		#	movement initialize
		movement = []
		sigma = []
		for p in parameters :
			movement.append( theano.shared( np.asarray( np.zeros(p.get_value().shape), dtype =theano.config.floatX )))
			sigma.append( theano.shared( np.asarray( np.zeros(p.get_value().shape), dtype =theano.config.floatX )))


		#	update function
		self.update_parameter = theano.function(	#	y_seq_T (batch_size, frame_number, y_dim)
			inputs = [x_seq, y_hat_seq],
			updates = self.MyUpdate_RMSProp(parameters, gradients, movement, sigma),
			outputs = [cost],
			allow_input_downcast = True
			)

 ######### PREDICTION #########

		output = T.argmax(y_seq_T, axis=2)	#	y_seq (frame_number, y_dim, batch_size)
		self.predict = theano.function(
			inputs = [x_seq],
			outputs = output,
			allow_input_downcast = True
			)
 		



 ######### HELPER #########

	def activation(self, z) :	
		return 1 / (1 + T.exp(-z))    # sigmoid


	def softmax(self, z_output) :
		self.total = T.sum(T.exp(z_output), axis = 0)
		return T.exp(z_output) / self.total


	def MyUpdate_Momentum(self, para, grad, move, sigma) :
		update = [(self.learning_rate, self.learning_rate*self.DECAY)]
		update += [( i, self.MOMENTUM*i - self.learning_rate*j ) for i,j in izip(move, grad)]
		update += [( i, i + self.MOMENTUM*k - self.learning_rate*j ) for i,j,k in izip(para, grad, move)]

		return update

	def MyUpdate_RMSProp(self, parameter, gradient, move, sigma) :
		update = []
		for p,g in izip(parameter, gradient) :
			acc = theano.shared(p.get_value() * 0.)
			acc_new = self.ALPHA * acc + (1 - self.ALPHA) * g ** 2

			scale = T.sqrt(acc_new + 1e-6)
			g = g / scale

			update += [(acc, acc_new)]
			update += [(p, p - self.learning_rate*g)]
		return update
			


 ######### Utility #########

	def train(self, training_seq_x, training_seq_y) :	#train one data in a epoch


		return self.update_parameter(training_seq_x, training_seq_y)



	def test(self, testing_data) :


		return self.predict(testing_data)

