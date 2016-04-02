
from random import shuffle
import numpy as np
import time


#########################################################
#														#
#						FUNCTIONS						#
#														#
#########################################################

def LoadTrainRNN(training_file_path, state_file_path) :
	#	read file
	print 'File \'train.ark\' Loading.....'
	training_file = open(training_file_path,'r')
	training_data_list = training_file.read().splitlines()


	#	read file
	print 'File \'train.lab\' Loading.....'
	state_file = open(state_file_path, 'r')
	state_file_list = state_file.read().splitlines()


	#	read file
	print 'File \'48_39.map\' Loading.....'
	map_file = open('data/48_39.map','r')
	map_list = map_file.read().splitlines()


	#
	filler = [[0]*48]
	training_data_x = {}
	training_data_y = {}
	training_data_l = {}

	print 'Parsing Files.....ID--2Ddata'	
	training_data = {}
	for line in training_data_list :
		line = line.strip()

		SpeakID_SentenceID = line.split()[0].rsplit('_',1)[0]
		FrameID = int( line.split()[0].rsplit('_',1)[1] )

		if FrameID == 1 :
			training_data[SpeakID_SentenceID] = []

		training_data[SpeakID_SentenceID].append( [float(data) for data in line.split()[1:]] )	#	append a (48 list) into []
		
	key =  list(training_data.keys())



	for keys in key :
		training_data_l[keys] = len(training_data[keys])
	longest = training_data_l[max(training_data_l, key=training_data_l.get)]
	print longest

	
	print 'Parsing Files.....ID--2Ddata(aligned)'
	for keys in key :
		training_data_x[keys] = training_data[keys] + filler*(longest-len(training_data[keys]))
	
	

	#
	print 'Parsing Files.....ID--STRINGphone'
	state_data = {}
	for line in state_file_list :
		line = line.strip()

		SpeakID_SentenceID = line.rsplit(',', 1)[0].rsplit('_',1)[0]
		FrameID = int( line.rsplit(',', 1)[0].rsplit('_',1)[1] )

		if FrameID == 1 :
			state_data[SpeakID_SentenceID] = []

		state_data[SpeakID_SentenceID].append( line.rsplit(',', 1)[1] )

	#
	map_data =  {}
	for idx,line in enumerate(map_list) :
		line = line.strip()
		map_data[line.split()[0]] = idx

	print 'Parsing Files.....ID--MATRIXphone(aligned)'
	for keys in key :
		label_num_list = [ map_data[phone] for phone in state_data[keys] ]
		training_sentence_y = []
		for i in label_num_list :
			training_phone_y = [0]*48
			training_phone_y[i] = 1
			training_sentence_y.append(training_phone_y)
		training_data_y[keys] = training_sentence_y + filler*(longest-len(training_sentence_y))

	return key, training_data_x, training_data_y, training_data_l




def TrainRNN(data, rnn_object ,batch_size, epoch) :	# train one epoch

	start = time.time()

	key = data[0]
	training_data_x = data[1]
	training_data_y = data[2]
	training_data_l = data[3]

	print ' *** START TRAINING *** '
	
	for turns in range(epoch) :
		print 'EPOCH %d ' % (turns)
		count = 0
		shuffle(key)

		training_batch_x=[]
		training_batch_y=[]
		training_batch_l = []
		for SpeakID_SentenceID in key :
			training_batch_x.append( training_data_x[SpeakID_SentenceID] )
			training_batch_y.append( training_data_y[SpeakID_SentenceID] )
			training_batch_l.append( training_data_l[SpeakID_SentenceID] )

			if count % batch_size == batch_size-1 :
				print 'EPOCH %d ; Batch Number : %d' % (turns, count//batch_size)

				temp = rnn_object.train( training_batch_x, training_batch_y )
				print temp
				print
				

				
				training_batch_x = []
				training_batch_y = []
				training_batch_l = []
				
			count = count + 1
	print time.time()-start

