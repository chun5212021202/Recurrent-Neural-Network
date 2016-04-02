# Recurrent-Neural-Network


1. Check the data-file-path before running

2. Run with command "python mainRNN.py"

3. Parameters setting of the network, such as width or depth, can be configured in mainRNN.py

4. Default parameters are as follows. You are strongly suggested to adjust those values properly, since they aren't quite robust.
        
        RNN_EPOCH = 300

        RNN_WIDTH = 1024

        RNN_DEPTH = 1
        
        RNN_X_DIMENSION = 48
        
        RNN_Y_DIMENSION = 48
        
        RNN_LEARNING_RATE = 0.000001
        
        RNN_BATCH_SIZE = 2
        
        RNN_MOMENTUM = 0.9
        
        RNN_DECAY = 0.99994
        
        RNN_ALPHA = 0.99
        
        RNN_GRAD_BOUND = 0.1
        
        RNN_OUTPUT_FILE = 'result_best.lab'

5. Optimizations should be implemented in RNN.py. You can use different activation function or optimizer, such as NAV, etc.

6. Due to copyright issue, files in "data" folder are mostly not complete dataset files. You should keep this in mind because it may cause some unexpected error. Of course, you can ignore this if you use your own datasets and parse them on your own. In this case, you might need to implement your own UtilRNN.py.

7. As I said, datasets are not complete. If you use those data for RNN training and testing, results might possibly turn out to be overfitted, and thus the testing accuracy would be quite low.
