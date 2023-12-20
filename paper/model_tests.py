import tensorflow.compat.v1 as tf

# Define the path to the saved model files
model_dir = './py1k_4epoch_backup'
meta_graph_file = 'model.meta'
checkpoint_file = 'model'

# Import the graph structure from the meta graph file
graph = tf.Graph()
with graph.as_default():
    saver = tf.train.import_meta_graph(model_dir + '/' + meta_graph_file)

# Restore the variable values from the checkpoint file
with tf.Session(graph=graph) as sess:
    saver.restore(sess, model_dir + '/' + checkpoint_file)

    # get all the operations in the graph
    # operations = graph.get_operations()
    # with open('operations.txt', 'w') as f:
    #     f.write(str(operations))
    # print(operations, sep='\n')

    restored_tensor = graph.get_tensor_by_name('Model/RNN/wm:0') # <- get W_m matrix (tensor) from https://arxiv.org/pdf/1711.09573.pdf, p. 3
    print(restored_tensor)

    # get tensor contents
    print(sess.run(restored_tensor))

