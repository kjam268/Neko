import tflearn
import tensorflow as tf
from scipy import ndimage


def Znet(input_size = (80, 576, 576, 2), feature_map=8, kernel_size=3, keep_rate=0.8, lr=0.001, log_dir = "logs"):

	# 2d convolution operation
	def tflearn_conv_2d(net, nb_filter, kernel, stride,dropout=1.0,activation=True):

		net = tflearn.layers.conv.conv_2d(net, nb_filter, kernel, stride, padding="same", activation="linear",bias=False)
		net = tflearn.layers.normalization.batch_normalization(net)
		
		if activation:
			net = tflearn.activations.prelu(net)
		
		net = tflearn.layers.core.dropout(net,keep_prob=dropout)
		
		return(net)

	# 2d deconvolution operation
	def tflearn_deconv_2d(net, nb_filter, kernel, stride, dropout=1.0):

		net = tflearn.layers.conv.conv_2d_transpose(net, nb_filter, kernel,
													[net.shape[1].value*stride, net.shape[2].value*stride, nb_filter],
													[1, stride, stride,1],padding="same",activation="linear",bias=False)
		net = tflearn.layers.normalization.batch_normalization(net)
		net = tflearn.activations.prelu(net)
		net = tflearn.layers.core.dropout(net, keep_prob=dropout)
		
		return(net)

	# merging operation
	def tflearn_merge_2d(layers, method):
		
		net = tflearn.layers.merge_ops.merge(layers, method, axis=3)
		
		return(net)


	# level 0 input
	layer_0a_input	= tflearn.layers.core.input_data(input_size) #shape=[None,n1,n2,n3,1])

	# level 1 down
	layer_1a_conv 	= tflearn_conv_2d(net=layer_0a_input, nb_filter=feature_map, kernel=5, stride=1,activation=False)
	layer_1a_stack	= tflearn_merge_2d([layer_0a_input]*feature_map, "concat")
	layer_1a_stack 	= tflearn.activations.prelu(layer_1a_stack)

	layer_1a_add	= tflearn_merge_2d([layer_1a_conv,layer_1a_stack], "elemwise_sum")
	layer_1a_down	= tflearn_conv_2d(net=layer_1a_add, nb_filter=feature_map*2, kernel=2, stride=2)

	# level 2 down
	layer_2a_conv 	= tflearn_conv_2d(net=layer_1a_down, nb_filter=feature_map*2, kernel=kernel_size, stride=1)
	layer_2a_conv 	= tflearn_conv_2d(net=layer_2a_conv, nb_filter=feature_map*2, kernel=kernel_size, stride=1)

	layer_2a_add	= tflearn_merge_2d([layer_1a_down,layer_2a_conv], "elemwise_sum")
	layer_2a_down	= tflearn_conv_2d(net=layer_2a_add, nb_filter=feature_map*4, kernel=2, stride=2)

	# level 3 down
	layer_3a_conv 	= tflearn_conv_2d(net=layer_2a_down, nb_filter=feature_map*4, kernel=kernel_size, stride=1)
	layer_3a_conv 	= tflearn_conv_2d(net=layer_3a_conv, nb_filter=feature_map*4, kernel=kernel_size, stride=1)
	layer_3a_conv 	= tflearn_conv_2d(net=layer_3a_conv, nb_filter=feature_map*4, kernel=kernel_size, stride=1)

	layer_3a_add	= tflearn_merge_2d([layer_2a_down,layer_3a_conv], "elemwise_sum")
	layer_3a_down	= tflearn_conv_2d(net=layer_3a_add, nb_filter=feature_map*8, kernel=2, stride=2, dropout=keep_rate)

	# level 4 down
	layer_4a_conv 	= tflearn_conv_2d(net=layer_3a_down, nb_filter=feature_map*8, kernel=kernel_size, stride=1, dropout=keep_rate)
	layer_4a_conv 	= tflearn_conv_2d(net=layer_4a_conv, nb_filter=feature_map*8, kernel=kernel_size, stride=1, dropout=keep_rate)
	layer_4a_conv 	= tflearn_conv_2d(net=layer_4a_conv, nb_filter=feature_map*8, kernel=kernel_size, stride=1, dropout=keep_rate)

	layer_4a_add	= tflearn_merge_2d([layer_3a_down,layer_4a_conv], "elemwise_sum")
	layer_4a_down	= tflearn_conv_2d(net=layer_4a_add, nb_filter=feature_map*16,kernel=2,stride=2,dropout=keep_rate)

	# level 5
	layer_5a_conv 	= tflearn_conv_2d(net=layer_4a_down, nb_filter=feature_map*16, kernel=kernel_size, stride=1, dropout=keep_rate)
	layer_5a_conv 	= tflearn_conv_2d(net=layer_5a_conv, nb_filter=feature_map*16, kernel=kernel_size, stride=1, dropout=keep_rate)
	layer_5a_conv 	= tflearn_conv_2d(net=layer_5a_conv, nb_filter=feature_map*16, kernel=kernel_size, stride=1, dropout=keep_rate)

	layer_5a_add	= tflearn_merge_2d([layer_4a_down,layer_5a_conv], "elemwise_sum")
	layer_5a_up		= tflearn_deconv_2d(net=layer_5a_add, nb_filter=feature_map*8, kernel=2, stride=2, dropout=keep_rate)

	# level 4 up
	layer_4b_concat	= tflearn_merge_2d([layer_4a_add,layer_5a_up], "concat")
	layer_4b_conv 	= tflearn_conv_2d(net=layer_4b_concat, nb_filter=feature_map*16, kernel=kernel_size, stride=1, dropout=keep_rate)
	layer_4b_conv 	= tflearn_conv_2d(net=layer_4b_conv, nb_filter=feature_map*16, kernel=kernel_size, stride=1, dropout=keep_rate)
	layer_4b_conv 	= tflearn_conv_2d(net=layer_4b_conv, nb_filter=feature_map*16, kernel=kernel_size, stride=1, dropout=keep_rate)

	layer_4b_add	= tflearn_merge_2d([layer_4b_conv,layer_4b_concat], "elemwise_sum")
	layer_4b_up		= tflearn_deconv_2d(net=layer_4b_add, nb_filter=feature_map*4, kernel=2, stride=2, dropout=keep_rate)

	# level 3 up
	layer_3b_concat	= tflearn_merge_2d([layer_3a_add,layer_4b_up], "concat")
	layer_3b_conv 	= tflearn_conv_2d(net=layer_3b_concat, nb_filter=feature_map*8, kernel=kernel_size, stride=1, dropout=keep_rate)
	layer_3b_conv 	= tflearn_conv_2d(net=layer_3b_conv, nb_filter=feature_map*8, kernel=kernel_size, stride=1, dropout=keep_rate)
	layer_3b_conv 	= tflearn_conv_2d(net=layer_3b_conv, nb_filter=feature_map*8, kernel=kernel_size, stride=1, dropout=keep_rate)

	layer_3b_add	= tflearn_merge_2d([layer_3b_conv,layer_3b_concat], "elemwise_sum")
	layer_3b_up		= tflearn_deconv_2d(net=layer_3b_add, nb_filter=feature_map*2, kernel=2, stride=2)

	# level 2 up
	layer_2b_concat	= tflearn_merge_2d([layer_2a_add,layer_3b_up], "concat")
	layer_2b_conv 	= tflearn_conv_2d(net=layer_2b_concat, nb_filter=feature_map*4, kernel=kernel_size, stride=1)
	layer_2b_conv 	= tflearn_conv_2d(net=layer_2b_conv, nb_filter=feature_map*4, kernel=kernel_size, stride=1)

	layer_2b_add	= tflearn_merge_2d([layer_2b_conv,layer_2b_concat], "elemwise_sum")
	layer_2b_up		= tflearn_deconv_2d(net=layer_2b_add, nb_filter=feature_map, kernel=2, stride=2)

	# level 1 up
	layer_1b_concat	= tflearn_merge_2d([layer_1a_add,layer_2b_up], "concat")
	layer_1b_conv 	= tflearn_conv_2d(net=layer_1b_concat, nb_filter=feature_map*2, kernel=kernel_size, stride=1)

	layer_1b_add	= tflearn_merge_2d([layer_1b_conv,layer_1b_concat], "elemwise_sum")

	# level 0 classifier
	layer_0b_conv	= tflearn_conv_2d(net=layer_1b_add, nb_filter=2, kernel=5, stride=1)

	#layer_0b_clf	= tflearn.layers.conv.conv_2d(layer_0b_conv, 2, 1, 1, activation="softmax")
	layer_0b_clf 	= tflearn.layers.core.fully_connected(layer_0b_conv, 2, activation='linear')

	# Optimizer
	regress = tflearn.layers.estimator.regression(layer_0b_clf, optimizer='adam', loss="mean_square", learning_rate=lr) # categorical_crossentropy/dice_loss_3d
	model   = tflearn.models.dnn.DNN(regress,tensorboard_dir=log_dir)

	#model.save("Weights/Neko_centroid")

	return model


def dice_loss_2d(y_pred, y_true):
	
	with tf.name_scope("dice_loss_2D_function"):
		
		y_pred = y_pred[:,:,:,1]
		y_true = y_true[:,:,:,1]

		smooth = 1.0
		
		intersection = tf.reduce_sum(y_pred * y_true)
		union = tf.reduce_sum(y_pred * y_pred) + tf.reduce_sum(y_true * y_true)
		
		dice = (2.0 * intersection + smooth) / (union + smooth)
		
	return(1 - dice)

