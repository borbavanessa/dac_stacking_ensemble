# Load libs standard python and custom
import numpy as np
import datetime
import sys

from keras.models import Sequential
from keras.layers import Dense, Dropout
from keras.layers import Conv1D, GlobalAveragePooling1D
from keras.layers import Embedding
from keras.optimizers import adadelta

from network_model.model_class import ModelClass
from utils.experiment_processes import ExperimentProcesses

import utils.definition_network as dn


def generate_model(exp, name_model, kernel_name, set_params, function):
		exp.pp_data.vocabulary_size = 5000
		
		exp.pp_data.embedding_size = 300
		exp.pp_data.max_posts = 1750
		exp.pp_data.max_terms_by_post = 300
		exp.pp_data.binary_classifier = True
		exp.pp_data.format_input_data = dn.InputData.POSTS_ONLY_TEXT
		exp.pp_data.remove_stopwords = False
		exp.pp_data.delete_low_tfid = False
		exp.pp_data.min_df = 0
		exp.pp_data.min_tf = 0
		exp.pp_data.random_posts = False
		exp.pp_data.random_users = False
		exp.pp_data.tokenizing_type = 'WE'
		exp.pp_data.type_prediction_label = dn.TypePredictionLabel.MULTI_LABEL_CATEGORICAL
		
		exp.use_custom_metrics = False
		exp.use_valid_set_for_train = True
		exp.valid_split_from_train_set = 0.0
		exp.imbalanced_classes = False
		
		cnn = ModelClass(1)
		cnn.loss_function = 'binary_crossentropy'
		cnn.optmizer_function = 'adadelta'
		
		filters_by_layer = set_params['filters_by_layer']
		kernels_size = set_params['kernels_size']
		epochs = set_params['epochs']
		batch_sizes = set_params['batch_sizes']
		dropouts = set_params['dropouts']
		
		np.random.seed(dn.SEED)
		
		time_ini_rep = datetime.datetime.now()

		for embedding_type in set_params['embedding_types']:
				for embedding_custom_file in set_params['embedding_custom_files']:
						for use_embedding in set_params['use_embeddings']:
								exp.pp_data.embedding_type = embedding_type
								exp.pp_data.word_embedding_custom_file = embedding_custom_file
								exp.pp_data.use_embedding = use_embedding
								exp.pp_data.load_dataset_type = dn.LoadDataset.TRAIN_DATA_MODEL

								exp.set_period_time_end(time_ini_rep, 'Load data')
								x_train, y_train, x_valid, y_valid, num_words, embedding_matrix = exp.pp_data.load_data()

								cnn.use_embedding_pre_train = exp.pp_data.use_embedding
								cnn.embed_trainable = (
														cnn.use_embedding_pre_train == (dn.UseEmbedding.RAND or dn.UseEmbedding.NON_STATIC))
								
								emb_name = function
								
								if embedding_custom_file != '':
										emb_name = exp.pp_data.word_embedding_custom_file.split('.')[0]

								we_file_name = 'ET_' + str(exp.pp_data.embedding_type.value) + '_UE_' + \
															 str(exp.pp_data.use_embedding.value) + '_EF_' + emb_name + kernel_name

								for filter in filters_by_layer:
										for kernel_size in kernels_size:
												for batch_size in batch_sizes:
														for epoch in epochs:
																for dropout in dropouts:
																		cnn.epochs = epoch
																		cnn.batch_size = batch_size
																		cnn.patience_train = epoch
																		exp.experiment_name = name_model + '_cnn' + '_F' + str(filter) + '_K' + str(kernel_size) +\
																													'_P' + 'None' + '_B' + str(batch_size) + '_E' +\
																													str(epoch) + '_D' + str(dropout) + '_HLN' + str(filter)  + '_' + \
																													we_file_name

																		cnn.model = Sequential()
																		cnn.model.add(Embedding(exp.pp_data.vocabulary_size, exp.pp_data.embedding_size,
																														 trainable=cnn.embed_trainable, name='emb_' + name_model))
																		cnn.model.add(Dropout(dropout, name='dropout_1_' + name_model))
																		cnn.model.add(Conv1D(filters=filter, kernel_size=kernel_size,
																												 kernel_initializer='glorot_uniform',
																												 # kernel_regularizer=regularizers.l2(0.03),
																												 padding='valid', activation='relu',
																												 name='conv_1_' + name_model))
																		cnn.model.add(GlobalAveragePooling1D(name='gloval_avg_pool_1_' + name_model))
																		cnn.model.add(Dense(filter, activation='relu', kernel_initializer='glorot_uniform',
																												name='dense_1_' + name_model))
																		cnn.model.add(Dropout(dropout, name='dropout_2_' + name_model))
																		cnn.model.add(Dense(3, activation='sigmoid', name='dense_2_' + name_model))

																		time_ini_exp = datetime.datetime.now()
																		exp.generate_model_hypeparams(cnn, x_train, y_train, x_valid, y_valid, embedding_matrix)
																		exp.set_period_time_end(time_ini_exp, 'Total experiment')

								del x_train, y_train, x_valid, y_valid, num_words, embedding_matrix
		
		# Test
		np.random.seed(dn.SEED)
		time_ini_rep = datetime.datetime.now()

		for embedding_type in set_params['embedding_types']:
				for embedding_custom_file in set_params['embedding_custom_files']:
						for use_embedding in set_params['use_embeddings']:
								exp.pp_data.embedding_type = embedding_type
								exp.pp_data.word_embedding_custom_file = embedding_custom_file
								exp.pp_data.use_embedding = use_embedding
								exp.pp_data.load_dataset_type = dn.LoadDataset.TEST_DATA_MODEL
								
								exp.set_period_time_end(time_ini_rep, 'Load data')
								x_test, y_test = exp.pp_data.load_data()

								cnn.use_embedding_pre_train = exp.pp_data.use_embedding
								cnn.embed_trainable = (
														cnn.use_embedding_pre_train == (dn.UseEmbedding.RAND or dn.UseEmbedding.NON_STATIC))
								
								emb_name = function
								
								if embedding_custom_file != '':
										emb_name = exp.pp_data.word_embedding_custom_file.split('.')[0]
								
								we_file_name = 'ET_' + str(exp.pp_data.embedding_type.value) + '_UE_' + \
															 str(exp.pp_data.use_embedding.value) + '_EF_' + emb_name + kernel_name
								
								for filter in filters_by_layer:
										for kernel_size in kernels_size:
												for batch_size in batch_sizes:
														for epoch in epochs:
																for dropout in dropouts:
																		cnn.epochs = epoch
																		cnn.batch_size = batch_size
																		cnn.patience_train = epoch
																		exp.experiment_name = name_model + '_cnn' + '_F' + str(filter) + '_K' + str(kernel_size) +\
																													'_P' + 'None' + '_B' + str(batch_size) + '_E' +\
																													str(epoch) + '_D' + str(dropout) + '_HLN' + str(filter)  + '_' + \
																													we_file_name
																		
																		cnn.model = exp.load_model(dn.PATH_PROJECT + exp.experiment_name + '.h5')
																		exp.save_geral_configs('Experiment Specific Configuration: ' + exp.experiment_name)
																		exp.save_summary_model(cnn.model)
																		exp.predict_samples(cnn, x_test, y_test)
		
								del x_test, y_test
								
		del cnn, exp


def generate_model_softmax(exp, name_model, kernel_name, set_params, function):
		exp.pp_data.vocabulary_size = 5000

		exp.pp_data.embedding_size = 300
		exp.pp_data.max_posts = 1750
		exp.pp_data.max_terms_by_post = 300
		exp.pp_data.binary_classifier = True
		exp.pp_data.format_input_data = dn.InputData.POSTS_ONLY_TEXT
		exp.pp_data.remove_stopwords = False
		exp.pp_data.delete_low_tfid = False
		exp.pp_data.min_df = 0
		exp.pp_data.min_tf = 0
		exp.pp_data.random_posts = False
		exp.pp_data.random_users = False
		exp.pp_data.tokenizing_type = 'WE'
		exp.pp_data.type_prediction_label = dn.TypePredictionLabel.SINGLE_LABEL_CATEGORICAL

		exp.use_custom_metrics = False
		exp.use_valid_set_for_train = True
		exp.valid_split_from_train_set = 0.0
		exp.imbalanced_classes = False

		cnn = ModelClass(1)
		cnn.loss_function = 'categorical_crossentropy'
		cnn.optmizer_function = adadelta(lr=set_params["learning_rate"])

		filters_by_layer = set_params['filters_by_layer']
		kernels_size = set_params['kernels_size']
		epochs = set_params['epochs']
		batch_sizes = set_params['batch_sizes']
		dropouts = set_params['dropouts']

		np.random.seed(dn.SEED)

		time_ini_rep = datetime.datetime.now()

		for embedding_type in set_params['embedding_types']:
				for embedding_custom_file in set_params['embedding_custom_files']:
						for use_embedding in set_params['use_embeddings']:
								exp.pp_data.embedding_type = embedding_type
								exp.pp_data.word_embedding_custom_file = embedding_custom_file
								exp.pp_data.use_embedding = use_embedding
								exp.pp_data.load_dataset_type = dn.LoadDataset.TRAIN_DATA_MODEL

								exp.set_period_time_end(time_ini_rep, 'Load data')
								x_train, y_train, x_valid, y_valid, num_words, embedding_matrix = exp.pp_data.load_data()

								cnn.use_embedding_pre_train = exp.pp_data.use_embedding
								cnn.embed_trainable = (
												cnn.use_embedding_pre_train == (dn.UseEmbedding.RAND or dn.UseEmbedding.NON_STATIC))

								emb_name = function

								if embedding_custom_file != '':
										emb_name = exp.pp_data.word_embedding_custom_file.split('.')[0]

								we_file_name = 'ET_' + str(exp.pp_data.embedding_type.value) + '_UE_' + \
															 str(exp.pp_data.use_embedding.value) + '_EF_' + emb_name + kernel_name

								for filter in filters_by_layer:
										for kernel_size in kernels_size:
												for batch_size in batch_sizes:
														for epoch in epochs:
																for dropout in dropouts:
																		cnn.epochs = epoch
																		cnn.batch_size = batch_size
																		cnn.patience_train = epoch
																		exp.experiment_name = name_model + '_cnn' + '_F' + str(filter) + '_K' + str(
																				kernel_size) + \
																													'_P' + 'None' + '_B' + str(batch_size) + '_E' + \
																													str(epoch) + '_D' + str(dropout) + '_HLN' + str(
																				filter) + '_' + \
																													we_file_name

																		cnn.model = Sequential()
																		cnn.model.add(Embedding(exp.pp_data.vocabulary_size, exp.pp_data.embedding_size,
																														trainable=cnn.embed_trainable, name='emb_' + name_model))
																		cnn.model.add(Dropout(dropout, name='dropout_1_' + name_model))
																		cnn.model.add(Conv1D(filters=filter, kernel_size=kernel_size,
																												 kernel_initializer='glorot_uniform',
																												 # kernel_regularizer=regularizers.l2(0.03),
																												 padding='valid', activation='relu',
																												 name='conv_1_' + name_model))
																		cnn.model.add(GlobalAveragePooling1D(name='gloval_avg_pool_1_' + name_model))
																		cnn.model.add(Dense(set_params["hidden_ds"], activation='relu', kernel_initializer='glorot_uniform',
																												name='dense_1_' + name_model))
																		cnn.model.add(Dropout(dropout, name='dropout_2_' + name_model))
																		cnn.model.add(Dense(3, activation='softmax', name='dense_2_' + name_model))

																		time_ini_exp = datetime.datetime.now()
																		exp.generate_model_hypeparams(cnn, x_train, y_train, x_valid, y_valid,
																																	embedding_matrix)
																		exp.set_period_time_end(time_ini_exp, 'Total experiment')

								del x_train, y_train, x_valid, y_valid, num_words, embedding_matrix

		# Test
		np.random.seed(dn.SEED)
		time_ini_rep = datetime.datetime.now()

		for embedding_type in set_params['embedding_types']:
				for embedding_custom_file in set_params['embedding_custom_files']:
						for use_embedding in set_params['use_embeddings']:
								exp.pp_data.embedding_type = embedding_type
								exp.pp_data.word_embedding_custom_file = embedding_custom_file
								exp.pp_data.use_embedding = use_embedding
								exp.pp_data.load_dataset_type = dn.LoadDataset.TEST_DATA_MODEL

								exp.set_period_time_end(time_ini_rep, 'Load data')
								x_test, y_test = exp.pp_data.load_data()

								cnn.use_embedding_pre_train = exp.pp_data.use_embedding
								cnn.embed_trainable = (
												cnn.use_embedding_pre_train == (dn.UseEmbedding.RAND or dn.UseEmbedding.NON_STATIC))

								emb_name = function

								if embedding_custom_file != '':
										emb_name = exp.pp_data.word_embedding_custom_file.split('.')[0]

								we_file_name = 'ET_' + str(exp.pp_data.embedding_type.value) + '_UE_' + \
															 str(exp.pp_data.use_embedding.value) + '_EF_' + emb_name + kernel_name

								for filter in filters_by_layer:
										for kernel_size in kernels_size:
												for batch_size in batch_sizes:
														for epoch in epochs:
																for dropout in dropouts:
																		cnn.epochs = epoch
																		cnn.batch_size = batch_size
																		cnn.patience_train = epoch
																		exp.experiment_name = name_model + '_cnn' + '_F' + str(filter) + '_K' + str(
																				kernel_size) + \
																													'_P' + 'None' + '_B' + str(batch_size) + '_E' + \
																													str(epoch) + '_D' + str(dropout) + '_HLN' + str(
																				filter) + '_' + \
																													we_file_name

																		cnn.model = exp.load_model(dn.PATH_PROJECT + exp.experiment_name + '.h5')
																		exp.save_geral_configs('Experiment Specific Configuration: ' + exp.experiment_name)
																		exp.save_summary_model(cnn.model)
																		exp.predict_samples(cnn, x_test, y_test)

								del x_test, y_test

		del cnn, exp


def test_glove6b(option, function, set_params, type_generate='default'):
		if option == '1':
				print('Initializer experiment '+option+' (model SMHD_cnn_gl_1040_A_D)\n' + \
							'Set: kernel_initializer=glorot_uniform=xavier_uniform, dataset=SMHD_1040 diffs')
				exp = ExperimentProcesses('t'+option+'_cnn_L1')
				exp.pp_data.set_dataset_source(dataset_name='SMHD', label_set=['control', 'anxiety', 'depression'],
																			 total_registers=1040, subdirectory="only_disorders/A_D")
				if type_generate == 'default':
						generate_model(exp, 't'+option+'_SMHD_cnn_gl_1040_A_D', '_glorot', set_params, function)
				else:
						generate_model_softmax(exp, 't'+option+'_SMHD_cnn_gl_1040_A_D_'+type_generate, '_glorot', set_params, function)

		elif option == '2':
				print('Initializer experiment '+option+' (model SMHD_cnn_gl_880_A_AD)\n' + \
							'Set: kernel_initializer=glorot_uniform=xavier_uniform, dataset=SMHD_880 diffs')
				exp = ExperimentProcesses('t'+option+'_cnn_L1')
				exp.pp_data.set_dataset_source(dataset_name='SMHD', label_set=['control', 'anxiety', 'depression'],
																			 total_registers=880, subdirectory="only_disorders/A_AD")
				if type_generate == 'default':
						generate_model(exp, 't'+option+'_SMHD_cnn_gl_880_A_AD', '_glorot', set_params, function)
				else:
						generate_model_softmax(exp, 't'+option+'_SMHD_cnn_gl_880_A_AD_'+type_generate, '_glorot', set_params, function)

		elif option == '3':
				print('Initializer experiment '+option+' (model SMHD_ml_gl_880_D_AD)\n' + \
							'Set: kernel_initializer=glorot_uniform=xavier_uniform, dataset=SMHD_880 diffs')
				exp = ExperimentProcesses('t'+option+'_cnn_L1')
				exp.pp_data.set_dataset_source(dataset_name='SMHD', label_set=['control', 'anxiety', 'depression'],
																			 total_registers=880, subdirectory="only_disorders/D_AD")
				if type_generate == 'default':
						generate_model(exp, 't'+option+'_SMHD_ml_gl_880_D_AD', '_glorot', set_params, function)
				else:
						generate_model_softmax(exp, 't'+option+'_SMHD_ml_gl_880_D_AD_'+type_generate, '_glorot', set_params, function)


def test_glove_twitter_emb(option, function, set_params, type_generate='default'):
		if option == '1':
				print('Initializer experiment '+option+' (model SMHD_cnn_gl_1040_A_D)\n' + \
							'Set: kernel_initializer=glorot_uniform=xavier_uniform, dataset=SMHD_1040 diffs')
				exp = ExperimentProcesses('t'+option+'_cnn_L1')
				exp.pp_data.set_dataset_source(dataset_name='SMHD', label_set=['control', 'anxiety', 'depression'],
																			 total_registers=1040, subdirectory="only_disorders/A_D")
				if type_generate == 'default':
						generate_model(exp, 't'+option+'_SMHD_cnn_gl_1040_A_D', '_glorot', set_params, function)
				else:
						generate_model_softmax(exp, 't'+option+'_SMHD_cnn_gl_1040_A_D_'+type_generate, '_glorot', set_params, function)

		elif option == '2':
				print('Initializer experiment '+option+' (model SMHD_cnn_gl_880_A_AD)\n' + \
							'Set: kernel_initializer=glorot_uniform=xavier_uniform, dataset=SMHD_880 diffs')
				exp = ExperimentProcesses('t'+option+'_cnn_L1')
				exp.pp_data.set_dataset_source(dataset_name='SMHD', label_set=['control', 'anxiety', 'depression'],
																			 total_registers=880, subdirectory="only_disorders/A_AD")
				if type_generate == 'default':
						generate_model(exp, 't'+option+'_SMHD_cnn_gl_880_A_AD', '_glorot', set_params, function)
				else:
						generate_model_softmax(exp, 't'+option+'_SMHD_cnn_gl_880_A_AD_'+type_generate, '_glorot', set_params, function)

		elif option == '3':
				print('Initializer experiment '+option+' (model SMHD_ml_gl_880_D_AD)\n' + \
							'Set: kernel_initializer=glorot_uniform=xavier_uniform, dataset=SMHD_880 diffs')
				exp = ExperimentProcesses('t'+option+'_cnn_L1')
				exp.pp_data.set_dataset_source(dataset_name='SMHD', label_set=['control', 'anxiety', 'depression'],
																			 total_registers=880, subdirectory="only_disorders/D_AD")
				if type_generate == 'default':
						generate_model(exp, 't'+option+'_SMHD_ml_gl_880_D_AD', '_glorot', set_params, function)
				else:
						generate_model_softmax(exp, 't'+option+'_SMHD_ml_gl_880_D_AD_'+type_generate, '_glorot', set_params, function)


def test_google_news_emb(option, function, set_params, type_generate='default'):
		if option == '1':
				print('Initializer experiment '+option+' (model SMHD_cnn_gl_1040_A_D)\n' + \
							'Set: kernel_initializer=glorot_uniform=xavier_uniform, dataset=SMHD_1040 diffs')
				exp = ExperimentProcesses('t'+option+'_cnn_L1')
				exp.pp_data.set_dataset_source(dataset_name='SMHD', label_set=['control', 'anxiety', 'depression'],
																			 total_registers=1040, subdirectory="only_disorders/A_D")
				if type_generate == 'default':
						generate_model(exp, 't'+option+'_SMHD_cnn_gl_1040_A_D', '_glorot', set_params, function)
				else:
						generate_model_softmax(exp, 't'+option+'_SMHD_cnn_gl_1040_A_D_'+type_generate, '_glorot', set_params, function)

		elif option == '2':
				print('Initializer experiment '+option+' (model SMHD_cnn_gl_880_A_AD)\n' + \
							'Set: kernel_initializer=glorot_uniform=xavier_uniform, dataset=SMHD_880 diffs')
				exp = ExperimentProcesses('t'+option+'_cnn_L1')
				exp.pp_data.set_dataset_source(dataset_name='SMHD', label_set=['control', 'anxiety', 'depression'],
																			 total_registers=880, subdirectory="only_disorders/A_AD")
				if type_generate == 'default':
						generate_model(exp, 't'+option+'_SMHD_cnn_gl_880_A_AD', '_glorot', set_params, function)
				else:
						generate_model_softmax(exp, 't'+option+'_SMHD_cnn_gl_880_A_AD_'+type_generate, '_glorot', set_params, function)

		elif option == '3':
				print('Initializer experiment '+option+' (model SMHD_ml_gl_880_D_AD)\n' + \
							'Set: kernel_initializer=glorot_uniform=xavier_uniform, dataset=SMHD_880 diffs')
				exp = ExperimentProcesses('t'+option+'_cnn_L1')
				exp.pp_data.set_dataset_source(dataset_name='SMHD', label_set=['control', 'anxiety', 'depression'],
																			 total_registers=880, subdirectory="only_disorders/D_AD")
				if type_generate == 'default':
						generate_model(exp, 't'+option+'_SMHD_cnn_gl_880_A_AD', '_glorot', set_params, function)
				else:
						generate_model_softmax(exp, 't'+option+'_SMHD_cnn_gl_880_A_AD_'+type_generate, '_glorot', set_params, function)


def test_w2v_custom_emb(option, function, set_params, type_generate='default'):
		if option == '1':
				print('Initializer experiment '+option+' (model SMHD_cnn_gl_1040_A_D)\n' + \
							'Set: kernel_initializer=glorot_uniform=xavier_uniform, dataset=SMHD_1040 diffs')
				exp = ExperimentProcesses('t'+option+'_cnn_L1')
				exp.pp_data.set_dataset_source(dataset_name='SMHD', label_set=['control', 'anxiety', 'depression'],
																			 total_registers=1040, subdirectory="only_disorders/A_D")
				if type_generate == 'default':
						generate_model(exp, 't'+option+'_SMHD_cnn_gl_1040_A_D', '_glorot', set_params, function)
				else:
						generate_model_softmax(exp, 't'+option+'_SMHD_cnn_gl_1040_A_D_'+type_generate, '_glorot', set_params, function)

		elif option == '2':
				print('Initializer experiment '+option+' (model SMHD_cnn_gl_880_A_AD)\n' + \
							'Set: kernel_initializer=glorot_uniform=xavier_uniform, dataset=SMHD_880 diffs')
				exp = ExperimentProcesses('t'+option+'_cnn_L1')
				exp.pp_data.set_dataset_source(dataset_name='SMHD', label_set=['control', 'anxiety', 'depression'],
																			 total_registers=880, subdirectory="only_disorders/A_AD")
				if type_generate == 'default':
						generate_model(exp, 't'+option+'_SMHD_cnn_gl_880_A_AD', '_glorot', set_params, function)
				else:
						generate_model_softmax(exp, 't'+option+'_SMHD_cnn_gl_880_A_AD_'+type_generate, '_glorot', set_params, function)

		elif option == '3':
				print('Initializer experiment '+option+' (model SMHD_ml_gl_880_D_AD)\n' + \
							'Set: kernel_initializer=glorot_uniform=xavier_uniform, dataset=SMHD_880 diffs')
				exp = ExperimentProcesses('t'+option+'_cnn_L1')
				exp.pp_data.set_dataset_source(dataset_name='SMHD', label_set=['control', 'anxiety', 'depression'],
																			 total_registers=880, subdirectory="only_disorders/D_AD")
				if type_generate == 'default':
						generate_model(exp, 't'+option+'_SMHD_ml_gl_880_D_AD', '_glorot', set_params, function)
				else:
						generate_model_softmax(exp, 't'+option+'_SMHD_ml_gl_880_D_AD_'+type_generate, '_glorot', set_params, function)


def test_glove_custom_emb(option, function, set_params, type_generate='default'):
		if option == '1':
				print('Initializer experiment '+option+' (model SMHD_cnn_gl_1040_A_D)\n' + \
							'Set: kernel_initializer=glorot_uniform=xavier_uniform, dataset=SMHD_1040 diffs')
				exp = ExperimentProcesses('t'+option+'_cnn_L1')
				exp.pp_data.set_dataset_source(dataset_name='SMHD', label_set=['control', 'anxiety', 'depression'],
																			 total_registers=1040, subdirectory="only_disorders/A_D")
				if type_generate == 'default':
						generate_model(exp, 't'+option+'_SMHD_cnn_gl_1040_A_D', '_glorot', set_params, function)
				else:
						generate_model_softmax(exp, 't'+option+'_SMHD_cnn_gl_1040_A_D_'+type_generate, '_glorot', set_params, function)

		elif option == '2':
				print('Initializer experiment '+option+' (model SMHD_cnn_gl_880_A_AD)\n' + \
							'Set: kernel_initializer=glorot_uniform=xavier_uniform, dataset=SMHD_880 diffs')
				exp = ExperimentProcesses('t'+option+'_cnn_L1')
				exp.pp_data.set_dataset_source(dataset_name='SMHD', label_set=['control', 'anxiety', 'depression'],
																			 total_registers=880, subdirectory="only_disorders/A_AD")
				if type_generate == 'default':
						generate_model(exp, 't'+option+'_SMHD_cnn_gl_880_A_AD', '_glorot', set_params, function)
				else:
						generate_model_softmax(exp, 't'+option+'_SMHD_cnn_gl_880_A_AD_'+type_generate, '_glorot', set_params, function)

		elif option == '3':
				print('Initializer experiment '+option+' (model SMHD_ml_gl_880_D_AD)\n' + \
							'Set: kernel_initializer=glorot_uniform=xavier_uniform, dataset=SMHD_880 diffs')
				exp = ExperimentProcesses('t'+option+'_cnn_L1')
				exp.pp_data.set_dataset_source(dataset_name='SMHD', label_set=['control', 'anxiety', 'depression'],
																			 total_registers=880, subdirectory="only_disorders/D_AD")
				if type_generate == 'default':
						generate_model(exp, 't'+option+'_SMHD_ml_gl_880_D_AD', '_glorot', set_params, function)
				else:
						generate_model_softmax(exp, 't'+option+'_SMHD_ml_gl_880_D_AD_'+type_generate, '_glorot', set_params, function)


def test_none_emb(option, function, set_params, type_generate='default'):
		if option == '1':
				print('Initializer experiment '+option+' (model SMHD_cnn_gl_1040_A_D)\n' + \
							'Set: kernel_initializer=glorot_uniform=xavier_uniform, dataset=SMHD_1040 diffs')
				exp = ExperimentProcesses('t'+option+'_cnn_L1')
				exp.pp_data.set_dataset_source(dataset_name='SMHD', label_set=['control', 'anxiety', 'depression'],
																			 total_registers=1040, subdirectory="only_disorders/A_D")
				if type_generate == 'default':
						generate_model(exp, 't'+option+'_SMHD_cnn_gl_1040_A_D', '_glorot', set_params, function)
				else:
						generate_model_softmax(exp, 't'+option+'_SMHD_cnn_gl_1040_A_D_'+type_generate, '_glorot', set_params, function)

		elif option == '2':
				print('Initializer experiment '+option+' (model SMHD_cnn_gl_880_A_AD)\n' + \
							'Set: kernel_initializer=glorot_uniform=xavier_uniform, dataset=SMHD_880 diffs')
				exp = ExperimentProcesses('t'+option+'_cnn_L1')
				exp.pp_data.set_dataset_source(dataset_name='SMHD', label_set=['control', 'anxiety', 'depression'],
																			 total_registers=880, subdirectory="only_disorders/A_AD")
				if type_generate == 'default':
						generate_model(exp, 't'+option+'_SMHD_cnn_gl_880_A_AD', '_glorot', set_params, function)
				else:
						generate_model_softmax(exp, 't'+option+'_SMHD_cnn_gl_880_A_AD_'+type_generate, '_glorot', set_params, function)

		elif option == '3':
				print('Initializer experiment '+option+' (model SMHD_ml_gl_880_D_AD)\n' + \
							'Set: kernel_initializer=glorot_uniform=xavier_uniform, dataset=SMHD_880 diffs')
				exp = ExperimentProcesses('t'+option+'_cnn_L1')
				exp.pp_data.set_dataset_source(dataset_name='SMHD', label_set=['control', 'anxiety', 'depression'],
																			 total_registers=880, subdirectory="only_disorders/D_AD")
				if type_generate == 'default':
						generate_model(exp, 't'+option+'_SMHD_ml_gl_880_D_AD', '_glorot', set_params, function)
				else:
						generate_model_softmax(exp, 't'+option+'_SMHD_ml_gl_880_D_AD_'+type_generate, '_glorot', set_params, function)


def first_test(option, function):
		if function == 'glove6B300d':
				set_params = dict({'filters_by_layer': [100, 250],
													 'kernels_size': [3, 4, 5],
													 'epochs': [10],
													 'batch_sizes': [20], #, 40],
													 'dropouts': [0.2, 0.5],
													 'embedding_types': [dn.EmbeddingType.GLOVE_6B],
													 'embedding_custom_files': [''],
													 'use_embeddings': [dn.UseEmbedding.STATIC, dn.UseEmbedding.NON_STATIC]})
				test_glove6b(option, function, set_params)
		
		elif function == 'gloveTwitter':
				set_params = dict({'filters_by_layer': [100, 250],
													 'kernels_size': [3, 4, 5],
													 'epochs': [10],
													 'batch_sizes': [20],
													 'dropouts': [0.2, 0.5],
													 'embedding_types': [dn.EmbeddingType.GLOVE_TWITTER],
													 'embedding_custom_files': [''],
													 'use_embeddings': [dn.UseEmbedding.STATIC, dn.UseEmbedding.NON_STATIC]})
				test_glove_twitter_emb(option, function, set_params)
		
		elif function == 'googleNews':
				set_params = dict({'filters_by_layer': [100, 250],
													 'kernels_size': [3, 4, 5],
													 'epochs': [10],
													 'batch_sizes': [20],
													 'dropouts': [0.2, 0.5],
													 'embedding_types': [dn.EmbeddingType.WORD2VEC],
													 'embedding_custom_files': [''],
													 'use_embeddings': [dn.UseEmbedding.STATIC, dn.UseEmbedding.NON_STATIC]})
				test_google_news_emb(option, function, set_params)
		
		elif function == 'w2vCustom':
				set_params = dict({'filters_by_layer': [100, 250],
													 'kernels_size': [3, 4, 5],
													 'epochs': [10],
													 'batch_sizes': [20],
													 'dropouts': [0.2, 0.5],
													 'embedding_types': [dn.EmbeddingType.WORD2VEC_CUSTOM],
													 'embedding_custom_files': ['SMHD-Skipgram-AllUsers-300.bin', 'SMHD-CBOW-AllUsers-300.bin',
																											'SMHD-Skipgram-A-D-ADUsers-300.bin', 'SMHD-CBOW-A-D-ADUsers-300.bin'],
													 'use_embeddings': [dn.UseEmbedding.STATIC, dn.UseEmbedding.NON_STATIC]})
				test_w2v_custom_emb(option, function, set_params)
		
		elif function == 'gloveCustom':
				set_params = dict({'filters_by_layer': [100, 250],
													 'kernels_size': [3, 4, 5],
													 'epochs': [10],
													 'batch_sizes': [20],
													 'dropouts': [0.2, 0.5],
													 'embedding_types': [dn.EmbeddingType.GLOVE_CUSTOM],
													 'embedding_custom_files': ['SMHD-glove-AllUsers-300.pkl', 'SMHD-glove-A-D-ADUsers-300.pkl'],
													 'use_embeddings': [dn.UseEmbedding.STATIC, dn.UseEmbedding.NON_STATIC]})
				test_glove_custom_emb(option, function, set_params)
		
		else:  # None
				set_params = dict({'filters_by_layer': [100, 250],
													 'kernels_size': [3, 4, 5],
													 'epochs': [10],
													 'batch_sizes': [20],
													 'dropouts': [0.2, 0.5],
													 'embedding_types': [dn.EmbeddingType.NONE],
													 'embedding_custom_files': [''],
													 'use_embeddings': [dn.UseEmbedding.RAND]})
				test_none_emb(option, function, set_params)

def outperformance_test1(option, function):
		print("Outperformance Test 1")
		
		if function == 'glove6B300d':
				if option == '1':
						# t1_SMHD_cnn_gl_1040_A_D_cnn_F100_K3_PNone_B20_E10_D0.5_HLN100_ET_2_UE_4_EF_glove6B300d_glorot
						set_params = dict({'filters_by_layer': [100],
															 'kernels_size': [3],
															 'epochs': [10, 20],
															 'batch_sizes': [5, 10, 20],
															 'dropouts': [0.5],
															 'embedding_types': [dn.EmbeddingType.GLOVE_6B],
															 'embedding_custom_files': [''],
															 'use_embeddings': [dn.UseEmbedding.NON_STATIC]})
						test_glove6b(option, function, set_params)
				
						# t1_SMHD_cnn_gl_1040_A_D_cnn_F250_K3_PNone_B20_E10_D0.2_HLN250_ET_2_UE_4_EF_glove6B300d_glorot
						set_params = dict({'filters_by_layer': [250],
																'kernels_size': [3],
																'epochs': [10, 20],
																'batch_sizes': [5, 10, 20],
																'dropouts': [0.2],
																'embedding_types': [dn.EmbeddingType.GLOVE_6B],
																'embedding_custom_files': [''],
																'use_embeddings': [dn.UseEmbedding.NON_STATIC]})
						test_glove6b(option, function, set_params)
				
				elif option == '2':
						# t2_SMHD_cnn_gl_880_A_AD_cnn_F100_K5_PNone_B20_E10_D0.2_HLN100_ET_2_UE_4_EF_glove6B300d_glorot
						set_params = dict({'filters_by_layer': [100],
															 'kernels_size': [5],
															 'epochs': [10, 20],
															 'batch_sizes': [5, 10, 20],
															 'dropouts': [0.2],
															 'embedding_types': [dn.EmbeddingType.GLOVE_6B],
															 'embedding_custom_files': [''],
															 'use_embeddings': [dn.UseEmbedding.NON_STATIC]})
						test_glove6b(option, function, set_params)
						
						# t2_SMHD_cnn_gl_880_A_AD_cnn_F250_K4_PNone_B20_E10_D0.5_HLN250_ET_2_UE_3_EF_glove6B300d_glorot
						set_params = dict({'filters_by_layer': [250],
															 'kernels_size': [4],
															 'epochs': [10, 20],
															 'batch_sizes': [5, 10, 20],
															 'dropouts': [0.5],
															 'embedding_types': [dn.EmbeddingType.GLOVE_6B],
															 'embedding_custom_files': [''],
															 'use_embeddings': [dn.UseEmbedding.STATIC]})
						test_glove6b(option, function, set_params)
						
				else:
						# t3_SMHD_ml_gl_880_D_AD_cnn_F100_K3_PNone_B20_E10_D0.5_HLN100_ET_2_UE_3_EF_glove6B300d_glorot
						set_params = dict({'filters_by_layer': [100],
															 'kernels_size': [3],
															 'epochs': [10, 20],
															 'batch_sizes': [5, 10, 20],
															 'dropouts': [0.5],
															 'embedding_types': [dn.EmbeddingType.GLOVE_6B],
															 'embedding_custom_files': [''],
															 'use_embeddings': [dn.UseEmbedding.STATIC]})
						test_glove6b(option, function, set_params)
						
						# t3_SMHD_ml_gl_880_D_AD_cnn_F250_K3_PNone_B20_E10_D0.2_HLN250_ET_2_UE_3_EF_glove6B300d_glorot
						set_params = dict({'filters_by_layer': [250],
															 'kernels_size': [3],
															 'epochs': [10, 20],
															 'batch_sizes': [5, 10, 20],
															 'dropouts': [0.2],
															 'embedding_types': [dn.EmbeddingType.GLOVE_6B],
															 'embedding_custom_files': [''],
															 'use_embeddings': [dn.UseEmbedding.STATIC]})
						test_glove6b(option, function, set_params)
						
		elif function == 'gloveTwitter':
				if option == '1':
						# t1_SMHD_cnn_gl_1040_A_D_cnn_F250_K4_PNone_B20_E10_D0.2_HLN250_ET_3_UE_4_EF_gloveTwitter_glorot
						set_params = dict({'filters_by_layer': [250],
															 'kernels_size': [4],
															 'epochs': [10, 20],
															 'batch_sizes': [5, 10, 20],
															 'dropouts': [0.2],
															 'embedding_types': [dn.EmbeddingType.GLOVE_TWITTER],
															 'embedding_custom_files': [''],
															 'use_embeddings': [dn.UseEmbedding.NON_STATIC]})
						test_glove_twitter_emb(option, function, set_params)
						
						# t1_SMHD_cnn_gl_1040_A_D_cnn_F250_K5_PNone_B20_E10_D0.5_HLN250_ET_3_UE_4_EF_gloveTwitter_glorot
						set_params = dict({'filters_by_layer': [250],
															 'kernels_size': [5],
															 'epochs': [10, 20],
															 'batch_sizes': [5, 10, 20],
															 'dropouts': [0.5],
															 'embedding_types': [dn.EmbeddingType.GLOVE_TWITTER],
															 'embedding_custom_files': [''],
															 'use_embeddings': [dn.UseEmbedding.NON_STATIC]})
						test_glove_twitter_emb(option, function, set_params)

				elif option == '2':
						# t2_SMHD_cnn_gl_880_A_AD_cnn_F100_K5_PNone_B20_E10_D0.5_HLN100_ET_3_UE_4_EF_gloveTwitter_glorot
						set_params = dict({'filters_by_layer': [100],
															 'kernels_size': [5],
															 'epochs': [10, 20],
															 'batch_sizes': [5, 10, 20],
															 'dropouts': [0.5],
															 'embedding_types': [dn.EmbeddingType.GLOVE_TWITTER],
															 'embedding_custom_files': [''],
															 'use_embeddings': [dn.UseEmbedding.NON_STATIC]})
						test_glove_twitter_emb(option, function, set_params)

						# t2_SMHD_cnn_gl_880_A_AD_cnn_F250_K3_PNone_B20_E10_D0.5_HLN250_ET_3_UE_4_EF_gloveTwitter_glorot
						set_params = dict({'filters_by_layer': [250],
															 'kernels_size': [3],
															 'epochs': [10, 20],
															 'batch_sizes': [5, 10, 20],
															 'dropouts': [0.5],
															 'embedding_types': [dn.EmbeddingType.GLOVE_TWITTER],
															 'embedding_custom_files': [''],
															 'use_embeddings': [dn.UseEmbedding.NON_STATIC]})
						test_glove_twitter_emb(option, function, set_params)
				else:
						# t3_SMHD_ml_gl_880_D_AD_cnn_F100_K5_PNone_B20_E10_D0.2_HLN100_ET_3_UE_3_EF_gloveTwitter_glorot
						set_params = dict({'filters_by_layer': [100],
															 'kernels_size': [5],
															 'epochs': [10, 20],
															 'batch_sizes': [5, 10, 20],
															 'dropouts': [0.2],
															 'embedding_types': [dn.EmbeddingType.GLOVE_TWITTER],
															 'embedding_custom_files': [''],
															 'use_embeddings': [dn.UseEmbedding.STATIC]})
						test_glove_twitter_emb(option, function, set_params)

						# t3_SMHD_ml_gl_880_D_AD_cnn_F250_K5_PNone_B20_E10_D0.2_HLN250_ET_3_UE_3_EF_gloveTwitter_glorot
						set_params = dict({'filters_by_layer': [250],
															 'kernels_size': [5],
															 'epochs': [10, 20],
															 'batch_sizes': [5, 10, 20],
															 'dropouts': [0.2],
															 'embedding_types': [dn.EmbeddingType.GLOVE_TWITTER],
															 'embedding_custom_files': [''],
															 'use_embeddings': [dn.UseEmbedding.STATIC]})
						test_glove_twitter_emb(option, function, set_params)

		elif function == 'googleNews':
				if option == '1':
						# t1_SMHD_cnn_gl_1040_A_D_cnn_F100_K4_PNone_B20_E10_D0.2_HLN100_ET_4_UE_4_EF_googleNews_glorot
						set_params = dict({'filters_by_layer': [100],
															 'kernels_size': [4],
															 'epochs': [10, 20],
															 'batch_sizes': [5, 10, 20],
															 'dropouts': [0.2],
															 'embedding_types': [dn.EmbeddingType.WORD2VEC],
															 'embedding_custom_files': [''],
															 'use_embeddings': [dn.UseEmbedding.NON_STATIC]})
						test_google_news_emb(option, function, set_params)

						# t1_SMHD_cnn_gl_1040_A_D_cnn_F250_K4_PNone_B20_E10_D0.2_HLN250_ET_4_UE_4_EF_googleNews_glorot
						set_params = dict({'filters_by_layer': [250],
															 'kernels_size': [4],
															 'epochs': [10, 20],
															 'batch_sizes': [5, 10, 20],
															 'dropouts': [0.2],
															 'embedding_types': [dn.EmbeddingType.WORD2VEC],
															 'embedding_custom_files': [''],
															 'use_embeddings': [dn.UseEmbedding.NON_STATIC]})
						test_google_news_emb(option, function, set_params)
				
				elif option == '2':
						# t2_SMHD_cnn_gl_880_A_AD_cnn_F100_K5_PNone_B20_E10_D0.2_HLN100_ET_4_UE_4_EF_googleNews_glorot
						set_params = dict({'filters_by_layer': [100],
															 'kernels_size': [5],
															 'epochs': [10, 20],
															 'batch_sizes': [5, 10, 20],
															 'dropouts': [0.2],
															 'embedding_types': [dn.EmbeddingType.WORD2VEC],
															 'embedding_custom_files': [''],
															 'use_embeddings': [dn.UseEmbedding.NON_STATIC]})
						test_google_news_emb(option, function, set_params)

						# t2_SMHD_cnn_gl_880_A_AD_cnn_F100_K5_PNone_B20_E10_D0.5_HLN100_ET_4_UE_4_EF_googleNews_glorot
						set_params = dict({'filters_by_layer': [100],
															 'kernels_size': [5],
															 'epochs': [10, 20],
															 'batch_sizes': [5, 10, 20],
															 'dropouts': [0.5],
															 'embedding_types': [dn.EmbeddingType.WORD2VEC],
															 'embedding_custom_files': [''],
															 'use_embeddings': [dn.UseEmbedding.NON_STATIC]})
						test_google_news_emb(option, function, set_params)
				else:
						# t3_SMHD_ml_gl_880_D_AD_cnn_F250_K5_PNone_B20_E10_D0.2_HLN250_ET_4_UE_3_EF_googleNews_glorot
						set_params = dict({'filters_by_layer': [250],
															 'kernels_size': [5],
															 'epochs': [10, 20],
															 'batch_sizes': [5, 10, 20],
															 'dropouts': [0.2],
															 'embedding_types': [dn.EmbeddingType.WORD2VEC],
															 'embedding_custom_files': [''],
															 'use_embeddings': [dn.UseEmbedding.STATIC]})
						test_google_news_emb(option, function, set_params)

						# t3_SMHD_ml_gl_880_D_AD_cnn_F250_K5_PNone_B20_E10_D0.5_HLN250_ET_4_UE_3_EF_googleNews_glorot
						set_params = dict({'filters_by_layer': [250],
															 'kernels_size': [5],
															 'epochs': [10, 20],
															 'batch_sizes': [5, 10, 20],
															 'dropouts': [0.5],
															 'embedding_types': [dn.EmbeddingType.WORD2VEC],
															 'embedding_custom_files': [''],
															 'use_embeddings': [dn.UseEmbedding.NON_STATIC]})
						test_google_news_emb(option, function, set_params)
		
		elif function == 'w2vCustom':
				if option == '1':
						# t1_SMHD_cnn_gl_1040_A_D_cnn_F100_K4_PNone_B20_E10_D0.2_HLN100_ET_6_UE_4_EF_SMHD-CBOW-A-D-ADUsers-300_glorot
						set_params = dict({'filters_by_layer': [100],
															 'kernels_size': [4],
															 'epochs': [10, 20],
															 'batch_sizes': [5, 10, 20],
															 'dropouts': [0.2],
															 'embedding_types': [dn.EmbeddingType.WORD2VEC_CUSTOM],
															 'embedding_custom_files': ['SMHD-CBOW-A-D-ADUsers-300.bin'],
															 'use_embeddings': [dn.UseEmbedding.NON_STATIC]})
						test_w2v_custom_emb(option, function, set_params)

						# t1_SMHD_cnn_gl_1040_A_D_cnn_F250_K4_PNone_B20_E10_D0.5_HLN250_ET_6_UE_4_EF_SMHD-CBOW-A-D-ADUsers-300_glorot
						set_params = dict({'filters_by_layer': [250],
															 'kernels_size': [4],
															 'epochs': [10, 20],
															 'batch_sizes': [5, 10, 20],
															 'dropouts': [0.5],
															 'embedding_types': [dn.EmbeddingType.WORD2VEC_CUSTOM],
															 'embedding_custom_files': ['SMHD-CBOW-A-D-ADUsers-300.bin'],
															 'use_embeddings': [dn.UseEmbedding.NON_STATIC]})
						test_w2v_custom_emb(option, function, set_params)

				elif option == '2':
						# t2_SMHD_cnn_gl_880_A_AD_cnn_F100_K3_PNone_B20_E10_D0.2_HLN100_ET_6_UE_4_EF_SMHD-CBOW-A-D-ADUsers-300_glorot
						set_params = dict({'filters_by_layer': [100],
															 'kernels_size': [3],
															 'epochs': [10, 20],
															 'batch_sizes': [5, 10, 20],
															 'dropouts': [0.2],
															 'embedding_types': [dn.EmbeddingType.WORD2VEC_CUSTOM],
															 'embedding_custom_files': ['SMHD-CBOW-A-D-ADUsers-300.bin'],
															 'use_embeddings': [dn.UseEmbedding.NON_STATIC]})
						test_w2v_custom_emb(option, function, set_params)

						# t2_SMHD_cnn_gl_880_A_AD_cnn_F100_K5_PNone_B20_E10_D0.5_HLN100_ET_6_UE_3_EF_SMHD-Skipgram-A-D-ADUsers-300_glorot
						set_params = dict({'filters_by_layer': [100],
															 'kernels_size': [5],
															 'epochs': [10, 20],
															 'batch_sizes': [5, 10, 20],
															 'dropouts': [0.5],
															 'embedding_types': [dn.EmbeddingType.WORD2VEC_CUSTOM],
															 'embedding_custom_files': ['SMHD-Skipgram-A-D-ADUsers-300.bin'],
															 'use_embeddings': [dn.UseEmbedding.STATIC]})
						test_w2v_custom_emb(option, function, set_params)
				else:
						# t3_SMHD_ml_gl_880_D_AD_cnn_F100_K4_PNone_B20_E10_D0.2_HLN100_ET_6_UE_4_EF_SMHD-Skipgram-AllUsers-300_glorot
						set_params = dict({'filters_by_layer': [100],
															 'kernels_size': [4],
															 'epochs': [10, 20],
															 'batch_sizes': [5, 10, 20],
															 'dropouts': [0.2],
															 'embedding_types': [dn.EmbeddingType.WORD2VEC_CUSTOM],
															 'embedding_custom_files': ['SMHD-Skipgram-AllUsers-300.bin'],
															 'use_embeddings': [dn.UseEmbedding.NON_STATIC]})
						test_w2v_custom_emb(option, function, set_params)

						# t3_SMHD_ml_gl_880_D_AD_cnn_F250_K4_PNone_B20_E10_D0.2_HLN250_ET_6_UE_3_EF_SMHD-Skipgram-AllUsers-300_glorot
						set_params = dict({'filters_by_layer': [250],
															 'kernels_size': [4],
															 'epochs': [10, 20],
															 'batch_sizes': [5, 10, 20],
															 'dropouts': [0.2],
															 'embedding_types': [dn.EmbeddingType.WORD2VEC_CUSTOM],
															 'embedding_custom_files': ['SMHD-Skipgram-AllUsers-300.bin'],
															 'use_embeddings': [dn.UseEmbedding.STATIC]})
						test_w2v_custom_emb(option, function, set_params)
		
		elif function == 'gloveCustom':
				if option == '1':
						# t1_SMHD_cnn_gl_1040_A_D_cnn_F100_K5_PNone_B20_E10_D0.5_HLN100_ET_7_UE_4_EF_SMHD-glove-AllUsers-300_glorot
						set_params = dict({'filters_by_layer': [100],
															 'kernels_size': [5],
															 'epochs': [10, 20],
															 'batch_sizes': [5, 10, 20],
															 'dropouts': [0.5],
															 'embedding_types': [dn.EmbeddingType.GLOVE_CUSTOM],
															 'embedding_custom_files': ['SMHD-glove-AllUsers-300.pkl'],
															 'use_embeddings': [dn.UseEmbedding.NON_STATIC]})
						test_glove_custom_emb(option, function, set_params)

						# t1_SMHD_cnn_gl_1040_A_D_cnn_F250_K5_PNone_B20_E10_D0.2_HLN250_ET_7_UE_4_EF_SMHD-glove-A-D-ADUsers-300_glorot
						set_params = dict({'filters_by_layer': [250],
															 'kernels_size': [5],
															 'epochs': [10, 20],
															 'batch_sizes': [5, 10, 20],
															 'dropouts': [0.2],
															 'embedding_types': [dn.EmbeddingType.GLOVE_CUSTOM],
															 'embedding_custom_files': ['SMHD-glove-A-D-ADUsers-300.pkl'],
															 'use_embeddings': [dn.UseEmbedding.NON_STATIC]})
						test_glove_custom_emb(option, function, set_params)

				elif option == '2':
						# t2_SMHD_cnn_gl_880_A_AD_cnn_F100_K3_PNone_B20_E10_D0.2_HLN100_ET_7_UE_4_EF_SMHD-glove-A-D-ADUsers-300_glorot
						set_params = dict({'filters_by_layer': [100],
															 'kernels_size': [3],
															 'epochs': [10, 20],
															 'batch_sizes': [5, 10, 20],
															 'dropouts': [0.2],
															 'embedding_types': [dn.EmbeddingType.GLOVE_CUSTOM],
															 'embedding_custom_files': ['SMHD-glove-A-D-ADUsers-300.pkl'],
															 'use_embeddings': [dn.UseEmbedding.NON_STATIC]})
						test_glove_custom_emb(option, function, set_params)
						
						# t2_SMHD_cnn_gl_880_A_AD_cnn_F100_K4_PNone_B20_E10_D0.5_HLN100_ET_7_UE_4_EF_SMHD-glove-AllUsers-300_glorot
						set_params = dict({'filters_by_layer': [100],
															 'kernels_size': [4],
															 'epochs': [10, 20],
															 'batch_sizes': [5, 10, 20],
															 'dropouts': [0.5],
															 'embedding_types': [dn.EmbeddingType.GLOVE_CUSTOM],
															 'embedding_custom_files': ['SMHD-glove-AllUsers-300.pkl'],
															 'use_embeddings': [dn.UseEmbedding.NON_STATIC]})
						test_glove_custom_emb(option, function, set_params)
				else:
						# t3_SMHD_ml_gl_880_D_AD_cnn_F100_K4_PNone_B20_E10_D0.2_HLN100_ET_7_UE_4_EF_SMHD-glove-AllUsers-300_glorot
						set_params = dict({'filters_by_layer': [100],
															 'kernels_size': [4],
															 'epochs': [10, 20],
															 'batch_sizes': [5, 10, 20],
															 'dropouts': [0.2],
															 'embedding_types': [dn.EmbeddingType.GLOVE_CUSTOM],
															 'embedding_custom_files': ['SMHD-glove-AllUsers-300.pkl'],
															 'use_embeddings': [dn.UseEmbedding.NON_STATIC]})
						test_glove_custom_emb(option, function, set_params)
						
						# t3_SMHD_ml_gl_880_D_AD_cnn_F250_K5_PNone_B20_E10_D0.2_HLN250_ET_7_UE_4_EF_SMHD-glove-AllUsers-300_glorot
						set_params = dict({'filters_by_layer': [250],
															 'kernels_size': [5],
															 'epochs': [10, 20],
															 'batch_sizes': [5, 10, 20],
															 'dropouts': [0.2],
															 'embedding_types': [dn.EmbeddingType.GLOVE_CUSTOM],
															 'embedding_custom_files': ['SMHD-glove-AllUsers-300.pkl'],
															 'use_embeddings': [dn.UseEmbedding.NON_STATIC]})
						test_glove_custom_emb(option, function, set_params)


def outperformance_test2(id_submodel_to_stacked, function):
		print("Outperformance Test 2 - Same set configuration, but replace last layer function sigmoid by softmax")
		set_params = dict()

		if function == 'glove6B300d':
				if id_submodel_to_stacked == '13':
						set_params = dict({'filters_by_layer': [250],
															 'kernels_size': [3],
															 # 'epochs': [20],
															 'epochs': [40],
															 'batch_sizes': [5],
															 'dropouts': [0.2],
															 'embedding_types': [dn.EmbeddingType.GLOVE_6B],
															 'embedding_custom_files': [''],
															 'use_embeddings': [dn.UseEmbedding.NON_STATIC]})

				test_glove6b('1', function, set_params, 'CAD13')

		elif function == 'gloveTwitter':
				if id_submodel_to_stacked == '1':
						set_params = dict({'filters_by_layer': [250],
															 'kernels_size': [4],
															 'epochs': [10, 20],
															 'batch_sizes': [5, 10, 20],
															 'dropouts': [0.2],
															 'embedding_types': [dn.EmbeddingType.GLOVE_TWITTER],
															 'embedding_custom_files': [''],
															 'use_embeddings': [dn.UseEmbedding.NON_STATIC]})

				test_glove_twitter_emb(option, function, set_params)

		elif function == 'googleNews':
				if id_submodel_to_stacked == '19':
						# set_params = dict({'filters_by_layer': [100],
						# 									 'kernels_size': [4],
						# 									 # 'epochs': [10],
						# 									 # 'batch_sizes': [20],
						# 									 'epochs': [40],
						# 									 'batch_sizes': [5],
						# 									 'dropouts': [0.2],
						# 									 'embedding_types': [dn.EmbeddingType.WORD2VEC],
						# 									 'embedding_custom_files': [''],
						# 									 'use_embeddings': [dn.UseEmbedding.NON_STATIC]})
						set_params = dict({'filters_by_layer': [135],
															 'kernels_size': [4],
															 'epochs': [22],
															 'batch_sizes': [6],
															 'dropouts': [0.401417],
															 'hidden_ds': 68,
															 'learning_rate': 0.0427706,
															 'embedding_types': [dn.EmbeddingType.WORD2VEC],
															 'embedding_custom_files': [''],
															 'use_embeddings': [dn.UseEmbedding.NON_STATIC]})

				test_google_news_emb('1', function, set_params, 'CAD19')

		elif function == 'w2vCustom':
				if id_submodel_to_stacked == '37':
						set_params = dict({'filters_by_layer': [100],
															 'kernels_size': [3],
															 'epochs': [10],
															 'batch_sizes': [20],
															 'dropouts': [0.2],
															 'embedding_types': [dn.EmbeddingType.WORD2VEC_CUSTOM],
															 'embedding_custom_files': ['SMHD-CBOW-A-D-ADUsers-300.bin'],
															 'use_embeddings': [dn.UseEmbedding.NON_STATIC]})

				test_w2v_custom_emb(option, function, set_params)

		elif function == 'gloveCustom':
				if id_submodel_to_stacked == '15':
						set_params = dict({'filters_by_layer': [250],
															 'kernels_size': [3],
															 # 'epochs': [10],
															 # 'batch_sizes': [20],
															 'epochs': [40],
															 'batch_sizes': [5],
															 'dropouts': [0.2],
															 'embedding_types': [dn.EmbeddingType.GLOVE_CUSTOM],
															 'embedding_custom_files': ['SMHD-glove-A-D-ADUsers-300.pkl'],
															 'use_embeddings': [dn.UseEmbedding.NON_STATIC]})

				test_glove_custom_emb('1', function, set_params, 'CAD15')


if __name__ == '__main__':
		function = sys.argv[1]
		option = sys.argv[2]
		opt_test = sys.argv[3]

		# function = 'gloveCustom'
		# option = '15'
		# opt_test = 'out_test2'
		
		import tensorflow as tf
		from keras.backend import tensorflow_backend as K
		
		config = tf.ConfigProto(intra_op_parallelism_threads=4)
		config.graph_options.optimizer_options.global_jit_level = tf.OptimizerOptions.ON_1 #TF_XLA_FLAGS=--tf_xla_cpu_global_jit
		
		with tf.Session(config=config) as sess:
				K.set_session(sess)
		
				if opt_test == 'first_test':
						first_test(option, function)
				elif opt_test == 'out_test1':
						outperformance_test1(option, function)
				else: # 'out_test2
						outperformance_test2(option, function)
				
				K.clear_session(sess)