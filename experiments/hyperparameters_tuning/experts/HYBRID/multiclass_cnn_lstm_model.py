# Load libs standard python and custom
import numpy as np
import datetime
import sys

from keras.models import Sequential
from keras import regularizers
from keras.models import Model
from keras.layers import Dense, Dropout, Flatten
from keras.layers import LSTM, Conv1D, MaxPooling1D, AveragePooling1D, GlobalAveragePooling1D
from keras.layers import Embedding, TimeDistributed, Input
from network_model.model_class import ModelClass
from utils.experiment_processes import ExperimentProcesses

import utils.definition_network as dn

# Gera modelo lstm_exp9_var_L3_N16_B40_E32_D0.2 static glove com kernel_initializer=glorot_uniform=xavier_uniform
def generate_model(exp, name_model, kernel_function, set_params):
		# Configura pre-processamento dos dados para importação
		exp.pp_data.vocabulary_size = 5000
		
		exp.pp_data.embedding_size = 300  # 300 obrigatório se for usar word_embedding word2vec google neg300
		exp.pp_data.max_posts = 1750
		exp.pp_data.max_terms_by_post = 300
		exp.pp_data.binary_classifier = True
		exp.pp_data.format_input_data = dn.InputData.POSTS_ONLY_TEXT
		exp.pp_data.remove_stopwords = False
		exp.pp_data.delete_low_tfid = False
		exp.pp_data.min_df = 0
		exp.pp_data.min_tf = 0
		exp.pp_data.random_posts = False  # False = ordem cronológica
		exp.pp_data.random_users = False  # Não usada, as amostras são sempre random no validation k-fold
		exp.pp_data.tokenizing_type = 'WE'
		exp.pp_data.word_embedding_custom_file = ''
		exp.pp_data.embedding_type = dn.EmbeddingType.GLOVE_6B
		exp.pp_data.use_embedding = dn.UseEmbedding.STATIC
		exp.pp_data.word_embedding_custom_file = ''
		exp.pp_data.load_dataset_type = dn.LoadDataset.TRAIN_DATA_MODEL
		
		exp.pp_data.type_prediction_label = dn.TypePredictionLabel.MULTI_LABEL_CATEGORICAL
		
		exp.use_custom_metrics = False
		exp.use_valid_set_for_train = True
		exp.valid_split_from_train_set = 0.0
		exp.imbalanced_classes = False
		
		we_file_name = 'ET_' + str(exp.pp_data.embedding_type.value) + '_UE_' + str(exp.pp_data.use_embedding.value) + \
									 '_EF_' + 'glove6B300d' + kernel_function
		
		## Gera dados conforme configuração
		cnn_lstm = ModelClass(1)
		cnn_lstm.loss_function = 'binary_crossentropy'
		cnn_lstm.optmizer_function = 'adadelta'
		cnn_lstm.epochs = 15
		cnn_lstm.batch_size = 32
		cnn_lstm.patience_train = 10
		cnn_lstm.use_embedding_pre_train = exp.pp_data.use_embedding
		cnn_lstm.embed_trainable = (cnn_lstm.use_embedding_pre_train == (dn.UseEmbedding.RAND or dn.UseEmbedding.NON_STATIC))
		
		# set_params is empty
		if not bool(set_params):
				filters_by_layer = [32, 64, 128]
				neuronios_by_lstm_layer = [64, 128, 256]
				dropouts = [0.2, 0.5]
				dropouts_lstm = [0.2, 0.5]
		else:
				filters_by_layer = set_params['filters_by_layer']
				neuronios_by_lstm_layer = set_params['neuronios_by_lstm_layer']
				dropouts = set_params['dropouts']
				dropouts_lstm = set_params['dropouts_lstm']
		
		kernels_size = [5]
		epochs = [10]
		batch_sizes = [20]
		# Expected input batch shape: (batch_size, timesteps, data_dim)
		# Note that we have to provide the full batch_input_shape since the network is stateful.
		# the sample of index i in batch k is the follow-up for the sample i in batch k-1.
		np.random.seed(dn.SEED)
		
		time_ini_rep = datetime.datetime.now()
		x_train, y_train, x_valid, y_valid, num_words, embedding_matrix = exp.pp_data.load_data()
		exp.set_period_time_end(time_ini_rep, 'Load data')
		
		for filter in filters_by_layer:
				for kernel_size in kernels_size:
						for batch_size in batch_sizes:
								for epoch in epochs:
										for dropout in dropouts:
												for dropout_lstm in dropouts_lstm:
														for neuronios in neuronios_by_lstm_layer:
																cnn_lstm.epochs = epoch
																cnn_lstm.batch_size = batch_size
																cnn_lstm.patience_train = epoch
																exp.experiment_name = name_model + '_cnn_lstm' + '_F' + str(filter) + '_K' + \
																											str(kernel_size) + '_P' + 'None' + '_B' + str(batch_size) + \
																											'_E' + str(epoch) + '_D' + str(dropout) + '_HLN' + \
																											str(filter) + '_LSTM_N' + str(neuronios) + \
																											'_D'+ str(dropout_lstm) +	'_' + we_file_name
																
																cnn_lstm.model = Sequential()
																cnn_lstm.model.add(Embedding(exp.pp_data.vocabulary_size, exp.pp_data.embedding_size,
																												trainable=cnn_lstm.embed_trainable, name='emb_' + name_model))
																cnn_lstm.model.add(Dropout(dropout, name='dropout_1_' + name_model))
																cnn_lstm.model.add(Conv1D(filters=filter, kernel_size=kernel_size,
																										 kernel_initializer='glorot_uniform',
																										 # kernel_regularizer=regularizers.l2(0.03),
																										 padding='valid', activation='relu',
																										 name='conv_1_' + name_model))
																cnn_lstm.model.add(MaxPooling1D(name='max_pool_1_' + name_model))
																cnn_lstm.model.add(LSTM(neuronios,
																												activation='tanh', dropout=dropout_lstm,
																												recurrent_dropout=dropout_lstm,
																												return_sequences=True, name='lstm_1_' + name_model))
																cnn_lstm.model.add(LSTM(neuronios,
																												activation='tanh', dropout=dropout_lstm,
																												recurrent_dropout=dropout_lstm,
																												return_sequences=True, name='lstm_2_' + name_model))
																cnn_lstm.model.add(LSTM(neuronios,
																												activation='tanh', dropout=dropout_lstm,
																												recurrent_dropout=dropout_lstm,
																												name='lstm_3_' + name_model))
																cnn_lstm.model.add(Dense(3, activation='sigmoid', name='dense_1_' + name_model))
										
																time_ini_exp = datetime.datetime.now()
																exp.generate_model_hypeparams(cnn_lstm, x_train, y_train, x_valid, y_valid, embedding_matrix)
																exp.set_period_time_end(time_ini_exp, 'Total experiment')
		
		del x_train, y_train, x_valid, y_valid, num_words, embedding_matrix
		
		# Test
		exp.pp_data.load_dataset_type = dn.LoadDataset.TEST_DATA_MODEL
		np.random.seed(dn.SEED)
		time_ini_rep = datetime.datetime.now()
		x_test, y_test = exp.pp_data.load_data()
		exp.set_period_time_end(time_ini_rep, 'Load data')
		
		for filter in filters_by_layer:
				for kernel_size in kernels_size:
						for batch_size in batch_sizes:
								for epoch in epochs:
										for dropout in dropouts:
												for dropout_lstm in dropouts_lstm:
														for neuronios in neuronios_by_lstm_layer:
																cnn_lstm.epochs = epoch
																cnn_lstm.batch_size = batch_size
																cnn_lstm.patience_train = epoch
																exp.experiment_name = name_model + '_cnn_lstm' + '_F' + str(filter) + '_K' + \
																											str(kernel_size) + '_P' + 'None' + '_B' + str(batch_size) + \
																											'_E' + str(epoch) + '_D' + str(dropout) + '_HLN' + \
																											str(filter) + '_LSTM_N' + str(neuronios) + \
																											'_D'+ str(dropout_lstm) +	'_' + we_file_name
										
																cnn_lstm.model = exp.load_model(dn.PATH_PROJECT + exp.experiment_name + '.h5')
																exp.save_geral_configs('Experiment Specific Configuration: ' + exp.experiment_name)
																exp.save_summary_model(cnn_lstm.model)
																exp.predict_samples(cnn_lstm, x_test, y_test)
		
		del x_test, y_test, cnn_lstm, exp

def test(option):
		set_params = dict()

		if option == '1':
				print('Initializer experiment 1 (model SMHD_cnn_lstm_gl_880)\n' + \
							'Set: kernel_initializer=glorot_uniform=xavier_uniform, dataset=SMHD_880 multi-label')
				exp = ExperimentProcesses('cnn_L1_lstm_L3')
				exp.pp_data.set_dataset_source(dataset_name='SMHD', label_set=['control', 'anxiety', 'depression'],
																			 total_registers=880, subdirectory="anxiety,depression")
				generate_model(exp, 'SMHD_cnn_lstm_gl_880', '_glorot', set_params)

		elif option == '2':
				print('Initializer experiment 2 (model SMHD_cnn_lstm_gl_1040)\n' + \
							'Set: kernel_initializer=glorot_uniform=xavier_uniform, dataset=SMHD_1040 multi-label')
				exp = ExperimentProcesses('cnn_L1_lstm_L3')
				exp.pp_data.set_dataset_source(dataset_name='SMHD', label_set=['control', 'anxiety', 'depression'],
																			 total_registers=1040, subdirectory="anxiety")
				generate_model(exp, 'SMHD_cnn_lstm_gl_1040', '_glorot', set_params)

		elif option == '3':
				print('Initializer experiment 3 (model SMHD_cnn_lstm_gl_2160)\n' + \
							'Set: kernel_initializer=glorot_uniform=xavier_uniform, dataset=SMHD_2160 multi-label')
				exp = ExperimentProcesses('cnn_L1_lstm_L3')
				exp.pp_data.set_dataset_source(dataset_name='SMHD', label_set=['control', 'anxiety', 'depression'],
																			 total_registers=2160, subdirectory="depression")
				generate_model(exp, 'SMHD_cnn_lstm_gl_2160', '_glorot', set_params)

		else: # division test 4 to run in PCAD, with total time experiment not excceding 24h
				if option == '1.1.1':
						set_params.update({'filters_by_layer': [32],
															 'neuronios_by_lstm_layer': [64],
															 'dropouts': [0.2],
															 'dropouts_lstm': [0.2]})
				
				elif option == '1.1.2':
						set_params.update({'filters_by_layer': [32],
															 'neuronios_by_lstm_layer': [64],
															 'dropouts': [0.2],
															 'dropouts_lstm': [0.5]})
				
				elif option == '1.1.3':
						set_params.update({'filters_by_layer': [32],
															 'neuronios_by_lstm_layer': [64],
															 'dropouts': [0.5],
															 'dropouts_lstm': [0.2]})
				
				elif option == '1.1.4':
						set_params.update({'filters_by_layer': [32],
															 'neuronios_by_lstm_layer': [64],
															 'dropouts': [0.5],
															 'dropouts_lstm': [0.5]})
				
				elif option == '1.2.1':
						set_params.update({'filters_by_layer': [32],
															 'neuronios_by_lstm_layer': [128],
															 'dropouts': [0.2],
															 'dropouts_lstm': [0.2]})
				
				elif option == '1.2.2':
						set_params.update({'filters_by_layer': [32],
															 'neuronios_by_lstm_layer': [128],
															 'dropouts': [0.2],
															 'dropouts_lstm': [0.5]})
				
				elif option == '1.2.3':
						set_params.update({'filters_by_layer': [32],
															 'neuronios_by_lstm_layer': [128],
															 'dropouts': [0.5],
															 'dropouts_lstm': [0.2]})
				
				elif option == '1.2.4':
						set_params.update({'filters_by_layer': [32],
															 'neuronios_by_lstm_layer': [128],
															 'dropouts': [0.5],
															 'dropouts_lstm': [0.5]})
				
				elif option == '1.3.1':
						set_params.update({'filters_by_layer': [32],
															 'neuronios_by_lstm_layer': [256],
															 'dropouts': [0.2],
															 'dropouts_lstm': [0.2]})
				
				elif option == '1.3.2':
						set_params.update({'filters_by_layer': [32],
															 'neuronios_by_lstm_layer': [256],
															 'dropouts': [0.2],
															 'dropouts_lstm': [0.5]})
				
				elif option == '1.3.3':
						set_params.update({'filters_by_layer': [32],
															 'neuronios_by_lstm_layer': [256],
															 'dropouts': [0.5],
															 'dropouts_lstm': [0.2]})
				
				elif option == '1.3.4':
						set_params.update({'filters_by_layer': [32],
															 'neuronios_by_lstm_layer': [256],
															 'dropouts': [0.5],
															 'dropouts_lstm': [0.5]})
				
				elif option == '2.1.1':
						set_params.update({'filters_by_layer': [64],
															 'neuronios_by_lstm_layer': [64],
															 'dropouts': [0.2],
															 'dropouts_lstm': [0.2]})
				
				elif option == '2.1.2':
						set_params.update({'filters_by_layer': [64],
															 'neuronios_by_lstm_layer': [64],
															 'dropouts': [0.2],
															 'dropouts_lstm': [0.5]})
				
				elif option == '2.1.3':
						set_params.update({'filters_by_layer': [64],
															 'neuronios_by_lstm_layer': [64],
															 'dropouts': [0.5],
															 'dropouts_lstm': [0.2]})
				
				elif option == '2.1.4':
						set_params.update({'filters_by_layer': [64],
															 'neuronios_by_lstm_layer': [64],
															 'dropouts': [0.5],
															 'dropouts_lstm': [0.5]})
				
				elif option == '2.2.1':
						set_params.update({'filters_by_layer': [64],
															 'neuronios_by_lstm_layer': [128],
															 'dropouts': [0.2],
															 'dropouts_lstm': [0.2]})
				
				elif option == '2.2.2':
						set_params.update({'filters_by_layer': [64],
															 'neuronios_by_lstm_layer': [128],
															 'dropouts': [0.2],
															 'dropouts_lstm': [0.5]})
				
				elif option == '2.2.3':
						set_params.update({'filters_by_layer': [64],
															 'neuronios_by_lstm_layer': [128],
															 'dropouts': [0.5],
															 'dropouts_lstm': [0.2]})
				
				elif option == '2.2.4':
						set_params.update({'filters_by_layer': [64],
															 'neuronios_by_lstm_layer': [128],
															 'dropouts': [0.5],
															 'dropouts_lstm': [0.5]})
				
				elif option == '2.3.1':
						set_params.update({'filters_by_layer': [64],
															 'neuronios_by_lstm_layer': [256],
															 'dropouts': [0.2],
															 'dropouts_lstm': [0.2]})
				
				elif option == '2.3.2':
						set_params.update({'filters_by_layer': [64],
															 'neuronios_by_lstm_layer': [256],
															 'dropouts': [0.2],
															 'dropouts_lstm': [0.5]})
				
				elif option == '2.3.3':
						set_params.update({'filters_by_layer': [64],
															 'neuronios_by_lstm_layer': [256],
															 'dropouts': [0.5],
															 'dropouts_lstm': [0.2]})
				
				elif option == '2.3.4':
						set_params.update({'filters_by_layer': [64],
															 'neuronios_by_lstm_layer': [256],
															 'dropouts': [0.5],
															 'dropouts_lstm': [0.5]})
						
				elif option == '3.1.1':
						set_params.update({'filters_by_layer': [128],
															 'neuronios_by_lstm_layer': [128],
															 'dropouts': [0.2],
															 'dropouts_lstm': [0.2]})
				
				elif option == '3.1.2':
						set_params.update({'filters_by_layer': [128],
															 'neuronios_by_lstm_layer': [128],
															 'dropouts': [0.2],
															 'dropouts_lstm': [0.5]})
				
				elif option == '3.1.3':
						set_params.update({'filters_by_layer': [128],
															 'neuronios_by_lstm_layer': [128],
															 'dropouts': [0.5],
															 'dropouts_lstm': [0.2]})
				
				elif option == '3.1.4':
						set_params.update({'filters_by_layer': [128],
															 'neuronios_by_lstm_layer': [128],
															 'dropouts': [0.5],
															 'dropouts_lstm': [0.5]})
				
				elif option == '3.2.1':
						set_params.update({'filters_by_layer': [128],
															 'neuronios_by_lstm_layer': [192],
															 'dropouts': [0.2],
															 'dropouts_lstm': [0.2]})
				
				elif option == '3.2.2':
						set_params.update({'filters_by_layer': [128],
															 'neuronios_by_lstm_layer': [192],
															 'dropouts': [0.2],
															 'dropouts_lstm': [0.5]})
				
				elif option == '3.2.3':
						set_params.update({'filters_by_layer': [128],
															 'neuronios_by_lstm_layer': [192],
															 'dropouts': [0.5],
															 'dropouts_lstm': [0.2]})
				
				elif option == '3.2.4':
						set_params.update({'filters_by_layer': [128],
															 'neuronios_by_lstm_layer': [192],
															 'dropouts': [0.5],
															 'dropouts_lstm': [0.5]})
				
				elif option == '3.3.1':
						set_params.update({'filters_by_layer': [128],
															 'neuronios_by_lstm_layer': [256],
															 'dropouts': [0.2],
															 'dropouts_lstm': [0.2]})
				
				elif option == '3.3.2':
						set_params.update({'filters_by_layer': [128],
															 'neuronios_by_lstm_layer': [256],
															 'dropouts': [0.2],
															 'dropouts_lstm': [0.5]})
				
				elif option == '3.3.3':
						set_params.update({'filters_by_layer': [128],
															 'neuronios_by_lstm_layer': [256],
															 'dropouts': [0.5],
															 'dropouts_lstm': [0.2]})
				
				elif option == '3.3.4':
						set_params.update({'filters_by_layer': [128],
															 'neuronios_by_lstm_layer': [256],
															 'dropouts': [0.5],
															 'dropouts_lstm': [0.5]})
				
				print('Initializer experiment ' + option + ' (model SMHD_cnn_lstm_gl_2640)\n' + \
							'Set: kernel_initializer=glorot_uniform=xavier_uniform, dataset=SMHD_2640 multi-label')
				exp = ExperimentProcesses('cnn_L1_lstm_L3')
				exp.pp_data.set_dataset_source(dataset_name='SMHD', label_set=['control', 'anxiety', 'depression'],
																			 total_registers=2640, subdirectory="anx_dep_multilabel")
				generate_model(exp, 'SMHD_cnn_lstm_gl_2640', '_glorot', set_params)


if __name__ == '__main__':
		option = sys.argv[1]
		test(option)
