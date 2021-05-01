"""
		Stacked Ensemble
"""
import numpy as np
import pandas as pd
import datetime

from keras.callbacks import EarlyStopping
from keras.callbacks import ModelCheckpoint
from utils.experiment_processes import ExperimentProcesses
from utils.preprocess_data import PreprocessData
from utils.metrics import Metrics
from utils.log import Log
from network_model.model_class import ModelClass
from sklearn.utils import shuffle
from keras.utils import plot_model
from keras.models import Model
from keras.layers import Dense
from keras.layers.merge import concatenate
import pandas.io.json as pd_json
from keras.models import load_model

import utils.definition_network as dn

class StackedEnsemble:
		def __init__(self, _name_class, _verbose=0, _log_file_path=dn.PATH_PROJECT):
				self.verbose = _verbose
				self.name_class = _name_class
				self.all_files_path = _log_file_path

				# train model params
				self.submodels = dict({'CA': [1,2,3],
															 'CD': [1,2,3],
															 'CAD':[1,2,3]})
				self.ensemble_stacked_model = ModelClass(self.verbose)
				self.ensemble_stacked_conf = dict()
				self.k_fold = 10
				self.list_report_metrics = []
				self.job_name = ''
				self.metrics = Metrics()
				self.metrics_based_sample = False
				self.log = Log(self.all_files_path + self.name_class, "txt")
				self.type_int_weights = 'constant'
				self.labels_set = None
				self.labels_ensemble = None
				self.type_predict_label = dn.TypePredictionLabel.MULTI_LABEL_CATEGORICAL
				self.path_submodels = dn.PATH_PROJECT + "weak_classifiers/"
				self.dataset_train_path = 'dataset/anx_dep_multilabel/SMHD_test_train_1200.df'
				self.dataset_test_path = 'dataset/anx_dep_multilabel/SMHD_test_test_360.df'

				# Final model data
				self.all_submodels = dict()

				# Set print array to 5 places precision
				np.set_printoptions(precision=5)

		def set_network_params_ensemble_stack(self):
				self.ensemble_stacked_model.epochs = self.ensemble_stacked_conf['epochs']
				self.ensemble_stacked_model.batch_size = self.ensemble_stacked_conf['batch_size']
				self.ensemble_stacked_model.patience_train = self.ensemble_stacked_conf['patient_train']
				self.ensemble_stacked_model.activation_output_fn = self.ensemble_stacked_conf['activation_output_fn']
				self.ensemble_stacked_model.loss_function = self.ensemble_stacked_conf['loss_function']
				self.ensemble_stacked_model.optmizer_function = self.ensemble_stacked_conf['optmizer_function']
				self.ensemble_stacked_model.main_metric = [self.ensemble_stacked_conf['main_metric']]
				self.submodels = self.ensemble_stacked_conf['submodels']
				self.dataset_train_path = self.ensemble_stacked_conf['dataset_train_path']
				self.dataset_test_path = self.ensemble_stacked_conf['dataset_test_path']
				self.path_submodels = self.ensemble_stacked_conf['path_submodels']
				self.type_submodels = self.ensemble_stacked_conf['type_submodels']
				
				self.log.save('Set Network Test\n' + str(self.ensemble_stacked_conf))
				
		def save_metrics(self, title, model_name, y, y_hat, labels_set):
				final_metrics = self.metrics.calc_metrics_multilabel(y, y_hat, labels_set, self.type_predict_label,
																														 self.metrics_based_sample)
				self.log.save(title + str(' ') + model_name)
				self.log.save('Correct Prediction per Label: ' + str(final_metrics['Correct Prediction per Label']))
				self.log.save('Exact Match Ratio: ' + str(final_metrics['Exact Match Ratio']))
				self.log.save('Hamming Loss: ' + str(final_metrics['Hamming Loss']))
				self.log.save('Confusion Matrix: \n' + str(final_metrics['Multi-label Confusion Matrix']))
				self.log.save('=== Model Performance - Multi-label Metrics ===\n' + str(final_metrics['Multi-label Report']))
				self.log.save(
						'\n\n=== Model Performance - Single-label Metrics ===\n' + str(final_metrics['Single-label Report']))

				self.list_report_metrics.append(dict({'test': self.name_class,
																							'iteraction': self.job_name,
																							'model': model_name,
																							'CPLC': final_metrics['Correct Prediction per Label'][0],
																							'CPLA': final_metrics['Correct Prediction per Label'][1],
																							'CPLD': final_metrics['Correct Prediction per Label'][2],
																							'EMR': final_metrics['Exact Match Ratio'],
																							'HL': final_metrics['Hamming Loss'],
																							'metrics_multilabel': final_metrics['Multi-label Report Dict'],
																							'metrics_singlelabel': final_metrics['Single-label Report Dict']}))


		def save_metrics_to_pandas(self):
				data_pd = pd_json.json_normalize(self.list_report_metrics)
				data_pd.to_pickle(dn.PATH_PROJECT  + self.all_files_path + self.name_class + str('_metrics.df'))

		def set_period_time_end(self, time_ini, task_desc):
				time_end = datetime.datetime.now()
				period = time_end - time_ini
				self.log.save(task_desc +
											' - Ini: '+ str(time_ini.strftime("%Y-%m-%d %H:%M:%S")) +
											'\tEnd: '+ str(time_end.strftime("%Y-%m-%d %H:%M:%S")) +
											'\tTotal: '+ str(period))

		def load_train_dataset(self):
				train_df = pd.read_pickle(dn.PATH_PROJECT + self.dataset_train_path)
				return train_df
		
		def load_test_dataset(self):
				test_df = pd.read_pickle(dn.PATH_PROJECT + self.dataset_test_path)
				return test_df
		
		# vLO = version LSTM Only, submodels number between 1 and 3
		# For SMHD dep, anx e anx_dep lstm_exp9_var_L3_N16_B40_E32_D0.2 static glove, kernel_initializer=lecun_uniform
		def load_submodel_vLO(self, exp, name_model, kernel_name):
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
				exp.pp_data.random_posts = False  # False = chronological order
				exp.pp_data.random_users = False
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
											 '_EF_' + 'glove6B300d'
				
				## Load model according configuration
				lstm = ModelClass(1)
				lstm.loss_function = 'binary_crossentropy'
				lstm.optmizer_function = 'adam'
				lstm.epochs = 15
				lstm.batch_size = 32
				lstm.patience_train = 10
				lstm.use_embedding_pre_train = exp.pp_data.use_embedding
				lstm.embed_trainable = (lstm.use_embedding_pre_train == (dn.UseEmbedding.RAND or dn.UseEmbedding.NON_STATIC))
				
				neuronios_by_layer = [16]
				epochs = [32]
				batch_sizes = [40]
				dropouts = [0.2]
				
				for neuronios in neuronios_by_layer:
						for batch_size in batch_sizes:
								for epoch in epochs:
										for dropout in dropouts:
												lstm.epochs = epoch
												lstm.batch_size = batch_size
												lstm.patience_train = epoch / 2
												exp.experiment_name = name_model + '_lstm_exp9_var_L3' + '_N' + str(neuronios) + '_B' + str(
														batch_size) + '_E' + str(epoch) + '_D' + str(dropout) + '_' + we_file_name + kernel_name
				
				lstm.model = exp.load_model(self.path_submodels + exp.experiment_name + '.h5')
				
				return exp, lstm
		
		# Generated model lstm_exp9_var_L3_N16_B40_E32_D0.2 static custom glove A-D-AD
		# For SMHD anx
		def load_submodel_anx_vLO(self, exp, name_model, kernel_name):
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
				exp.pp_data.embedding_type = dn.EmbeddingType.GLOVE_CUSTOM
				exp.pp_data.use_embedding = dn.UseEmbedding.STATIC
				exp.pp_data.word_embedding_custom_file = 'SMHD-glove-A-D-ADUsers-300.pkl'
				exp.pp_data.type_prediction_label= dn.TypePredictionLabel.MULTI_LABEL_CATEGORICAL
				
				we_file_name = 'ET_' + str(exp.pp_data.embedding_type.value) + '_UE_' + str(exp.pp_data.use_embedding.value) + \
											 '_EF_' + exp.pp_data.word_embedding_custom_file.split('.')[0] + kernel_name
				
				lstm = ModelClass(self.verbose)
				lstm.loss_function = 'binary_crossentropy'
				lstm.optmizer_function = 'adam'
				lstm.epochs = 15
				lstm.batch_size = 32
				lstm.patience_train = 10
				lstm.use_embedding_pre_train = exp.pp_data.use_embedding
				lstm.embed_trainable = (lstm.use_embedding_pre_train == (dn.UseEmbedding.RAND or dn.UseEmbedding.NON_STATIC))
				
				neuronios_by_layer = [16]
				# epochs = [32]
				# batch_sizes = [40]
				# dropouts = [0.2]
				epochs = [96]
				batch_sizes = [20]
				dropouts = [0.1]

				for neuronios in neuronios_by_layer:
						for batch_size in batch_sizes:
								for epoch in epochs:
										for dropout in dropouts:
												lstm.epochs = epoch
												lstm.batch_size = batch_size
												lstm.patience_train = epoch / 2
												exp.experiment_name = name_model + '_lstm_exp9_var_L3' + '_N' + str(neuronios) + '_B' + str(
														batch_size) + '_E' + str(epoch) + '_D' + str(dropout) + '_' + we_file_name
				
				lstm.model = exp.load_model(self.path_submodels + exp.experiment_name + '.h5')
				
				return exp, lstm
		
		# Generated model lstm_exp9_var_L3_N16_B40_E32_D0.2 non-static custom w2v CBOW A-D-AD
		# For SMHD dep
		def load_submodel_dep_vLO(self, exp, name_model, kernel_name):
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
				exp.pp_data.embedding_type = dn.EmbeddingType.WORD2VEC_CUSTOM
				exp.pp_data.use_embedding = dn.UseEmbedding.NON_STATIC
				exp.pp_data.word_embedding_custom_file = 'SMHD-CBOW-A-D-ADUsers-300.bin'
				exp.pp_data.type_prediction_label= dn.TypePredictionLabel.MULTI_LABEL_CATEGORICAL
				
				we_file_name = 'ET_' + str(exp.pp_data.embedding_type.value) + '_UE_' + str(exp.pp_data.use_embedding.value) + \
											 '_EF_' + exp.pp_data.word_embedding_custom_file.split('.')[0] + kernel_name
				
				lstm = ModelClass(self.verbose)
				lstm.loss_function = 'binary_crossentropy'
				lstm.optmizer_function = 'adam'
				lstm.epochs = 15
				lstm.batch_size = 32
				lstm.patience_train = 10
				lstm.use_embedding_pre_train = exp.pp_data.use_embedding
				lstm.embed_trainable = (lstm.use_embedding_pre_train == (dn.UseEmbedding.RAND or dn.UseEmbedding.NON_STATIC))
				
				neuronios_by_layer = [16]
				# epochs = [32]
				# batch_sizes = [40]
				# dropouts = [0.2]
				epochs = [96]
				batch_sizes = [20]
				dropouts = [0.1]

				for neuronios in neuronios_by_layer:
						for batch_size in batch_sizes:
								for epoch in epochs:
										for dropout in dropouts:
												lstm.epochs = epoch
												lstm.batch_size = batch_size
												lstm.patience_train = epoch / 2
												exp.experiment_name = name_model + '_lstm_exp9_var_L3' + '_N' + str(neuronios) + '_B' + str(
														batch_size) + '_E' + str(epoch) + '_D' + str(dropout) + '_' + we_file_name
				
				lstm.model = exp.load_model(self.path_submodels + exp.experiment_name + '.h5')

				return exp, lstm

		# Generated model lstm_exp9_var_L3_N16_B40_E32_D0.2 non-static custom w2v CBOW AllUser
		# For SMHD anx_dep
		def load_submodel_anx_dep_vLO(self, exp, name_model, kernel_name):
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
				exp.pp_data.embedding_type = dn.EmbeddingType.WORD2VEC_CUSTOM
				exp.pp_data.use_embedding = dn.UseEmbedding.NON_STATIC
				exp.pp_data.word_embedding_custom_file = 'SMHD-CBOW-AllUsers-300.bin'
				exp.pp_data.type_prediction_label = dn.TypePredictionLabel.MULTI_LABEL_CATEGORICAL
				exp.pp_data.load_dataset_type = dn.LoadDataset.TRAIN_DATA_MODEL

				exp.use_custom_metrics = True
				exp.use_valid_set_for_train = True
				exp.valid_split_from_train_set = 0.0
				exp.imbalanced_classes = False

				we_file_name = 'ET_' + str(exp.pp_data.embedding_type.value) + '_UE_' + str(exp.pp_data.use_embedding.value) + \
											 '_EF_' + exp.pp_data.word_embedding_custom_file.split('.')[0] + kernel_name
				
				lstm = ModelClass(1)
				lstm.loss_function = 'binary_crossentropy'
				lstm.optmizer_function = 'adam'
				lstm.use_embedding_pre_train = exp.pp_data.use_embedding
				lstm.embed_trainable = (lstm.use_embedding_pre_train == (dn.UseEmbedding.RAND or dn.UseEmbedding.NON_STATIC))

				neuronios_by_layer = [16]
				# epochs = [32]
				# batch_sizes = [40]
				# dropouts = [0.2]
				epochs = [96]
				batch_sizes = [20]
				dropouts = [0.1]

				for neuronios in neuronios_by_layer:
						for batch_size in batch_sizes:
								for epoch in epochs:
										for dropout in dropouts:
												lstm.epochs = epoch
												lstm.batch_size = batch_size
												lstm.patience_train = epoch / 2
												exp.experiment_name = name_model + '_lstm_exp9_var_L3' + '_N' + str(neuronios) + '_B' + str(
														batch_size) + '_E' + str(epoch) + '_D' + str(dropout) + '_' + we_file_name

				lstm.model = exp.load_model(self.path_submodels + exp.experiment_name + '.h5')

				return exp, lstm

		# Generated model CNN, feel submodels of number 4
		def load_submodel_vCO(self, exp, name_model, kernel_name, set_params):
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
											 '_EF_' + 'glove6B300d' + kernel_name

				submodel = ModelClass(1)
				submodel.loss_function = 'binary_crossentropy'
				submodel.optmizer_function = 'adadelta'
				submodel.use_embedding_pre_train = exp.pp_data.use_embedding
				submodel.embed_trainable = (
										submodel.use_embedding_pre_train == (dn.UseEmbedding.RAND or dn.UseEmbedding.NON_STATIC))

				for filter in set_params['filters_by_layer']:
						for kernel_size in set_params['kernels_size']:
								for batch_size in set_params['batch_sizes']:
										for epoch in set_params['epochs']:
												for dropout in set_params['dropouts']:
														submodel.epochs = epoch
														submodel.batch_size = batch_size
														submodel.patience_train = epoch
														exp.experiment_name = name_model + '_cnn' + '_F' + str(filter) + '_K' + str(kernel_size) + \
																									'_P' + 'None' + '_B' + str(batch_size) + '_E' + \
																									str(epoch) + '_D' + str(dropout) + '_HLN' + str(filter) + '_' + \
																									we_file_name

				submodel.model = exp.load_model(self.path_submodels + exp.experiment_name + '.h5')

				return exp, submodel

		
		def load_submodel_vLCO_anx_dep(self, exp, name_model, kernel_name, set_params):
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

				exp.pp_data.embedding_type = dn.EmbeddingType.WORD2VEC_CUSTOM
				exp.pp_data.use_embedding = dn.UseEmbedding.NON_STATIC
				exp.pp_data.word_embedding_custom_file = 'SMHD-CBOW-AllUsers-300.bin'
				exp.pp_data.type_prediction_label = dn.TypePredictionLabel.MULTI_LABEL_CATEGORICAL
				exp.pp_data.load_dataset_type = dn.LoadDataset.TRAIN_DATA_MODEL
				
				exp.use_custom_metrics = False
				exp.use_valid_set_for_train = True
				exp.valid_split_from_train_set = 0.0
				exp.imbalanced_classes = False
				
				we_file_name = 'ET_' + str(exp.pp_data.embedding_type.value) + '_UE_' + str(exp.pp_data.use_embedding.value) + \
											 '_EF_' + exp.pp_data.word_embedding_custom_file.split('.')[0] + kernel_name
						
				submodel = ModelClass(1)
				submodel.loss_function = 'binary_crossentropy'
				submodel.optmizer_function = 'adadelta'
				submodel.use_embedding_pre_train = exp.pp_data.use_embedding
				submodel.embed_trainable = (
								submodel.use_embedding_pre_train == (dn.UseEmbedding.RAND or dn.UseEmbedding.NON_STATIC))
				
				kernels_size = set_params['kernels_size']
				epochs = set_params['epochs']
				batch_sizes = set_params['batch_sizes']
				
				for filter in set_params['filters_by_layer']:
						for kernel_size in kernels_size:
								for batch_size in batch_sizes:
										for epoch in epochs:
												for dropout in set_params['dropouts']:
														for dropout_lstm in set_params['dropouts_lstm']:
																for neuronios in set_params['neuronios_by_lstm_layer']:
																		submodel.epochs = epoch
																		submodel.batch_size = batch_size
																		submodel.patience_train = epoch
																		exp.experiment_name = name_model + '_cnn_lstm' + '_F' + str(filter) + '_K' + \
																													str(kernel_size) + '_P' + 'None' + '_B' + str(batch_size) + \
																													'_E' + str(epoch) + '_D' + str(dropout) + '_HLN' + \
																													str(filter) + '_LSTM_N' + str(neuronios) + \
																													'_D' + str(dropout_lstm) + '_' + we_file_name
				
				submodel.model = exp.load_model(self.path_submodels + exp.experiment_name + '.h5')
				
				return exp, submodel
		
		# Loaded model LSTM specialist in disorders with Glove 6B
		# A-D, D-AD
		def load_submodel_spec_lstm(self, set_params, function, use_custom_metrics=False, path_subdir='differenciators/lstm/'):
				# Configura pre-processamento dos dados para importação
				exp = ExperimentProcesses('lstm_exp9_var_L'+str(set_params['total_layers_lstm']))
				exp.pp_data.set_dataset_source(dataset_name='SMHD', label_set=['control', 'anxiety', 'depression'],
																			 total_registers=set_params['total_registers'],
																			 subdirectory=set_params['subdirectory'])

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
				exp.pp_data.random_posts = False  # False = chronological order
				exp.pp_data.random_users = False
				exp.pp_data.tokenizing_type = 'WE'
				exp.pp_data.type_prediction_label = dn.TypePredictionLabel.MULTI_LABEL_CATEGORICAL

				exp.use_custom_metrics = use_custom_metrics
				exp.use_valid_set_for_train = True
				exp.valid_split_from_train_set = 0.0
				exp.imbalanced_classes = False

				## Load model according configuration
				lstm = ModelClass(1)
				lstm.loss_function = 'binary_crossentropy'
				lstm.optmizer_function = 'adam'
				lstm.epochs = 15
				lstm.batch_size = 32
				lstm.patience_train = 10
				lstm.use_embedding_pre_train = exp.pp_data.use_embedding
				lstm.embed_trainable = (lstm.use_embedding_pre_train == (dn.UseEmbedding.RAND or dn.UseEmbedding.NON_STATIC))

				neuronios_by_layer = set_params['neuronios_by_layer']
				epochs = set_params['epochs']
				batch_sizes = set_params['batch_sizes']
				dropouts = set_params['dropouts']

				for embedding_type in set_params['embedding_types']:
						for embedding_custom_file in set_params['embedding_custom_files']:
								for use_embedding in set_params['use_embeddings']:
										exp.pp_data.embedding_type = embedding_type
										exp.pp_data.word_embedding_custom_file = embedding_custom_file
										exp.pp_data.use_embedding = use_embedding
										exp.pp_data.load_dataset_type = dn.LoadDataset.TEST_DATA_MODEL

										lstm.use_embedding_pre_train = exp.pp_data.use_embedding
										lstm.embed_trainable = (
														lstm.use_embedding_pre_train == (dn.UseEmbedding.RAND or dn.UseEmbedding.NON_STATIC))

										emb_name = function

										if embedding_custom_file != '':
												emb_name = exp.pp_data.word_embedding_custom_file.split('.')[0]

										we_file_name = 'ET_' + str(exp.pp_data.embedding_type.value) + '_UE_' + \
																	 str(exp.pp_data.use_embedding.value) + '_EF_' + emb_name + set_params['kernel_name']

										for neuronios in neuronios_by_layer:
												for batch_size in batch_sizes:
														for epoch in epochs:
																for dropout in dropouts:
																		lstm.epochs = epoch
																		lstm.batch_size = batch_size
																		lstm.patience_train = epoch / 2
																		exp.experiment_name = set_params['name_model'] + '_' + exp.experiment_name + '_N' + str(neuronios) + '_B' + str(
																				batch_size) + '_E' + str(epoch) + '_D' + str(dropout) + '_' + we_file_name
												
				lstm.model = exp.load_model(self.path_submodels + path_subdir + exp.experiment_name + '.h5')
				
				return exp, lstm


		# Loaded model CNN specialist in disorders
		def load_submodel_spec_cnn(self, option, set_params, function, path_subdir='differenciators/cnn/'):
				name_model = 't' + option + set_params['name_model']

				# Configura pre-processamento dos dados para importação
				exp = ExperimentProcesses('t' + option + '_cnn_L1')
				exp.pp_data.set_dataset_source(dataset_name='SMHD', label_set=['control', 'anxiety', 'depression'],
																			 total_registers=set_params['total_registers'],
																			 subdirectory= set_params['subdirectory'])

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
				exp.pp_data.type_prediction_label = dn.TypePredictionLabel.MULTI_LABEL_CATEGORICAL

				exp.use_custom_metrics = False
				exp.use_valid_set_for_train = True
				exp.valid_split_from_train_set = 0.0
				exp.imbalanced_classes = False

				## Gera dados conforme configuração
				cnn = ModelClass(1)
				cnn.loss_function = 'binary_crossentropy'
				cnn.optmizer_function = 'adadelta'

				filters_by_layer = set_params['filters_by_layer']
				kernels_size = set_params['kernels_size']
				epochs = set_params['epochs']
				batch_sizes = set_params['batch_sizes']
				dropouts = set_params['dropouts']

				# Test
				np.random.seed(dn.SEED)

				for embedding_type in set_params['embedding_types']:
						for embedding_custom_file in set_params['embedding_custom_files']:
								for use_embedding in set_params['use_embeddings']:
										exp.pp_data.embedding_type = embedding_type
										exp.pp_data.word_embedding_custom_file = embedding_custom_file
										exp.pp_data.use_embedding = use_embedding
										exp.pp_data.load_dataset_type = dn.LoadDataset.TEST_DATA_MODEL

										cnn.use_embedding_pre_train = exp.pp_data.use_embedding
										cnn.embed_trainable = (
														cnn.use_embedding_pre_train == (dn.UseEmbedding.RAND or dn.UseEmbedding.NON_STATIC))

										emb_name = function

										if embedding_custom_file != '':
												emb_name = exp.pp_data.word_embedding_custom_file.split('.')[0]

										we_file_name = 'ET_' + str(exp.pp_data.embedding_type.value) + '_UE_' + \
																	 str(exp.pp_data.use_embedding.value) + '_EF_' + emb_name + set_params['kernel_name']

										for filter in filters_by_layer:
												for kernel_size in kernels_size:
														for batch_size in batch_sizes:
																for epoch in epochs:
																		for dropout in dropouts:
																				cnn.epochs = epoch
																				cnn.batch_size = batch_size
																				cnn.patience_train = epoch
																				exp.experiment_name = name_model + '_cnn' + '_F' + str(filter) + '_K' + \
																															str(kernel_size) + '_P' + 'None' + '_B' + str(batch_size) + '_E' + \
																															str(epoch) + '_D' + str(dropout) + '_HLN' + str(filter) + '_' + \
																															we_file_name

				cnn.model = exp.load_model(self.path_submodels + path_subdir + exp.experiment_name + '.h5')

				return exp, cnn
		
		
		# Generated model LSTM_CNN, feel submodels of number 5
		def load_submodel_spec_hybrid(self, exp, name_model, kernel_name, set_params):
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
											 '_EF_' + 'glove6B300d' + kernel_name
				
				submodel = ModelClass(1)
				submodel.loss_function = 'binary_crossentropy'
				submodel.optmizer_function = 'adadelta'
				submodel.use_embedding_pre_train = exp.pp_data.use_embedding
				submodel.embed_trainable = (
								submodel.use_embedding_pre_train == (dn.UseEmbedding.RAND or dn.UseEmbedding.NON_STATIC))
				
				kernels_size = set_params['kernels_size']
				epochs = set_params['epochs']
				batch_sizes = set_params['batch_sizes']
				
				for filter in set_params['filters_by_layer']:
						for kernel_size in kernels_size:
								for batch_size in batch_sizes:
										for epoch in epochs:
												for dropout in set_params['dropouts']:
														for dropout_lstm in set_params['dropouts_lstm']:
																for neuronios in set_params['neuronios_by_lstm_layer']:
																		submodel.epochs = epoch
																		submodel.batch_size = batch_size
																		submodel.patience_train = epoch
																		exp.experiment_name = name_model + '_cnn_lstm' + '_F' + str(filter) + '_K' + \
																													str(kernel_size) + '_P' + 'None' + '_B' + str(batch_size) + \
																													'_E' + str(epoch) + '_D' + str(dropout) + '_HLN' + \
																													str(filter) + '_LSTM_N' + str(neuronios) + \
																													'_D' + str(dropout_lstm) + '_' + we_file_name
				
				submodel.model = exp.load_model(self.path_submodels + exp.experiment_name + '.h5')
				
				return exp, submodel

		
		def rename_layers_diff_submodels(self, exp, submodel, sufix_name, path_files):
				for i, layer in enumerate(submodel.model.layers):
						layer.name = layer.name + '_' + str(sufix_name)

				submodel.model.save(self.path_submodels + path_files + exp.experiment_name + '.h5')


		# Submodels multilabel, multiclass
		def multi_label_model(self,	set_params):
				exp = ExperimentProcesses(set_params['log_file'])
				exp.pp_data.set_dataset_source(dataset_name='SMHD', label_set=['control', 'anxiety', 'depression'],
																			 total_registers=2640, subdirectory="anx_dep_multilabel")
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
				exp.pp_data.embedding_type = set_params['embedding_type']
				exp.pp_data.word_embedding_custom_file = set_params['embedding_custom_file']
				exp.pp_data.use_embedding = set_params['use_embedding']
				exp.pp_data.load_dataset_type = dn.LoadDataset.TEST_DATA_MODEL

				model_ml = ModelClass(1)
				model_ml.loss_function = 'binary_crossentropy'
				model_ml.optmizer_function = dn.OPTIMIZER_FUNCTIONS[set_params['optimizer_function']]
				model_ml.epochs = set_params['epochs']
				model_ml.batch_size = set_params['batch_size']
				model_ml.patience_train = set_params['epochs'] / 2

				model_ml.use_embedding_pre_train = exp.pp_data.use_embedding
				model_ml.embed_trainable = (
				model_ml.use_embedding_pre_train == (dn.UseEmbedding.RAND or dn.UseEmbedding.NON_STATIC))

				emb_name = set_params['function']
				if set_params['embedding_custom_file'] != '':
						emb_name = exp.pp_data.word_embedding_custom_file.split('.')[0]

				we_file_name = 'ET_' + str(exp.pp_data.embedding_type.value) + '_UE_' + str(exp.pp_data.use_embedding.value) +\
											 '_EF_' + emb_name + '_' + set_params['kernel_function'].split('_')[0]

				if set_params['model_type'] == 'LSTM':
						exp.experiment_name = set_params['name_model'] + '_lstm_exp9_var_L3' + '_N' +\
																	str(set_params['neuronios_by_layer']) +	'_B' + str(set_params['batch_size']) + '_E' +\
																	str(set_params['epochs']) + '_D' + str(set_params['dropouts']) + '_OF' +\
																	model_ml.optmizer_function + '_HL' + str(set_params['hidden_layer']) + '_' +\
																	we_file_name

				elif set_params['model_type'] == 'CNN':
						exp.experiment_name = set_params['name_model'] + '_cnn' + '_F' + str(set_params['filter_by_layer']) + '_K' +\
																	str(set_params['kernel_size']) + '_P' + 'None' + '_B' +\
																	str(set_params['batch_size']) + '_E' + str(set_params['epochs']) + '_D' +\
																	str(set_params['dropouts']) + '_HLN' + str(set_params['filter_by_layer']) + '_' +\
																	we_file_name

				else:
						exp.experiment_name = set_params['name_model'] + '_cnn_lstm_F' + str(set_params['filter_by_layer']) +\
																	'_K' + str(set_params['kernel_size']) + '_P' + 'None' + '_B' + \
																	str(set_params['batch_size']) + '_E' + str(set_params['epochs']) + '_D' +\
																	str(set_params['dropouts']) + '_HLN' + str(set_params['filter_by_layer']) +\
																	'_LSTM_N' + str(set_params['neuronios_by_lstm_layer']) + '_D'\
																	+ str(set_params['dropouts_lstm']) + '_' + we_file_name

				model_ml.model = exp.load_model(self.path_submodels + 'multitask_models/' +set_params['model_type'] + '/' +\
																				exp.experiment_name + '.h5')
				return exp, model_ml


		# Load submodel accordance options
		def load_submodels_1(self, all_models, key_model):
				if key_model == 'CA':
						self.log.save('Load model SMHD_anx_1')
						exp = ExperimentProcesses('lstm_exp9_var_L3')
						exp.pp_data.set_dataset_source(dataset_name='SMHD', label_set=['control', 'anxiety', 'depression'],
																					 total_registers=1040, subdirectory="anxiety")
						exp, lstm = self.load_submodel_vLO(exp, 'SMHD_ml_gl_1040', '_glorot')
						all_models.update({'CA1': {'exp': exp, 'model_class': lstm}})
						del exp, lstm
				elif key_model == 'CD':
						self.log.save('Load model SMHD_dep_1')
						exp = ExperimentProcesses('lstm_exp9_var_L3')
						exp.pp_data.set_dataset_source(dataset_name='SMHD', label_set=['control', 'anxiety', 'depression'],
																					 total_registers=2160, subdirectory="depression")
						exp, lstm = self.load_submodel_vLO(exp, 'SMHD_ml_gl_2160', '_glorot')
						all_models.update({'CD1': {'exp': exp, 'model_class': lstm}})
						del exp, lstm
				else:
						self.log.save('Load model SMHD_anx_dep_1')
						exp = ExperimentProcesses('lstm_exp9_var_L3')
						exp.pp_data.set_dataset_source(dataset_name='SMHD', label_set=['control', 'anxiety', 'depression'],
																					 total_registers=880, subdirectory="anxiety,depression")
						exp, lstm = self.load_submodel_vLO(exp, 'SMHD_ml_gl_880', '_glorot')
						all_models.update({'CAD1': {'exp': exp, 'model_class': lstm}})
						del exp, lstm

				return all_models

		def load_submodels_2(self, all_models, key_model):
				if key_model == 'CA':
						self.log.save('Load model SMHD_anx_2')
						exp = ExperimentProcesses('lstm_exp9_var_L3')
						exp.pp_data.set_dataset_source(dataset_name='SMHD', label_set=['control', 'anxiety', 'depression'],
																					 total_registers=1040)
						exp, lstm = self.load_submodel_vLO(exp, 'SMHD_ml_le_1040', '_lecun')
						all_models.update({'CA2': {'exp': exp, 'model_class': lstm}})
						del exp, lstm
				elif key_model == 'CD':
						self.log.save('Load model SMHD_dep_2')
						exp = ExperimentProcesses('lstm_exp9_var_L3')
						exp.pp_data.set_dataset_source(dataset_name='SMHD', label_set=['control', 'anxiety', 'depression'],
																					 total_registers=2160, subdirectory="depression")
						exp, lstm = self.load_submodel_vLO(exp, 'SMHD_ml_le_2160', '_lecun')
						all_models.update({'CD2': {'exp': exp, 'model_class': lstm}})
						del exp, lstm
				else:
						self.log.save('Load model SMHD_anx_dep_2')
						exp = ExperimentProcesses('lstm_exp9_var_L3')
						exp.pp_data.set_dataset_source(dataset_name='SMHD', label_set=['control', 'anxiety', 'depression'],
																					 total_registers=880, subdirectory="anxiety,depression")
						exp, lstm = self.load_submodel_vLO(exp, 'SMHD_ml_le_880', '_lecun')
						all_models.update({'CAD2': {'exp': exp, 'model_class': lstm}})
						del exp, lstm

				return all_models

		def load_submodels_3(self, all_models, key_model):
				if key_model == 'CA':
						self.log.save('Load model SMHD_anx_3')
						exp = ExperimentProcesses('lstm_exp9_var_L3')
						exp.pp_data.set_dataset_source(dataset_name='SMHD', label_set=['control', 'anxiety', 'depression'],
																					 total_registers=1040)
						# exp, lstm = self.load_submodel_anx_vLO(exp, 'SMHD_ml_gl_1040', '_glorot')
						exp, lstm = self.load_submodel_anx_vLO(exp, 'SMHD_ml_gl_1040_gloveadad', '_glorot')
						all_models.update({'CA3': {'exp': exp, 'model_class': lstm}})
						del exp, lstm
				elif key_model == 'CD':
						self.log.save('Load model SMHD_dep_3')
						exp = ExperimentProcesses('lstm_exp9_var_L3')
						exp.pp_data.set_dataset_source(dataset_name='SMHD', label_set=['control', 'anxiety', 'depression'],
																					 total_registers=2160, subdirectory="depression")
						# exp, lstm = self.load_submodel_dep_vLO(exp, 'SMHD_ml_gl_2160', '_glorot')
						exp, lstm = self.load_submodel_dep_vLO(exp, 'SMHD_ml_gl_2160_cbow_adad', '_glorot')
						all_models.update({'CD3': {'exp': exp, 'model_class': lstm}})
						del exp, lstm
				else:
						self.log.save('Load model SMHD_anx_dep_3')
						exp = ExperimentProcesses('lstm_exp9_var_L3')
						exp.pp_data.set_dataset_source(dataset_name='SMHD', label_set=['control', 'anxiety', 'depression'],
																					 total_registers=880, subdirectory="anxiety,depression")
						# exp, lstm = self.load_submodel_anx_dep_vLO(exp, 'SMHD_ml_gl_880', '_glorot')
						exp, lstm = self.load_submodel_anx_dep_vLO(exp, 'SMHD_ml_gl_880_cbow_alluser', '_glorot')
						all_models.update({'CAD3': {'exp': exp, 'model_class': lstm}})
						del exp, lstm

				return all_models

		def load_submodels_cnn(self, all_models, key_model, id_model):
				if key_model == 'CA':
						self.log.save('Load model SMHD_cnn_gl_1040 CA' + str(id_model))
						if id_model == 4:
								set_params = dict({'filters_by_layer': [250],
																	 'kernels_size': [5],
																	 'epochs': [10],
																	 'batch_sizes': [20],
																	 'dropouts': [0.5]})
								exp = ExperimentProcesses('cnn_L1')
								exp.pp_data.set_dataset_source(dataset_name='SMHD', label_set=['control', 'anxiety', 'depression'],
																							 total_registers=1040, subdirectory="anxiety")
								
								exp, submodel = self.load_submodel_vCO(exp, 'SMHD_cnn_gl_1040', '_glorot', set_params)
								
						elif id_model == 101: # t2_gt_SMHD_cnn_gl_1040_cnn_F100_K5_PNone_B20_E10_D0.5_HLN100_ET_3_UE_3_EF_gloveTwitter_glorot
								set_params = dict({'filters_by_layer': [100],
																	 'kernels_size': [5],
																	 'batch_sizes': [20],
																	 'epochs': [10],
																	 'dropouts': [0.5],
																	 'kernel_name': '_glorot',
																	 'total_registers': 1040,
																	 'subdirectory': 'anxiety',
																	 'name_model': '_gt_SMHD_cnn_gl_1040',
																	 'embedding_types': [dn.EmbeddingType.GLOVE_TWITTER],
																	 'embedding_custom_files': [''],
																	 'use_embeddings': [dn.UseEmbedding.STATIC]})
								exp, submodel = self.load_submodel_spec_cnn('2', set_params, 'gloveTwitter', path_subdir='')

						else: #102 t2_wc_SMHD_cnn_gl_1040_cnn_F250_K4_PNone_B20_E10_D0.2_HLN250_ET_6_UE_3_EF_SMHD-Skipgram-AllUsers-300_glorot
								set_params = dict({'filters_by_layer': [250],
																	 'kernels_size': [4],
																	 'batch_sizes': [20],
																	 'epochs': [10],
																	 'dropouts': [0.2],
																	 'kernel_name': '_glorot',
																	 'total_registers': 1040,
																	 'subdirectory': 'anxiety',
																	 'name_model': '_wc_SMHD_cnn_gl_1040',
																	 'embedding_types': [dn.EmbeddingType.WORD2VEC_CUSTOM],
																	 'embedding_custom_files': ['SMHD-Skipgram-AllUsers-300.bin'],
																	 'use_embeddings': [dn.UseEmbedding.STATIC]})
								exp, submodel = self.load_submodel_spec_cnn('2', set_params, '', path_subdir='')
						
				elif key_model == 'CD':
						self.log.save('Load model SMHD_cnn_gl_2160 CD'+ str(id_model))
						if id_model == 4:
								set_params = dict({'filters_by_layer': [250],
																	 'kernels_size': [4],
																	 'epochs': [10],
																	 'batch_sizes': [20],
																	 'dropouts': [0.2]})
								exp = ExperimentProcesses('lstm_exp9_var_L3')
								exp.pp_data.set_dataset_source(dataset_name='SMHD', label_set=['control', 'anxiety', 'depression'],
																							 total_registers=2160, subdirectory="depression")
								exp, submodel = self.load_submodel_vCO(exp, 'SMHD_cnn_gl_2160', '_glorot', set_params)
						
						elif id_model == 101: # t3_gt_SMHD_cnn_gl_2160_cnn_F250_K3_PNone_B20_E10_D0.5_HLN250_ET_3_UE_3_EF_gloveTwitter_glorot
								set_params = dict({'filters_by_layer': [250],
																	 'kernels_size': [3],
																	 'batch_sizes': [20],
																	 'epochs': [10],
																	 'dropouts': [0.5],
																	 'kernel_name': '_glorot',
																	 'total_registers': 2160,
																	 'subdirectory': 'depression',
																	 'name_model': '_gt_SMHD_cnn_gl_2160',
																	 'embedding_types': [dn.EmbeddingType.GLOVE_TWITTER],
																	 'embedding_custom_files': [''],
																	 'use_embeddings': [dn.UseEmbedding.STATIC]})
								exp, submodel = self.load_submodel_spec_cnn('3', set_params, 'gloveTwitter', path_subdir='')

						else:  # 102 t3_wc_SMHD_cnn_gl_2160_cnn_F100_K3_PNone_B20_E10_D0.2_HLN100_ET_6_UE_3_EF_SMHD-Skipgram-A-D-ADUsers-300_glorot
								set_params = dict({'filters_by_layer': [100],
																	 'kernels_size': [3],
																	 'batch_sizes': [20],
																	 'epochs': [10],
																	 'dropouts': [0.2],
																	 'kernel_name': '_glorot',
																	 'total_registers': 2160,
																	 'subdirectory': 'depression',
																	 'name_model': '_wc_SMHD_cnn_gl_2160',
																	 'embedding_types': [dn.EmbeddingType.WORD2VEC_CUSTOM],
																	 'embedding_custom_files': ['SMHD-Skipgram-A-D-ADUsers-300.bin'],
																	 'use_embeddings': [dn.UseEmbedding.STATIC]})
								exp, submodel = self.load_submodel_spec_cnn('3', set_params, '', path_subdir='')
								
				else:
						self.log.save('Load model SMHD_cnn_gl_880 CAD'+ str(id_model))
						if id_model == 4:
								set_params = dict({'filters_by_layer': [100],
																	 'kernels_size': [4],
																	 'epochs': [10],
																	 'batch_sizes': [20],
																	 'dropouts': [0.5]})
								exp = ExperimentProcesses('lstm_exp9_var_L3')
								exp.pp_data.set_dataset_source(dataset_name='SMHD', label_set=['control', 'anxiety', 'depression'],
																							 total_registers=880, subdirectory="anxiety,depression")
								exp, submodel = self.load_submodel_vCO(exp, 'SMHD_cnn_gl_880', '_glorot', set_params)

						elif id_model == 100: # t1_gn_SMHD_cnn_gl_880_cnn_F100_K5_PNone_B20_E10_D0.5_HLN100_ET_4_UE_3_EF_googleNews_glorot
								set_params = dict({'filters_by_layer': [100],
																	 'kernels_size': [5],
																	 'batch_sizes': [20],
																	 'epochs': [10],
																	 'dropouts': [0.5],
																	 'kernel_name': '_glorot',
																	 'total_registers': 880,
																	 'subdirectory': 'anxiety,depression',
																	 'name_model': '_gn_SMHD_cnn_gl_880',
																	 'embedding_types': [dn.EmbeddingType.WORD2VEC],
																	 'embedding_custom_files': [''],
																	 'use_embeddings': [dn.UseEmbedding.STATIC]})
								exp, submodel = self.load_submodel_spec_cnn('1', set_params, 'googleNews', path_subdir='')
						
						elif id_model == 101:  # t1_gt_SMHD_cnn_gl_880_cnn_F250_K3_PNone_B20_E10_D0.5_HLN250_ET_3_UE_3_EF_gloveTwitter_glorot
								set_params = dict({'filters_by_layer': [250],
																	 'kernels_size': [3],
																	 'batch_sizes': [20],
																	 'epochs': [10],
																	 'dropouts': [0.5],
																	 'kernel_name': '_glorot',
																	 'total_registers': 880,
																	 'subdirectory': 'anxiety,depression',
																	 'name_model': '_gt_SMHD_cnn_gl_880',
																	 'embedding_types': [dn.EmbeddingType.GLOVE_TWITTER],
																	 'embedding_custom_files': [''],
																	 'use_embeddings': [dn.UseEmbedding.STATIC]})
								exp, submodel = self.load_submodel_spec_cnn('1', set_params, 'gloveTwitter', path_subdir='')
						
						else:  #102 t1_wc_SMHD_cnn_gl_880_cnn_F250_K5_PNone_B20_E10_D0.2_HLN250_ET_6_UE_3_EF_SMHD-Skipgram-A-D-ADUsers-300_glorot
								set_params = dict({'filters_by_layer': [250],
																	 'kernels_size': [5],
																	 'batch_sizes': [20],
																	 'epochs': [10],
																	 'dropouts': [0.2],
																	 'kernel_name': '_glorot',
																	 'total_registers': 880,
																	 'subdirectory': 'anxiety,depression',
																	 'name_model': '_wc_SMHD_cnn_gl_880',
																	 'embedding_types': [dn.EmbeddingType.WORD2VEC_CUSTOM],
																	 'embedding_custom_files': ['SMHD-Skipgram-A-D-ADUsers-300.bin'],
																	 'use_embeddings': [dn.UseEmbedding.STATIC]})
								exp, submodel = self.load_submodel_spec_cnn('1', set_params, '', path_subdir='')
				
				all_models.update({key_model + str(id_model): {'exp': exp, 'model_class': submodel}})
				del exp, submodel
				
				return all_models

		def load_submodels_hybrid(self, all_models, key_model, id_model):
				self.log.save('Load model SMHD_cnn_lstm_gl_'+	key_model + str(id_model))
				if key_model == 'CA':
						if id_model == 5: # SMHD_cnn_lstm_gl_1040_cnn_lstm_F128_K5_PNone_B20_E25_D0.2_HLN128_LSTM_N64_D0.2_ET_2_UE_3_EF_glove6B300d_glorot
								set_params = dict({'filters_by_layer': [128],
																	 'kernels_size': [5],
																	 'dropouts': [0.2],
																	 'neuronios_by_lstm_layer': [64],
																	 'dropouts_lstm': [0.2],
																	 'epochs': [25],
																	 'batch_sizes': [20]})
								exp = ExperimentProcesses('cnn_L1')
								exp.pp_data.set_dataset_source(dataset_name='SMHD', label_set=['control', 'anxiety', 'depression'],
																							 total_registers=1040, subdirectory="anxiety")
								exp, submodel = self.load_submodel_spec_hybrid(exp, 'SMHD_cnn_lstm_gl_1040', '_glorot', set_params)
						
						elif id_model == 110: # t6_SMHD_cnn_lstm_gl_1040_cnn_lstm_F128_K4_PNone_B20_E50_D0.2_HLN128_LSTM_N64_D0.2_ET_2_UE_3_EF_glove6B300d_glorot
								set_params = dict({'filters_by_layer': [128],
																	 'kernels_size': [4],
																	 'dropouts': [0.2],
																	 'neuronios_by_lstm_layer': [64],
																	 'dropouts_lstm': [0.2],
																	 'epochs': [50],
																	 'batch_sizes': [20]})
								exp = ExperimentProcesses('cnn_L1')
								exp.pp_data.set_dataset_source(dataset_name='SMHD', label_set=['control', 'anxiety', 'depression'],
																							 total_registers=1040, subdirectory="anxiety")
								exp, submodel = self.load_submodel_spec_hybrid(exp, 't6_SMHD_cnn_lstm_gl_1040', '_glorot', set_params)
						
						else: #111 SMHD_cnn_lstm_gl_1040_cnn_lstm_F32_K5_PNone_B20_E10_D0.2_HLN32_LSTM_N256_D0.2_ET_2_UE_3_EF_glove6B300d_glorot
								set_params = dict({'filters_by_layer': [32],
																	 'kernels_size': [5],
																	 'dropouts': [0.2],
																	 'neuronios_by_lstm_layer': [256],
																	 'dropouts_lstm': [0.2],
																	 'epochs': [10],
																	 'batch_sizes': [20]})
								exp = ExperimentProcesses('cnn_L1')
								exp.pp_data.set_dataset_source(dataset_name='SMHD', label_set=['control', 'anxiety', 'depression'],
																							 total_registers=1040, subdirectory="anxiety")
								exp, submodel = self.load_submodel_spec_hybrid(exp, 'SMHD_cnn_lstm_gl_1040', '_glorot', set_params)
						
				elif key_model == 'CD':
						if id_model == 5: # SMHD_cnn_lstm_gl_2160_cnn_lstm_F128_K5_PNone_B20_E20_D0.5_HLN128_LSTM_N128_D0.2_ET_2_UE_3_EF_glove6B300d_glorot
								set_params = dict({'filters_by_layer': [128],
																	 'kernels_size': [5],
																	 'dropouts': [0.5],
																	 'neuronios_by_lstm_layer': [128],
																	 'dropouts_lstm': [0.2],
																	 'epochs': [20],
																	 'batch_sizes': [20]})
								exp = ExperimentProcesses('lstm_exp9_var_L3')
								exp.pp_data.set_dataset_source(dataset_name='SMHD', label_set=['control', 'anxiety', 'depression'],
																							 total_registers=2160, subdirectory="depression")
								exp, submodel = self.load_submodel_spec_hybrid(exp, 'SMHD_cnn_lstm_gl_2160', '_glorot', set_params)
						
						elif id_model == 110: # t23_SMHD_cnn_lstm_gl_2160_cnn_lstm_F128_K5_PNone_B20_E50_D0.5_HLN128_LSTM_N128_D0.2_ET_2_UE_3_EF_glove6B300d_glorot
								set_params = dict({'filters_by_layer': [128],
																	 'kernels_size': [5],
																	 'dropouts': [0.5],
																	 'neuronios_by_lstm_layer': [128],
																	 'dropouts_lstm': [0.2],
																	 'epochs': [50],
																	 'batch_sizes': [20]})
								exp = ExperimentProcesses('lstm_exp9_var_L3')
								exp.pp_data.set_dataset_source(dataset_name='SMHD', label_set=['control', 'anxiety', 'depression'],
																							 total_registers=2160, subdirectory="depression")
								exp, submodel = self.load_submodel_spec_hybrid(exp, 't23_SMHD_cnn_lstm_gl_2160', '_glorot', set_params)
						
						else:  # 111 SMHD_cnn_lstm_gl_2160_cnn_lstm_F128_K5_PNone_B20_E10_D0.5_HLN128_LSTM_N128_D0.2_ET_2_UE_3_EF_glove6B300d_glorot
								set_params = dict({'filters_by_layer': [128],
																	 'kernels_size': [5],
																	 'dropouts': [0.5],
																	 'neuronios_by_lstm_layer': [128],
																	 'dropouts_lstm': [0.2],
																	 'epochs': [10],
																	 'batch_sizes': [20]})
								exp = ExperimentProcesses('lstm_exp9_var_L3')
								exp.pp_data.set_dataset_source(dataset_name='SMHD', label_set=['control', 'anxiety', 'depression'],
																							 total_registers=2160, subdirectory="depression")
								exp, submodel = self.load_submodel_spec_hybrid(exp, 'SMHD_cnn_lstm_gl_2160', '_glorot', set_params)
				
				else:
						if id_model == 5: #SMHD_cnn_lstm_gl_880_cnn_lstm_F32_K5_PNone_B20_E10_D0.2_HLN32_LSTM_N256_D0.5_ET_2_UE_3_EF_glove6B300d_glorot
								set_params = dict({'filters_by_layer': [32],
																	 'kernels_size': [5],
																	 'dropouts': [0.2],
																	 'neuronios_by_lstm_layer': [256],
																	 'dropouts_lstm': [0.5],
																	 'epochs': [10],
																	 'batch_sizes': [20]})
								exp = ExperimentProcesses('lstm_exp9_var_L3')
								exp.pp_data.set_dataset_source(dataset_name='SMHD', label_set=['control', 'anxiety', 'depression'],
																							 total_registers=880, subdirectory="anxiety,depression")
								exp, submodel = self.load_submodel_spec_hybrid(exp, 'SMHD_cnn_lstm_gl_880', '_glorot', set_params)

						elif id_model == 110: # t63_SMHD_cnn_lstm_gl_880_cnn_lstm_F64_K5_PNone_B20_E15_D0.5_HLN64_LSTM_N256_D0.2_ET_2_UE_3_EF_glove6B300d_glorot
								set_params = dict({'filters_by_layer': [64],
																	 'kernels_size': [5],
																	 'dropouts': [0.5],
																	 'neuronios_by_lstm_layer': [256],
																	 'dropouts_lstm': [0.2],
																	 'epochs': [15],
																	 'batch_sizes': [20]})
								exp = ExperimentProcesses('lstm_exp9_var_L3')
								exp.pp_data.set_dataset_source(dataset_name='SMHD', label_set=['control', 'anxiety', 'depression'],
																							 total_registers=880, subdirectory="anxiety,depression")
								exp, submodel = self.load_submodel_spec_hybrid(exp, 't63_SMHD_cnn_lstm_gl_880', '_glorot', set_params)

						else:  # 111 SMHD_cnn_lstm_gl_880_cnn_lstm_F64_K5_PNone_B20_E10_D0.2_HLN64_LSTM_N256_D0.2_ET_2_UE_3_EF_glove6B300d_glorot
								set_params = dict({'filters_by_layer': [64],
																	 'kernels_size': [5],
																	 'dropouts': [0.2],
																	 'neuronios_by_lstm_layer': [256],
																	 'dropouts_lstm': [0.2],
																	 'epochs': [10],
																	 'batch_sizes': [20]})
						
								exp = ExperimentProcesses('lstm_exp9_var_L3')
								exp.pp_data.set_dataset_source(dataset_name='SMHD', label_set=['control', 'anxiety', 'depression'],
																							 total_registers=880, subdirectory="anxiety,depression")
								exp, submodel = self.load_submodel_spec_hybrid(exp, 'SMHD_cnn_lstm_gl_880', '_glorot', set_params)
						

				all_models.update({key_model+str(id_model): {'exp': exp, 'model_class': submodel}})
				del exp, submodel

				return all_models

		def load_submodels_6_to_11(self, all_models, id_model):
				if id_model == 6: # A-D t2
						self.log.save('Load model SMHD_ml_le_1040_A_D 6')
						
						set_params = dict({'neuronios_by_layer': [16],
															 'epochs': [32],
															 'batch_sizes': [20],
															 'dropouts': [0.1],
															 'total_layers_lstm': 3,
															 'kernel_name': '_lecun',
															 'total_registers': 1040,
															 'subdirectory': 'only_disorders/A_D',
															 'name_model': 'SMHD_ml_le_1040_A_D',
															 'embedding_types': [dn.EmbeddingType.GLOVE_6B],
															 'embedding_custom_files': [''],
															 'use_embeddings': [dn.UseEmbedding.STATIC]})
						exp, submodel = self.load_submodel_spec_lstm(set_params, 'glove6B300d')

				elif id_model == 9: # A-D t6
						self.log.save('Load model SMHD_ml_le_1040_A_D 9')
						
						set_params = dict({'neuronios_by_layer': [16],
															 'epochs': [96],
															 'batch_sizes': [20],
															 'dropouts': [0.1],
															 'total_layers_lstm': 3,
															 'kernel_name': '_glorot',
															 'total_registers': 1040,
															 'subdirectory': 'only_disorders/A_D',
															 'name_model': 'SMHD_ml_gl_1040_A_D_glove_a-d-aduser',
															 'embedding_types': [dn.EmbeddingType.GLOVE_CUSTOM],
															 'embedding_custom_files': ['SMHD-glove-A-D-ADUsers-300.pkl'],
															 'use_embeddings': [dn.UseEmbedding.STATIC]})
						exp, submodel = self.load_submodel_spec_lstm(set_params, '', use_custom_metrics=True)

				elif id_model == 10: # A-D t20
						self.log.save('Load model SMHD_ml_le_1040_A_D 10')
						
						set_params = dict({'neuronios_by_layer': [16],
															 'epochs': [64],
															 'batch_sizes': [10],
															 'dropouts': [0.1],
															 'total_layers_lstm': 3,
															 'kernel_name': '_glorot',
															 'total_registers': 1040,
															 'subdirectory': 'only_disorders/A_D',
															 'name_model': 'SMHD_ml_gl_1040_A_D_cbow_alluser',
															 'embedding_types': [dn.EmbeddingType.WORD2VEC_CUSTOM],
															 'embedding_custom_files': ['SMHD-CBOW-AllUsers-300.bin'],
															 'use_embeddings': [dn.UseEmbedding.NON_STATIC]})
						exp, submodel = self.load_submodel_spec_lstm(set_params, '', use_custom_metrics=True)

				elif id_model == 7: # t9
						self.log.save('Load model SMHD_ml_gl_880_A_AD_glove_a-d-aduser')
						set_params = dict({'neuronios_by_layer': [16],
															 'epochs': [96],
															 'batch_sizes': [20],
															 'dropouts': [0.1],
															 'total_layers_lstm': 3,
															 'kernel_name': '_glorot',
															 'total_registers': 880,
															 'subdirectory': 'only_disorders/A_AD',
															 'name_model': 'SMHD_ml_gl_880_A_AD_glove_a-d-aduser',
															 'embedding_types': [dn.EmbeddingType.GLOVE_CUSTOM],
															 'embedding_custom_files': ['SMHD-glove-A-D-ADUsers-300.pkl'],
															 'use_embeddings': [dn.UseEmbedding.STATIC]})
						exp, submodel = self.load_submodel_spec_lstm(set_params, '', use_custom_metrics=True)

				elif id_model == 8: # t11 D0.2, teste t13 stacked
						self.log.save('Load model SMHD_ml_le_880_D_AD')
						#t11_SMHD_ml_le_880_D_AD_lstm_exp9_var_L3_N16_B40_E32_D0.2_ET_2_UE_3_EF_glove6B300d.h5
						set_params = dict({'neuronios_by_layer': [16],
															 'epochs': [32],
															 'batch_sizes': [40],
															 'dropouts': [0.2],
															 'total_layers_lstm': 3,
															 'kernel_name': '_lecun',
															 'total_registers': 880,
															 'subdirectory': 'only_disorders/D_AD',
															 'name_model': 'SMHD_ml_le_880_D_AD',
															 'embedding_types': [dn.EmbeddingType.GLOVE_6B],
															 'embedding_custom_files': [''],
															 'use_embeddings': [dn.UseEmbedding.STATIC]})
						exp, submodel = self.load_submodel_spec_lstm(set_params, 'glove6B300d')

				else: # model 11 corresponde a t11 D0.1, teste t15 stacked
						self.log.save('Load model SMHD_ml_le_880_D_AD')
						
						# t11_SMHD_ml_le_880_D_AD_lstm_exp9_var_L3_N16_B40_E32_D0.1_ET_2_UE_3_EF_glove6B300d.h5
						set_params = dict({'neuronios_by_layer': [16],
															 'epochs': [32],
															 'batch_sizes': [40],
															 'dropouts': [0.1],
															 'total_layers_lstm': 3,
															 'kernel_name': '_lecun',
															 'total_registers': 880,
															 'subdirectory': 'only_disorders/D_AD',
															 'name_model': 'SMHD_ml_le_880_D_AD',
															 'embedding_types': [dn.EmbeddingType.GLOVE_6B],
															 'embedding_custom_files': [''],
															 'use_embeddings': [dn.UseEmbedding.STATIC]})
						exp, submodel = self.load_submodel_spec_lstm(set_params, 'glove6B300d', use_custom_metrics=True)

				all_models.update({'CAD'+str(id_model): {'exp': exp, 'model_class': submodel}})
				del exp, submodel

				return all_models

		def load_submodels_12_to_54(self, all_models, id_model):
				if id_model == 12:
						self.log.save('Load model t1_SMHD_cnn_gl_1040_A_D_cnn_F250_K3_PNone_B10_E20_D0.2_HLN250_ET_2_UE_4_EF_glove6B300d_glorot CAD'+ str(id_model))
						set_params = dict({'filters_by_layer': [250],
																'kernels_size': [3],
																'epochs': [20],
																'batch_sizes': [10],
																'dropouts': [0.2],
															  'kernel_name': '_glorot',
															  'total_registers': 1040,
															  'subdirectory': 'only_disorders/A_D',
															  'name_model': '_SMHD_cnn_gl_1040_A_D',
																'embedding_types': [dn.EmbeddingType.GLOVE_6B],
																'embedding_custom_files': [''],
																'use_embeddings': [dn.UseEmbedding.NON_STATIC]})
						exp, submodel = self.load_submodel_spec_cnn('1', set_params, 'glove6B300d')

				elif id_model == 13:
						self.log.save(
								'Load model t1_SMHD_cnn_gl_1040_A_D_cnn_F250_K3_PNone_B5_E20_D0.2_HLN250_ET_2_UE_4_EF_glove6B300d_glorot CAD'+ str(id_model))
						set_params = dict({'filters_by_layer': [250],
															 'kernels_size': [3],
															 'epochs': [20],
															 'batch_sizes': [5],
															 'dropouts': [0.2],
															 'kernel_name': '_glorot',
															 'total_registers': 1040,
															 'subdirectory': 'only_disorders/A_D',
															 'name_model': '_SMHD_cnn_gl_1040_A_D',
															 'embedding_types': [dn.EmbeddingType.GLOVE_6B],
															 'embedding_custom_files': [''],
															 'use_embeddings': [dn.UseEmbedding.NON_STATIC]})
						exp, submodel = self.load_submodel_spec_cnn('1', set_params, 'glove6B300d')

				elif id_model == 14:
						self.log.save('Load model t1_SMHD_cnn_gl_1040_A_D_cnn_F100_K5_PNone_B20_E10_D0.5_HLN100_ET_7_UE_4_EF_SMHD-glove-AllUsers-300_glorot CAD'+ str(id_model))
						set_params = dict({'filters_by_layer': [100],
															 'kernels_size': [5],
															 'epochs': [10],
															 'batch_sizes': [20],
															 'dropouts': [0.5],
															 'kernel_name': '_glorot',
															 'total_registers': 1040,
															 'subdirectory': 'only_disorders/A_D',
															 'name_model': '_SMHD_cnn_gl_1040_A_D',
															 'embedding_types': [dn.EmbeddingType.GLOVE_CUSTOM],
															 'embedding_custom_files': ['SMHD-glove-AllUsers-300.pkl'],
															 'use_embeddings': [dn.UseEmbedding.NON_STATIC]})
						exp, submodel = self.load_submodel_spec_cnn('1', set_params, 'gloveCustom')

				elif id_model == 15:
						self.log.save(
								'Load model t1_SMHD_cnn_gl_1040_A_D_cnn_F250_K3_PNone_B20_E10_D0.2_HLN250_ET_7_UE_4_EF_SMHD-glove-A-D-ADUsers-300_glorot CAD'+ str(id_model))
						set_params = dict({'filters_by_layer': [250],
															 'kernels_size': [3],
															 'epochs': [10],
															 'batch_sizes': [20],
															 'dropouts': [0.2],
															 'kernel_name': '_glorot',
															 'total_registers': 1040,
															 'subdirectory': 'only_disorders/A_D',
															 'name_model': '_SMHD_cnn_gl_1040_A_D',
															 'embedding_types': [dn.EmbeddingType.GLOVE_CUSTOM],
															 'embedding_custom_files': ['SMHD-glove-A-D-ADUsers-300.pkl'],
															 'use_embeddings': [dn.UseEmbedding.NON_STATIC]})
						exp, submodel = self.load_submodel_spec_cnn('1', set_params, 'gloveCustom')

				elif id_model == 16:
						self.log.save('Load model t1_SMHD_cnn_gl_1040_A_D_cnn_F250_K5_PNone_B20_E10_D0.2_HLN250_ET_7_UE_4_EF_SMHD-glove-A-D-ADUsers-300_glorot CAD'+ str(id_model))
						set_params = dict({'filters_by_layer': [250],
															 'kernels_size': [5],
															 'epochs': [10],
															 'batch_sizes': [20],
															 'dropouts': [0.2],
															 'kernel_name': '_glorot',
															 'total_registers': 1040,
															 'subdirectory': 'only_disorders/A_D',
															 'name_model': '_SMHD_cnn_gl_1040_A_D',
															 'embedding_types': [dn.EmbeddingType.GLOVE_CUSTOM],
															 'embedding_custom_files': ['SMHD-glove-A-D-ADUsers-300.pkl'],
															 'use_embeddings': [dn.UseEmbedding.NON_STATIC]})
						exp, submodel = self.load_submodel_spec_cnn('1', set_params, 'gloveCustom')

				elif id_model == 17:
						self.log.save('Load model t1_SMHD_cnn_gl_1040_A_D_cnn_F250_K4_PNone_B5_E10_D0.2_HLN250_ET_3_UE_4_EF_gloveTwitter_glorot CAD'+ str(id_model))
						set_params = dict({'filters_by_layer': [250],
															 'kernels_size': [4],
															 'epochs': [10],
															 'batch_sizes': [5],
															 'dropouts': [0.2],
															 'kernel_name': '_glorot',
															 'total_registers': 1040,
															 'subdirectory': 'only_disorders/A_D',
															 'name_model': '_SMHD_cnn_gl_1040_A_D',
															 'embedding_types': [dn.EmbeddingType.GLOVE_TWITTER],
															 'embedding_custom_files': [''],
															 'use_embeddings': [dn.UseEmbedding.NON_STATIC]})
						exp, submodel = self.load_submodel_spec_cnn('1', set_params, 'gloveTwitter')

				elif id_model == 18:
						self.log.save(
								'Load model t1_SMHD_cnn_gl_1040_A_D_cnn_F250_K5_PNone_B20_E10_D0.5_HLN250_ET_3_UE_4_EF_gloveTwitter_glorot CAD'+ str(id_model))
						set_params = dict({'filters_by_layer': [250],
															 'kernels_size': [5],
															 'epochs': [10],
															 'batch_sizes': [20],
															 'dropouts': [0.5],
															 'kernel_name': '_glorot',
															 'total_registers': 1040,
															 'subdirectory': 'only_disorders/A_D',
															 'name_model': '_SMHD_cnn_gl_1040_A_D',
															 'embedding_types': [dn.EmbeddingType.GLOVE_TWITTER],
															 'embedding_custom_files': [''],
															 'use_embeddings': [dn.UseEmbedding.NON_STATIC]})
						exp, submodel = self.load_submodel_spec_cnn('1', set_params, 'gloveTwitter')

				elif id_model == 19:
						self.log.save(
								'Load model t1_SMHD_cnn_gl_1040_A_D_cnn_F100_K4_PNone_B20_E10_D0.2_HLN100_ET_4_UE_4_EF_googleNews_glorot CAD'+ str(id_model))
						set_params = dict({'filters_by_layer': [100],
															 'kernels_size': [4],
															 'epochs': [10],
															 'batch_sizes': [20],
															 'dropouts': [0.2],
															 'kernel_name': '_glorot',
															 'total_registers': 1040,
															 'subdirectory': 'only_disorders/A_D',
															 'name_model': '_SMHD_cnn_gl_1040_A_D',
															 'embedding_types': [dn.EmbeddingType.WORD2VEC],
															 'embedding_custom_files': [''],
															 'use_embeddings': [dn.UseEmbedding.NON_STATIC]})
						exp, submodel = self.load_submodel_spec_cnn('1', set_params, 'googleNews')

				elif id_model == 20:
						self.log.save('Load model t1_SMHD_cnn_gl_1040_A_D_cnn_F100_K4_PNone_B20_E20_D0.2_HLN100_ET_4_UE_4_EF_googleNews_glorot CAD'+ str(id_model))
						set_params = dict({'filters_by_layer': [100],
															 'kernels_size': [4],
															 'epochs': [20],
															 'batch_sizes': [20],
															 'dropouts': [0.2],
															 'kernel_name': '_glorot',
															 'total_registers': 1040,
															 'subdirectory': 'only_disorders/A_D',
															 'name_model': '_SMHD_cnn_gl_1040_A_D',
															 'embedding_types': [dn.EmbeddingType.WORD2VEC],
															 'embedding_custom_files': [''],
															 'use_embeddings': [dn.UseEmbedding.NON_STATIC]})
						exp, submodel = self.load_submodel_spec_cnn('1', set_params, 'googleNews')

				elif id_model == 22:
						self.log.save('Load model t1_SMHD_cnn_gl_1040_A_D_cnn_F100_K5_PNone_B20_E10_D0.2_HLN100_ET_6_UE_4_EF_SMHD-Skipgram-A-D-ADUsers-300_glorot CAD'+ str(id_model))
						set_params = dict({'filters_by_layer': [100],
															 'kernels_size': [5],
															 'epochs': [10],
															 'batch_sizes': [20],
															 'dropouts': [0.2],
															 'kernel_name': '_glorot',
															 'total_registers': 1040,
															 'subdirectory': 'only_disorders/A_D',
															 'name_model': '_SMHD_cnn_gl_1040_A_D',
															 'embedding_types': [dn.EmbeddingType.WORD2VEC_CUSTOM],
															 'embedding_custom_files': ['SMHD-Skipgram-A-D-ADUsers-300.bin'],
															 'use_embeddings': [dn.UseEmbedding.NON_STATIC]})
						exp, submodel = self.load_submodel_spec_cnn('1', set_params, 'w2vCustom')

				elif id_model == 23:
						self.log.save('Load model t1_SMHD_cnn_gl_1040_A_D_cnn_F100_K3_PNone_B20_E10_D0.5_HLN100_ET_6_UE_4_EF_SMHD-CBOW-AllUsers-300_glorot CAD'+ str(id_model))
						set_params = dict({'filters_by_layer': [100],
															 'kernels_size': [3],
															 'epochs': [10],
															 'batch_sizes': [20],
															 'dropouts': [0.5],
															 'kernel_name': '_glorot',
															 'total_registers': 1040,
															 'subdirectory': 'only_disorders/A_D',
															 'name_model': '_SMHD_cnn_gl_1040_A_D',
															 'embedding_types': [dn.EmbeddingType.WORD2VEC_CUSTOM],
															 'embedding_custom_files': ['SMHD-CBOW-AllUsers-300.bin'],
															 'use_embeddings': [dn.UseEmbedding.NON_STATIC]})
						exp, submodel = self.load_submodel_spec_cnn('1', set_params, 'w2vCustom')

				elif id_model == 24:
						self.log.save('Load model t1_SMHD_cnn_gl_1040_A_D_cnn_F250_K4_PNone_B5_E10_D0.5_HLN250_ET_6_UE_4_EF_SMHD-CBOW-A-D-ADUsers-300_glorot CAD'+ str(id_model))
						set_params = dict({'filters_by_layer': [250],
															 'kernels_size': [4],
															 'epochs': [10],
															 'batch_sizes': [5],
															 'dropouts': [0.5],
															 'kernel_name': '_glorot',
															 'total_registers': 1040,
															 'subdirectory': 'only_disorders/A_D',
															 'name_model': '_SMHD_cnn_gl_1040_A_D',
															 'embedding_types': [dn.EmbeddingType.WORD2VEC_CUSTOM],
															 'embedding_custom_files': ['SMHD-CBOW-A-D-ADUsers-300.bin'],
															 'use_embeddings': [dn.UseEmbedding.NON_STATIC]})
						exp, submodel = self.load_submodel_spec_cnn('1', set_params, 'w2vCustom')

				elif id_model == 25:
						self.log.save(
								'Load model t2_SMHD_cnn_gl_880_A_AD_cnn_F100_K5_PNone_B20_E10_D0.2_HLN100_ET_2_UE_4_EF_glove6B300d_glorot CAD'+ str(id_model))
						set_params = dict({'filters_by_layer': [100],
															 'kernels_size': [5],
															 'epochs': [10],
															 'batch_sizes': [20],  # , 40],
															 'dropouts': [0.2],
															 'kernel_name': '_glorot',
															 'total_registers': 880,
															 'subdirectory': 'only_disorders/A_AD',
															 'name_model': '_SMHD_cnn_gl_880_A_AD',
															 'embedding_types': [dn.EmbeddingType.GLOVE_6B],
															 'embedding_custom_files': [''],
															 'use_embeddings': [dn.UseEmbedding.NON_STATIC]})
						exp, submodel = self.load_submodel_spec_cnn('2', set_params, 'glove6B300d')

				elif id_model == 26:
						self.log.save(
								'Load model t2_SMHD_cnn_gl_880_A_AD_cnn_F250_K4_PNone_B20_E10_D0.5_HLN250_ET_2_UE_3_EF_glove6B300d_glorot CAD'+ str(id_model))
						set_params = dict({'filters_by_layer': [250],
															 'kernels_size': [4],
															 'epochs': [10],
															 'batch_sizes': [20],
															 'dropouts': [0.5],
															 'kernel_name': '_glorot',
															 'total_registers': 880,
															 'subdirectory': 'only_disorders/A_AD',
															 'name_model': '_SMHD_cnn_gl_880_A_AD',
															 'embedding_types': [dn.EmbeddingType.GLOVE_6B],
															 'embedding_custom_files': [''],
															 'use_embeddings': [dn.UseEmbedding.STATIC]})
						exp, submodel = self.load_submodel_spec_cnn('2', set_params, 'glove6B300d')

				elif id_model == 27:
						self.log.save(
								'Load model t2_SMHD_cnn_gl_880_A_AD_cnn_F100_K5_PNone_B5_E20_D0.2_HLN100_ET_2_UE_4_EF_glove6B300d_glorot CAD'+ str(id_model))
						set_params = dict({'filters_by_layer': [100],
															 'kernels_size': [5],
															 'epochs': [20],
															 'batch_sizes': [5],
															 'dropouts': [0.2],
															 'kernel_name': '_glorot',
															 'total_registers': 880,
															 'subdirectory': 'only_disorders/A_AD',
															 'name_model': '_SMHD_cnn_gl_880_A_AD',
															 'embedding_types': [dn.EmbeddingType.GLOVE_6B],
															 'embedding_custom_files': [''],
															 'use_embeddings': [dn.UseEmbedding.NON_STATIC]})
						exp, submodel = self.load_submodel_spec_cnn('2', set_params, 'glove6B300d')

				elif id_model == 28:
						self.log.save(
								'Load model t2_SMHD_cnn_gl_880_A_AD_cnn_F250_K3_PNone_B20_E10_D0.5_HLN250_ET_7_UE_4_EF_SMHD-glove-AllUsers-300_glorot CAD'+ str(id_model))
						set_params = dict({'filters_by_layer': [250],
															 'kernels_size': [3],
															 'epochs': [10],
															 'batch_sizes': [20],
															 'dropouts': [0.5],
															 'kernel_name': '_glorot',
															 'total_registers': 880,
															 'subdirectory': 'only_disorders/A_AD',
															 'name_model': '_SMHD_cnn_gl_880_A_AD',
															 'embedding_types': [dn.EmbeddingType.GLOVE_CUSTOM],
															 'embedding_custom_files': ['SMHD-glove-AllUsers-300.pkl'],
															 'use_embeddings': [dn.UseEmbedding.NON_STATIC]})
						exp, submodel = self.load_submodel_spec_cnn('2', set_params, 'gloveCustom')

				elif id_model == 29:
						self.log.save(
								'Load model t2_SMHD_cnn_gl_880_A_AD_cnn_F250_K5_PNone_B20_E10_D0.5_HLN250_ET_7_UE_4_EF_SMHD-glove-A-D-ADUsers-300_glorot CAD'+ str(id_model))
						set_params = dict({'filters_by_layer': [250],
															 'kernels_size': [5],
															 'epochs': [10],
															 'batch_sizes': [20],
															 'dropouts': [0.5],
															 'kernel_name': '_glorot',
															 'total_registers': 880,
															 'subdirectory': 'only_disorders/A_AD',
															 'name_model': '_SMHD_cnn_gl_880_A_AD',
															 'embedding_types': [dn.EmbeddingType.GLOVE_CUSTOM],
															 'embedding_custom_files': ['SMHD-glove-A-D-ADUsers-300.pkl'],
															 'use_embeddings': [dn.UseEmbedding.NON_STATIC]})
						exp, submodel = self.load_submodel_spec_cnn('2', set_params, 'gloveCustom')

				elif id_model == 30:
						self.log.save(
								'Load model t2_SMHD_cnn_gl_880_A_AD_cnn_F100_K3_PNone_B5_E10_D0.2_HLN100_ET_7_UE_4_EF_SMHD-glove-A-D-ADUsers-300_glorot CAD'+ str(id_model))
						set_params = dict({'filters_by_layer': [100],
															 'kernels_size': [3],
															 'epochs': [10],
															 'batch_sizes': [5],
															 'dropouts': [0.2],
															 'kernel_name': '_glorot',
															 'total_registers': 880,
															 'subdirectory': 'only_disorders/A_AD',
															 'name_model': '_SMHD_cnn_gl_880_A_AD',
															 'embedding_types': [dn.EmbeddingType.GLOVE_CUSTOM],
															 'embedding_custom_files': ['SMHD-glove-A-D-ADUsers-300.pkl'],
															 'use_embeddings': [dn.UseEmbedding.NON_STATIC]})
						exp, submodel = self.load_submodel_spec_cnn('2', set_params, 'gloveCustom')

				elif id_model == 31:
						self.log.save(
								'Load model t2_SMHD_cnn_gl_880_A_AD_cnn_F100_K4_PNone_B5_E20_D0.5_HLN100_ET_7_UE_4_EF_SMHD-glove-AllUsers-300_glorot CAD'+ str(id_model))
						set_params = dict({'filters_by_layer': [100],
															 'kernels_size': [4],
															 'epochs': [20],
															 'batch_sizes': [5],
															 'dropouts': [0.5],
															 'kernel_name': '_glorot',
															 'total_registers': 880,
															 'subdirectory': 'only_disorders/A_AD',
															 'name_model': '_SMHD_cnn_gl_880_A_AD',
															 'embedding_types': [dn.EmbeddingType.GLOVE_CUSTOM],
															 'embedding_custom_files': ['SMHD-glove-AllUsers-300.pkl'],
															 'use_embeddings': [dn.UseEmbedding.NON_STATIC]})
						exp, submodel = self.load_submodel_spec_cnn('2', set_params, 'gloveCustom')

				elif id_model == 32:
						self.log.save(
								'Load model t2_SMHD_cnn_gl_880_A_AD_cnn_F100_K5_PNone_B20_E10_D0.5_HLN100_ET_3_UE_4_EF_gloveTwitter_glorot CAD'+ str(id_model))
						set_params = dict({'filters_by_layer': [100],
															 'kernels_size': [5],
															 'epochs': [10],
															 'batch_sizes': [20],
															 'dropouts': [0.5],
															 'kernel_name': '_glorot',
															 'total_registers': 880,
															 'subdirectory': 'only_disorders/A_AD',
															 'name_model': '_SMHD_cnn_gl_880_A_AD',
															 'embedding_types': [dn.EmbeddingType.GLOVE_TWITTER],
															 'embedding_custom_files': [''],
															 'use_embeddings': [dn.UseEmbedding.NON_STATIC]})
						exp, submodel = self.load_submodel_spec_cnn('2', set_params, 'gloveTwitter')

				elif id_model == 33:
						self.log.save(
								'Load model t2_SMHD_cnn_gl_880_A_AD_cnn_F250_K3_PNone_B20_E10_D0.5_HLN250_ET_3_UE_4_EF_gloveTwitter_glorot CAD'+ str(id_model))
						set_params = dict({'filters_by_layer': [250],
															 'kernels_size': [3],
															 'epochs': [10],
															 'batch_sizes': [20],
															 'dropouts': [0.5],
															 'kernel_name': '_glorot',
															 'total_registers': 880,
															 'subdirectory': 'only_disorders/A_AD',
															 'name_model': '_SMHD_cnn_gl_880_A_AD',
															 'embedding_types': [dn.EmbeddingType.GLOVE_TWITTER],
															 'embedding_custom_files': [''],
															 'use_embeddings': [dn.UseEmbedding.NON_STATIC]})
						exp, submodel = self.load_submodel_spec_cnn('2', set_params, 'gloveTwitter')

				elif id_model == 34:
						self.log.save(
								'Load model t2_SMHD_cnn_gl_880_A_AD_cnn_F100_K5_PNone_B5_E20_D0.5_HLN100_ET_3_UE_4_EF_gloveTwitter_glorot CAD'+ str(id_model))
						set_params = dict({'filters_by_layer': [100],
															 'kernels_size': [5],
															 'epochs': [20],
															 'batch_sizes': [5],
															 'dropouts': [0.5],
															 'kernel_name': '_glorot',
															 'total_registers': 880,
															 'subdirectory': 'only_disorders/A_AD',
															 'name_model': '_SMHD_cnn_gl_880_A_AD',
															 'embedding_types': [dn.EmbeddingType.GLOVE_TWITTER],
															 'embedding_custom_files': [''],
															 'use_embeddings': [dn.UseEmbedding.NON_STATIC]})
						exp, submodel = self.load_submodel_spec_cnn('2', set_params, 'gloveTwitter')

				elif id_model == 35:
						self.log.save(
								'Load model t2_SMHD_cnn_gl_880_A_AD_cnn_F250_K5_PNone_B20_E10_D0.2_HLN250_ET_4_UE_4_EF_googleNews_glorot CAD'+ str(id_model))
						set_params = dict({'filters_by_layer': [250],
															 'kernels_size': [5],
															 'epochs': [10],
															 'batch_sizes': [20],
															 'dropouts': [0.2],
															 'kernel_name': '_glorot',
															 'total_registers': 880,
															 'subdirectory': 'only_disorders/A_AD',
															 'name_model': '_SMHD_cnn_gl_880_A_AD',
															 'embedding_types': [dn.EmbeddingType.WORD2VEC],
															 'embedding_custom_files': [''],
															 'use_embeddings': [dn.UseEmbedding.NON_STATIC]})
						exp, submodel = self.load_submodel_spec_cnn('2', set_params, 'googleNews')

				elif id_model == 36:
						self.log.save(
								'Load model t2_SMHD_cnn_gl_880_A_AD_cnn_F100_K5_PNone_B5_E20_D0.2_HLN100_ET_4_UE_4_EF_googleNews_glorot CAD'+ str(id_model))
						set_params = dict({'filters_by_layer': [100],
															 'kernels_size': [5],
															 'epochs': [20],
															 'batch_sizes': [5],
															 'dropouts': [0.2],
															 'kernel_name': '_glorot',
															 'total_registers': 880,
															 'subdirectory': 'only_disorders/A_AD',
															 'name_model': '_SMHD_cnn_gl_880_A_AD',
															 'embedding_types': [dn.EmbeddingType.WORD2VEC],
															 'embedding_custom_files': [''],
															 'use_embeddings': [dn.UseEmbedding.NON_STATIC]})
						exp, submodel = self.load_submodel_spec_cnn('2', set_params, 'googleNews')

				elif id_model == 37:
						self.log.save(
								'Load model t2_SMHD_cnn_gl_880_A_AD_cnn_F100_K3_PNone_B20_E10_D0.2_HLN100_ET_6_UE_4_EF_SMHD-CBOW-A-D-ADUsers-300_glorot CAD'+ str(id_model))
						set_params = dict({'filters_by_layer': [100],
															 'kernels_size': [3],
															 'epochs': [10],
															 'batch_sizes': [20],
															 'dropouts': [0.2],
															 'kernel_name': '_glorot',
															 'total_registers': 880,
															 'subdirectory': 'only_disorders/A_AD',
															 'name_model': '_SMHD_cnn_gl_880_A_AD',
															 'embedding_types': [dn.EmbeddingType.WORD2VEC_CUSTOM],
															 'embedding_custom_files': ['SMHD-CBOW-A-D-ADUsers-300.bin'],
															 'use_embeddings': [dn.UseEmbedding.NON_STATIC]})
						exp, submodel = self.load_submodel_spec_cnn('2', set_params, 'w2vCustom')

				elif id_model == 38:
						self.log.save(
								'Load model t2_SMHD_cnn_gl_880_A_AD_cnn_F100_K3_PNone_B20_E10_D0.5_HLN100_ET_6_UE_3_EF_SMHD-Skipgram-AllUsers-300_glorot CAD'+ str(id_model))
						set_params = dict({'filters_by_layer': [100],
															 'kernels_size': [3],
															 'epochs': [10],
															 'batch_sizes': [20],
															 'dropouts': [0.5],
															 'kernel_name': '_glorot',
															 'total_registers': 880,
															 'subdirectory': 'only_disorders/A_AD',
															 'name_model': '_SMHD_cnn_gl_880_A_AD',
															 'embedding_types': [dn.EmbeddingType.WORD2VEC_CUSTOM],
															 'embedding_custom_files': ['SMHD-Skipgram-AllUsers-300.bin'],
															 'use_embeddings': [dn.UseEmbedding.STATIC]})
						exp, submodel = self.load_submodel_spec_cnn('2', set_params, 'w2vCustom')

				elif id_model == 39:
						self.log.save(
								'Load model t2_SMHD_cnn_gl_880_A_AD_cnn_F100_K5_PNone_B20_E10_D0.5_HLN100_ET_6_UE_3_EF_SMHD-Skipgram-A-D-ADUsers-300_glorot CAD'+ str(id_model))
						set_params = dict({'filters_by_layer': [100],
															 'kernels_size': [5],
															 'epochs': [10],
															 'batch_sizes': [20],
															 'dropouts': [0.5],
															 'kernel_name': '_glorot',
															 'total_registers': 880,
															 'subdirectory': 'only_disorders/A_AD',
															 'name_model': '_SMHD_cnn_gl_880_A_AD',
															 'embedding_types': [dn.EmbeddingType.WORD2VEC_CUSTOM],
															 'embedding_custom_files': ['SMHD-Skipgram-A-D-ADUsers-300.bin'],
															 'use_embeddings': [dn.UseEmbedding.STATIC]})
						exp, submodel = self.load_submodel_spec_cnn('2', set_params, 'w2vCustom')

				elif id_model == 40:
						self.log.save(
								'Load model t3_SMHD_ml_gl_880_D_AD_cnn_F250_K3_PNone_B20_E10_D0.2_HLN250_ET_2_UE_3_EF_glove6B300d_glorot CAD'+ str(id_model))
						set_params = dict({'filters_by_layer': [250],
															 'kernels_size': [3],
															 'epochs': [10],
															 'batch_sizes': [20],  # , 40],
															 'dropouts': [0.2],
															 'kernel_name': '_glorot',
															 'total_registers': 880,
															 'subdirectory': 'only_disorders/D_AD',
															 'name_model': '_SMHD_ml_gl_880_D_AD',
															 'embedding_types': [dn.EmbeddingType.GLOVE_6B],
															 'embedding_custom_files': [''],
															 'use_embeddings': [dn.UseEmbedding.STATIC]})
						exp, submodel = self.load_submodel_spec_cnn('3', set_params, 'glove6B300d')

				elif id_model == 41:
						self.log.save(
								'Load model t3_SMHD_ml_gl_880_D_AD_cnn_F100_K3_PNone_B20_E10_D0.5_HLN100_ET_2_UE_3_EF_glove6B300d_glorot CAD'+ str(id_model))
						set_params = dict({'filters_by_layer': [100],
															 'kernels_size': [3],
															 'epochs': [10],
															 'batch_sizes': [20],  # , 40],
															 'dropouts': [0.5],
															 'kernel_name': '_glorot',
															 'total_registers': 880,
															 'subdirectory': 'only_disorders/D_AD',
															 'name_model': '_SMHD_ml_gl_880_D_AD',
															 'embedding_types': [dn.EmbeddingType.GLOVE_6B],
															 'embedding_custom_files': [''],
															 'use_embeddings': [dn.UseEmbedding.STATIC]})
						exp, submodel = self.load_submodel_spec_cnn('3', set_params, 'glove6B300d')

				elif id_model == 43:
						self.log.save(
								'Load model t3_SMHD_ml_gl_880_D_AD_cnn_F100_K4_PNone_B20_E10_D0.2_HLN100_ET_6_UE_4_EF_SMHD-Skipgram-AllUsers-300_glorot CAD'+ str(id_model))
						set_params = dict({'filters_by_layer': [100],
															 'kernels_size': [4],
															 'epochs': [10],
															 'batch_sizes': [20],
															 'dropouts': [0.2],
															 'kernel_name': '_glorot',
															 'total_registers': 880,
															 'subdirectory': 'only_disorders/D_AD',
															 'name_model': '_SMHD_ml_gl_880_D_AD',
															 'embedding_types': [dn.EmbeddingType.WORD2VEC_CUSTOM],
															 'embedding_custom_files': ['SMHD-Skipgram-AllUsers-300.bin'],
															 'use_embeddings': [dn.UseEmbedding.NON_STATIC]})
						exp, submodel = self.load_submodel_spec_cnn('3', set_params, 'w2vCustom')

				# Ray Tune LSTM Diffs Models
				elif id_model == 44:
						self.log.save('Load model rt_SMHD_ml_le_1040_A_D_lstm_exp9_var_L3_N49_B10_E60_D0.32_ET_6_UE_4_EF_SMHD-CBOW-A-D-ADUsers-300_lecun CAD' + str(id_model))
						set_params = dict({'neuronios_by_layer': [49],
															 'epochs': [60],
															 'batch_sizes': [10],
															 'dropouts': [0.32],
															 'total_layers_lstm': 3,
															 'kernel_name': '_lecun',
															 'total_registers': 1040,
															 'subdirectory': 'only_disorders/A_D',
															 'name_model': 'rt_SMHD_ml_le_1040_A_D',
															 'embedding_types': [dn.EmbeddingType.WORD2VEC_CUSTOM],
															 'embedding_custom_files': ['SMHD-CBOW-A-D-ADUsers-300.bin'],
															 'use_embeddings': [dn.UseEmbedding.NON_STATIC]})
						exp, submodel = self.load_submodel_spec_lstm(set_params, '')

				# elif id_model == 45:
				# 		self.log.save('Load model rt_SMHD_ml_le_1040_A_D_lstm_exp9_var_L3_N29_B40_E60_D0.3_ET_2_UE_3_EF_glove6B300d_lecun CAD' + str(id_model))
				# 		set_params = dict({'neuronios_by_layer': [29],
				# 											 'epochs': [60],
				# 											 'batch_sizes': [40],
				# 											 'dropouts': [0.3],
				# 											 'total_layers_lstm': 3,
				# 											 'kernel_name': '_lecun',
				# 											 'total_registers': 1040,
				# 											 'subdirectory': 'only_disorders/A_D',
				# 											 'name_model': 'rt_SMHD_ml_le_1040_A_D',
				# 											 'embedding_types': [dn.EmbeddingType.GLOVE_6B],
				# 											 'embedding_custom_files': [''],
				# 											 'use_embeddings': [dn.UseEmbedding.STATIC]})
				# 		exp, submodel = self.load_submodel_spec_lstm(set_params, 'glove6B300d')

				elif id_model == 45:
						self.log.save('Load model rt_SMHD_ml_le_880_A_AD_lstm_exp9_var_L4_N118_B20_E68_D0.31_ET_7_UE_3_EF_SMHD-glove-A-D-ADUsers-300_lecun CAD' + str(id_model))
						set_params = dict({'neuronios_by_layer': [118],
															 'epochs': [68],
															 'batch_sizes': [20],
															 'dropouts': [0.31],
															 'total_layers_lstm': 4,
															 'kernel_name': '_lecun',
															 'total_registers': 880,
															 'subdirectory': 'only_disorders/A_AD',
															 'name_model': 'rt_SMHD_ml_le_880_A_AD',
															 'embedding_types': [dn.EmbeddingType.GLOVE_CUSTOM],
															 'embedding_custom_files': ['SMHD-glove-A-D-ADUsers-300.pkl'],
															 'use_embeddings': [dn.UseEmbedding.STATIC]})
						exp, submodel = self.load_submodel_spec_lstm(set_params, '')

				elif id_model == 46:
						self.log.save('Load model rt_SMHD_ml_le_880_A_AD_lstm_exp9_var_L3_N94_B20_E66_D0.37_ET_2_UE_3_EF_glove6B300d_lecun CAD' + str(id_model))
						set_params = dict({'neuronios_by_layer': [94],
															 'epochs': [66],
															 'batch_sizes': [20],
															 'dropouts': [0.37],
															 'total_layers_lstm': 3,
															 'kernel_name': '_lecun',
															 'total_registers': 880,
															 'subdirectory': 'only_disorders/A_AD',
															 'name_model': 'rt_SMHD_ml_le_880_A_AD',
															 'embedding_types': [dn.EmbeddingType.GLOVE_6B],
															 'embedding_custom_files': [''],
															 'use_embeddings': [dn.UseEmbedding.STATIC]})
						exp, submodel = self.load_submodel_spec_lstm(set_params, 'glove6B300d')

				elif id_model == 47:
						self.log.save('Load model rt_SMHD_ml_le_880_A_AD_lstm_exp9_var_L4_N51_B10_E42_D0.27_ET_2_UE_3_EF_glove6B300d_lecun CAD' + str(id_model))
						set_params = dict({'neuronios_by_layer': [51],
															 'epochs': [42],
															 'batch_sizes': [10],
															 'dropouts': [0.27],
															 'total_layers_lstm': 4,
															 'kernel_name': '_lecun',
															 'total_registers': 880,
															 'subdirectory': 'only_disorders/A_AD',
															 'name_model': 'rt_SMHD_ml_le_880_A_AD',
															 'embedding_types': [dn.EmbeddingType.GLOVE_6B],
															 'embedding_custom_files': [''],
															 'use_embeddings': [dn.UseEmbedding.STATIC]})
						exp, submodel = self.load_submodel_spec_lstm(set_params, 'glove6B300d')

				elif id_model == 48:
						self.log.save('Load model rt_SMHD_ml_le_880_D_AD_lstm_exp9_var_L3_N30_B40_E75_D0.3_ET_2_UE_3_EF_glove6B300d_lecun CAD' + str(id_model))
						set_params = dict({'neuronios_by_layer': [30],
															 'epochs': [75],
															 'batch_sizes': [40],
															 'dropouts': [0.3],
															 'total_layers_lstm': 3,
															 'kernel_name': '_lecun',
															 'total_registers': 880,
															 'subdirectory': 'only_disorders/D_AD',
															 'name_model': 'rt_SMHD_ml_le_880_D_AD',
															 'embedding_types': [dn.EmbeddingType.GLOVE_6B],
															 'embedding_custom_files': [''],
															 'use_embeddings': [dn.UseEmbedding.STATIC]})
						exp, submodel = self.load_submodel_spec_lstm(set_params, 'glove6B300d')

				# Ray Tune CNN
				elif id_model == 49:
						self.log.save('Load model t1_SMHD_cnn_gl_1040_A_D_cnn_F497_K5_PNone_B5_E38_D0.42_HLN497_ET_2_UE_3_EF_glove6B300d_glorot CAD'+ str(id_model))
						set_params = dict({'filters_by_layer': [497],
																'kernels_size': [5],
																'epochs': [38],
																'batch_sizes': [5],
																'dropouts': [0.42],
															  'kernel_name': '_glorot',
															  'total_registers': 1040,
															  'subdirectory': 'only_disorders/A_D',
															  'name_model': '_SMHD_cnn_gl_1040_A_D',
																'embedding_types': [dn.EmbeddingType.GLOVE_6B],
																'embedding_custom_files': [''],
																'use_embeddings': [dn.UseEmbedding.STATIC]})
						exp, submodel = self.load_submodel_spec_cnn('1', set_params, 'glove6B300d')

				elif id_model == 50:
						self.log.save(
								'Load model t2_SMHD_cnn_gl_880_A_AD_cnn_F277_K5_PNone_B5_E35_D0.24_HLN277_ET_2_UE_3_EF_glove6B300d_glorot CAD'+ str(id_model))
						set_params = dict({'filters_by_layer': [277],
															 'kernels_size': [5],
															 'epochs': [35],
															 'batch_sizes': [5],
															 'dropouts': [0.24],
															 'kernel_name': '_glorot',
															 'total_registers': 880,
															 'subdirectory': 'only_disorders/A_AD',
															 'name_model': '_SMHD_cnn_gl_880_A_AD',
															 'embedding_types': [dn.EmbeddingType.GLOVE_6B],
															 'embedding_custom_files': [''],
															 'use_embeddings': [dn.UseEmbedding.STATIC]})
						exp, submodel = self.load_submodel_spec_cnn('2', set_params, 'glove6B300d')

				elif id_model == 51:
						self.log.save(
								'Load model t2_SMHD_cnn_gl_880_A_AD_cnn_F297_K3_PNone_B10_E36_D0.23_HLN297_ET_3_UE_3_EF_googleTwitter_glorot CAD'+ str(id_model))
						set_params = dict({'filters_by_layer': [297],
															 'kernels_size': [3],
															 'epochs': [36],
															 'batch_sizes': [10],
															 'dropouts': [0.23],
															 'kernel_name': '_glorot',
															 'total_registers': 880,
															 'subdirectory': 'only_disorders/A_AD',
															 'name_model': '_SMHD_cnn_gl_880_A_AD',
															 'embedding_types': [dn.EmbeddingType.GLOVE_TWITTER],
															 'embedding_custom_files': [''],
															 'use_embeddings': [dn.UseEmbedding.STATIC]})
						exp, submodel = self.load_submodel_spec_cnn('2', set_params, 'googleTwitter')

				elif id_model == 52:
						self.log.save(
								'Load model t2_SMHD_cnn_gl_880_A_AD_cnn_F279_K3_PNone_B50_E20_D0.4_HLN279_ET_3_UE_3_EF_googleTwitter_glorot CAD'+ str(id_model))
						set_params = dict({'filters_by_layer': [279],
															 'kernels_size': [3],
															 'epochs': [20],
															 'batch_sizes': [50],
															 'dropouts': [0.4],
															 'kernel_name': '_glorot',
															 'total_registers': 880,
															 'subdirectory': 'only_disorders/A_AD',
															 'name_model': '_SMHD_cnn_gl_880_A_AD',
															 'embedding_types': [dn.EmbeddingType.GLOVE_TWITTER],
															 'embedding_custom_files': [''],
															 'use_embeddings': [dn.UseEmbedding.STATIC]})
						exp, submodel = self.load_submodel_spec_cnn('2', set_params, 'googleTwitter')

				elif id_model == 53:
						self.log.save(
								'Load model t3_SMHD_cnn_gl_880_D_AD_cnn_F421_K3_PNone_B40_E39_D0.35_HLN421_ET_3_UE_3_EF_googleTwitter_glorot CAD'+ str(id_model))
						set_params = dict({'filters_by_layer': [421],
															 'kernels_size': [3],
															 'epochs': [39],
															 'batch_sizes': [40],
															 'dropouts': [0.35],
															 'kernel_name': '_glorot',
															 'total_registers': 880,
															 'subdirectory': 'only_disorders/D_AD',
															 'name_model': '_SMHD_cnn_gl_880_D_AD',
															 'embedding_types': [dn.EmbeddingType.GLOVE_TWITTER],
															 'embedding_custom_files': [''],
															 'use_embeddings': [dn.UseEmbedding.STATIC]})
						exp, submodel = self.load_submodel_spec_cnn('3', set_params, 'googleTwitter')

				elif id_model == 54:
						self.log.save(
								'Load model t3_SMHD_cnn_gl_880_D_AD_cnn_F459_K3_PNone_B50_E12_D0.21_HLN459_ET_3_UE_3_EF_googleTwitter_glorot CAD'+ str(id_model))
						set_params = dict({'filters_by_layer': [459],
															 'kernels_size': [3],
															 'epochs': [12],
															 'batch_sizes': [50],
															 'dropouts': [0.21],
															 'kernel_name': '_glorot',
															 'total_registers': 880,
															 'subdirectory': 'only_disorders/D_AD',
															 'name_model': '_SMHD_cnn_gl_880_D_AD',
															 'embedding_types': [dn.EmbeddingType.GLOVE_TWITTER],
															 'embedding_custom_files': [''],
															 'use_embeddings': [dn.UseEmbedding.STATIC]})
						exp, submodel = self.load_submodel_spec_cnn('3', set_params, 'googleTwitter')

				all_models.update({'CAD' + str(id_model): {'exp': exp, 'model_class': submodel}})
				# self.rename_layers_diff_submodels(exp, submodel, 'CAD' + str(id_model), 'differenciators/cnn/')
				del exp, submodel

				return all_models

		def load_submodels_multilabel(self, all_models, id_model):
				# Submodels LSTM
				if id_model == 90:
						self.log.save('Load model SMHD_ml_gl_2640_lstm_exp9_var_L3_N16_B40_E32_D0.2_OFadam_HL2_ET_6_UE_3_EF_SMHD-Skipgram-AllUsers-300_glorot.h5 CA'+str(id_model))
						set_params = dict({'model_type': 'LSTM',
															 'log_file': 'SMHD_ml_gl_2640_lstm_L3',
															 'name_model': 'SMHD_ml_gl_2640',
															 'function': 'w2vCustom',
															 'embedding_type': dn.EmbeddingType.WORD2VEC_CUSTOM,
															 'embedding_custom_file': 'SMHD-Skipgram-AllUsers-300.bin',
															 'use_embedding': dn.UseEmbedding.STATIC,
															 'optimizer_function': dn.OptimizerFunction.ADAM.value,
															 'hidden_layer': 2,
															 'neuronios_by_layer': 16,
															 'batch_size': 40,
															 'epochs': 32,
															 'dropouts': 0.2,
															 'kernel_function': 'glorot_uniform'})
						exp, submodel = self.multi_label_model(set_params)
						key_model = 'CA' + str(id_model)

				elif id_model == 91:
						self.log.save('Load model SMHD_ml_gl_2640_var_nl_lstm_exp9_var_L3_N32_B40_E32_D0.2_OFadam_HL4_ET_2_UE_3_EF_glove6B300d_glorot CA' + str(id_model))
						set_params = dict({'model_type': 'LSTM',
															 'log_file': 'SMHD_ml_gl_2640_var_nl_lstm_L3',
															 'name_model': 'SMHD_ml_gl_2640_var_nl',
															 'function': 'glove6B300d',
															 'embedding_type': dn.EmbeddingType.GLOVE_6B,
															 'embedding_custom_file': '',
															 'use_embedding': dn.UseEmbedding.STATIC,
															 'optimizer_function': dn.OptimizerFunction.ADAM.value,
															 'hidden_layer': 4,
															 'neuronios_by_layer': 32,
															 'batch_size': 40,
															 'epochs': 32,
															 'dropouts': 0.2,
															 'kernel_function': 'glorot_uniform'})
						exp, submodel = self.multi_label_model(set_params)
						key_model = 'CA' + str(id_model)

				elif id_model == 92:
						self.log.save('Load model t1_gt_SMHD_ml_gl_2640_var_hl_lstm_exp9_var_L3_N32_B40_E64_D0.2_OFadam_HL4_ET_3_UE_3_EF_gloveTwitter_glorot CA' + str(id_model))
						set_params = dict({'model_type': 'LSTM',
															 'log_file': 't1_gt_SMHD_ml_gl_2640_var_hl',
															 'name_model': 't1_gt_SMHD_ml_gl_2640_var_hl',
															 'function': 'gloveTwitter',
															 'embedding_type': dn.EmbeddingType.GLOVE_TWITTER,
															 'embedding_custom_file': '',
															 'use_embedding': dn.UseEmbedding.STATIC,
															 'optimizer_function': dn.OptimizerFunction.ADAM.value,
															 'hidden_layer': 4,
															 'neuronios_by_layer': 32,
															 'batch_size': 40,
															 'epochs': 64,
															 'dropouts': 0.2,
															 'kernel_function': 'glorot_uniform'})
						exp, submodel = self.multi_label_model(set_params)
						key_model = 'CA' + str(id_model)

				# Submodels CNN
				elif id_model == 93:
						self.log.save('Load model t4_wc_SMHD_cnn_gl_2460_cnn_F100_K3_PNone_B20_E10_D0.2_HLN100_ET_6_UE_3_EF_SMHD-Skipgram-AllUsers-300_glorot CA' + str(id_model))
						set_params = dict({'model_type': 'CNN',
															 'log_file': 't4_wc_SMHD_cnn_gl_2460_cnn',
															 'name_model': 't4_wc_SMHD_cnn_gl_2460',
															 'function': 'w2vCustom',
															 'embedding_type': dn.EmbeddingType.WORD2VEC_CUSTOM,
															 'embedding_custom_file': 'SMHD-Skipgram-AllUsers-300.bin',
															 'use_embedding': dn.UseEmbedding.STATIC,
															 'optimizer_function': dn.OptimizerFunction.ADADELTA.value,
															 'filter_by_layer': 100,
															 'kernel_size': 3,
															 'batch_size': 20,
															 'epochs': 10,
															 'dropouts': 0.2,
															 'kernel_function': 'glorot_uniform'})
						exp, submodel = self.multi_label_model(set_params)
						key_model = 'CD' + str(id_model)

				elif id_model == 94:
						self.log.save('Load model t4_gn_SMHD_cnn_gl_2460_cnn_F250_K5_PNone_B20_E10_D0.5_HLN250_ET_4_UE_3_EF_googleNews_glorot CA' + str(id_model))
						set_params = dict({'model_type': 'CNN',
															 'log_file': 't4_gn_SMHD_cnn_gl_2460_cnn',
															 'name_model': 't4_gn_SMHD_cnn_gl_2460',
															 'function': 'googleNews',
															 'embedding_type': dn.EmbeddingType.WORD2VEC,
															 'embedding_custom_file': '',
															 'use_embedding': dn.UseEmbedding.STATIC,
															 'optimizer_function': dn.OptimizerFunction.ADADELTA.value,
															 'filter_by_layer': 250,
															 'kernel_size': 5,
															 'batch_size': 20,
															 'epochs': 10,
															 'dropouts': 0.5,
															 'kernel_function': 'glorot_uniform'})
						exp, submodel = self.multi_label_model(set_params)
						key_model = 'CD' + str(id_model)

				elif id_model == 95:
						self.log.save('Load model t4_gt_SMHD_cnn_gl_2460_cnn_F250_K3_PNone_B20_E10_D0.5_HLN250_ET_3_UE_3_EF_gloveTwitter_glorot CA' + str(id_model))
						set_params = dict({'model_type': 'CNN',
															 'log_file': 't4_gt_SMHD_cnn_gl_2460_cnn',
															 'name_model': 't4_gt_SMHD_cnn_gl_2460',
															 'function': 'gloveTwitter',
															 'embedding_type': dn.EmbeddingType.GLOVE_TWITTER,
															 'embedding_custom_file': '',
															 'use_embedding': dn.UseEmbedding.STATIC,
															 'optimizer_function': dn.OptimizerFunction.ADADELTA.value,
															 'filter_by_layer': 250,
															 'kernel_size': 3,
															 'batch_size': 20,
															 'epochs': 10,
															 'dropouts': 0.5,
															 'kernel_function': 'glorot_uniform'})
						exp, submodel = self.multi_label_model(set_params)
						key_model = 'CD' + str(id_model)

				# Submodels CNN-LSTM
				elif id_model == 96:
						self.log.save('Load model t83_SMHD_cnn_lstm_gl_2640_cnn_lstm_F64_K4_PNone_B20_E50_D0.5_HLN64_LSTM_N64_D0.2_ET_2_UE_3_EF_glove6B300d_glorot CA' + str(id_model))
						set_params = dict({'model_type': 'CNN-LSTM',
															 'log_file': 't83_SMHD_cnn_lstm_gl_2640_cnn_lstm',
															 'name_model': 't83_SMHD_cnn_lstm_gl_2640',
															 'function': 'glove6B300d',
															 'embedding_type': dn.EmbeddingType.GLOVE_6B,
															 'embedding_custom_file': '',
															 'use_embedding': dn.UseEmbedding.STATIC,
															 'optimizer_function': dn.OptimizerFunction.ADADELTA.value,
															 'filter_by_layer': 64,
															 'kernel_size': 4,
															 'dropouts': 0.5,
															 'neuronios_by_lstm_layer': 64,
															 'dropouts_lstm': 0.2,
															 'epochs': 50,
															 'batch_size': 20,
															 'kernel_function': 'glorot_uniform'})
						exp, submodel = self.multi_label_model(set_params)
						key_model = 'CAD' + str(id_model)

				elif id_model == 97:
						self.log.save(
								'Load model t72_SMHD_cnn_lstm_gl_2640_cnn_lstm_F128_K5_PNone_B20_E25_D0.5_HLN128_LSTM_N256_D0.5_ET_2_UE_3_EF_glove6B300d_glorot CA' + str(
										id_model))
						set_params = dict({'model_type': 'CNN-LSTM',
															 'log_file': 't72_SMHD_cnn_lstm_gl_2640_cnn_lstm',
															 'name_model': 't72_SMHD_cnn_lstm_gl_2640',
															 'function': 'glove6B300d',
															 'embedding_type': dn.EmbeddingType.GLOVE_6B,
															 'embedding_custom_file': '',
															 'use_embedding': dn.UseEmbedding.STATIC,
															 'optimizer_function': dn.OptimizerFunction.ADADELTA.value,
															 'filter_by_layer': 128,
															 'kernel_size': 5,
															 'dropouts': 0.5,
															 'neuronios_by_lstm_layer': 256,
															 'dropouts_lstm': 0.5,
															 'epochs': 25,
															 'batch_size': 20,
															 'kernel_function': 'glorot_uniform'})
						exp, submodel = self.multi_label_model(set_params)
						key_model = 'CAD' + str(id_model)

				else: #98
						self.log.save(
								'Load model t69_SMHD_cnn_lstm_gl_2640_cnn_lstm_F128_K4_PNone_B20_E25_D0.5_HLN128_LSTM_N256_D0.5_ET_2_UE_3_EF_glove6B300d_glorot CA' + str(
										id_model))
						set_params = dict({'model_type': 'CNN-LSTM',
															 'log_file': 't69_SMHD_cnn_lstm_gl_2640_cnn_lstm',
															 'name_model': 't69_SMHD_cnn_lstm_gl_2640',
															 'function': 'glove6B300d',
															 'embedding_type': dn.EmbeddingType.GLOVE_6B,
															 'embedding_custom_file': '',
															 'use_embedding': dn.UseEmbedding.STATIC,
															 'optimizer_function': dn.OptimizerFunction.ADADELTA.value,
															 'filter_by_layer': 128,
															 'kernel_size': 4,
															 'dropouts': 0.5,
															 'neuronios_by_lstm_layer': 256,
															 'dropouts_lstm': 0.5,
															 'epochs': 25,
															 'batch_size': 20,
															 'kernel_function': 'glorot_uniform'})
						exp, submodel = self.multi_label_model(set_params)
						key_model = 'CAD' + str(id_model)

				all_models.update({key_model: {'exp': exp, 'model_class': submodel}})
				del exp, submodel
				return all_models

		def load_submodels(self):
				all_models = dict()
				LSTM_diffs = list(range(6, 11))
				CNN_diffs = list(range(12, 54))

				for key_model, id_models in self.submodels.items():
						for id_model in id_models:
								if id_model == 1:
										all_models = self.load_submodels_1(all_models, key_model)
								elif id_model == 2:
										all_models = self.load_submodels_2(all_models, key_model)
								elif id_model == 3:
										all_models = self.load_submodels_3(all_models, key_model)
								elif id_model in [4, 100, 101, 102]:
										all_models = self.load_submodels_cnn(all_models, key_model, id_model)
								elif id_model in [5, 110, 111]:
										all_models = self.load_submodels_hybrid(all_models, key_model, id_model)
								elif id_model in LSTM_diffs:
										all_models = self.load_submodels_6_to_11(all_models, id_model)
								elif id_model in CNN_diffs:
										all_models = self.load_submodels_12_to_54(all_models, id_model)
								else: #
										all_models = self.load_submodels_multilabel(all_models, id_model)

				self.all_submodels = all_models


		def format_dataset_stacked(self, data_df):
				x_data_lst = []
				y_test = []
				for key_model, value in self.all_submodels.items():
						# The flag type_prediction_label only affects the formatting of y
						self.all_submodels[key_model]['exp'].pp_data.type_prediction_label = self.type_predict_label
						x_test, y_test = self.all_submodels[key_model]['exp'].pp_data.load_subdataset_generic(data_df,
																																																	self.labels_set)
						x_data_lst.append(x_test)

				return x_data_lst, y_test
		
		def training_step(self, train_df, valid_df):
				self.log.save('ensemble_stacked_train_test results')
				x_train_lst, y_train = self.format_dataset_stacked(train_df)
				x_valid_lst, y_valid = self.format_dataset_stacked(valid_df)

				early_stopping_train = EarlyStopping(monitor='val_loss', mode='min', verbose=self.verbose,
																						 patience=self.ensemble_stacked_model.patience_train)
				model_checkpoint = ModelCheckpoint(dn.PATH_PROJECT + self.all_files_path + self.ensemble_stacked_model.model_name + '.h5',
																					 monitor='val_acc', mode='max', save_best_only=True, save_weights_only=False)

				history = self.ensemble_stacked_model.model.fit(x_train_lst, y_train,
																												validation_data=(x_valid_lst, y_valid),
																												batch_size=self.ensemble_stacked_model.batch_size,
																												epochs=self.ensemble_stacked_model.epochs,
																												verbose=self.verbose,
																												callbacks=[early_stopping_train, model_checkpoint])
				hist_df = pd.DataFrame(history.history)

				with open(dn.PATH_PROJECT + self.all_files_path + self.name_class + '_'+ self.job_name + '.csv', mode='w') as file:
						hist_df.to_csv(file)

				# self.ensemble_stacked_model.model.save(dn.PATH_PROJECT + self.all_files_path + self.ensemble_stacked_model.model_name + '.h5')

		def training_test_step(self, data_df):
				self.log.save('ensemble_stacked_test results')
				x_test_lst, y_test = self.format_dataset_stacked(data_df)

				y_hat = self.ensemble_stacked_model.model.predict(x_test_lst, verbose=self.verbose)
				self.save_metrics('ensemble_stacked_test_' + self.job_name, 'ensemble_stacked', y_test, y_hat,
													self.labels_ensemble)
				file_name = self.all_files_path + self.name_class + '_' + 'predict_ensemble_stack_' + self.job_name + '_predict_results'
				self.metrics.save_predict_results(file_name, self.type_predict_label, y_test, y_hat)
				
				return y_test, y_hat

		def get_total_submodels(self):
				total_models = 0
				for key, values in self.submodels.items():
						total_models = total_models + len(values)

				return total_models

		def generate_model(self):
				ensemble_visible = []
				ensemble_outputs = []

				for key_model, value in self.all_submodels.items():
						# The flag type_prediction_label only affects the formatting of y
						self.all_submodels[key_model]['exp'].pp_data.type_prediction_label = self.type_predict_label

						for i, layer in enumerate(self.all_submodels[key_model]['model_class'].model.layers):
								# make not trainable
								layer.trainable = False
								# rename to avoid 'unique layer name' issue
								layer.name = 'ensemble_' + key_model + '_' + str(i + 1) + '_' + layer.name

						# define multi-headed input
						ensemble_visible.append(self.all_submodels[key_model]['model_class'].model.input)
						# concatenate merge output from each model
						ensemble_outputs.append(self.all_submodels[key_model]['model_class'].model.output)

				# Build model
				merge = concatenate(ensemble_outputs)
				first_hidden_layer = True
				total_submodels = self.get_total_submodels()

				for set_hidden in self.ensemble_stacked_conf['hidden_layers']:
						if first_hidden_layer:
								hidden = Dense(units=total_submodels * set_hidden['units'],
															 activation=set_hidden['activation'],
															 use_bias=set_hidden['use_bias'],
															 kernel_initializer=set_hidden['kernel_initializer'],
															 bias_initializer=set_hidden['bias_initializer'],
															 kernel_regularizer=set_hidden['kernel_regularizer'],
															 bias_regularizer=set_hidden['bias_regularizer'],
															 activity_regularizer=set_hidden['activity_regularizer'],
															 kernel_constraint=set_hidden['kernel_constraint'],
															 bias_constraint=set_hidden['bias_constraint'])(merge)
								first_hidden_layer = False
						else:
								hidden = Dense(units=total_submodels * set_hidden['units'],
															 activation=set_hidden['activation'],
															 use_bias=set_hidden['use_bias'],
															 kernel_initializer=set_hidden['kernel_initializer'],
															 bias_initializer=set_hidden['bias_initializer'],
															 kernel_regularizer=set_hidden['kernel_regularizer'],
															 bias_regularizer=set_hidden['bias_regularizer'],
															 activity_regularizer=set_hidden['activity_regularizer'],
															 kernel_constraint=set_hidden['kernel_constraint'],
															 bias_constraint=set_hidden['bias_constraint'])(hidden)

				output = Dense(len(self.labels_ensemble), activation=self.ensemble_stacked_conf['activation_output_fn'])(hidden)
				self.ensemble_stacked_model.model = Model(inputs=ensemble_visible, outputs=output)
				
				self.ensemble_stacked_model.model_name = self.name_class + '_' + self.job_name + '_ens_stk_model'
				plot_model(self.ensemble_stacked_model.model, show_shapes=True,
									 to_file=dn.PATH_PROJECT  + self.all_files_path + self.ensemble_stacked_model.model_name + '.png')

				self.ensemble_stacked_model.model.compile(loss=self.ensemble_stacked_conf['loss_function'],
																									optimizer=self.ensemble_stacked_conf['optmizer_function'],
																									metrics=[self.ensemble_stacked_conf['main_metric']])
				
		def load_pre_trained_model(self, file_name):
				self.ensemble_stacked_model.model = load_model(file_name, custom_objects=None, compile=True)
				
		def generate_stratifeid_folds(self, data_df):
				# Identify total samples by label
				data_df = shuffle(data_df)
				samples_label_by_fold = data_df.groupby('label').size()/self.k_fold

				range_sample = dict()
				for label, total in samples_label_by_fold.items():
						range_sample.update({label: {'ini': 0, 'end': int(total)}})

				# Generate subset with total sample by label in fold,
				# In total, k-folds generate balanced
				folds_list = []
				for k in range(self.k_fold):
						subset = []
						for label in range_sample.keys():
								offset_range = range_sample[label]['end'] - range_sample[label]['ini']
								subset.append(data_df[data_df.label == label][range_sample[label]['ini']:range_sample[label]['end']])
								range_sample[label]['ini'] = range_sample[label]['end']
								range_sample[label]['end'] = range_sample[label]['end'] + offset_range

						folds_list.append(shuffle(pd.concat(subset)).index)

				# Organize folds in train-test, k-1 folds to train and 1 fold to valid
				k_fold_list = []
				for cross_valid_idx in range(self.k_fold):
						fold_test = []
						folds_train = []
						for folds_idx in range(self.k_fold):
								if cross_valid_idx == folds_idx:
										fold_test = folds_list[folds_idx]
								else:
										folds_train.extend(folds_list[folds_idx])

						k_fold_list.append([folds_train, fold_test])

				return k_fold_list

		def generate_folds(self):
				train_df = self.load_train_dataset()
				train_df = train_df.reset_index()
				x = train_df.drop('label', axis=1)
				y = train_df.drop('texts', axis=1)

				folds_list = self.generate_stratifeid_folds(train_df)

				return x, y, folds_list
		
		def test_final_model(self, data_df=pd.DataFrame()):
				self.log.save("\n======= Test Ensemble Model =======")
				
				if data_df.empty:
						test_df = self.load_test_dataset()
				else:
						test_df = data_df
				
				time_ini_rep = datetime.datetime.now()
				if self.job_name == '':
						self.job_name = 'final_test'
				
				y, y_hat = self.training_test_step(test_df)
				
				self.set_period_time_end(time_ini_rep, 'Test Ensemble Model')
				return y, y_hat
		
		def model_training(self):
				self.log.save("\n======= Train-validation Ensemble Model =======")
				time_ini_rep = datetime.datetime.now()
				
				self.load_submodels()
				
				for iteraction_cross in range(1,6):
						print("Interaction cross-validation " % (iteraction_cross))
						x, y, folds_list = self.generate_folds()
						idx_fold = 0
						
						# Train-valid process
						for train_index, test_index in folds_list:
								self.log.save('Iteraction k_fold = ' + str(idx_fold))
		
								x_train, x_test = x.iloc[train_index], x.iloc[test_index]
								y_train, y_test = y.iloc[train_index], y.iloc[test_index]
								
								x_train_df = pd.concat([x_train, y_train], axis=1)
								x_test_df = pd.concat([x_test, y_test], axis=1)
								
								self.job_name = 'train_valid_kf_' + str(idx_fold)
								
								# Train
								self.generate_model()
								self.training_step(x_train_df, x_test_df)
								
								# Validation
								self.job_name = 'test_kf_' + str(idx_fold)
								self.training_test_step(x_test_df)
								
								# Evaluation
								self.load_pre_trained_model(dn.PATH_PROJECT + self.all_files_path + self.ensemble_stacked_model.model_name + '.h5')
								self.job_name = 'final_test_kf_' + str(idx_fold)
								self.test_final_model()
								
								idx_fold = idx_fold + 1

						self.set_period_time_end(time_ini_rep, 'Test Ensemble Model')

		# This function can be applied if all submodels use the same tokenizer,
		# x_data the dataset already encoded using the generic tokenizer.
		def prediction_format_interpretation(self, x_data):
				assert self.ensemble_stacked_model.model != None, "No stacked models were loaded"

				x_data_lst = []
				for idx in range(self.all_submodels.__len__()):
						x_data_lst.append(x_data)

				y_hat = self.ensemble_stacked_model.model.predict(x_data_lst, verbose=self.verbose)

				return y_hat

