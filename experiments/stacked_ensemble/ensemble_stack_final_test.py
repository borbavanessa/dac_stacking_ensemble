# Load libs standard python and custom
import sys
import utils.definition_network as dn

from network_model.stacked_ensemble import StackedEnsemble

def set_params_stacked_ensemble(dataset_train_path, dataset_test_path, use_submodel, b_size, h_layer, epochs):
		epoch = epochs
		batch_size = b_size
		neurons_by_submodel = 12
		hidden_layer = h_layer

		metric = 'accuracy'
		loss_fn = 'binary_crossentropy'
		activation_output_fn = 'sigmoid'
		optimizer_fn = 'adam'
		activation_hidden_fn = 'tanh'
		kernel_initializer = 'glorot_uniform'
		use_bias = True
		bias_initializer = 'zeros'
		kernel_regularizer = None
		bias_regularizer = None
		activity_regularizer = None
		kernel_constraint = None
		bias_constraint = None
		path_submodels = dn.PATH_PROJECT + "weak_classifiers/"
		type_submodels = dn.TypePredictionLabel.MULTI_LABEL_CATEGORICAL

		hidden_layers_set = []
		for idx in range(hidden_layer):
						hidden_layers_set.append(
										dict({'units': neurons_by_submodel,
																'activation': activation_hidden_fn,
																'use_bias': use_bias,
																'kernel_initializer': kernel_initializer,
																'bias_initializer': bias_initializer,
																'kernel_regularizer': kernel_regularizer,
																'bias_regularizer': bias_regularizer,
																'activity_regularizer': activity_regularizer,
																'kernel_constraint': kernel_constraint,
																'bias_constraint': bias_constraint}))

		set_network = dict({'epochs': epoch,
																						'batch_size': batch_size,
																						'patient_train': int(
																										epoch / 2),
																						'activation_output_fn': activation_output_fn,
																						'loss_function': loss_fn,
																						'optmizer_function': optimizer_fn,
																						'main_metric': metric,
																						'dataset_train_path': dataset_train_path,
																						'dataset_test_path': dataset_test_path,
																						'path_submodels': path_submodels,
																						'type_submodels': type_submodels,
																						'submodels': use_submodel,
																						'hidden_layers': hidden_layers_set
																						})

		name_test = 'E_' + str(epoch) + '_BS_' + str(batch_size) + \
														'_US_' + str(len(use_submodel)) + '_N_' + str(neurons_by_submodel) + \
														'_HL_' + str(hidden_layer) + '_M_' + str(metric)[0:2] + \
														'_AO_' + str(bias_constraint)[0:2] + \
														'_LF_' + str(loss_fn)[0:2] + '_OP_' + str(optimizer_fn) + \
														'_AH_' + str(activation_hidden_fn)[0:2] + '_KI_' + str(kernel_initializer)[0:2] + \
														'_UB_' + str(use_bias)[0] + '_BI_' + str(bias_initializer)[0:2] + \
														'_KR_' + str(kernel_regularizer) + '_BR_' + str(bias_regularizer) + \
														'_AR_' + str(activity_regularizer) + '_KC_' + str(kernel_constraint)[0:2] + \
														'_BC_' + str(bias_constraint)[0:2]

		return name_test, set_network

def load_stacked_ensemble(name_test, set_network):
		print("Experiment: " + name_test)
		ensemble_stk = StackedEnsemble(name_test, 1, '')

		ensemble_stk.list_report_metrics = []
		ensemble_stk.ensemble_stacked_conf = set_network
		ensemble_stk.k_fold = 5
		ensemble_stk.labels_set = ['control', 'anxiety', 'depression']
		ensemble_stk.labels_ensemble = ['control', 'anxiety', 'depression']

		ensemble_stk.type_predict_label = dn.TypePredictionLabel.MULTI_LABEL_CATEGORICAL
		ensemble_stk.metrics_based_sample = False

		ensemble_stk.set_network_params_ensemble_stack()
		ensemble_stk.model_training()
		
		# ensemble_stk.load_submodels()
		# ensemble_stk.load_pre_trained_model(dn.PATH_PROJECT + "DAC Stacking/t25_E_16_BS_8_US_2_N_12_HL_3_M_ac_AO_No_LF_bi_OP_adam_AH_ta_KI_gl_UB_T_BI_ze_KR_None_BR_None_AR_None_KC_No_BC_No_train_valid_kf_2_ens_stk_model.h5")
		# ensemble_stk.test_final_model()

		return ensemble_stk


def generate_test(option):
		dataset_train_path = 'dataset/anx_dep_multilabel/SMHD_multi_label_test_train_2112.df'
		dataset_test_path = 'dataset/anx_dep_multilabel/SMHD_multi_label_test_test_528.df'

		# 1 - 4: lstm only
		if option == '1':
				use_submodel = dict({'CA': [1, 2, 3], 'CD': [1, 2, 3], 'CAD': [1, 2, 3]})
				name_test, set_network = set_params_stacked_ensemble(dataset_train_path, dataset_test_path, use_submodel, 8, 3, 16)
			
		elif option == '2':
				use_submodel = dict({'CA': [1, 2, 3], 'CD': [1, 2, 3]})
				name_test, set_network = set_params_stacked_ensemble(dataset_train_path, dataset_test_path, use_submodel, 8, 3, 16)

		# 3 - 4: test LSTM + CNN
		elif option == '3':
				use_submodel = dict({'CA': [2, 3, 4], 'CD': [2, 3, 4], 'CAD': [2, 3, 4]})
				name_test, set_network = set_params_stacked_ensemble(dataset_train_path, dataset_test_path, use_submodel, 8, 3, 16)

		elif option == '4':
				use_submodel = dict({'CA': [2, 3, 4], 'CD': [2, 3, 4]})
				name_test, set_network = set_params_stacked_ensemble(dataset_train_path, dataset_test_path, use_submodel, 8, 3, 16)

		# 5 - 6: LSTM + LSTM-CNN
		elif option == '5':
				use_submodel = dict({'CA': [2, 3, 5], 'CD': [2, 3, 5], 'CAD': [2, 3, 5]})
				name_test, set_network = set_params_stacked_ensemble(dataset_train_path, dataset_test_path, use_submodel, 8, 3, 16)

		elif option == '6':
				use_submodel = dict({'CA': [2, 3, 5], 'CD': [2, 3, 5]})
				name_test, set_network = set_params_stacked_ensemble(dataset_train_path, dataset_test_path, use_submodel, 8, 3, 16)

		# 7 - 12: LSTM lecun, CNN, LSTM_CNN
		elif option == '7':
				use_submodel = dict({'CA': [2, 4, 5], 'CD': [2, 4, 5], 'CAD': [2, 4, 5]})
				name_test, set_network = set_params_stacked_ensemble(dataset_train_path, dataset_test_path, use_submodel, 8, 3, 16)

		elif option == '8':
				use_submodel = dict({'CA': [2, 4, 5], 'CD': [2, 4, 5]})
				name_test, set_network = set_params_stacked_ensemble(dataset_train_path, dataset_test_path, use_submodel, 8, 3, 16)

		elif option == '9':
				use_submodel = dict({'CA': [3, 4, 5], 'CD': [3, 4, 5], 'CAD': [3, 4, 5]})
				name_test, set_network = set_params_stacked_ensemble(dataset_train_path, dataset_test_path, use_submodel, 8, 3, 16)

		elif option == '10':
				use_submodel = dict({'CA': [3, 4, 5], 'CD': [3, 4, 5]})
				name_test, set_network = set_params_stacked_ensemble(dataset_train_path, dataset_test_path, use_submodel, 8, 3, 16)

		elif option == '11':
				use_submodel = dict({'CA': [4, 5], 'CD': [4, 5], 'CAD': [4, 5]})
				name_test, set_network = set_params_stacked_ensemble(dataset_train_path, dataset_test_path, use_submodel, 8, 3, 16)

		elif option == '12':
				use_submodel = dict({'CA': [4, 5], 'CD': [4, 5]})
				name_test, set_network = set_params_stacked_ensemble(dataset_train_path, dataset_test_path, use_submodel, 8, 3, 16)
		
		# 13 - 15: lstm only with diffenciators submodels
		elif option == '13':
				use_submodel = dict({'CA': [1, 2, 3], 'CD': [1, 2, 3], 'CAD': [6, 7, 8]})
				name_test, set_network = set_params_stacked_ensemble(dataset_train_path, dataset_test_path, use_submodel, 8, 3, 16)

		elif option == '14':
				use_submodel = dict({'CA': [1, 2, 3], 'CD': [1, 2, 3], 'CAD': [6, 9, 10]})
				name_test, set_network = set_params_stacked_ensemble(dataset_train_path, dataset_test_path, use_submodel, 8, 3, 16)

		else:
				use_submodel = dict({'CA': [1, 2, 3], 'CD': [1, 2, 3], 'CAD': [6, 7, 11]})
				name_test, set_network = set_params_stacked_ensemble(dataset_train_path, dataset_test_path, use_submodel, 8, 3, 16)

		ens_stk = load_stacked_ensemble('t'+option+'_'+name_test, set_network)
		
		if option == '1':
				for i in [0, 1, 2, 3]:
						ens_stk.job_name = 'final_test_k' + str(i)
						ens_stk.load_pre_trained_model(dn.PATH_PROJECT + "DAC Stacking EC/t1_E_16_BS_8_US_3_N_12_HL_3_M_ac_AO_No_LF_bi_OP_adam_AH_ta_KI_gl_UB_T_BI_ze_KR_None_BR_None_AR_None_KC_No_BC_No_train_valid_kf_"+str(i)+"_ens_stk_model.h5")
						ens_stk.test_final_model()

		elif option == '2':
				for i in [0, 1, 2, 3]:
						ens_stk.job_name = 'final_test_k' + str(i)
						ens_stk.load_pre_trained_model(dn.PATH_PROJECT + "DAC Stacking/t2_E_16_BS_8_US_2_N_12_HL_3_M_ac_AO_No_LF_bi_OP_adam_AH_ta_KI_gl_UB_T_BI_ze_KR_None_BR_None_AR_None_KC_No_BC_No_train_valid_kf_"+str(i)+"_ens_stk_model.h5")
						ens_stk.test_final_model()


def generate_outperformance_diffs(group_test_diff_mlp):
		dataset_train_path = 'dataset/anx_dep_multilabel/SMHD_multi_label_test_train_2112.df'
		dataset_test_path = 'dataset/anx_dep_multilabel/SMHD_multi_label_test_test_528.df'

		# First, using same stacking topology that LSTM differenciators was tested (t2), to compare if news diffs are the best

		use_submodel = dict({'CA': [1, 2, 3], 'CD': [1, 2, 3]})

		# Models CNNs
		if group_test_diff_mlp == '1':
				use_submodel.update({'CAD': [20, 23, 24]}) # only A-D
		elif group_test_diff_mlp == '2':
				use_submodel.update({'CAD': [12, 17, 22]}) # only A-D
		elif group_test_diff_mlp == '3':
				use_submodel.update({'CAD': [14, 16, 18]}) # only A-D
		elif group_test_diff_mlp == '4':
				use_submodel.update({'CAD': [13, 15, 19]}) # only A-D
		elif group_test_diff_mlp == '5':
				use_submodel.update({'CAD': [26, 37, 39]}) # only A-AD
		elif group_test_diff_mlp == '6':
				use_submodel.update({'CAD': [28, 31, 32]}) # only A-AD
		elif group_test_diff_mlp == '7':
				use_submodel.update({'CAD': [27, 35, 38]}) # only A-AD
		elif group_test_diff_mlp == '8':
				use_submodel.update({'CAD': [25, 33, 36]}) # only A-AD
		elif group_test_diff_mlp == '9':
				use_submodel.update({'CAD': [29, 30, 34]}) # only A-AD
		elif group_test_diff_mlp == '10':
				use_submodel.update({'CAD': [40, 41, 43]}) # only D-AD
		elif group_test_diff_mlp == '11':
				use_submodel.update({'CAD': [20, 35, 40]}) # A-D, A-AD, D-AD

		# Models Ray Tunning
		elif group_test_diff_mlp == '12':
				use_submodel.update({'CAD': [6, 45, 48]})  # Lstm only A-D, A-AD, D-AD
		elif group_test_diff_mlp == '13':
				use_submodel.update({'CAD': [6, 47, 48]})  # Lstm only A-D, A-AD, D-AD
		elif group_test_diff_mlp == '14':
				use_submodel.update({'CAD': [49, 50, 54]}) # CNN only A-D, A-AD, D-AD
		elif group_test_diff_mlp == '15':
				use_submodel.update({'CAD': [49, 51, 54]}) # CNN only A-D, A-AD, D-AD
		elif group_test_diff_mlp == '16':
				use_submodel.update({'CAD': [49, 50, 53]}) # CNN only A-D, A-AD, D-AD
		elif group_test_diff_mlp == '17':
				use_submodel.update({'CAD': [6, 45, 53]})  # Lstm e CNN A-D, A-AD, D-AD
		elif group_test_diff_mlp == '17':
				use_submodel.update({'CAD': [6, 45, 53]})  # Lstm e CNN A-D, A-AD, D-AD
		elif group_test_diff_mlp == '18':
				use_submodel.update({'CAD': [49, 45, 53]})  # Lstm e CNN A-D, A-AD, D-AD
		else: #'19'
				use_submodel.update({'CAD': [45, 53]})  # Lstm e CNN A-D, A-AD, D-AD

		name_test, set_network = set_params_stacked_ensemble(dataset_train_path, dataset_test_path, use_submodel, 8, 3, 16)
		ens_stk = load_stacked_ensemble('t2_df' + group_test_diff_mlp + '_' + name_test, set_network)


def generate_multilabel_stacked(option):
		dataset_train_path = 'dataset/anx_dep_multilabel/SMHD_multi_label_test_train_2112.df'
		dataset_test_path = 'dataset/anx_dep_multilabel/SMHD_multi_label_test_test_528.df'

		if option == '1':
				use_submodel = dict({'CA': [90, 91, 92]})
		elif option == '2':
				use_submodel = dict({'CD': [93, 94, 95]})
		elif option == '3':
				use_submodel = dict({'CAD': [96, 97, 98]})
		elif option == '4':
				use_submodel = dict({'CA': [90, 91, 92], 'CD': [93, 94, 95], 'CAD': [96, 97, 98]})
		elif option == '5':
				use_submodel = dict({'CA': [90], 'CD': [93], 'CAD': [96]})

		name_test, set_network = set_params_stacked_ensemble(dataset_train_path, dataset_test_path, use_submodel, 8, 3, 16)
		load_stacked_ensemble('t2_df_opt' + option + '_' + name_test, set_network)


def generate_stacked_best_results(option):
		dataset_train_path = 'dataset/anx_dep_multilabel/SMHD_multi_label_test_train_2112.df'
		dataset_test_path = 'dataset/anx_dep_multilabel/SMHD_multi_label_test_test_528.df'
		
		# Only best CNNs
		if option == '21':
				use_submodel = dict({'CA': [4, 101, 102], 'CD': [4, 101, 102], 'CAD': [100, 101, 102]}) # DAC EC
		
		elif option == '22':
				use_submodel = dict({'CA': [4, 101, 102], 'CD': [4, 101, 102]}) # DAC

		elif option == '23':
				use_submodel = dict({'CA': [4, 101, 102], 'CD': [4, 101, 102], 'CAD': [17, 32, 40]}) # DAC DT
		
		# Only best Hybrid
		elif option == '24':
				use_submodel = dict({'CA': [5, 110, 111], 'CD': [5, 110, 111], 'CAD': [5, 110, 111]}) #DAC EC

		# Only best Hybrid
		elif option == '25':
				use_submodel = dict({'CA': [5, 110, 111], 'CD': [5, 110, 111]}) # DAC

		# Only best LSTM
		elif option == '26':
				use_submodel = dict({'CA': [1, 2, 3], 'CD': [1, 2, 3], 'CAD': [6, 45, 47]}) # DAC DT

		# Only best each
		elif option == '27':
				use_submodel = dict({'CA': [2, 102, 5], 'CD': [2, 101, 110]}) # DAC

		elif option == '28':
				use_submodel = dict({'CA': [2, 102, 5], 'CD': [2, 101, 110], 'CAD': [3, 100, 110]}) # DAC EC

		elif option == '29': # Best each LSTM DF
				use_submodel = dict({'CA': [2, 102, 5], 'CD': [2, 101, 110], 'CAD': [17, 45, 40]}) # DAC DT

		elif option == '30': # Best each CNN DF
				use_submodel = dict({'CA': [2, 102, 5], 'CD': [2, 101, 110], 'CAD': [17, 32, 40]}) # DAC DT

		name_test, set_network = set_params_stacked_ensemble(dataset_train_path, dataset_test_path, use_submodel, 8, 3, 16)
		ens_stk = load_stacked_ensemble('t' + option + '_' + name_test, set_network)
		
		if option == '27':
				for i in [0, 2, 3]:
						ens_stk.job_name = 'final_test_k' + str(i)
						ens_stk.load_pre_trained_model(dn.PATH_PROJECT + "DAC Stacking/t27_E_16_BS_8_US_2_N_12_HL_3_M_ac_AO_No_LF_bi_OP_adam_AH_ta_KI_gl_UB_T_BI_ze_KR_None_BR_None_AR_None_KC_No_BC_No_train_valid_kf_"+str(i)+"_ens_stk_model.h5")
						ens_stk.test_final_model()

		elif option == '21':
				for i in [0, 1, 2, 3]:
						ens_stk.job_name = 'final_test_k' + str(i)
						ens_stk.load_pre_trained_model(dn.PATH_PROJECT + "DAC Stacking EC/t21_E_16_BS_8_US_3_N_12_HL_3_M_ac_AO_No_LF_bi_OP_adam_AH_ta_KI_gl_UB_T_BI_ze_KR_None_BR_None_AR_None_KC_No_BC_No_train_valid_kf_"+str(i)+"_ens_stk_model.h5")
						ens_stk.test_final_model()

		elif option == '28':
				for i in [0, 1, 2, 3]:
						ens_stk.job_name = 'final_test_k' + str(i)
						ens_stk.load_pre_trained_model(dn.PATH_PROJECT + "DAC Stacking EC/t28_E_16_BS_8_US_3_N_12_HL_3_M_ac_AO_No_LF_bi_OP_adam_AH_ta_KI_gl_UB_T_BI_ze_KR_None_BR_None_AR_None_KC_No_BC_No_train_valid_kf_"+str(i)+"_ens_stk_model.h5")
						ens_stk.test_final_model()

if __name__ == '__main__':
		arg = sys.argv[1]

		generate_multilabel_stacked(arg)

		generate_outperformance_diffs(arg)

		generate_test(arg)
		
		generate_stacked_best_results(arg)