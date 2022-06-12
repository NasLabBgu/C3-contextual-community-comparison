# Authors: Shani Cohen (ssarusi)
# Python version: 3.7
# Last update: 14.6.2019

# The general flow of code is:
# 1. this code (main) - generates all folders+files of the 3com algo
# 2. tagging file (from 'analysis' folder) - enrich data (e.g., intersection/union, gold label information)
# 3. clustering analysis (from 'analysis' folder) - statistical test  (within, between based)
# 4. correlation analysis (from 'analysis' folder) - generated corr between 3com to other measures
# 5. heatmap analysis (from 'analysis' folder) - generated heatmaps of the distances between 2 communities
# this can be done, only 'main' runs again with the option to save distances between communities
# ("save_dis_matrix": "False"

import sys
import os
# adding the parent path so we'll be able to import internal packages
sys.path.append(os.path.dirname(os.getcwd()))
from three_com_algo.utils import get_models_names
from three_com_algo.tf_idf import calc_tf_idf_all_models, calc_vocab_distribution
from three_com_algo.compare import calc_scores_all_models
import time
from os.path import join
import numpy as np
import pickle
import datetime as dt
import random as rn
import commentjson
import pandas as pd

###################################################### Configurations ##################################################
config_dict = commentjson.load(open(os.path.join(os.path.dirname((os.getcwd())), 'configurations',
                                                 'three_com_algo_config.json')))
machine = 'yalla' if sys.platform == 'linux' else os.environ['COMPUTERNAME']
config_dict['machine'] = machine
# adjusting the config_dict to current run (e.g., model_type to paths)
config_dict['data_path'][machine] = join(config_dict['data_path'][machine], config_dict['general_config']['model_type'])
model_output_path = config_dict['model_output_path'][machine]
config_dict['vocab_distr_path'] = {machine: join(model_output_path, 'vocab_distribution',
                                                          config_dict['general_config']['model_type'])}
config_dict['tf_idf_path'] = {machine: join(model_output_path, 'tf_idf', config_dict['general_config']['model_type'])}
config_dict['user_based_word_weights_path'] = {machine: join(model_output_path, 'user_based_word_weights',
                                                             config_dict['general_config']['model_type'])}
config_dict['scores_path'] = {machine: join(model_output_path, 'scores', config_dict['general_config']['model_type'])}
config_dict['dis_path'] = {machine: join(model_output_path, 'dis_path', config_dict['general_config']['model_type'])}
config_dict['combinations_path'] = {machine: join(model_output_path, 'combinations',
                                                  config_dict['general_config']['model_type'])}
pairs_list = list(set((pd.read_csv('/data/home/isabrah/3com_project_data/sr_names_only_the_chosen_1565.csv')['name'])))
pairs_list = [n for n in pairs_list if n != 'movies']
pairs_list = sorted([n+'_parta' for n in pairs_list] + [n+'_partb' for n in pairs_list])
config_dict['current_run_flags']['ad_hoc_names'] = str(pairs_list)
#config_dict['current_run_flags']['ad_hoc_names'] = str(list(set((pd.read_csv('/data/home/isabrah/3com_project_data/smart_sampled_102_srs_out_of_the_1565_ones.csv')['name']))))

np.random.seed(config_dict["random_seed"])
rn.seed(config_dict["random_seed"])
for p in [config_dict['vocab_distr_path'][machine], config_dict['tf_idf_path'][machine],
          config_dict['scores_path'][machine], config_dict['dis_path'][machine],
          config_dict['combinations_path'][machine], config_dict['labels_path'][machine],
          config_dict['user_based_word_weights_path'][machine]]:
    if not os.path.exists(p):
        os.makedirs(p)
########################################################################################################################

if __name__ == "__main__":
    print(f"start time {dt.datetime.now().strftime('%Y-%m-%d %H:%M')}")
    print(f"MODEL_TYPE = {config_dict['general_config']['model_type']}")
    print(f"VOCAB_PERC_THRES = {config_dict['algo_parmas']['vocab_perc_thres']}")
    #print(f"COMMUNITIES TO COMPARE = {c.N}")
    print(f"METHOD = {config_dict['algo_parmas']['method']}")
    # in case ad_hoc_names is a real list and not None - means we want to run the algorithm only to subset of SRs
    if eval(config_dict['current_run_flags']['ad_hoc_names']) is not None:
        print(f"AD_HOC_NAMES = {config_dict['current_run_flags']['ad_hoc_names']}")
        # valid_m_names will hold the names of the srs to compare
        valid_m_names = eval(config_dict['current_run_flags']['ad_hoc_names'])
    # case we want to run the algorithm for ALL SRs
    else:
        m_names = get_models_names(config_dict=config_dict)
        m_names = sorted(m_names)
        # case we wish to calc to distributions of vocab (case it was not done in the past)
        if eval(config_dict['current_run_flags']['apply_vocab_thres']):
            valid_m_names = calc_vocab_distribution(m_names=m_names, config_dict=config_dict)
            # saving the names of valid models to be used (passed the threshold)
            with open(join(config_dict['vocab_distr_path'][machine], 'valid_m_names' + '.pickle'), 'wb') as handle:
                pickle.dump(valid_m_names, handle, protocol=pickle.HIGHEST_PROTOCOL)
        # case the valid names of SRs to be used already been calculated (pickle exists)
        elif eval(config_dict['current_run_flags']['load_valid_names']):
            with open(join(config_dict['vocab_distr_path'][machine], 'valid_m_names' + '.pickle'), 'rb') as handle:
                valid_m_names = pickle.load(handle)
            # valid_m_names.to_csv(join(c.vocab_distr_path, 'valid_m_names' + '.csv'))
            print(f"valid_m_names: {len(valid_m_names)}")
            #assert (len(valid_m_names) == c.N)
        # case we wish to run the algorithm over ALL SRs (no threhosld, no usage of vocabulary already created in past)
        else:
            valid_m_names = m_names

    print(f"\n1 - CALC_TF_IDF -> {config_dict['current_run_flags']['calc_tf_idf']}")
    start_1 = time.time()
    if eval(config_dict['current_run_flags']['calc_tf_idf']):
        calc_tf_idf_all_models(m_names=valid_m_names, config_dict=config_dict)
    print(f"CALC_TF_IDF - elapsed time (min): {(time.time()-start_1)/60}")

    print(f"\n2 - CALC_SCORES -> {config_dict['current_run_flags']['calc_scores']}")
    print(f"2.1 - SAVE_DIS_MATRIX -> {config_dict['current_run_flags']['save_dis_matrix']}")
    start_2 = time.time()
    if eval(config_dict['current_run_flags']['calc_scores']):
        calc_scores_all_models(m_names=valid_m_names, config_dict=config_dict)
    print(f"CALC_SCORES - elapsed time (min): {(time.time()-start_2)/60}")
