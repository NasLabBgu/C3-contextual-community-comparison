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

import os
import time
from os.path import join
import numpy as np
import datetime as dt
import random as rn
import commentjson
from C3_algo.utils import get_models_names
from C3_algo.tf_idf import calc_tf_idf_all_models
from C3_algo.compare import calc_scores_all_models

###################################################### Configurations ##################################################
config_dict = commentjson.load(open(join(os.getcwd(), 'config.json')))
np.random.seed(config_dict["random_seed"])
rn.seed(config_dict["random_seed"])

if not os.path.exists(config_dict['model_output_path']):
        os.makedirs(config_dict['model_output_path'])
########################################################################################################################

if __name__ == "__main__":
    print(f"start time {dt.datetime.now().strftime('%Y-%m-%d %H:%M')}")
    print(f"MODEL_TYPE = {config_dict['model_version']}")
    valid_m_names = get_models_names(path=config_dict['embedding_models_path'])
    valid_m_names = sorted(valid_m_names)
    print(f"\n1 - CALC_TF_IDF -> {config_dict['current_run_flags']['calc_tf_idf']}")
    start_1 = time.time()
    if eval(config_dict['current_run_flags']['calc_tf_idf']):
        calc_tf_idf_all_models(m_names=valid_m_names, config_dict=config_dict)
    print(f"CALC_TF_IDF - elapsed time (min): {(time.time()-start_1)/60}")
    print(f"2.1 - SAVE_DIS_MATRIX -> {config_dict['current_run_flags']['save_dis_matrix']}")
    start_2 = time.time()
    calc_scores_all_models(m_names=valid_m_names, config_dict=config_dict)
    print(f"CALC_SCORES - elapsed time (min): {(time.time()-start_2)/60}")
