# Authors: Abraham Israeli (isabrah@post.bgu.ac.il)
# Python version: 3.7
# Last update: 13.6.2022

# this is the main file of the C3 algorithm
# see publication here: https://papers.ssrn.com/sol3/papers.cfm?abstract_id=4051307)
# Note that you have to configure the 'config.json' file first before you run this file
import os
import time
from os.path import join
import numpy as np
import datetime as dt
import random as rn
import commentjson
from C3_algo.utils import get_models_names, check_input_validity
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
    # making sure that the input we got is valid (configurations and data)
    check_input_validity(config_dict)
    valid_m_names = get_models_names(path=config_dict['embedding_models_path'])
    valid_m_names = sorted(valid_m_names)
    start_1 = time.time()
    if eval(config_dict['current_run_flags']['calc_tf_idf']):
        calc_tf_idf_all_models(m_names=valid_m_names, config_dict=config_dict)
    print(f"CALC_TF_IDF finished - elapsed time (min): {(time.time()-start_1)/60}")
    start_2 = time.time()
    calc_scores_all_models(m_names=valid_m_names, config_dict=config_dict)
    print(f"CALC_SCORES finished - elapsed time (min): {(time.time()-start_2)/60}")
