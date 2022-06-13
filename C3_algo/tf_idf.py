# Authors: Abraham Israeli (isabrah@post.bgu.ac.il)
# Python version: 3.7
# Last update: 13.6.2022

from os.path import join as opj
from C3_algo.utils import load_model
import pickle
import math
import multiprocessing as mp
import numpy as np
import pandas as pd


def add_wd_count(model, all_wdc, n_wc, config_dict):
    """
    a function supporting the idf calculation. The code does 2 main things:
    1. calculates the word-count per the given model and adds the counting to the all_wdc dictionary (this supports the
    IDF calculation in later phases).
    2. calculates the word-count (wc) per the given model and save it to csv. This is useful for later usage (if needed)
    for the tf-idf calculation per community.

    :param model: Gensim model
        the model to use for retrieving its words for the calculation. Can be either word2vec of fastext.
    :param all_wdc: Dict
        the cumulative count of documents containing each word (including words from all models)
    :param n_wc: string
        file name for wc (word-count).
    :param config_dict: dict
        the configuration dictionary with all the relevant parameters. This dictionary is the loaded dictionary from the
        config.json file.
    :return: dict
        the updated cumulative count of documents containing each word.
        Note that in addition, the function also saves the wc for the given model as a csv file.
    """
    tf_idf_path = config_dict['tf_idf_path']
    # document counter - in how many documents (communities) the word appears (indicator)
    dc = dict()
    # word counter - how many times the word appears in each document (int counter)
    wc = dict()
    # looping over each word in the vocabulary of the model
    for (k, v) in model.wv.vocab.items():
        dc[k] = 1  # add 1 document count to the words appear in the current document (community)
        wc[k] = v.count  # add 1 document count to the words appear in the current document (community)
    # saving the results as csv
    wc_as_df = pd.DataFrame.from_dict(wc, orient='index', columns=['value'])
    wc_as_df.reset_index(inplace=True)
    wc_as_df = wc_as_df.rename(columns={'index': 'word'})
    wc_as_df.to_csv(opj(tf_idf_path, n_wc + '.csv'), index=False)
    # aggregation
    all_wdc = {k: all_wdc.get(k, 0) + dc.get(k, 0) for k in set(all_wdc) | set(dc)}
    return all_wdc


def calc_idf(m_names, config_dict):
    """
    calculate the IDF (inverse document frequency) of the corpus. In addition, creates word-count csv per model --
    these are used while calculating the tf-idf per community.

    :param m_names: list
        names of the models to load.
    :param config_dict: dict
        the configuration dictionary with all the relevant parameters. This dictionary is the loaded dictionary from the
        config.json file.
    :return: dict
        idf score for each word in the corpus (words from all communities).
    """
    idf_f_name = 'idf_dict.csv'
    data_path = config_dict['embedding_models_path']
    tf_idf_path = config_dict['tf_idf_path']
    N = len(m_names)
    total_wdc = dict()  # document-word count in all documents (communities)
    # looping over each model (community) to calculate the IDF + saving wc (word count) csv, so later we'll be able
    # to use it for the tf-idf calculation
    for i, m_name in enumerate(m_names):
        try:
            curr_model = load_model(path=data_path, name=m_name, config_dict=config_dict)
        except IndexError:
            print(f"Model of SR {m_name} could not be found. Be aware - you should add this embedding "
                  f"model to the required folder ({data_path})")
            continue
        n_wc = 'wc_' + m_name
        # the actual calculation + saving of the wc to a csv file
        total_wdc = add_wd_count(model=curr_model, all_wdc=total_wdc, n_wc=n_wc, config_dict=config_dict)
    # in case we wish to calculate (and save!) the idf from scratch
    if eval(config_dict['current_run_flags']['calc_idf']):
        # idf (inverse document frequency):  idf(t, D) = log(N / |{d in D : t in T}|)
        #      dividing the total number of documents by the number of documents containing the term,
        #      and then taking the logarithm of that quotient
        idf_as_dict = {k: math.log(N/v) for (k, v) in total_wdc.items()}
        # saving the IDF result as a csv file
        idf_as_df = pd.DataFrame.from_dict(idf_as_dict, orient='index', columns=['value'])
        idf_as_df.reset_index(inplace=True)
        idf_as_df = idf_as_df.rename(columns={'index': 'word'})
        idf_as_df.to_csv(opj(tf_idf_path, 'idf_dict.csv'), index=False)
    # in case we wish to load the idf from an existing csv file (we assume this file exists)
    else:
        # loading the file
        idf_as_df = pd.read_csv(opj(tf_idf_path, idf_f_name))
        idf_as_dict = idf_as_df.set_index('word')['value'].to_dict()
    return idf_as_dict


def calc_tf_idf(m_name, idf, config_dict):
    """
    calculates the tf-idf for a given model (community).
    Note that the wc (word-count) file most exists when calculation is done. In this code, the creation of the wc csv
    file is done by the 'calc_idf' function.

    :param m_name: string
        the model's name. This is actually the community name (e.g., 'java').
    :param idf: dict
        idf score for each word in the corpus (words from all communities).
    :param config_dict: dict
        the configuration dictionary with all the relevant parameters. This dictionary is the loaded dictionary from the
        config.json file.
    :return: None
        the function saves 'tf-idf' vector per the given community as a csv file under the location specified in the
        config file ('tf_idf_path')
    """
    # tf (term frequency):  tf(t,d)= f[t,d] (the raw count of term t in document d)
    # tf_idf(t,d,D) = tf(t,d) * idf(t,D)

    wc_f_name = 'wc_' + m_name
    tf_idf_path = config_dict['tf_idf_path']
    wc_path = opj(tf_idf_path, wc_f_name + '.csv')
    wc_as_df = pd.read_csv(wc_path)
    wc_as_dict = wc_as_df.set_index('word')['value'].to_dict()

    # the 0 in the get is in case when the value does not appear in the IDF dict (should not occur, but just in case)
    tf_idf = {k: v * idf.get(k, 0) for (k, v) in wc_as_dict.items()}
    m_f_name = 'tf_idf_' + m_name
    # saving to disk
    tf_idf_as_df = pd.DataFrame.from_dict(tf_idf, orient='index', columns=['value'])
    tf_idf_as_df.reset_index(inplace=True)
    tf_idf_as_df = tf_idf_as_df.rename(columns={'index': 'word'})
    tf_idf_as_df.to_csv(opj(tf_idf_path, m_f_name + '.csv'), index=False)


def calc_tf_idf_all_models(m_names, config_dict):
    """
    calculates the tf-idf value per model (community). We are doing it in a multiprocess way to run the code faster.
    :param m_names: list
        list of the model names to take into account while calculating.
    :param config_dict: dict
        the configuration dictionary with all the relevant parameters. This dictionary is the loaded dictionary from the
        config.json file.
    :return: None
        the function saves 'tf-idf' vector for all the communities in the corpus as a csv files under the location
        specified in the config file ('tf_idf_path').
    """
    print("Starting the IDF calculation")
    idf = calc_idf(m_names=m_names, config_dict=config_dict)
    lst = []
    for m in m_names:
        lst.append((m, idf, config_dict))
    print("Ended the IDF calculation. Starting the TF-IDF calculation per community (multiprocess)")
    pool = mp.Pool(processes=config_dict['general_config']['cpu_count'])
    with pool as pool:
        pool.starmap(calc_tf_idf, lst)


def get_vocab_length(m_name, config_dict):
    """
    returnes the vocabulary length of a given model.
    :param m_name: str
        the name of the model (community) to analyze.
    :param config_dict: dict
        the configuration dictionary with all the relevant parameters. This dictionary is the loaded dictionary from the
        config.json file.
    :return: dict
        a dictionary with two keys:
        1. 'm_name' -- name of the community (e.g., 'java').
        2. 'vocab_length -- vocabulary length (int).
    """
    model = load_model(path=config_dict['data_path'], name=m_name, config_dict=config_dict)
    res = {'m_name': m_name, 'vocab_length': len(model.wv.index2entity)}
    return res