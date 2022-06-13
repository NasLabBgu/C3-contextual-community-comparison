# Authors: Abraham Israeli (isabrah@post.bgu.ac.il)
# Python version: 3.7
# Last update: 13.6.2022

import os
from os.path import join
from gensim.models import Word2Vec, FastText
import re
import pandas as pd


def get_models_names(path):
    """
    retrieve all model names found in the path given. Model file names must include the word 'model' and must have a
    '.model' suffix.
    :param path: string.
        path of the files to look for the models.
    :return: list.
        a list with all file names found.
    """
    lst = [f for f in os.listdir(path) if re.match(r".*_model_" + '.*.' + r"\.model$", f)]
    return [x.split('_model_')[0] for x in lst]


def load_model(path, name, config_dict):
    """
    loads a given model . The model file name must starts with the 'name' given and must ends with the '.model' suffix.
    :param path: str
        path to where the model is saved.
    :param name: str
        name of the community. The file name must starts with this string (e.g., 'java_best_model.model).
    :param config_dict: dict
        the configuration dictionary with all the relevant parameters. This dictionary is the loaded dictionary from the
        config.json file.
    :return: model
        a gensim model (w2v of fastText one).
    """
    # if this one raises an error, it means that the model we expect to see doesn't exist
    full_name = [f for f in os.listdir(path)
                 if f.startswith(name) and f.endswith('.model')][0]
    if config_dict['general_config']['embedding_model_name'] == 'w2v':
        return Word2Vec.load(join(path, full_name))
    elif config_dict['general_config']['embedding_model_name'] == 'fastext':
        return FastText.load(join(path, full_name))
    else:
        raise IOError("'embedding_model_name' (in the config file) is illegal - must be 'w2v' or 'fastext'")


def load_tfidf(path, name):
    """
    loads a tf-idf dictionary from a csv file
    :param path: str
        the path to the csv file.
    :param name: str
        the name of the tf-idf file to load.
    :return: dict
        the tf-idf dictionary
    """
    full_name = 'tf_idf_' + name
    full_name = join(path, full_name + '.csv')
    tf_idf_as_df = pd.read_csv(full_name)
    tfidf_as_dict = tf_idf_as_df.set_index('word')['value'].to_dict()
    return tfidf_as_dict


def load_user_based_word_weights(path, name):
    """
    loads a user_based_word_weights dictionary from a csv file. This is the same like the 'load_tfidf' function.
    The only difference is that in this function the dictionary is expected to rely on user information rather than
    only a tf-idf information.
    :param path: str
        the path to the csv file.
    :param name: str
        the name of the user_based_word_weights file to load.
    :return: dict
        the user_based_word_weights dictionary
    """
    full_name = 'user_based_word_weights_' + name
    full_name = join(path, full_name + '.csv')
    user_based_word_weights_as_df = pd.read_csv(full_name)
    user_based_word_weights_as_dict = user_based_word_weights_as_df.set_index('word')['value'].to_dict()
    return user_based_word_weights_as_dict


def check_input_validity(config_dict):
    """
    testing the validity of the input. In case something is invalid, raises an error.
    :param config_dict: dict
        the configuration dictionary with all the relevant parameters. This dictionary is the loaded dictionary from the
        config.json file
    :return: None
        raises and error if anything in invalid
    """
    # checking the paths are valid
    embedding_path = os.path.isdir(config_dict['embedding_models_path'])
    if not embedding_path:
        raise IOError(f"embedding_models_path was provided as {embedding_path}. However, it does'nt exist. Please fix.")
    # if one of the specified directories does not exist - we create it
    tf_idf_path = os.path.isdir(config_dict['tf_idf_path'])
    model_output_path = os.path.isdir(config_dict['model_output_path'])
    user_based_word_weights_path = os.path.isdir(config_dict['user_based_word_weights_path'])
    try:
        if not tf_idf_path:
            os.makedirs(tf_idf_path)
        if not model_output_path:
            os.makedirs(model_output_path)
        if not user_based_word_weights_path:
            os.makedirs(user_based_word_weights_path)
    except FileNotFoundError:
        print(f"One of the specified tf_idf/model_output/user_based_word_weights folders is invalid. Please fix.")
    # general_config validity
    embedding_model_name = config_dict['general_config']['embedding_model_name']
    if embedding_model_name not in {'w2v', 'fastext'}:
        raise IOError(f"embedding_model_name was provided as {embedding_model_name}. However, has to be either"
                      f"'w2v' or 'fastext'. Please fix.")



