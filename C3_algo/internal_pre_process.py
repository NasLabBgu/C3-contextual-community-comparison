import pickle
from os.path import join as opj
import pandas as pd
# a. Converting the tf-idf pickle files into csv files (with 2 columns: word and tf-idf value)
data_path = 'C:\\Users\\avrahami\\Documents\\Private\\Uni\\BGU\\PhD\\courses\\advanced NLP\\project\\' \
            'C3-contextual-community-comparison\\data\\tf_idf'
srs_corpus_names = ['acmilan', 'c_programming', 'football', 'java', 'timbers']
for cur_sr in srs_corpus_names:
    cur_tf_idf_f_name = 'tf_idf_' + cur_sr + '.p'
    cur_tf_idf_file = pickle.load(open(opj(data_path, cur_tf_idf_f_name), "rb" ))
    cur_tf_idf_as_df = pd.DataFrame.from_dict(cur_tf_idf_file, orient='index', columns=['value'])
    cur_tf_idf_as_df.reset_index(inplace=True)
    cur_tf_idf_as_df = cur_tf_idf_as_df.rename(columns={'index': 'word'})
    cur_tf_idf_as_df.to_csv(opj(data_path, 'tf_idf_' + cur_sr + '.csv'), index=False)

# in case we wish to load it to a dict object (used while loading the data in the C3 algo):
#cur_tf_idf_path = opj(data_path, 'tf_idf_' + srs_corpus_names[0] + '.csv')
#tf_idf_as_df = pd.read_csv(cur_tf_idf_path)
#tfidf_as_dict = tf_idf_as_df.set_index('word')['value'].to_dict()


# b. Converting the user_based_word_frequencies pickle files into csv files (with 2 columns: word and tf-idf value)
data_path = 'C:\\Users\\avrahami\\Documents\\Private\\Uni\\BGU\\PhD\\courses\\advanced NLP\\project\\' \
            'C3-contextual-community-comparison\\data\\user_based_word_weights'
srs_corpus_names = ['acmilan', 'c_programming', 'football', 'java', 'timbers']
for cur_sr in srs_corpus_names:
    cur_word_weights_f_name = 'user_based_word_weights_' + cur_sr + '.p'
    cur_word_weights_file = pickle.load(open(opj(data_path, cur_word_weights_f_name), "rb"))
    cur_word_weights_file = cur_word_weights_file['freq_upvote_based']['factorized_idf_unfiltered']
    cur_tf_idf_as_df = pd.DataFrame.from_dict(cur_word_weights_file, orient='index', columns=['value'])
    cur_tf_idf_as_df.reset_index(inplace=True)
    cur_tf_idf_as_df = cur_tf_idf_as_df.rename(columns={'index': 'word'})
    cur_tf_idf_as_df.to_csv(opj(data_path, 'user_based_word_weights_' + cur_sr + '.csv'), index=False)

# in case we wish to load it to a dict object (used while loading the data in the C3 algo):
cur_user_based_word_weights_path = opj(data_path, 'user_based_word_weights_' + srs_corpus_names[0] + '.csv')
user_based_word_weights_as_df = pd.read_csv(cur_user_based_word_weights_path)
user_based_word_weights_as_dict = user_based_word_weights_as_df.set_index('word')['value'].to_dict()
"""
OLD and not relevant way
for cur_sr in srs_corpus_names:
    cur_word_weights_f_name = 'user_based_word_frequencies_' + cur_sr + '.p'
    cur_word_weights_file = pickle.load(open(opj(data_path, cur_word_weights_f_name), "rb"))
    cur_word_weights_file = cur_word_weights_file['users_words_dist']
    # converting the nested dict to a normalized dict
    normalized_dict = dict()
    cur_idx = 0
    # outer loop, each key is a user, values is more complicated
    for u_name, values in cur_word_weights_file.items():
        # internal loop, each key is an object (e.g. 'words_dist')
        for internal_key, internal_value in values.items():
            if internal_key == 'words_dist':
                for word, freq in internal_value.items():
                    normalized_dict[cur_idx] = {'user_name': u_name, 'word': word, 'freq': freq}
                    cur_idx += 1
            else:
                normalized_dict[cur_idx] = {'user_name': u_name, 'word': internal_key, 'freq': internal_value}
                cur_idx += 1
    cur_word_weights_as_df = pd.DataFrame.from_dict(normalized_dict, orient='index')
    cur_word_weights_as_df.to_csv(opj(data_path, 'user_based_word_weights_' + cur_sr + '.csv'), index=False)
"""

