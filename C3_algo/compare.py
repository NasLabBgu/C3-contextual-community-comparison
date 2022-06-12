
import sys
import os
# adding the parent path so we'll be able to import internal packages
sys.path.append(os.path.dirname(os.getcwd()))
from three_com_algo.utils import load_model, load_tfidf, load_user_based_word_weights, filter_pairs
from os.path import join
import numpy as np
from itertools import combinations
import pandas as pd
from scipy.spatial.distance import pdist, squareform
from operator import itemgetter
import pickle
import time
import datetime
import multiprocessing as mp
import datetime as dt
import traceback
import operator
from collections import defaultdict


def generate_weights(model):
    """

    :param model:
    :return:
    """
    words = model.wv.index2entity
    r = np.random.random(len(words))
    weights = r/r.sum()
    return dict(zip(words, weights))


def calc_vec_distances(name, vectors_matrix, metric, config_dict):
    """
    calc the distance between vectors representing words (for each SR seperatly)
    :param name: String. file name for distances matrix
    :param vectors_matrix: Numpy array. he vectors to calc distances between them
    :param metric: String. distance metric.
    :return: Numpy array - distances matrix.
    """
    if eval(config_dict['current_run_flags']['calc_dis']):
        dis_matrix = pdist(X=vectors_matrix, metric=metric)
        if eval(config_dict['current_run_flags']['save_dis_matrix']):
            with open(join(config_dict['dis_path'][config_dict['machine']], name + '.pickle'), 'wb') as handle:
                pickle.dump(dis_matrix, handle, protocol=pickle.HIGHEST_PROTOCOL)
    else:
        with open(join(config_dict['dis_path'][config_dict['machine']], name + '.pickle'), 'rb') as handle:
            dis_matrix = pickle.load(handle)
    return dis_matrix


def get_selected_weights(w_dict, keys_lst, normalize, config_dict):
    """

    :param w_dict: dict
        the full dictionary of the SR (including words we don't really care about in a specific comparison
    :param keys_lst: list
        includes the desired words to take into account (either the intersected words between SRs or unions between them)
    :param normalize:
    :return: ndarray (numpy), where unselected words have a zero weight
    """
    # case we have to add to the dictionary some new words which belong the compared SR - hence zero value words are added
    #if config_dict['algo_parmas']['method'] == 'union':
    ### CHANGE HERE!!! ##########
    new_w = np.setdiff1d(keys_lst, np.array(list(w_dict.keys())), assume_unique=True)
    w_dict.update(dict(zip(new_w, np.zeros(len(new_w)))))  # add new words to dict with weight=0
    v = np.array(itemgetter(*keys_lst)(w_dict))
    if normalize:
        return v/v.sum()
    return v


def calc_pairwise_weights(arr, agg_function='mean', normalize=False):
    """

    :param arr: Numpy array. array of positive weights.
    :param f_name: String. file name for the pairwise_weights (save/load).
    :param normalize: whether to normalize the weights vector.
    :param agg_function: string
        the function to apply between the two vectors. Currently it can be mean or max
    :return: condensed matrix of max weight for each two elements in arr.
    """
    n = len(arr)
    N = n * (n - 1) // 2
    idx = np.concatenate(([0], np.arange(n - 1, 0, -1).cumsum()))
    start, stop = idx[:-1], idx[1:]
    pairs_w = np.empty(N, dtype=float)
    for j, i in enumerate(range(n - 1)):
        s0, s1 = start[j], stop[j]
        curr = np.array([np.repeat(arr[i], len(arr[i + 1:])), np.array(arr[i + 1:])])
        # taking the maximum or average out of the two. If average is used - we take average over the non-zero cases
        # only, zero ones are left with the zero value
        if agg_function == 'mean':
            pairs_w[s0:s1] = np.mean(np.array([curr[0], curr[1]]), axis=0) * \
                             np.array([1 if a > 0 and b > 0 else 0 for a, b in zip(curr[0], curr[1])])
        elif agg_function == 'max':
            pairs_w[s0:s1] = np.amax(curr, axis=0)
    if normalize:
        return pairs_w/pairs_w.sum()
    return pairs_w


def generate_pairwise_names(arr_names):
    """

    :param arr: Numpy array. array of positive weights.
    :param f_name: String. file name for the pairwise_weights (save/load).
    :param normalize: whether to normalize the weights vector.
    :param calc: Boolean.
    :return: condensed matrix of max weight for each two elements in arr.
    """
    n = len(arr_names)
    pairwise_names = list()
    for j, i in enumerate(range(n - 1)):
        curr = np.array([np.repeat(arr_names[i], len(arr_names[i + 1:])), np.array(arr_names[i + 1:])])
        pairwise_names.extend(list(zip(curr[0], curr[1])))
    return pairwise_names


def keep_top_weights(arr, top_perc):
    """

    :param arr: Numpy array- original weights.
    :param top_perc: Int - the percentage of top (highest) weights to keep.
    :return: Numpy array- new weights after selecting top.
    """
    min_thres = np.percentile(arr, 100-top_perc)
    arr_t = np.where(arr < min_thres, 0, arr)
    # normalizing the vector (those which are at the top 'top_perc')
    return arr_t/arr_t.sum()


def apply_extreme_distances_filter(dis_vector_1, dis_vector_2, apply_both_ends, top_perc):
    dis_1_array = np.array(dis_vector_1)
    dis_2_array = np.array(dis_vector_2)
    bottom_thres_1 = np.percentile(dis_vector_1, top_perc)
    bottom_thres_2 = np.percentile(dis_vector_2, top_perc)
    if apply_both_ends:
        top_thres_1 = np.percentile(dis_vector_1, 100 - top_perc)
        top_thres_2 = np.percentile(dis_vector_2, 100 - top_perc)
        keep_dis_1 = (dis_1_array < bottom_thres_1) | (top_thres_1 < dis_1_array)
        keep_dis_2 = (dis_2_array < bottom_thres_2) | (top_thres_2 < dis_2_array)
        keep_dis_combined = keep_dis_1 | keep_dis_2
    else:
        keep_dis_1 = (dis_1_array < bottom_thres_1)
        keep_dis_2 = (dis_2_array < bottom_thres_2)
        keep_dis_combined = keep_dis_1 | keep_dis_2
    return keep_dis_combined


def calc_distance_between_comm(d1, d2, w):
    """

    :param d1: condensed matrix of distances between intersection words - community 1
    :param d2: condensed matrix of distances between intersection words - community 2
    :param w: condensed matrix of weights for pairs of intersection words
    Note: words' order should be identical in d1, d2, w

    :return: float [0, 2] - distance between community 1 and 2 = weighted average of absolute difference between intersection words
    """
    abs_d = abs(d1 - d2)
    return np.matmul(abs_d, w.T)


def compare(i, n_model_1, n_model_2, config_dict):
    """

    :param i: Int. current iteration.
    :param n_model_1: String. name of model_1.
    :param n_model_2: String. name of model_2.
    :return:
    """
    try:
        start = time.time()
        tf_idf_path = config_dict['tf_idf_path'][config_dict['machine']]
        user_based_word_weights_path = config_dict['user_based_word_weights_path'][config_dict['machine']]
        data_path = config_dict['data_path'][config_dict['machine']]
        tf_idf_weights_1 = load_tfidf(path=tf_idf_path, name=n_model_1)
        tf_idf_weights_2 = load_tfidf(path=tf_idf_path, name=n_model_2)
        user_based_word_weights_1 = load_user_based_word_weights(path=user_based_word_weights_path, name=n_model_1)
        user_based_word_weights_2 = load_user_based_word_weights(path=user_based_word_weights_path, name=n_model_2)
        # taking the specific type of tf-idf in the current settings (new update, 28/6/2020)
        user_based_word_weights_1 = user_based_word_weights_1['freq_upvote_based']['factorized_idf_unfiltered']
        user_based_word_weights_2 = user_based_word_weights_2['freq_upvote_based']['factorized_idf_unfiltered']
        model_1 = load_model(path=data_path, name=n_model_1, config_dict=config_dict)
        model_2 = load_model(path=data_path, name=n_model_2, config_dict=config_dict)
        # 1- word vectors
        wv_1, wv_2 = model_1.wv, model_2.wv
        # 2- intersection and union
        intersec = np.intersect1d(wv_1.index2entity, wv_2.index2entity)
        wc1, wc2, wc_inter = len(wv_1.index2entity), len(wv_2.index2entity), len(intersec)
        method = config_dict['algo_parmas']['method']
        # w_lst holds the list of words to include in the comparison
        if method == 'intersection':
            w_lst = list(intersec)
        elif method == 'union':
            union = np.union1d(wv_1.index2entity, wv_2.index2entity)
            # inter_idx = np.in1d(union, intersec)
            w_lst = list(union)
        else:
            raise IOError("method must be either 'intersection' or 'union'")
        wv_1_selected, wv_2_selected = wv_1[w_lst], wv_2[w_lst]

        # 3- calc vectors distances (within each SR separately)
        # Choosing the distance metric to be used when comparing the word pairs
        #dis_metric = 'euclidean'
        dis_metric = 'cosine'
        name_1 = n_model_1 + '_' + method + '_' + n_model_2
        name_2 = n_model_2 + '_' + method + '_' + n_model_1
        dis_1 = calc_vec_distances(name=name_1, vectors_matrix=wv_1_selected, metric=dis_metric,
                                   config_dict=config_dict)
        dis_2 = calc_vec_distances(name=name_2, vectors_matrix=wv_2_selected, metric=dis_metric,
                                   config_dict=config_dict)
        unweighted_score = np.mean(abs(dis_1 - dis_2))
        # 3.1 - find indexes of intersection
        # union = np.union1d(wv_1.index2entity, wv_2.index2entity)
        # inter_idx = np.in1d(union, intersec)
        # inter_dis_idx = pdist(X=inter_idx.reshape(-1, 1), metric=lambda u, v: np.logical_and(u, v))
        # dis_1, dis_2 = dis_1[inter_dis_idx], dis_2[inter_dis_idx]

        f_name = 'pair_w_' + n_model_1 + '_' + n_model_2
        # 4- calc weights for intersection words (tf-idf based)
        # get weights of selected words per community
        w_1_tf_idf = get_selected_weights(w_dict=tf_idf_weights_1, keys_lst=w_lst, normalize=False,
                                          config_dict=config_dict)
        w_2_tf_idf = get_selected_weights(w_dict=tf_idf_weights_2, keys_lst=w_lst, normalize=False,
                                          config_dict=config_dict)

        w_1_user_based = get_selected_weights(w_dict=user_based_word_weights_1, keys_lst=w_lst, normalize=False,
                                              config_dict=config_dict)
        w_2_user_based = get_selected_weights(w_dict=user_based_word_weights_2, keys_lst=w_lst, normalize=False,
                                              config_dict=config_dict)
        ## TAKING ONLY 2% of the User-Based data
        #min_thres = np.percentile(w_1_user_based, 70)
        #w_1_user_based = np.where(w_1_user_based < min_thres, 0, w_1_user_based)
        #min_thres = np.percentile(w_2_user_based, 70)
        #w_2_user_based = np.where(w_2_user_based < min_thres, 0, w_2_user_based)

        # weight per word (max or mean). If we take the mean, we take only mean in cases where the weight > 0
        w_mean_tf_idf = \
            np.mean(np.array([w_1_tf_idf, w_2_tf_idf]), axis=0) * np.array([1 if a > 0 and b > 0 else 0
                                                                            for a, b in zip(w_1_tf_idf, w_2_tf_idf)])
        w_mean_user_based = \
            np.mean(np.array([w_1_user_based, w_2_user_based]), axis=0) * np.array([1 if a > 0 and b > 0 else 0
                                                                                    for a, b in zip(w_1_user_based, w_2_user_based)])

        #w_max = np.amax(np.array([w_1, w_2]), axis=0)
        # weight per pair of words (max of the 2 elements in pair)
        w_pairs_tf_idf = calc_pairwise_weights(arr=w_mean_tf_idf, agg_function='max', normalize=False)
        w_pairs_user_based = calc_pairwise_weights(arr=w_mean_user_based, agg_function='max', normalize=False)

        # 5.1 - keeping only those pair of words with high/low distance (if flag set to True)
        if eval(config_dict['algo_parmas']['keep_only_extreme_distances']):
            apply_both_ends = eval(config_dict['algo_parmas']['apply_both_ends'])
            top_perc = config_dict['algo_parmas']['top_bottom_distances_perc'] * 100
            keep_distance_indic = apply_extreme_distances_filter(dis_vector_1=dis_1, dis_vector_2=dis_2,
                                                                 apply_both_ends=apply_both_ends, top_perc=top_perc)
            # replacing values with zero in cases the distance value should not be kept
            w_pairs_tf_idf = np.where(keep_distance_indic, w_pairs_tf_idf, 0)
            w_pairs_user_based = np.where(keep_distance_indic, w_pairs_user_based, 0)

        # 5.2 - keep top weights + normalize. Currently taking only top 10% out of the selected words
        # (those with the highest tf-idf)
        top_perc = int(config_dict['algo_parmas']['top_tf_idf_weights_perc'] * 100)

        w_pairs_tf_idf = keep_top_weights(arr=w_pairs_tf_idf, top_perc=100)
        w_pairs_user_based = keep_top_weights(arr=w_pairs_user_based, top_perc=top_perc)

        w_pairs_hybrid = w_pairs_tf_idf * w_pairs_user_based
        w_pairs_hybrid = w_pairs_hybrid / w_pairs_hybrid.sum()

        non_zero_pairs_user_based = len([1 for w in w_pairs_user_based if w > 0])
        # 6- compare communities
        score_tf_idf_weight = calc_distance_between_comm(d1=dis_1, d2=dis_2, w=w_pairs_tf_idf)
        score_user_based_weight = calc_distance_between_comm(d1=dis_1, d2=dis_2, w=w_pairs_user_based)
        score_hybrid_weight = calc_distance_between_comm(d1=dis_1, d2=dis_2, w=w_pairs_hybrid)
        res = {'name_m1': n_model_1, 'name_m2': n_model_2, 'unweighted_score': unweighted_score,
               'score_tf_idf_based': score_tf_idf_weight, 'score_user_based_weight': score_user_based_weight,
               'score_hybrid_weight': score_hybrid_weight, 'wc_m1': wc1, 'wc_m2': wc2, 'wc_inter': wc_inter,
               'non_zero_pairs_user_based': non_zero_pairs_user_based}
        print(f"iteration:{i}, {res}, elapsed time (min): {(time.time() - start) / 60}")
        return res

    except Exception as e:
        print("type error: " + str(e))
        print(traceback.format_exc())
        res = {'name_m1': n_model_1, 'name_m2': n_model_2, 'score': None, 'wc_m1': None, 'wc_m2': None,
               'wc_inter': None}
        return res


def calc_scores_all_models(m_names, config_dict):
    """

    :param m_names: list
        list of names of all communities
    :return:
    """
    m_type = config_dict['general_config']['model_type']
    metrics = pd.DataFrame(columns=['name_m1', 'name_m2', 'score', 'wc_m1', 'wc_m2', 'wc_inter'])
    #print(f"tot_i = {int(len(m_names)*(len(m_names)-1)/2)}")
    if eval(config_dict['current_run_flags']['calc_combinations']):
        lst = list()
        unsorted_lst = list()
        # case the models we got are from the type of those we created only as a self-chcek cases
        if all(['_parta' in name or '_partb' in name for name in m_names]):
            lst = _get_model_pairs(m_names, config_dict, consider_only_complementary=False)
        # 'regular case' - all combinations of pairs should be computed
        else:
            for m1, m2 in combinations(iterable=m_names, r=2):
                unsorted_lst.append((m1, m2))
            # now sorting the list
            lst = sorted(unsorted_lst, key=lambda element: (element[0], element[1]))
            lst = [(idx, tup[0], tup[1], config_dict) for idx, tup in enumerate(lst)]
        with open(join(config_dict['combinations_path'][config_dict['machine']], 'combinations_' + str(len(m_names)) + '.pickle'), 'wb') as handle:
            pickle.dump(lst, handle, protocol=pickle.HIGHEST_PROTOCOL)
        #raise IOError("I Want to stop here :)")
    # case some of the combinations were already calc
    else:
        with open(join(config_dict['combinations_path'][config_dict['machine']], 'combinations_' + str(len(m_names)) + '.pickle'), 'rb') as handle:
            lst = pickle.load(handle)
    if eval(config_dict['current_run_flags']['filter_pairs']):
        lst = filter_pairs(lst=lst, config_dict=config_dict)
        with open(join(config_dict['combinations_path'][config_dict['machine']],
                       'combinations_after_filter_' + str(len(m_names)) + '_' +
                       config_dict['general_config']['model_type'] + '.pickle'), 'wb') as handle:
            pickle.dump(lst, handle, protocol=pickle.HIGHEST_PROTOCOL)
        # with open(join(c.combinations_path, 'combinations_after_filter_' + str(len(m_names)) + '.pickle'), 'rb') as handle:
        #     lst = pickle.load(handle)
    print(f"Potential total combination of communities = {len(lst)}")
    # New option by Avrahami (24/2/2020) - checking which combinations were already calculated and not use them
    lst = _get_model_chunk(lst, config_dict)
    print(f"Updated_tot_i after taking current chunk only = {len(lst)}")
    # starting a multi=process job
    pool = mp.Pool(processes=config_dict['general_config']['cpu_count'])
    print('start compare')
    with pool as pool:
        results = pool.starmap(compare, lst)
    for res in results:
        metrics = metrics.append(res, ignore_index=True)
    for col in ['name_m1', 'name_m2']:
        metrics[col] = metrics[col].apply(lambda x: x.replace('_model_' +
                                                              config_dict['general_config']['model_type'] + '.model', ''))
    metrics_f_name = 'pairs_similarity_results' + m_type
    # in the past we used to save pickle files - no need for that any more
    #with open(join(config_dict['scores_path'][config_dict['machine']], metrics_f_name + '.pickle'), 'wb') as handle:
    #    pickle.dump(metrics, handle, protocol=pickle.HIGHEST_PROTOCOL)
    metrics_full_path_name = join(config_dict['scores_path'][config_dict['machine']], metrics_f_name + '.csv')
    # case file exist already - we'll add results to the exisitng file
    if os.path.isfile(metrics_full_path_name):
        with open(metrics_full_path_name, 'a') as f:
            metrics.to_csv(f, header=False, index=False)
    # case it doesn't exist - we'll create one
    else:
        metrics.to_csv(metrics_full_path_name, header=True, index=False)
    print(f"calc_scores_all_models function ended, {metrics.shape[0]} rows were written to {metrics_full_path_name}")


def _get_model_pairs(m_names, config_dict, consider_only_complementary=True, top_n_to_consider=100):
    lst = []
    if consider_only_complementary:
        for cur_idx in range(len(m_names)-1):
            m1 = m_names[cur_idx]
            m2 = m_names[cur_idx+1]
            first_name_list = m_names[cur_idx].split('_')
            first_name_list = [i for i in first_name_list if i != 'parta']
            second_name_list = m_names[cur_idx+1].split('_')
            second_name_list = [i for i in second_name_list if i != 'partb']
            if first_name_list == second_name_list:
                lst.append((len(lst) + 1, m1, m2, config_dict))
        return lst
    # in case we wish to pull 'top_n' other srs for each sr we test (and not only the complementary one)
    # this option takes ~2 hours to calc over the 1565 communities
    else:
        tf_idf_path = config_dict['tf_idf_path'][config_dict['machine']]
        # looping over each sr and finding the best condidates for him
        intersection_union_ratio_dict = defaultdict(lambda: defaultdict(float))
        start_time = datetime.datetime.now()
        # loading all tf-idf models to memory
        tf_idf_models = dict()
        for cur_sr in m_names:
            tf_idf_models[cur_sr] = load_tfidf(path=tf_idf_path, name=cur_sr)
        # looping over all srs (besides the last one)
        for sr_idx, sr_in_focus in enumerate(m_names[0:len(m_names)-1]):
            if sr_idx % 5 == 0:
                duration = (datetime.datetime.now() - start_time).seconds
                print(f"in the _get_model_pairs, passed over {sr_idx} items. Time up to now: {duration} second")
            tf_idf_weights_1 = tf_idf_models[sr_in_focus]
            #tf_idf_weights_1 = load_tfidf(path=tf_idf_path, name=sr_in_focus)
            # looping over all other srs (from index i to the end)
            for sr_to_compare in m_names[sr_idx+1:]:
                tf_idf_weights_2 = tf_idf_models[sr_to_compare]
                #tf_idf_weights_2 = load_tfidf(path=tf_idf_path, name=sr_to_compare)
                # the overlap between the two
                intersection_size = len(set(tf_idf_weights_1.keys()).intersection(set(tf_idf_weights_2.keys())))
                union_size = len(tf_idf_weights_1.keys()) + len(tf_idf_weights_2.keys()) - 2*intersection_size
                #union_size = len(tf_idf_weights_1.keys() | tf_idf_weights_2.keys())
                try:
                    intersection_union_ratio_dict[sr_in_focus][sr_to_compare] = intersection_size / union_size
                    intersection_union_ratio_dict[sr_to_compare][sr_in_focus] = intersection_size / union_size
                except ZeroDivisionError:
                    intersection_union_ratio_dict[sr_in_focus][sr_to_compare] = 0.0
                    intersection_union_ratio_dict[sr_to_compare][sr_in_focus] = 0.0
                # by now, the focused sr should be filled with perfect information
                cur_focused_sr_sorted = sorted(intersection_union_ratio_dict[sr_in_focus].items(),
                                               key=lambda item: item[1], reverse=True)
            if sr_in_focus.split("_part")[0] != cur_focused_sr_sorted[0][0].split("_part")[0]:
                print(f"the sr with the highest overlap with {sr_in_focus} was {cur_focused_sr_sorted[0][0]}. not good")
            # adding the top_n items to the final list
            for chosen_sr in cur_focused_sr_sorted[0:top_n_to_consider]:
                lst.append((len(lst) + 1, sr_in_focus, chosen_sr[0], config_dict))
        return lst


def word_pairs_extractor(n_model_1, n_model_2, config_dict, positive_relations=True):
    """

    :param n_model_1: String. name of model_1.
    :param n_model_2: String. name of model_2.
    :return:
    """
    tf_idf_path = config_dict['tf_idf_path'][config_dict['machine']]
    user_based_word_weights_path = config_dict['user_based_word_weights_path'][config_dict['machine']]
    data_path = config_dict['data_path'][config_dict['machine']]
    tf_idf_weights_1 = load_tfidf(path=tf_idf_path, name=n_model_1)
    tf_idf_weights_2 = load_tfidf(path=tf_idf_path, name=n_model_2)
    user_based_word_weights_1 = load_user_based_word_weights(path=user_based_word_weights_path, name=n_model_1)
    user_based_word_weights_2 = load_user_based_word_weights(path=user_based_word_weights_path, name=n_model_1)
    model_1 = load_model(path=data_path, name=n_model_1, config_dict=config_dict)
    model_2 = load_model(path=data_path, name=n_model_2, config_dict=config_dict)

    # 1- word vectors
    wv_1, wv_2 = model_1.wv, model_2.wv

    # 2- intersection and union
    intersec = np.intersect1d(wv_1.index2entity, wv_2.index2entity)
    wc1, wc2, wc_inter = len(wv_1.index2entity), len(wv_2.index2entity), len(intersec)
    method = config_dict['algo_parmas']['method']
    if method == 'intersection':
        w_lst = list(intersec)
    elif method == 'union':
        union = np.union1d(wv_1.index2entity, wv_2.index2entity)
        # inter_idx = np.in1d(union, intersec)
        w_lst = list(union)
    else:
        raise IOError("method must be either 'intersection' or 'union'")
    wv_1_selected, wv_2_selected = wv_1[w_lst], wv_2[w_lst]

    # 3- calc vectors distances (within each SR separately)
    dis_metric = 'cosine'
    name_1 = n_model_1 + '_' + method + '_' + n_model_2
    name_2 = n_model_2 + '_' + method + '_' + n_model_1
    dis_1 = calc_vec_distances(name=name_1, vectors_matrix=wv_1_selected, metric=dis_metric,
                               config_dict=config_dict)
    dis_2 = calc_vec_distances(name=name_2, vectors_matrix=wv_2_selected, metric=dis_metric,
                               config_dict=config_dict)
    dis_1_10_quantile_value = np.quantile(dis_1, 0.1)
    dis_2_10_quantile_value = np.quantile(dis_2, 0.1)
    is_in_top_quantile_w_pairs = np.logical_and(dis_1 < dis_1_10_quantile_value, dis_2 < dis_2_10_quantile_value)

    # 3.1 - calc weights for intersection words (tf-idf based)
    # get weights of selected words per community
    w_1_tf_idf = get_selected_weights(w_dict=tf_idf_weights_1, keys_lst=w_lst, normalize=False,
                                      config_dict=config_dict)
    w_2_tf_idf = get_selected_weights(w_dict=tf_idf_weights_2, keys_lst=w_lst, normalize=False,
                                      config_dict=config_dict)

    w_1_user_based = get_selected_weights(w_dict=user_based_word_weights_1, keys_lst=w_lst, normalize=False,
                                          config_dict=config_dict)
    w_2_user_based = get_selected_weights(w_dict=user_based_word_weights_2, keys_lst=w_lst, normalize=False,
                                          config_dict=config_dict)

    # weight per word (max or average) - currently we take the AVG
    #w_max = np.amax(np.array([w_1, w_2]), axis=0)
    w_mean_tf_idf = \
        np.mean(np.array([w_1_tf_idf, w_2_tf_idf]), axis=0) * np.array([1 if a > 0 and b > 0 else 0
                                                                        for a, b in zip(w_1_tf_idf, w_2_tf_idf)])
    w_mean_user_based = \
        np.mean(np.array([w_1_user_based, w_2_user_based]), axis=0) * np.array([1 if a > 0 and b > 0 else 0
                                                                                for a, b in
                                                                                zip(w_1_user_based, w_2_user_based)])

    # weight per pair of words (max of the 2 elements in pair)
    f_name = 'pair_w_' + n_model_1 + '_' + n_model_2
    pairwise_names = generate_pairwise_names(arr_names=w_lst)
    w_pairs_tf_idf = calc_pairwise_weights(arr=w_mean_tf_idf, agg_function='max', normalize=False)
    w_pairs_user_based = calc_pairwise_weights(arr=w_mean_user_based, agg_function='max', normalize=False)

    # 4 - keep top weights + normalize. Currently taking only top 25% out of the selected words (those with the highest tf-idf)
    top_perc = int(config_dict['algo_parmas']['top_tf_idf_weights_perc'] * 100)
    w_pairs_tf_idf = keep_top_weights(arr=w_pairs_tf_idf, top_perc=top_perc)
    w_pairs_user_based = keep_top_weights(arr=w_pairs_user_based, top_perc=top_perc)

    # 5 - now, with dis1, dis2 and w_pairs we can get all information needed
    sr_pair_dist = abs(dis_1 - dis_2)
    w_tf_idf_sr_pair_dist = np.multiply(sr_pair_dist, w_pairs_tf_idf)
    w_user_based_sr_pair_dist = np.multiply(sr_pair_dist, w_pairs_user_based)
    w_tf_idf_sr_pair_dist_filtered = \
        [(cur_idx, cur_w) for cur_idx, (cur_w, is_rel) in enumerate(zip(w_tf_idf_sr_pair_dist, is_in_top_quantile_w_pairs))
         if cur_w > 0 and is_rel]
    w_user_based_sr_pair_dist_filtered = \
        [(cur_idx, cur_w) for cur_idx, (cur_w, is_rel) in enumerate(zip(w_user_based_sr_pair_dist, is_in_top_quantile_w_pairs))
         if cur_w > 0 and is_rel]

    if positive_relations:
        w_tf_idf_sr_pair_dist_filtered.sort(key=operator.itemgetter(1))
        w_user_based_sr_pair_dist_filtered.sort(key=operator.itemgetter(1))
    else:
        w_tf_idf_sr_pair_dist_filtered.sort(key=operator.itemgetter(1), reverse=True)
        w_user_based_sr_pair_dist_filtered.sort(key=operator.itemgetter(1), reverse=True)
    word_pairs_amount = config_dict['word_pairs_config']['word_pairs_amount']
    most_rel_word_pairs_tf_idf = [(pairwise_names[word_pair_idx], distance)
                                  for word_pair_idx, distance in w_tf_idf_sr_pair_dist_filtered[0:word_pairs_amount]]
    most_rel_word_pairs_user_based = [(pairwise_names[word_pair_idx], distance)
                                      for word_pair_idx, distance in w_user_based_sr_pair_dist_filtered[0:word_pairs_amount]]

    # 6 - calculating the distance (although it is not a must)
    score_tf_idf = calc_distance_between_comm(d1=dis_1, d2=dis_2, w=w_pairs_tf_idf)
    score_user_based = calc_distance_between_comm(d1=dis_1, d2=dis_2, w=w_pairs_user_based)
    return (most_rel_word_pairs_tf_idf, most_rel_word_pairs_user_based) , (score_tf_idf, score_user_based)


def _get_model_chunk(pairs_list, config_dict):
    scores_f_name = 'pairs_similarity_results_' + config_dict['general_config']['model_type']
    scores_full_path = join(config_dict['scores_path'][config_dict['machine']], scores_f_name + '.csv')
    # if the chunk_size is negative, we will not use the logic of taking a chunck from the data - will run all
    if config_dict['general_config']['chunk_size'] < 0:
        return pairs_list
    else:
        chunk_size = config_dict['general_config']['chunk_size']
        chunk_index = config_dict['general_config']['chunk_index']
        starting_point = chunk_index * chunk_size
        ending_point = starting_point + chunk_size
        pairs_list = pairs_list[starting_point:ending_point]
    # if a results file exists, we will pull it out and check the existing pair of SRs
    if os.path.isfile(scores_full_path):
        current_score_df = pd.read_csv(scores_full_path)
        existing_pairs = {(str(row['name_m1']), str(row['name_m2'])) for idx, row in current_score_df.iterrows()}
        filtered_list = [element for element in pairs_list if (element[1], element[2]) not in existing_pairs]
    # case the results file doesn't exist
    else:
        filtered_list = pairs_list
    return filtered_list
