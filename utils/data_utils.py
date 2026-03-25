import pandas as pd
import numpy as np
from collections import defaultdict
from tqdm import tqdm
import pickle
import random
from tensorflow.keras.preprocessing.sequence import pad_sequences

# debug模式：从训练集中划出一部分数据来调试代码
def get_all_click_sample(data_path, sample_nums=10000):
    """
        训练集中采样一部分数据调试
        data_path: 原数据的存储路径
        sample_nums: 采样数目（这里由于机器的内存限制，可以采样用户做）
    """
    all_click = pd.read_csv(data_path + 'train_click_log.csv')
    all_user_ids = all_click.user_id.unique()

    sample_user_ids = np.random.choice(all_user_ids, size=sample_nums, replace=False) 
    all_click = all_click[all_click['user_id'].isin(sample_user_ids)]
    
    all_click = all_click.drop_duplicates((['user_id', 'click_article_id', 'click_timestamp']))
    return all_click



# 读取点击数据，这里分成线上和线下，如果是为了获取线上提交结果应该讲测试集中的点击数据合并到总的数据中
# 如果是为了线下验证模型的有效性或者特征的有效性，可以只使用训练集
def get_all_click_df(data_path='./tcdata/', offline=True):
    if offline:
        all_click = pd.read_csv(data_path + 'train_click_log.csv')
    else:
        trn_click = pd.read_csv(data_path + 'train_click_log.csv')
        tst_click = pd.read_csv(data_path + 'testA_click_log.csv')

        # 用 concat 替代 append
        all_click = pd.concat([trn_click, tst_click], ignore_index=True)
    
    all_click = all_click.drop_duplicates(['user_id', 'click_article_id', 'click_timestamp'])
    return all_click


# 根据点击时间获取用户的点击文章序列   {user1: [(item1, time1), (item2, time2)..]...}
def get_user_item_time(click_df):

    def make_item_time_pair(df):
        return list(zip(df['click_article_id'], df['click_timestamp']))

    user_item_time_df = (
        click_df
        .groupby('user_id')[['click_article_id', 'click_timestamp']]  
        .apply(lambda x: make_item_time_pair(x))
        .reset_index()
        .rename(columns={0: 'item_time_list'})
    )

    user_item_time_dict = dict(
        zip(user_item_time_df['user_id'], user_item_time_df['item_time_list'])
    )

    return user_item_time_dict
    
# 根据时间获取商品被点击的用户序列  {item1: [(user1, time1), (user2, time2)...]...}
# 这里的时间是用户点击当前商品的时间，好像没有直接的关系。
def get_item_user_time_dict(click_df):
    def make_user_time_pair(df):
        return list(zip(df['user_id'], df['click_timestamp']))
    
    click_df = click_df.sort_values('click_timestamp')
    item_user_time_df = click_df.groupby('click_article_id')[['user_id', 'click_timestamp']].apply(lambda x: make_user_time_pair(x))\
                                                            .reset_index().rename(columns={0: 'user_time_list'})
    
    item_user_time_dict = dict(zip(item_user_time_df['click_article_id'], item_user_time_df['user_time_list']))
    return item_user_time_dict

# 获取近期点击最多的文章
def get_item_topk_click(click_df, k):
    topk_click = click_df['click_article_id'].value_counts().index[:k]
    return topk_click


# 读取文章的基本属性
def get_item_info_df(data_path):
    item_info_df = pd.read_csv(data_path + 'articles.csv')
    
    # 为了方便与训练集中的click_article_id拼接，需要把article_id修改成click_article_id
    item_info_df = item_info_df.rename(columns={'article_id': 'click_article_id'})
    
    return item_info_df
    

# 获取文章id对应的基本属性，保存成字典的形式，方便后面召回阶段，冷启动阶段直接使用
def get_item_info_dict(item_info_df):
    max_min_scaler = lambda x : (x-np.min(x))/(np.max(x)-np.min(x))
    item_info_df['created_at_ts'] = item_info_df[['created_at_ts']].apply(max_min_scaler)
    
    item_type_dict = dict(zip(item_info_df['click_article_id'], item_info_df['category_id']))
    item_words_dict = dict(zip(item_info_df['click_article_id'], item_info_df['words_count']))
    item_created_time_dict = dict(zip(item_info_df['click_article_id'], item_info_df['created_at_ts']))
    
    return item_type_dict, item_words_dict, item_created_time_dict


def bucketize_words_count(words_count_series, bucket_num=20):
    words_count_series = words_count_series.fillna(0)
    bucket_num = min(bucket_num, words_count_series.nunique())
    if bucket_num <= 1:
        return pd.Series(np.ones(len(words_count_series), dtype=np.int64), index=words_count_series.index)

    words_bucket = pd.qcut(words_count_series, q=bucket_num, duplicates='drop', labels=False)
    return words_bucket.fillna(0).astype(np.int64)

    
# 获取当前数据的历史点击和最后一次点击

def get_hist_and_last_click(all_click):
    all_click = all_click.sort_values(['user_id', 'click_timestamp']).reset_index(drop=True)

    click_last_df = all_click.groupby('user_id').tail(1).reset_index(drop=True)

    user_cnt = all_click.groupby('user_id')['click_article_id'].transform('count')
    rank = all_click.groupby('user_id').cumcount()

    # 多次点击用户：保留除最后一次外的记录
    # 单次点击用户：保留唯一那条
    click_hist_df = all_click[(user_cnt == 1) | (rank < user_cnt - 1)].reset_index(drop=True)

    return click_hist_df, click_last_df

    

def get_user_hist_item_info_dict(all_click):
    
    # 获取user_id对应的用户历史点击文章类型的集合字典
    user_hist_item_typs = all_click.groupby('user_id')['category_id'].agg(set).reset_index()
    user_hist_item_typs_dict = dict(zip(user_hist_item_typs['user_id'], user_hist_item_typs['category_id']))
    
    # 获取user_id对应的用户点击文章的集合
    user_hist_item_ids_dict = all_click.groupby('user_id')['click_article_id'].agg(set).reset_index()
    user_hist_item_ids_dict = dict(zip(user_hist_item_ids_dict['user_id'], user_hist_item_ids_dict['click_article_id']))
    
    # 获取user_id对应的用户历史点击的文章的平均字数字典
    user_hist_item_words = all_click.groupby('user_id')['words_count'].agg('mean').reset_index()
    user_hist_item_words_dict = dict(zip(user_hist_item_words['user_id'], user_hist_item_words['words_count']))
    
    # 获取user_id对应的用户最后一次点击的文章的创建时间
    all_click_ = all_click.sort_values('click_timestamp')
    user_last_item_created_time = all_click_.groupby('user_id')['created_at_ts'].apply(lambda x: x.iloc[-1]).reset_index()
    
    max_min_scaler = lambda x : (x-np.min(x))/(np.max(x)-np.min(x))
    user_last_item_created_time['created_at_ts'] = user_last_item_created_time[['created_at_ts']].apply(max_min_scaler)
    
    user_last_item_created_time_dict = dict(zip(user_last_item_created_time['user_id'], \
                                                user_last_item_created_time['created_at_ts']))
    
    return user_hist_item_typs_dict, user_hist_item_ids_dict, user_hist_item_words_dict, user_last_item_created_time_dict


# 读取文章的Embedding数据
def get_item_emb_dict(data_path, save_path):
    item_emb_df = pd.read_csv(data_path + 'articles_emb.csv')
    
    item_emb_cols = [x for x in item_emb_df.columns if 'emb' in x]
    item_emb_np = np.ascontiguousarray(item_emb_df[item_emb_cols])
    # 进行归一化
    item_emb_np = item_emb_np / np.linalg.norm(item_emb_np, axis=1, keepdims=True)

    item_emb_dict = dict(zip(item_emb_df['article_id'], item_emb_np))
    pickle.dump(item_emb_dict, open(save_path + 'item_content_emb.pkl', 'wb'))
    
    return item_emb_dict

def get_user_activate_degree_dict(all_click_df):
    all_click_df_ = all_click_df.groupby('user_id')['click_article_id'].count().reset_index()
    
    # 用户活跃度归一化
    mm = MinMaxScaler()
    all_click_df_['click_article_id'] = mm.fit_transform(all_click_df_[['click_article_id']])
    user_activate_degree_dict = dict(zip(all_click_df_['user_id'], all_click_df_['click_article_id']))
    
    return user_activate_degree_dict



#####双塔模型数据函数
# 获取双塔召回时的训练验证数据
# negsample指的是通过滑窗构建样本的时候，负样本的数量
def gen_data_set(data, negsample=0):
    data.sort_values("click_timestamp", inplace=True)
    item_ids = data['click_article_id'].unique()

    train_set = []
    test_set = []
    for reviewerID, hist in tqdm(data.groupby('user_id')):
        pos_list = hist['click_article_id'].tolist()
        
        if negsample > 0:
            candidate_set = list(set(item_ids) - set(pos_list))   # 用户没看过的文章里面选择负样本
            neg_list = np.random.choice(candidate_set,size=len(pos_list)*negsample,replace=True)  # 对于每个正样本，选择n个负样本
            
        # 长度只有一个的时候，需要把这条数据也放到训练集中，不然的话最终学到的embedding就会有缺失
        if len(pos_list) == 1:
            train_set.append((reviewerID, [pos_list[0]], pos_list[0],1,len(pos_list)))
            test_set.append((reviewerID, [pos_list[0]], pos_list[0],1,len(pos_list)))
            
        # 滑窗构造正负样本
        for i in range(1, len(pos_list)):
            hist = pos_list[:i]
            
            if i != len(pos_list) - 1:
                train_set.append((reviewerID, hist[::-1], pos_list[i], 1, len(hist[::-1])))  # 正样本 [user_id, his_item, pos_item, label, len(his_item)]
                for negi in range(negsample):
                    train_set.append((reviewerID, hist[::-1], neg_list[i*negsample+negi], 0,len(hist[::-1]))) # 负样本 [user_id, his_item, neg_item, label, len(his_item)]
            else:
                # 将最长的那一个序列长度作为测试数据
                test_set.append((reviewerID, hist[::-1], pos_list[i],1,len(hist[::-1])))
                
    random.shuffle(train_set)
    random.shuffle(test_set)
    
    return train_set, test_set

# 将输入的数据进行padding，使得序列特征的长度都一致
def gen_model_input(train_set, user_profile, seq_max_len, item_feat_dict=None):

    train_uid = np.array([line[0] for line in train_set])
    train_seq = [line[1] for line in train_set]
    train_iid = np.array([line[2] for line in train_set])
    train_label = np.array([line[3] for line in train_set])
    train_hist_len = np.array([line[4] for line in train_set])

    train_seq_pad = pad_sequences(train_seq, maxlen=seq_max_len, padding='post', truncating='post', value=0)
    train_model_input = {"user_id": train_uid, "click_article_id": train_iid, "hist_article_id": train_seq_pad,
                         "hist_len": train_hist_len}

    if item_feat_dict is not None:
        if "category_id" in item_feat_dict:
            category_map = item_feat_dict["category_id"]
            train_model_input["item_category_id"] = np.array(
                [category_map.get(item_id, 0) for item_id in train_iid],
                dtype=np.int64
            )
            train_model_input["hist_category_id"] = np.vectorize(
                lambda x: category_map.get(x, 0),
                otypes=[np.int64]
            )(train_seq_pad)

        if "words_count" in item_feat_dict:
            words_map = item_feat_dict["words_count"]
            train_model_input["item_words_count"] = np.array(
                [words_map.get(item_id, 0) for item_id in train_iid],
                dtype=np.int64
            )
            train_model_input["hist_words_count"] = np.vectorize(
                lambda x: words_map.get(x, 0),
                otypes=[np.int64]
            )(train_seq_pad)

    return train_model_input, train_label
