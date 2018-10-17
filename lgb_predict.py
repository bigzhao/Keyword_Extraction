import pandas as pd
import numpy as np
import jieba
import jieba.analyse
from sklearn.preprocessing import MinMaxScaler
import re
from operator import itemgetter
from scipy import stats
from sklearn.preprocessing import LabelEncoder
import gc
import math
from sklearn.model_selection import KFold, StratifiedKFold
from sklearn.model_selection import train_test_split
import lightgbm as lgb
from sklearn.metrics import roc_auc_score, f1_score
from scipy.stats import skew, kurtosis
from collections import Counter


## 预处理
def preprocessing(train_df, test_df):
	enc = LabelEncoder().fit(test_df.cixing)

	test_df['cixing_enc'] = enc.transform(test_df.cixing)
	train_df['cixing_enc'] = enc.transform(train_df.cixing)

	## 统计
	counter = Counter(test_df.tags.values)

	freq = train_df.tags.apply(lambda x: counter[x]).reset_index(drop=True)
	train_df['tag_freq'] = freq
	test_df['tag_freq'] = test_df.tags.apply(lambda x: counter[x]).reset_index(drop=True)

	test_df['score'] = np.load('lgb_sub_9524764012949717.npy')
	positive_counter = Counter(test_df[test_df.score > 0.5].tags.values)

	train_df['positive_tag_freq'] =  train_df.tags.apply(lambda x: positive_counter[x]).reset_index(drop=True)
	test_df['positive_tag_freq'] =  test_df.tags.apply(lambda x: positive_counter[x]).reset_index(drop=True)

	return train_df, test_df


def evaluate_5_fold(train_df, test_df, cols, test=False):

    kf = KFold(n_splits=5, shuffle=True, random_state=42)
    y_test = 0
    oof_train = np.zeros((train_df.shape[0],))
    for i, (train_index, val_index) in enumerate(kf.split(train_df[cols])):
        X_train, y_train = train_df.loc[train_index, cols], train_df.label.values[train_index]
        X_val, y_val = train_df.loc[val_index, cols], train_df.label.values[val_index]

        lgb_train = lgb.Dataset(
            X_train, y_train)
        lgb_eval = lgb.Dataset(
            X_val, y_val,
            reference=lgb_train)
    #     print('开始训练......')

        gbm = lgb.train(params,
                        lgb_train,
                        num_boost_round=40000,
                        valid_sets=lgb_eval,
                        early_stopping_rounds=50,
                        verbose_eval=False,
                        )
        y_pred = gbm.predict(X_val)
        if test:
        	## 防止爆内存
            t_1 = gbm.predict(test_df.loc[:533028, cols])
            t_2 = gbm.predict(test_df.loc[533028+1:2*533028, cols])
            t_3 = gbm.predict(test_df.loc[2*533028+1:3*533028, cols])
            t_4 = gbm.predict(test_df.loc[3*533028+1:, cols])
            y_test += np.concatenate([t_1, t_2, t_3, t_4])
        oof_train[val_index] = y_pred
    auc = roc_auc_score(train_df.label.values, oof_train)
    y_test /= 5
    print('5 Fold auc:', auc)
    gc.collect()
    return auc, oof_train, y_test


def get_keywords(x):
#     print(x)
    score = x.score.values
    tags = x.tags.values
    ret = pd.Series()
    ret['id'] = x['id'].values[0]
    if len(tags) == 0:
        ret['label1'] = ''
        ret['label2'] = ''
    elif len(tags) == 1:
        ret['label1'] = tags[0]
        ret['label2'] = ''
    else:
        sort = np.argsort(score)[::-1]
        ret['label1'] = tags[sort[0]]
        ret['label2'] = tags[sort[1]]
    return ret


## 后处理 之前写得 没什么用 主要是置换逗号
def postprocessing(x):
    x['label1'] = x['label1'].replace(',', '，')
    x['label2'] = x['label2'].replace(',', '，')
    if '妇联' in x['label1']:
        # print('found')
        x['label1'] = x['label1'].replace('妇', '复')
    if '妇联' in x['label2']:
        # print('found')
        x['label2'] = x['label2'].replace('妇', '复')
    if '霉霉' in x['label1']:
        # print('found')
        x['label1'] = x['label1'].replace('霉霉', '泰勒·斯威夫特')
    if '霉霉' in x['label2']:
        # print('found')
        x['label2'] = x['label2'].replace('霉霉', '泰勒·斯威夫特')
    return x


## 参考豆腐的baseline 不过后期也没什么用 基本标题都能找出来
def judge_title_list(x):
    ret = pd.Series()
    ret['id'] = x['id']
    title_regex = x['title_regex']
    for word in title_regex:
        if ',' in word:
            word = word.replace(',', '，')
    length = len(title_regex)
    
    if length == 0:
        ret['label1'] = x['label1']
        ret['label2'] = x['label2']
    elif length == 1:
        if title_regex[0] in [ x['label1'], x['label2']]:
            ret['label1'] = x['label1']
            ret['label2'] = x['label2']
        else:
            ret['label1'] = x['label1']
            ret['label2'] =title_regex[0]
    else:
        ret['label1'] = title_regex[0]
        ret['label2'] = title_regex[1]
    ## 针对食物
    if '家常菜' in x['title'] and '家常菜' not in (x['label1'], x['label2']):
        ret['label1'] = x['label1']
        ret['label2'] = '家常菜'
    elif '菜谱' in  x['title'] and '菜谱' not in (x['label1'], x['label2']):
        ret['label1'] = x['label1']
        ret['label2'] = '菜谱'
    elif '创新菜' in  x['title'] and '创新菜' not in (x['label1'], x['label2']):
        ret['label1'] = x['label1']
        ret['label2'] = '创新菜'  
    elif '凉菜' in  x['title'] and '凉菜' not in (x['label1'], x['label2']):
        ret['label1'] = x['label1']
        ret['label2'] = '凉菜'  
    elif '农家菜' in  x['title'] and '农家菜' not in (x['label1'], x['label2']):
        ret['label1'] = x['label1']
        ret['label2'] = '农家菜'  
    elif '乡土菜' in  x['title'] and '乡土菜' not in (x['label1'], x['label2']):
        ret['label1'] = x['label1']
        ret['label2'] = '乡土菜'  
    elif '热卖菜' in  x['title'] and '热卖菜' not in (x['label1'], x['label2']):
        ret['label1'] = x['label1']
        ret['label2'] = '热卖菜'          
    return ret


if __name__ == '__main__':
    all_docs = pd.read_csv('all_docs.txt', sep='\001', header=None)
    all_docs.columns = ['id', 'title', 'content']
    all_docs.fillna('', inplace=True)

    print('Loading Data...\n')
    train_df = pd.read_csv('train_df_v7.csv')
    test_df = pd.read_csv('test_df_v7.csv')
    print('Done.\n')

    print('Preprocessing...\n')	
    train_df, test_df = preprocessing(train_df, test_df)
    print('Done.\n')

    params = {
            'task': 'train',
            'boosting_type': 'gbdt',
            'objective': 'binary',
            'metric': {'auc', 'binary_logloss'},
            'learning_rate': 0.025,
            'num_leaves': 38,
            'min_data_in_leaf': 170,
            'bagging_fraction': 0.85,
            'bagging_freq': 1,
            'seed':42
    }

    cols = [col for col in train_df.columns if col not in ['tags', 'label', 'cixing', 'id', 'cixing_z_bili']]

    print('Predicting...\n')	
    # auc, lgb_oof_train, lgb_sub = evaluate_5_fold(train_df, None, cols, test=False)
    auc, lgb_oof_train, lgb_sub = evaluate_5_fold(train_df, test_df, cols, test=True)
    print('Done.\n')


    ## 输出提交文件
    print('Output...\n')    
    test_df['score'] = lgb_sub
    id_ = test_df.id.unique()
    sub = pd.DataFrame()
    sub['id'] = id_

    sub = test_df.groupby('id').apply(get_keywords)

    sub.fillna('', inplace=True)
    sub = sub.apply(postprocessing, axis=1)

    ## 将标题里被《...》的内容提取出来直接当关键词 
    sub = pd.merge(all_docs[['id', 'title']], sub, on=['id'], how='left')
    sub['title_regex'] = sub['title'].apply(lambda x:re.findall(r"《(.+?)》",x))
    sub = sub.apply(judge_title_list, axis=1)

    sub = sub.apply(postprocessing, axis=1)

    ## LGB单模线下auc 0.9578656266391324 A榜698.5 B榜828.5
    sub.to_csv('sub.csv', index=False)
    print('Done.\n')

