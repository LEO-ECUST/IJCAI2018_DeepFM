import pandas as pd
import numpy as np
import lightgbm as lgb
import sklearn
from sklearn.model_selection import train_test_split
from sklearn.metrics import log_loss
from sklearn.metrics import average_precision_score
from sklearn import preprocessing
import warnings
from model.DeepFM import DeepFM
from model.NFM import NFM
from scipy import stats
import math

warnings.filterwarnings("ignore")

path_train = 'D:/DeepLearning/CONTEST/TIANCHI/DATA/round1_ijcai_18_train_20180301/round1_ijcai_18_train_20180301.txt'
path_test = 'D:/DeepLearning/CONTEST/TIANCHI/DATA/round1_ijcai_18_test_a_20180301/round1_ijcai_18_test_a_20180301.txt'
path_result = 'D:/DeepLearning/CONTEST/TIANCHI/DATA/'
path_save_features = 'D:/DeepLearning/CONTEST/TIANCHI/DATA/features.csv'
path_save_header = 'D:/DeepLearning/CONTEST/TIANCHI/DATA/header.txt'
path_save_corr = 'D:/DeepLearning/CONTEST/TIANCHI/DATA/corr.txt'

use_saved_features = True
training_info = 'DeepFM_GitHubTest'
features_select_rate = 0.7

import time


def timestamp_datetime(value):
    format = '%Y-%m-%d %H:%M:%S'
    value = time.localtime(value)
    dt = time.strftime(format, value)
    return dt


def base_process(data):
    lbl = preprocessing.LabelEncoder()
    print(
        '--------------------------------------------------------------item--------------------------------------------------------------')
    data['len_item_category'] = data['item_category_list'].map(lambda x: len(str(x).split(';')))
    data['len_item_property'] = data['item_property_list'].map(lambda x: len(str(x).split(';')))
    for i in range(1, 3):
        data['item_category_list' + str(i)] = lbl.fit_transform(data['item_category_list'].map(
            lambda x: str(str(x).split(';')[i]) if len(str(x).split(';')) > i else ''))  # item_category_list的第0列全部都一样
    for i in range(10):
        data['item_property_list' + str(i)] = lbl.fit_transform(data['item_property_list'].map(lambda x: str(str(x).split(';')[i]) if len(str(x).split(';')) > i else ''))
    for col in ['item_id', 'item_brand_id', 'item_city_id']:
        data[col] = lbl.fit_transform(data[col])
    print(
        '--------------------------------------------------------------user--------------------------------------------------------------')
    for col in ['user_id']:
        data[col] = lbl.fit_transform(data[col])
    print('user 0,1 feature')
    data['gender0'] = data['user_gender_id'].apply(lambda x: 1 if x == -1 else 2)
    data['age0'] = data['user_age_level'].apply(lambda x: 1 if x == 1004 | x == 1005 | x == 1006 | x == 1007  else 2)
    data['occupation0'] = data['user_occupation_id'].apply(lambda x: 1 if x == -1 | x == 2003  else 2)
    data['star0'] = data['user_star_level'].apply(lambda x: 1 if x == -1 | x == 3000 | x == 3001  else 2)
    print(
        '--------------------------------------------------------------context--------------------------------------------------------------')
    data['realtime'] = data['context_timestamp'].apply(timestamp_datetime)
    data['realtime'] = pd.to_datetime(data['realtime'])
    data['day'] = data['realtime'].dt.day
    data['hour'] = data['realtime'].dt.hour
    data['len_predict_category_property'] = data['predict_category_property'].map(lambda x: len(str(x).split(';')))
    for i in range(5):
        data['predict_category_property' + str(i)] = lbl.fit_transform(data['predict_category_property'].map(
            lambda x: str(str(x).split(';')[i]) if len(str(x).split(';')) > i else ''))
    print('context 0,1 feature')
    data['context_page0'] = data['context_page_id'].apply(
        lambda x: 1 if x == 4001 | x == 4002 | x == 4003 | x == 4004 | x == 4007  else 2)
    print(
        '--------------------------------------------------------------shop--------------------------------------------------------------')
    for col in ['shop_id']:
        data[col] = lbl.fit_transform(data[col])
    data['shop_score_delivery0'] = data['shop_score_delivery'].apply(lambda x: 0 if x <= 0.98 and x >= 0.96  else 1)
    return data


def map_hour(x):
    if (x>=7)&(x<=12):
        return 1
    elif (x>=13)&(x<=20):
        return 2
    else:
        return 3

def deliver(x):
    #x=round(x,6)
    jiange=0.1
    for i in range(1,20):
        if (x>=4.1+jiange*(i-1))&(x<=4.1+jiange*i):
            return i+1
    if x==-5:
        return 1

def deliver1(x):
    if (x>=2)&(x<=4):
        return 1
    elif (x>=5)&(x<=7):
        return 2
    else:
        return 3


def review(x):
    # x=round(x,6)
    jiange = 0.02
    for i in range(1, 30):
        if (x >= 0.714 + jiange * (i - 1)) & (x <= 0.714 + jiange * i):
            return i + 1
    if x == -1:
        return 1

def review1(x):
    # x=round(x,6)
    if (x>=2)&(x<=12):
        return 1
    elif (x>=13)&(x<=15):
        return 2
    else:
        return 3


def service(x):
    #x=round(x,6)
    jiange=0.1
    for i in range(1,20):
        if (x>=3.93+jiange*(i-1))&(x<=3.93+jiange*i):
            return i+1
    if x==-1:
        return 1

def service1(x):
    if (x>=2)&(x<=7):
        return 1
    elif (x>=8)&(x<=9):
        return 2
    else:
        return 3


def describe(x):
    #x=round(x,6)
    jiange=0.1
    for i in range(1,30):
        if (x>=3.93+jiange*(i-1))&(x<=3.93+jiange*i):
            return i+1
    if x==-1:
        return 1

def describe1(x):
    if (x>=2)&(x<=8):
        return 1
    elif (x>=9)&(x<=10):
        return 2
    else:
        return 3

def shijian(data):
    data['hour_map'] = data['hour'].apply(map_hour)
    return data

def shop_fenduan(data):
    data['shop_score_delivery'] = data['shop_score_delivery'] * 5
    data = data[data['shop_score_delivery'] != -5]
    data['deliver_map'] = data['shop_score_delivery'].apply(deliver)
    data['deliver_map'] = data['deliver_map'].apply(deliver1)
    # del data['shop_score_delivery']
    print(data.deliver_map.value_counts())

    data['shop_score_service'] = data['shop_score_service'] * 5
    data = data[data['shop_score_service'] != -5]
    data['service_map'] = data['shop_score_service'].apply(service)
    data['service_map'] = data['service_map'].apply(service1)
    # del data['shop_score_service']
    print(data.service_map.value_counts())  # 视为好评，中评，差评
    #
    data['shop_score_description'] = data['shop_score_description'] * 5
    data = data[data['shop_score_description'] != -5]
    data['de_map'] = data['shop_score_description'].apply(describe)
    data['de_map'] = data['de_map'].apply(describe1)
    # del data['shop_score_description']
    print(data.de_map.value_counts())

    data = data[data['shop_review_positive_rate'] != -1]
    data['review_map'] = data['shop_review_positive_rate'].apply(review)
    data['review_map'] = data['review_map'].apply(review1)
    print(data.review_map.value_counts())

    data['normal_shop'] = data.apply(
        lambda x: 1 if (x.deliver_map == 3) & (x.service_map == 3) & (x.de_map == 3) & (x.review_map == 3) else 0,
        axis=1)
    del data['de_map']
    del data['service_map']
    del data['deliver_map']
    del data['review_map']
    return data


def slide_cnt(data):
    # item_cnt = data.groupby(by='item_id').count()['instance_id'].to_dict()
    # data['item_cnt'] = data['item_id'].apply(lambda x: item_cnt[x])
    # user_cnt = data.groupby(by='user_id').count()['instance_id'].to_dict()
    # data['user_cnt'] = data['user_id'].apply(lambda x: user_cnt[x])
    # shop_cnt = data.groupby(by='shop_id').count()['instance_id'].to_dict()
    # data['shop_cnt'] = data['shop_id'].apply(lambda x: shop_cnt[x])

    print('当前日期前一天的cnt')
    for d in range(19, 26):  # 18到24号
        df1 = data[data['day'] == d - 1]
        df2 = data[data['day'] == d]  # 19到25号
        user_cnt = df1.groupby(by='user_id').count()['instance_id'].to_dict()
        item_cnt = df1.groupby(by='item_id').count()['instance_id'].to_dict()
        shop_cnt = df1.groupby(by='shop_id').count()['instance_id'].to_dict()
        df2['user_cnt1'] = df2['user_id'].apply(lambda x: user_cnt.get(x, 0))
        df2['item_cnt1'] = df2['item_id'].apply(lambda x: item_cnt.get(x, 0))
        df2['shop_cnt1'] = df2['shop_id'].apply(lambda x: shop_cnt.get(x, 0))
        df2 = df2[['user_cnt1', 'item_cnt1', 'shop_cnt1', 'instance_id']]
        if d == 19:
            Df2 = df2
        else:
            Df2 = pd.concat([df2, Df2])
    data = pd.merge(data, Df2, on=['instance_id'], how='left')
    print('当前日期之前的cnt')
    for d in range(19, 26):
        # 19到25，25是test
        df1 = data[data['day'] < d]
        df2 = data[data['day'] == d]
        user_cnt = df1.groupby(by='user_id').count()['instance_id'].to_dict()
        item_cnt = df1.groupby(by='item_id').count()['instance_id'].to_dict()
        shop_cnt = df1.groupby(by='shop_id').count()['instance_id'].to_dict()
        df2['user_cntx'] = df2['user_id'].apply(lambda x: user_cnt.get(x, 0))
        df2['item_cntx'] = df2['item_id'].apply(lambda x: item_cnt.get(x, 0))
        df2['shop_cntx'] = df2['shop_id'].apply(lambda x: shop_cnt.get(x, 0))
        df2 = df2[['user_cntx', 'item_cntx', 'shop_cntx', 'instance_id']]
        if d == 19:
            Df2 = df2
        else:
            Df2 = pd.concat([df2, Df2])
    data = pd.merge(data, Df2, on=['instance_id'], how='left')

    print("前一个小时的统计量")

    return data


def zuhe(data):
    for col in ['user_gender_id','user_age_level','user_occupation_id','user_star_level']:
        data[col] = data[col].apply(lambda x: 0 if x == -1 else x)

    for col in ['item_sales_level', 'item_price_level', 'item_collected_level',
                'user_gender_id','user_age_level','user_occupation_id','user_star_level',
                'shop_review_num_level', 'shop_star_level']:
        data[col] = data[col].astype(str)

    print('item两两组合')
    data['sale_price'] = data['item_sales_level'] + data['item_price_level']
    data['sale_collect'] = data['item_sales_level'] + data['item_collected_level']
    data['price_collect'] = data['item_price_level'] + data['item_collected_level']

    print('user两两组合')
    data['gender_age'] = data['user_gender_id'] + data['user_age_level']
    data['gender_occ'] = data['user_gender_id'] + data['user_occupation_id']
    data['gender_star'] = data['user_gender_id'] + data['user_star_level']

    print('shop两两组合')
    data['review_star'] = data['shop_review_num_level'] + data['shop_star_level']


    for col in ['item_sales_level', 'item_price_level', 'item_collected_level',  'sale_price','sale_collect', 'price_collect',
                'user_gender_id', 'user_age_level', 'user_occupation_id', 'user_star_level','gender_age','gender_occ','gender_star',
                'shop_review_num_level','shop_star_level','review_star']:
        data[col] = data[col].astype(int)

    del data['review_star']

    return data

def item(data):
    print('一个item有多少brand,price salse collected level……')

    itemcnt = data.groupby(['item_id'], as_index=False)['instance_id'].agg({'item_cnt': 'count'})
    data = pd.merge(data, itemcnt, on=['item_id'], how='left')

    for col in ['item_brand_id','item_city_id', 'item_price_level', 'item_sales_level', 'item_collected_level', 'item_pv_level']:
        itemcnt = data.groupby([col, 'item_id'], as_index=False)['instance_id'].agg({str(col) + '_item_cnt': 'count'})
        data = pd.merge(data, itemcnt, on=[col, 'item_id'], how='left')
        data[str(col) + '_item_prob']=data[str(col) + '_item_cnt']/data['item_cnt']
    del data['item_cnt']

    print('一个brand有多少price salse collected level……')

    itemcnt = data.groupby(['item_brand_id'], as_index=False)['instance_id'].agg({'item_brand_cnt': 'count'})
    data = pd.merge(data, itemcnt, on=['item_brand_id'], how='left')

    for col in ['item_city_id', 'item_price_level', 'item_sales_level', 'item_collected_level', 'item_pv_level']:
        itemcnt = data.groupby([col, 'item_brand_id'], as_index=False)['instance_id'].agg({str(col) + '_brand_cnt': 'count'})
        data = pd.merge(data, itemcnt, on=[col, 'item_brand_id'], how='left')
        data[str(col) + '_brand_prob'] = data[str(col) + '_brand_cnt'] / data['item_brand_cnt']
    del data['item_brand_cnt']

    print('一个city有多少item_price_level，item_sales_level，item_collected_level，item_pv_level')

    itemcnt = data.groupby(['item_city_id'], as_index=False)['instance_id'].agg({'item_city_cnt': 'count'})
    data = pd.merge(data, itemcnt, on=['item_city_id'], how='left')
    for col in ['item_price_level', 'item_sales_level', 'item_collected_level', 'item_pv_level']:
        itemcnt = data.groupby([col, 'item_city_id'], as_index=False)['instance_id'].agg({str(col) + '_city_cnt': 'count'})
        data = pd.merge(data, itemcnt, on=[col, 'item_city_id'], how='left')
        data[str(col) + '_city_prob'] = data[str(col) + '_city_cnt'] / data['item_city_cnt']
    del data['item_city_cnt']

    print('一个price有多少item_sales_level，item_collected_level，item_pv_level')

    itemcnt = data.groupby(['item_price_level'], as_index=False)['instance_id'].agg({'item_price_cnt': 'count'})
    data = pd.merge(data, itemcnt, on=['item_price_level'], how='left')
    for col in ['item_sales_level', 'item_collected_level', 'item_pv_level']:
        itemcnt = data.groupby([col, 'item_city_id'], as_index=False)['instance_id'].agg({str(col) + '_price_cnt': 'count'})
        data = pd.merge(data, itemcnt, on=[col, 'item_city_id'], how='left')
        data[str(col) + '_price_prob'] = data[str(col) + '_price_cnt'] / data['item_price_cnt']
    del data['item_price_cnt']

    print('一个item_sales_level有多少item_collected_level，item_pv_level')

    itemcnt = data.groupby(['item_sales_level'], as_index=False)['instance_id'].agg({'item_salse_cnt': 'count'})
    data = pd.merge(data, itemcnt, on=['item_sales_level'], how='left')
    for col in ['item_collected_level', 'item_pv_level']:
        itemcnt = data.groupby([col, 'item_sales_level'], as_index=False)['instance_id'].agg({str(col) + '_salse_cnt': 'count'})
        data = pd.merge(data, itemcnt, on=[col, 'item_sales_level'], how='left')
        data[str(col) + '_salse_prob'] = data[str(col) + '_salse_cnt'] / data['item_salse_cnt']
    del data['item_salse_cnt']

    print('一个item_collected_level有多少item_pv_level')

    itemcnt = data.groupby(['item_collected_level'], as_index=False)['instance_id'].agg({'item_coll_cnt': 'count'})
    data = pd.merge(data, itemcnt, on=['item_collected_level'], how='left')
    for col in ['item_pv_level']:
        itemcnt = data.groupby([col, 'item_collected_level'], as_index=False)['instance_id'].agg({str(col) + '_coll_cnt': 'count'})
        data = pd.merge(data, itemcnt, on=[col, 'item_collected_level'], how='left')
        data[str(col) + '_coll_prob'] = data[str(col) + '_coll_cnt'] / data['item_coll_cnt']
    del data['item_coll_cnt']

    return data

def user(data):
    print('用户有多少性别')
    itemcnt = data.groupby(['user_id'], as_index=False)['instance_id'].agg({'user_cnt': 'count'})
    data = pd.merge(data, itemcnt, on=['user_id'], how='left')

    for col in ['user_gender_id','user_age_level', 'user_occupation_id', 'user_star_level']:
        itemcnt = data.groupby([col, 'user_id'], as_index=False)['instance_id'].agg({str(col) + '_user_cnt': 'count'})
        data = pd.merge(data, itemcnt, on=[col, 'user_id'], how='left')
        data[str(col) + '_user_prob']=data[str(col) + '_user_cnt']/data['user_cnt']
    del data['user_cnt']

    print('性别的年龄段，职业有多少')
    itemcnt = data.groupby(['user_gender_id'], as_index=False)['instance_id'].agg({'user_gender_cnt': 'count'})
    data = pd.merge(data, itemcnt, on=['user_gender_id'], how='left')

    for col in ['user_age_level', 'user_occupation_id', 'user_star_level']:
        itemcnt = data.groupby([col, 'user_gender_id'], as_index=False)['instance_id'].agg({str(col) + '_user_gender_cnt': 'count'})
        data = pd.merge(data, itemcnt, on=[col, 'user_gender_id'], how='left')
        data[str(col) + '_user_gender_prob']=data[str(col) + '_user_gender_cnt']/data['user_gender_cnt']
    del data['user_gender_cnt']

    print('user_age_level对应的user_occupation_id，user_star_level')
    itemcnt = data.groupby(['user_age_level'], as_index=False)['instance_id'].agg({'user_age_cnt': 'count'})
    data = pd.merge(data, itemcnt, on=['user_age_level'], how='left')

    for col in ['user_occupation_id', 'user_star_level']:
        itemcnt = data.groupby([col, 'user_age_level'], as_index=False)['instance_id'].agg({str(col) + '_user_age_cnt': 'count'})
        data = pd.merge(data, itemcnt, on=[col, 'user_age_level'], how='left')
        data[str(col) + '_user_age_prob']=data[str(col) + '_user_age_cnt']/data['user_age_cnt']
    del data['user_age_cnt']

    print('user_occupation_id对应的user_star_level')
    itemcnt = data.groupby(['user_occupation_id'], as_index=False)['instance_id'].agg({'user_occ_cnt': 'count'})
    data = pd.merge(data, itemcnt, on=['user_occupation_id'], how='left')
    for col in ['user_star_level']:
        itemcnt = data.groupby([col, 'user_occupation_id'], as_index=False)['instance_id'].agg({str(col) + '_user_occ_cnt': 'count'})
        data = pd.merge(data, itemcnt, on=[col, 'user_occupation_id'], how='left')
        data[str(col) + '_user_occ_prob']=data[str(col) + '_user_occ_cnt']/data['user_occ_cnt']
    del data['user_occ_cnt']

    return data

def user_item(data):
    itemcnt = data.groupby(['user_id'], as_index=False)['instance_id'].agg({'user_cnt': 'count'})
    data = pd.merge(data, itemcnt, on=['user_id'], how='left')
    print('一个user有多少item_id,item_brand_id……')
    for col in ['item_id',
                'item_brand_id','item_city_id','item_price_level',
                'item_sales_level','item_collected_level','item_pv_level']:
        item_shop_cnt = data.groupby([col, 'user_id'], as_index=False)['instance_id'].agg({str(col)+'_user_cnt': 'count'})
        data = pd.merge(data, item_shop_cnt, on=[col, 'user_id'], how='left')
        data[str(col) + '_user_prob'] = data[str(col) + '_user_cnt'] / data['user_cnt']

    print('一个user_gender有多少item_id,item_brand_id……')
    itemcnt = data.groupby(['user_gender_id'], as_index=False)['instance_id'].agg({'user_gender_cnt': 'count'})
    data = pd.merge(data, itemcnt, on=['user_gender_id'], how='left')
    for col in ['item_id',
                'item_brand_id','item_city_id','item_price_level',
                'item_sales_level','item_collected_level','item_pv_level']:
        item_shop_cnt = data.groupby([col, 'user_gender_id'], as_index=False)['instance_id'].agg({str(col)+'_user_gender_cnt': 'count'})
        data = pd.merge(data, item_shop_cnt, on=[col, 'user_gender_id'], how='left')
        data[str(col) + '_user_gender_prob'] = data[str(col) + '_user_gender_cnt'] / data['user_gender_cnt']

    print('一个user_age_level有多少item_id,item_brand_id……')
    itemcnt = data.groupby(['user_age_level'], as_index=False)['instance_id'].agg({'user_age_cnt': 'count'})
    data = pd.merge(data, itemcnt, on=['user_age_level'], how='left')
    for col in ['item_id',
                'item_brand_id','item_city_id','item_price_level',
                'item_sales_level','item_collected_level','item_pv_level']:
        item_shop_cnt = data.groupby([col, 'user_age_level'], as_index=False)['instance_id'].agg({str(col)+'_user_age_cnt': 'count'})
        data = pd.merge(data, item_shop_cnt, on=[col, 'user_age_level'], how='left')
        data[str(col) + '_user_age_prob'] = data[str(col) + '_user_age_cnt'] / data['user_age_cnt']

    print('一个user_occupation_id有多少item_id,item_brand_id…')
    itemcnt = data.groupby(['user_occupation_id'], as_index=False)['instance_id'].agg({'user_occ_cnt': 'count'})
    data = pd.merge(data, itemcnt, on=['user_occupation_id'], how='left')
    for col in ['item_id',
                'item_brand_id','item_city_id','item_price_level',
                'item_sales_level','item_collected_level','item_pv_level']:
        item_shop_cnt = data.groupby([col, 'user_occupation_id'], as_index=False)['instance_id'].agg({str(col)+'_user_occ_cnt': 'count'})
        data = pd.merge(data, item_shop_cnt, on=[col, 'user_occupation_id'], how='left')
        data[str(col) + '_user_occ_prob'] = data[str(col) + '_user_occ_cnt'] / data['user_occ_cnt']

    return data


def user_shop(data):
    print('一个user有多少shop_id,shop_review_num_level……')

    for col in ['shop_id', 'shop_review_num_level', 'shop_star_level']:
        item_shop_cnt = data.groupby([col, 'user_id'], as_index=False)['instance_id'].agg(
            {str(col) + '_user_cnt': 'count'})
        data = pd.merge(data, item_shop_cnt, on=[col, 'user_id'], how='left')
        data[str(col) + '_user_prob'] = data[str(col) + '_user_cnt'] / data['user_cnt']
    del data['user_cnt']

    print('一个user_gender有多少shop_id,shop_review_num_level……')
    for col in ['shop_id', 'shop_review_num_level', 'shop_star_level']:
        item_shop_cnt = data.groupby([col, 'user_gender_id'], as_index=False)['instance_id'].agg(
            {str(col) + '_user_gender_cnt': 'count'})
        data = pd.merge(data, item_shop_cnt, on=[col, 'user_gender_id'], how='left')
        data[str(col) + '_user_gender_prob'] = data[str(col) + '_user_gender_cnt'] / data['user_gender_cnt']
    del data['user_gender_cnt']

    print('一个user_age_level有多少shop_id,shop_review_num_level……')
    for col in ['shop_id', 'shop_review_num_level', 'shop_star_level']:
        item_shop_cnt = data.groupby([col, 'user_age_level'], as_index=False)['instance_id'].agg(
            {str(col) + '_user_age_cnt': 'count'})
        data = pd.merge(data, item_shop_cnt, on=[col, 'user_age_level'], how='left')
        data[str(col) + '_user_age_prob'] = data[str(col) + '_user_age_cnt'] / data['user_age_cnt']
    del data['user_age_cnt']

    print('一个user_occupation_id有多少shop_id,shop_review_num_level……')
    for col in ['shop_id', 'shop_review_num_level', 'shop_star_level']:
        item_shop_cnt = data.groupby([col, 'user_occupation_id'], as_index=False)['instance_id'].agg(
            {str(col) + '_user_occ_cnt': 'count'})
        data = pd.merge(data, item_shop_cnt, on=[col, 'user_occupation_id'], how='left')
        data[str(col) + '_user_occ_prob'] = data[str(col) + '_user_occ_cnt'] / data['user_occ_cnt']
    del data['user_occ_cnt']

    return data


def shop_item(data):
    print('一个shop有多少item_id,item_brand_id,item_city_id,item_price_level……')
    itemcnt = data.groupby(['shop_id'], as_index=False)['instance_id'].agg({'shop_cnt': 'count'})
    data = pd.merge(data, itemcnt, on=['shop_id'], how='left')
    for col in ['item_id',
                'item_brand_id','item_city_id','item_price_level',
                'item_sales_level','item_collected_level','item_pv_level']:
        item_shop_cnt = data.groupby([col, 'shop_id'], as_index=False)['instance_id'].agg({str(col)+'_shop_cnt': 'count'})
        data = pd.merge(data, item_shop_cnt, on=[col, 'shop_id'], how='left')
        data[str(col) + '_shop_prob'] = data[str(col) + '_shop_cnt'] / data['shop_cnt']
    del data['shop_cnt']

    print('一个shop_review_num_level有多少item_id,item_brand_id,item_city_id,item_price_level……')
    itemcnt = data.groupby(['shop_review_num_level'], as_index=False)['instance_id'].agg({'shop_rev_cnt': 'count'})
    data = pd.merge(data, itemcnt, on=['shop_review_num_level'], how='left')
    for col in ['item_id',
                'item_brand_id','item_city_id','item_price_level',
                'item_sales_level','item_collected_level','item_pv_level']:
        item_shop_cnt = data.groupby([col, 'shop_review_num_level'], as_index=False)['instance_id'].agg({str(col)+'_shop_rev_cnt': 'count'})
        data = pd.merge(data, item_shop_cnt, on=[col, 'shop_review_num_level'], how='left')
        data[str(col) + '_shop_rev_prob'] = data[str(col) + '_shop_rev_cnt'] / data['shop_rev_cnt']
    del data['shop_rev_cnt']

    # print('一个shop_star_level有多少item_id,item_brand_id,item_city_id,item_price_level……')
    # itemcnt = data.groupby(['shop_star_level'], as_index=False)['instance_id'].agg({'shop_star_cnt': 'count'})
    # data = pd.merge(data, itemcnt, on=['shop_star_level'], how='left')
    # for col in ['item_id',
    #             'item_brand_id', 'item_city_id', 'item_price_level',
    #             'item_sales_level', 'item_collected_level', 'item_pv_level']:
    #     item_shop_cnt = data.groupby([col, 'shop_star_level'], as_index=False)['instance_id'].agg({str(col) + '_shop_star_cnt': 'count'})
    #     data = pd.merge(data, item_shop_cnt, on=[col, 'shop_star_level'], how='left')
    #     data[str(col) + '_shop_star_prob'] = data[str(col) + '_shop_star_cnt'] / data['shop_star_cnt']
    # del data['shop_star_cnt']
    return data


def lgbCV(train, test):
    col = [c for c in train if
           c not in ['is_trade', 'item_category_list', 'item_property_list', 'predict_category_property', 'instance_id',
                     'context_id', 'realtime', 'context_timestamp']]
    # cat = ['sale_price', 'gender_star', 'user_age_level', 'item_price_level', 'item_sales_level', 'sale_collect',
    #        'price_collect', 'item_brand_id', 'user_star_level', 'item_id', 'shop_id',
    #        'item_city_id', 'context_page_id', 'gender_age', 'shop_star_level', 'item_pv_level', 'user_occupation_id',
    #        'day', 'gender_occ', 'user_gender_id']
    X = train[col]
    y = train['is_trade'].values
    X_tes = test[col]
    y_tes = test['is_trade'].values
    print('Training LGBM model...')
    lgb0 = lgb.LGBMClassifier(
        objective='binary',
        # metric='binary_error',
        num_leaves=35,
        depth=8,
        learning_rate=0.05,
        seed=2018,
        colsample_bytree=0.8,
        # min_child_samples=8,
        subsample=0.9,
        n_estimators=20000)
    lgb_model = lgb0.fit(X, y, eval_set=[(X_tes, y_tes)], early_stopping_rounds=200)
    best_iter = lgb_model.best_iteration_
    predictors = [i for i in X.columns]
    feat_imp = pd.Series(lgb_model.feature_importance(), predictors).sort_values(ascending=False)
    print(feat_imp)
    print(feat_imp.shape)
    # pred= lgb_model.predict(test[col])
    pred = lgb_model.predict_proba(test[col])[:, 1]
    test['pred'] = pred
    test['index'] = range(len(test))
    # print(test[['is_trade','pred']])
    print('误差 ', log_loss(test['is_trade'], test['pred']))
    return best_iter

def sub(train, test, best_iter):
    col = [c for c in train if
           c not in ['is_trade', 'item_category_list', 'item_property_list', 'predict_category_property', 'instance_id',
                     'context_id', 'realtime', 'context_timestamp']]
    X = train[col]
    y = train['is_trade'].values
    print('Training LGBM model...')
    lgb0 = lgb.LGBMClassifier(
        objective='binary',
        # metric='binary_error',
        num_leaves=35,
        depth=8,
        learning_rate=0.05,
        seed=2018,
        colsample_bytree=0.8,
        # min_child_samples=8,
        subsample=0.9,
        n_estimators=best_iter)
    lgb_model = lgb0.fit(X, y)
    predictors = [i for i in X.columns]
    feat_imp = pd.Series(lgb_model.feature_importance(), predictors).sort_values(ascending=False)
    print(feat_imp)
    print(feat_imp.shape)
    # pred= lgb_model.predict(test[col])
    pred = lgb_model.predict_proba(test[col])[:, 1]
    test['predicted_score'] = pred
    sub1 = test[['instance_id', 'predicted_score']]
    sub=pd.read_csv(path_test, sep="\s+")
    sub=pd.merge(sub,sub1,on=['instance_id'],how='left')
    sub=sub.fillna(0)
    #sub[['instance_id', 'predicted_score']].to_csv('result/result0320.csv',index=None,sep=' ')
    sub[['instance_id', 'predicted_score']].to_csv(path_result,sep=" ",index=False)


def DeepFM_sub(train, test):
    def evaluate_by_batch():
        pred = np.ndarray([n_instances_test])
        n_iter_pred = int(math.ceil(n_instances_test / batch_size))
        print('evaluating')
        for i_eval in range(n_iter_pred ):
            print("{}/{}".format(i_eval, n_iter_pred))
            idx_start, idx_end = i_eval*batch_size, min((i_eval+1)*batch_size, n_instances_test)
            pred[idx_start:idx_end] = classifier.predict_proba(Xi_test[idx_start:idx_end], Xv_test[idx_start:idx_end])
        return pred
    def save_checkpoint(pred, postfix):
        test['predicted_score'] = pred
        sub1 = test[['instance_id', 'predicted_score']]
        sub=pd.read_csv(path_test, sep="\s+")
        sub=pd.merge(sub,sub1,on=['instance_id'],how='left')
        sub=sub.fillna(0)
        path_save_result = path_result + 'result_' + training_info + '_' + postfix + '.txt'
        sub[['instance_id', 'predicted_score']].to_csv(path_save_result, sep=" ",index=False)
        print(path_save_result+' saved')

    '''
    col = [c for c in train if
           c not in ['is_trade', 'item_category_list', 'item_property_list', 'predict_category_property', 'instance_id',
                     'context_id', 'realtime', 'context_timestamp',
                     'user_cnt1', 'item_cnt1', 'shop_cnt1', 'user_cntx',	'item_cntx', 'shop_cntx',
                     ]]
    col_cate = [c for c in col if issubclass(np.dtype('intp').type, type(train[c][0]))]
    col_real = [c for c in col if not issubclass(np.dtype('intp').type, type(train[c][0]))]
    '''
    batch_size = 16
    col_cate, col_real = select_features_by_corr(train)

    #n_fields = len(col)
    n_fields_cate = len(col_cate)
    n_fields_real = len(col_real)
    n_fields = n_fields_cate + n_fields_real
    #assert n_fields == n_fields_cate + n_fields_real

    n_instances_train = len(train)
    n_instances_test = len(test)

    enc_lbl = preprocessing.LabelEncoder()

    feature_sizes_cate = np.ndarray((n_fields_cate,), dtype=np.dtype('int'))
    feature_sizes_real = np.ndarray((n_fields_real,), dtype=np.dtype('int'))

    Xi_train_cate = np.ndarray((n_fields_cate, n_instances_train), dtype=np.dtype('int'))
    Xv_train_cate = np.ndarray((n_fields_cate, n_instances_train), dtype=np.dtype('int'))
    Xi_train_real = np.ndarray((n_fields_real, n_instances_train))
    Xv_train_real = np.ndarray((n_fields_real, n_instances_train))

    y_train = train['is_trade'].values
    dist = np.bincount(y_train.astype(np.dtype('int64'))).tolist()
    assert y_train.size == dist[0] + dist[1]
    print('label distribution')
    print('label 0 = ' + str(dist[0]))
    print('label 1 = ' + str(dist[1]))
    class_weight = [dist[1]/(dist[0]+dist[1]), dist[0]/(dist[0]+dist[1])]
    print(class_weight )
    print(type(class_weight ))

    Xi_test_cate = np.ndarray((n_fields_cate, n_instances_test), dtype=np.dtype('int'))
    Xv_test_cate = np.ndarray((n_fields_cate, n_instances_test), dtype=np.dtype('int'))
    Xi_test_real = np.ndarray((n_fields_real, n_instances_test))
    Xv_test_real = np.ndarray((n_fields_real, n_instances_test))

    # category features
    for i, c in enumerate(col_cate):
        enc_lbl.fit(pd.concat([train[c], test[c]]))
        n_features = len(enc_lbl.classes_)
        feature_sizes_cate[i] = n_features

        train[c] = enc_lbl.transform(train[c])
        test[c] = enc_lbl.fit_transform(test[c])

        Xi_train_cate[i] = train[c].astype(int)
        Xv_train_cate[i] = np.ones((n_instances_train,))
        Xi_test_cate[i] = test[c].astype(int)
        Xv_test_cate[i] = np.ones((n_instances_test,))

    # real number features
    for i, c in enumerate(col_real):
        n_features = 1
        feature_sizes_real[i] = n_features

        #train[c] = enc_lbl.transform(train[c])
        #test[c] = enc_lbl.fit_transform(test[c])

        Xi_train_real[i] = np.zeros((n_instances_train), dtype=np.dtype('int'))
        Xv_train_real[i] = train[c]

        '''
        print('===================================real value feature : ' + c + '===================================')
        print(Xv_train_real.shape)
        print(Xv_train_real[i][:10])
        '''
        Xi_test_real[i] = np.zeros((n_instances_test), dtype=np.dtype('int'))
        Xv_test_real[i] = test[c]

    feature_sizes = np.concatenate((feature_sizes_cate, feature_sizes_real)).tolist()
    Xi_train = np.concatenate((Xi_train_cate, Xi_train_real))
    Xv_train = np.concatenate((Xv_train_cate, Xv_train_real))
    Xi_test = np.concatenate((Xi_test_cate, Xi_test_real))
    Xv_test = np.concatenate((Xv_test_cate, Xv_test_real))

    Xi_train = Xi_train.swapaxes(0,1)
    Xv_train = Xv_train.swapaxes(0,1)
    Xi_test = Xi_test.swapaxes(0,1)
    Xv_test = Xv_test.swapaxes(0,1)

    Xi_valid = None
    Xv_valid = None
    y_valid =  None

    print('n_fields : ' + str(n_fields))
    print('len(feature_sizes : )' + str(len(feature_sizes)))
    print('max(feature_sizes : )' + str(max(feature_sizes)))
    print('create '+training_info+' classifier')
    classifier = DeepFM(n_fields, feature_sizes, embedding_size = 20,
                             h_depth=2, deep_layers=[32, 32], is_deep_dropout=True, dropout_deep=[0.0, 0.2, 0.2],
                             is_batch_norm=True, verbose=True, weight_decay= 0.002,
                             n_epochs=1, batch_size=batch_size,
                             eval_metric=sklearn.metrics.average_precision_score,
                             use_fm=True, use_ffm=False,
                             use_cuda=False,
                             class_weight=class_weight,
                             )
    """
    print("Xi_train.shape : "+str(Xi_train.shape))
    print("Xv_train.shape : "+str(Xv_train.shape))
    print("y_train.shape : "+str(y_train.shape))
    """
    print(training_info+' classifier fitting')

    #pred = evaluate_by_batch()
    #save_checkpoint(pred, 'pre')
    for i in range(64):
        classifier.fit(Xi_train, Xv_train, y_train,
                           ealry_stopping=True)
        pred = evaluate_by_batch()
        save_checkpoint(pred, '{:02d}'.format(i))
        '''
        pred = np.ndarray([n_instances_test])
        n_iter_pred = int(math.ceil(n_instances_test / batch_size))
        print('evaluating')
        for i_eval in range(n_iter_pred ):
            print("{}/{}".format(i_eval, n_iter_pred))
            idx_start, idx_end = i_eval*batch_size, min((i_eval+1)*batch_size, n_instances_test)
            pred[idx_start:idx_end] = classifier.predict_proba(Xi_test[idx_start:idx_end], Xv_test[idx_start:idx_end])

        test['predicted_score'] = pred
        sub1 = test[['instance_id', 'predicted_score']]
        sub=pd.read_csv(path_test, sep="\s+")
        sub=pd.merge(sub,sub1,on=['instance_id'],how='left')
        sub=sub.fillna(0)
        path_save_result = path_result+'result_{:03d}.txt'.format(i)
        sub[['instance_id', 'predicted_score']].to_csv(path_save_result, sep=" ",index=False)
        print(path_save_result+' saved')
        '''


def save_header(train):
    col = [c for c in train if
           c not in ['is_trade', 'item_category_list', 'item_property_list', 'predict_category_property', 'instance_id',
                     'context_id', 'realtime', 'context_timestamp',
                     'user_cnt1', 'item_cnt1', 'shop_cnt1', 'user_cntx',	'item_cntx', 'shop_cntx',
                     ]]
    col_cate = [c for c in col if
                issubclass(np.dtype('intp').type, type(train[c][0])) and
                issubclass(np.dtype('intp').type, type(train[c][10])) and
                issubclass(np.dtype('intp').type, type(train[c][20]))]
    col_real = [c for c in col if not (
            issubclass(np.dtype('intp').type, type(train[c][0])) and
            issubclass(np.dtype('intp').type, type(train[c][10])) and
            issubclass(np.dtype('intp').type, type(train[c][20])))]

    with open(path_save_header , 'w') as f:
        f.write('======================categorical======================')
        for c in col_cate :
            f.write(c+',\n')
        f.write('======================real======================')
        for c in col_real:
            f.write(c + ',\n')
    print('header saved')


def select_features_by_corr(train):
    print('selecting features')
    col = [c for c in data if
           c not in ['is_trade', 'item_category_list', 'item_property_list', 'predict_category_property',
                     'instance_id',
                     'context_id', 'realtime', 'context_timestamp',
                     'user_cnt1', 'item_cnt1', 'shop_cnt1', 'user_cntx',	'item_cntx', 'shop_cntx',
                     ]]
    ## 把那些无关的变量去除掉
    col_cate = [c for c in col if issubclass(np.dtype('intp').type, type(train[c][0]))]
    col_real = [c for c in col if not issubclass(np.dtype('intp').type, type(train[c][0]))]
    print("categorical field number : {}".format(len(col_cate)))
    print("real value field number : {}".format(len(col_real)))

    cor_select_set = []
    for col in [col_cate, col_real]:
        cor_list = []
        cnt = 0
        for var in col:
            spearman_corr = stats.spearmanr(train['is_trade'], train[var])
            cor_list.append(spearman_corr.pvalue)

        cor_series = pd.Series(cor_list, index=col)

        cor_series.sort_values(ascending=False)

        n_features_selected = int(len(cor_series) * features_select_rate)
        cor_select = list(cor_series[:n_features_selected].index)
        cor_select_set.append(cor_select)
    print('features selected by corr， after selection, ')
    print("categorical field number : {}".format(len(cor_select_set[0])))
    print("real value field number : {}".format(len(cor_select_set[1])))

    return cor_select_set[0],cor_select_set[1]




if __name__ == "__main__":
    if use_saved_features:
        data = pd.read_csv(path_save_features)
        if 'pandas_index' in data:
            del data['pandas_index']
        print('pre-processed features loaded')
    else:
        train = pd.read_csv(path_train, sep="\s+")
        test = pd.read_csv(path_test, sep="\s+")
        data = pd.concat([train, test])
        data = data.drop_duplicates(subset='instance_id')  # 把instance id去重
        print('make feature')
        data = base_process(data)
        data=shijian(data)
        data=shop_fenduan(data)
        data = slide_cnt(data)
        data = zuhe(data)
        print('----------------------------全局统计特征---------------------------------------------------')
        data = item(data)
        data = user(data)
        data = user_item(data)
        data = user_shop(data)
        data=shop_item(data)
        "----------------------------------------------------保存----------------------------------------"
        data.to_csv(path_save_features, header = True, index= True, index_label='pandas_index')
    '''
    "----------------------------------------------------线下----------------------------------------"
    train= data[(data['day'] >= 18) & (data['day'] <= 23)]
    test= data[(data['day'] == 24)]
    best_iter = lgbCV(train, test)
    "----------------------------------------------------线上----------------------------------------"
    train = data[data.is_trade.notnull()]
    test = data[data.is_trade.isnull()]
    sub(train, test, best_iter)
    '''
    train = data[data.is_trade.notnull()]
    #train = train[:int(len(train)*0.02)]
    test = data[data.is_trade.isnull()]
    #test = test[:int(len(test)*0.4)]
    DeepFM_sub(train, test)
