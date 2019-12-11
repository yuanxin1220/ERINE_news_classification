# kaggle standard imports
import numpy as np
import pandas as pd
import jieba

# extra imports
np.random.seed(235)
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score
import gc

# lightgbm related
from sklearn.feature_extraction.text import TfidfVectorizer
from scipy.sparse import csr_matrix,hstack
import lightgbm as lgb

def seg_sentence(sentence,stopwords):
    sentence_seged=jieba.cut(sentence)
    outstr=''
    for word in sentence_seged:
        if word not in stopwords:
            outstr+=word
            outstr+=" "
    return outstr

def data_Prepare():
    """Prepare data"""
    print('load data')
    # load training and testing data
    train_df=pd.read_csv('data/news_classification_dataset.csv')

    train_df,test_df=train_test_split(train_df,train_size=0.9,random_state=235)

    train_df,val_df=train_test_split(train_df,train_size=0.9,random_state=235)
    print("train_df shape = ",train_df.shape)
    print("load data success !")

    """对句子进行分词和去停用词"""
    # 创建停用词列表
    stopwords=['的', '呀', '这', '那', '就', '的话', '如果','了','啊']
    train_df["text"]=train_df["text"].apply(lambda x:seg_sentence(x,stopwords))
    val_df["text"]=val_df["text"].apply(lambda x:seg_sentence(x,stopwords))
    test_df["text"]=test_df["text"].apply(lambda x:seg_sentence(x,stopwords))

    print('fill missing and get values')
    X_train=train_df["text"].fillna("na_").values
    X_val=val_df["text"].fillna("na_").values
    X_test=test_df["text"].fillna("na_").values

    y_train=train_df["label"].values
    y_val=val_df["label"].values
    y_test=test_df["label"].values

    print('size of training data: ',X_train.shape) # size of training data: (31160,)
    print("finished")
    return train_df,val_df,test_df,X_train,X_val,X_test,y_train,y_val,y_test

def vector_Prepare(X_train,X_val,X_test):
    """Prepare Vector For XGboost input"""
    all_text=list(X_train)+list(X_test)

    word_vector=TfidfVectorizer(max_features=9000)

    print('fit word vector')
    word_vector.fit(all_text)
    print("finished")

    print('transfer data based on word vector')
    # transform 后得到的tfidf矩阵，tocsr对稀疏矩阵进行压缩
    train_word_vector=word_vector.transform(X_train).tocsr()
    valid_word_vector=word_vector.transform(X_val).tocsr()
    test_word_vector=word_vector.transform(X_test).tocsr()
    print("finished!")

    del all_text
    gc.collect()

    return train_word_vector,valid_word_vector,test_word_vector

def get_features(data):
    for dataframe in data:
        # dataframe 添加列
        dataframe["text_size"]=dataframe["text"].apply(len).astype('uint16')    # 句子长度
        dataframe["exc_count"]=dataframe["text"].apply(lambda x: x.count("！")).astype('uint16') # 感叹号数量
        dataframe["question_count"]=dataframe["text"].apply(lambda x: x.count("？")).astype('uint16')    # 问号数量
        dataframe["unq_punctuation_count"]=dataframe["text"].apply(lambda x:sum(x.count(p) for p in '∞θ÷α•à−β∅³π‘₹´°£€\×™√²')).astype('uint16') # 特殊符号
        dataframe["punctuation_count"]=dataframe["text"].apply(lambda x:sum(x.count(p) for p in '.,;:^_`')).astype('uint16')    # 标点符号数量
        dataframe["symbol_count"]=dataframe["text"].apply(lambda x: sum(x.count(p) for p in '*&$%')).astype('uint16')   # 数学符号的数量
        dataframe["words_count"]=dataframe["text"].apply(lambda x:len(x.split())).astype('uint16')  # 单词数量
        dataframe["unique_words"]=dataframe["text"].apply(lambda x:(len(set(1 for w in x.split())))).astype('uint16')   # 不同单词的数量
        dataframe["unique_rate"]=dataframe["unique_words"]/dataframe["words_count"]
        dataframe["word_max_length"]=dataframe["text"].apply(lambda x:max([len(word) for word in x.split()])).astype('uint16')  # 最大单词长度
    return data

def feature_engineer(data):
    """Feature Engineering"""
    print('generate the feature')
    data=get_features(data)

    feature_cols=["text_size","exc_count","question_count","unq_punctuation_count","punctuation_count",
                  "symbol_count","words_count","unique_words","unique_rate","word_max_length"]

    X_train=csr_matrix(train_df[feature_cols].values)
    X_val=csr_matrix(val_df[feature_cols].values)
    X_test=csr_matrix(test_df[feature_cols].values)
    print("finished!")
    return data,X_train,X_val,X_test

def build_lgb(train_X,train_y,valid_X,valid_y):
    # create dataset for lightgbm
    lgb_train=lgb.Dataset(train_X,train_y)
    lgb_eval=lgb.Dataset(valid_X,valid_y,reference=lgb_train)
    # specify your configurations as a dict
    model_params={
        'boosting_type':'gbdt',
        'objective':'binary',
        'metrics':{'binary_logloss','auc'}, # 二进制对数损失
        'num_leaves':5,
        'max_depth':6,
        'min_data_in_leaf':450,
        'learning_rate':0.1,
        'feature_fraction':0.9,
        'bagging_fraction':0.95,
        'bagging_freq':5,
        'lambda_l1':1,
        'lambda_l2':0.001,  #l2越小正则程度越高
        'min_gain_to_split':0.2,
        'verbose':5,
        'is_unbalance':True
    }
    return lgb_train,lgb_eval,model_params

def train_lgb(lgb_train, lgb_eval, model_params):
    gbm=lgb.train(model_params,
                  lgb_train,
                  num_boost_round=10000,
                  valid_sets=lgb_eval,
                  early_stopping_rounds=500)
    return gbm

def save_results(submit,y_hat,y_test,threshold):
    print('threshold is : ',threshold)  # 0.45
    results=(y_hat>threshold).astype(int)
    score=f1_score(y_test,results)
    print("F1 score for test : ",score) # 0.9232409381663114
    submit['prediction']=results
    save_to='submission-lightgbm.csv'
    submit.to_csv(save_to,index=False)

if __name__ == '__main__':
    """Prepare Data"""
    train_df, val_df, test_df, X_train, X_val, X_test, y_train, y_val, y_test=data_Prepare()

    """Prepare Vector For XGboost input"""
    train_word_vector, valid_word_vector, test_word_vector=vector_Prepare(X_train,X_val,X_test)
    del X_train
    del X_val
    del X_test
    gc.collect()

    """Feature Engineering"""
    data=[train_df,val_df,test_df]
    data, X_train, X_val, X_test=feature_engineer(data)

    """按列将数组组合"""
    print('final preparation for input')
    input_train=hstack([X_train,train_word_vector])
    input_valid=hstack([X_val,valid_word_vector])
    input_test=hstack([X_test,test_word_vector])

    train_word_vector=None
    valid_word_vector=None
    test_word_vector=None
    print("finished")

    print("train the model")
    lgb_train, lgb_eval, model_params=build_lgb(input_train,y_train,input_valid,y_val)
    model=train_lgb(lgb_train,lgb_eval,model_params)
    print("finished!")

    print("predict validation")
    preds=model.predict(input_valid,num_iteration=model.best_iteration) # 输出的是概率结果

    scores_list=[]
    for threshold in [0.2,0.3,0.31,0.33,0.4,0.45,0.5]:
        score=f1_score(y_val,(preds>threshold).astype(int))
        scores_list.append([threshold,score])
        print('F1 score: {} for threshold: {}'.format(score,threshold))

        scores_list.sort(key=lambda x:x[1],reverse=True)
        best_threshold=scores_list[0][0]
        print('best threshold to generate predictions: ',best_threshold)
        print('best score: ',scores_list[0][1])

    # 导出特征重要性
    importance=model.feature_importance()
    names=model.feature_name()
    with open('feature_importance.txt','w+') as file:
        for index, im in enumerate(importance):
            string=names[index]+', '+str(im)+'\n'
            file.write(string)
    print('finished!')

    print('predict results')
    predictions=model.predict(input_test,num_iteration=model.best_iteration)
    print('save results')
    save_results(test_df,predictions,y_test,threshold=best_threshold)
    print('finished!')