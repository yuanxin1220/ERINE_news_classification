# kaggle standard imports
import numpy as np
import pandas as pd
import jieba

# extra imports
np.random.seed(235)
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score
import gc

# XGboost related
from sklearn.feature_extraction.text import TfidfVectorizer
import xgboost as xgb
from scipy.sparse import csr_matrix,hstack

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
    print("train_df shape = ", train_df.shape)  # (31160, 3)
    print("load data success !")

    """对句子进行分词和去停用词"""
    #创建停用词列表
    stopwords = ['的', '呀', '这', '那', '就', '的话', '如果']
    train_df["text"]=train_df["text"].apply(lambda x: seg_sentence(x,stopwords))
    val_df["text"]=val_df["text"].apply(lambda x: seg_sentence(x,stopwords))
    test_df["text"]=test_df["text"].apply(lambda x: seg_sentence(x,stopwords))

    print('fill missing and get values')
    X_train=train_df["text"].fillna("na_").values
    X_val=val_df["text"].fillna("na_").values
    X_test=test_df["text"].fillna("na_").values

    y_train=train_df['label'].values
    y_val=val_df['label'].values
    y_test=test_df['label'].values

    print('size of training data: ',X_train.shape)  # size of training data: (31160,)
    return train_df,val_df,test_df,X_train,X_val,X_test,y_train,y_val,y_test

def vector_Prepare(X_train, X_val, X_test):
    """Prepare Vector For XGbost input"""
    all_text=list(X_train)+list(X_test)

    word_vector=TfidfVectorizer(max_features=9000)

    print('fit word vector')
    word_vector.fit(all_text)
    print("finished")
    #print(word_vector.get_feature_names())

    print('transfer data based on word vector')
    # transform 后得到tfidf矩阵，tocsr对稀疏矩阵进行压缩
    train_word_vector=word_vector.transform(X_train).tocsr()    # (31160, 9000)
    valid_word_vector=word_vector.transform(X_val).tocsr()  # (3463, 9000)
    test_word_vector=word_vector.transform(X_test).tocsr()  # (3848, 9000)
    print("finished!")

    del all_text
    gc.collect()

    return train_word_vector,valid_word_vector,test_word_vector

def get_features(data):
    for dataframe in data:
        # dataframe 添加列
        dataframe["text_size"]=dataframe["text"].apply(len).astype('uint16')    # 句子长度
        dataframe["exc_count"]=dataframe["text"].apply(lambda x:x.count("！")).astype('uint16')  # 感叹号数量
        dataframe["question_count"]=dataframe["text"].apply(lambda x:x.count("？")).astype('uint16') # 问号数量
        dataframe["unq_punctuation_count"]=dataframe["text"].apply(lambda x:sum(x.count(p) for p in '∞θ÷α•à−β∅³π‘₹´°£€\×™√²')).astype('uint16') # 特殊符号
        dataframe["punctuation_count"]=dataframe["text"].apply(lambda x: sum(x.count(p) for p in '.,;:^_`')).astype('uint16')   # 标点符号数量
        dataframe["symbol_count"]=dataframe["text"].apply(lambda x:sum(x.count(p) for p in '*&$%')).astype('uint16')    # 数学符号的数量
        dataframe["words_count"]=dataframe["text"].apply(lambda x:len(x.split())).astype('uint16')  #单词数量
        dataframe["unique_words"]=dataframe["text"].apply(lambda x:(len(set(1 for w in x.split())))).astype('uint16') # 不同单词的数量
        dataframe["unique_rate"]=dataframe["unique_words"]/dataframe["words_count"]
        dataframe["word_max_length"]=dataframe["text"].apply(lambda x:max([len(word) for word in x.split()])).astype('uint16')  #最大单词长度
    return data

def feature_engineer(data):
    """Feature Engineering"""
    print('generate the feature')
    data=get_features(data)
    print("finished!")

    feature_cols=["text_size","exc_count","question_count","unq_punctuation_count","punctuation_count",
                  "symbol_count","words_count","unique_words","unique_rate","word_max_length"]


    print('final preparation for input')
    X_train=csr_matrix(train_df[feature_cols].values)
    X_val=csr_matrix(val_df[feature_cols].values)
    X_test=csr_matrix(test_df[feature_cols].values)
    return data,X_train,X_val,X_test

def build_xgb(train_X,train_y,valid_X,valid_y=None,subsample=0.75):
    xgtrain=xgb.DMatrix(train_X,label=train_y)
    if valid_y is not None:
        xgvalid=xgb.DMatrix(valid_X,label=valid_y)
    else:
        xgvalid=None

    model_params={}
    # binary 0 or 1
    model_params['objective']='binary:logistic'
    # eta is the learning_rate, [default=0.3]
    model_params['eta']=0.3
    # depth of the tree, deeper more complex.
    model_params['max_depth']=6
    # 0 [default] print running message, 1 means silent mode
    model_params['silent']=1
    model_params['eval_metric']='auc'
    # will give up further partitioning [default=1]
    model_params['subsample']=subsample
    # subsample ration of columns when constructing each tree
    model_params['colsample_bytree']=subsample
    # random seed
    model_params['seed']=2019

    model_params=list(model_params.items())
    return xgtrain,xgvalid,model_params

def train_xgboost(xgtrain,xgvalid,model_params,num_rounds=500,patience=20):
    if xgvalid is not None:
        # watchlist what information should be printed. specify validation monitoring
        wachlist=[(xgtrain,'train'),(xgvalid,'test')]
        model=xgb.train(model_params,xgtrain,num_rounds,wachlist,early_stopping_rounds=patience)
    else:
        model=xgb.train(model_params,xgtrain,num_rounds)
    return model

def save_results(submit,y_hat,y_test,threshold):
    print('threshold is: ',threshold)   # 0.48999999999999977
    results=(y_hat>threshold).astype(int)
    score=f1_score(y_test,results)
    print("F1 score for test : ",score) # 0.9711075441412521
    # print(results[:100])
    submit['prediction']=results
    save_to=('submission-xgboost.csv')
    submit.to_csv(save_to,index=False)

if __name__ == '__main__':
    """Prepare Data"""
    train_df, val_df, test_df, X_train, X_val, X_test, y_train, y_val, y_test=data_Prepare()

    """Prepare Vector For XGboost input"""
    train_word_vector, valid_word_vector, test_word_vector=vector_Prepare(X_train, X_val, X_test)
    # (31160, 9000) (3463, 9000) (3848, 9000)
    del X_train
    del X_val
    del X_test
    gc.collect()

    """Features Engineering"""
    data = [train_df, val_df, test_df]
    print("finished!")
    data, X_train, X_val, X_test=feature_engineer(data)
    # (31160, 10) (3463, 10) (3848, 10)
    del val_df
    del train_df
    # del test_df
    gc.collect()

    """按列将数组堆叠"""
    input_train=hstack([X_train,train_word_vector])
    input_valid=hstack([X_val,valid_word_vector])
    input_test=hstack([X_test,test_word_vector])

    train_word_vector=None
    valid_word_vector=None
    test_word_vector=None
    print("finished")

    print("train the model")
    xgtrain,xgvalid,model_params=build_xgb(input_train,y_train,input_valid,y_val)
    model=train_xgboost(xgtrain,xgvalid,model_params)
    print("finished!")

    print('predict validation')
    validate_hat=np.zeros((X_val.shape[0],1))
    validate_hat[:,0]=model.predict(xgb.DMatrix(input_valid),ntree_limit=model.best_ntree_limit)

    scores_list=[]
    for threshold in np.arange(0.1,0.501,0.01):
        score=f1_score(y_val,(validate_hat>threshold).astype(int))
        scores_list.append([threshold,score])
        print('F1 score: {} for threshold: {}'.format(score,threshold))

        scores_list.sort(key=lambda x:x[1],reverse=True)
        best_threshold=scores_list[0][0]
        print('best threshold th generate predictions: ',best_threshold)
        print('best score: ', scores_list[0][1])

    print('predict results')
    predictions=np.zeros((X_test.shape[0],1))
    predictions[:,0]=model.predict(xgb.DMatrix(input_test),ntree_limit=model.best_ntree_limit)

    print('save results')
    save_results(test_df,predictions,y_test,threshold=
                  best_threshold)