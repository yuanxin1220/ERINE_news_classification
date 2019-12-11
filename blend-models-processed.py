import time
import os
import gc
import jieba
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer,CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn import metrics
import pandas as pd
import numpy as np

from gensim.models import Word2Vec

from keras.layers import *
from keras.models import *
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.engine.topology import Layer

from tqdm import tqdm

embed_size=200
maxlen = 100
max_features = 73915 #73662 # 95000

"去除指定无用的符号，这里我们主要拿空格举例"
puncts = [' ']
def clean_text(x):
    x=x.strip()
    for punct in puncts:
        x=x.replace(punct,'')
    return x

"让文本只保留文字"
def is_chinese(xchar):
    if xchar>=u'\u4e00' and xchar<=u'\u9fa5':
        return True
    else:
        return False
def keep_chinese_text(x):
    out_str=''
    for i in x:
        if is_chinese(i):
            out_str=out_str+i
    return out_str

def seg_sentence(sentence,stopwords):
    "对句子进行分词和去除停用词"
    sentence_seged=jieba.cut(sentence)
    outstr=''
    for word in sentence_seged:
        if word not in stopwords:
                outstr+=word
                outstr+=" "
    return outstr

def build_vocab(sentences,verbose=True):
    "追踪训练词汇表，遍历所有文本对单词进行计数"
    vocab={}
    for sentence in tqdm(sentences,disable=(not verbose)):
        for word in sentence.split():
            try:
                vocab[word]+=1
            except KeyError:
                vocab[word]=1
    #vocab=sorted(vocab.items(),key=lambda d:d[1],reverse=True)   # 分词后共出现73915个单词
    # vocab=vocab[:vocab_size]
    return vocab

def texts_to_sequences(sentences,vocab,verbose=True):
    seq_sentences=[]
    unk_vec=np.random.random(embed_size)*0.5
    unk_vec=unk_vec-unk_vec.mean()
    for sentence in tqdm(sentences,disable=(not verbose)):
        seq_sentence=[]
        for word in sentence.split():
            seq_sentence.append(vocab.get(word,unk_vec))
        seq_sentences.append(seq_sentence)
    return seq_sentences

def load_and_prec():
    #文件读取
    train_df=pd.read_csv('data/news_classification_dataset.csv')

    #train_df, test_df = train_test_split(train_df, test_size=0.01, random_state=2019)

    #创建停用词列表
    stopwords = ['的', '呀', '这', '那', '就', '的话', '如果']

    train_df["text"]=train_df["text"].apply(lambda x:clean_text(x))
    #test_df["text"]=test_df["text"].apply(lambda x:clean_text(x))

    train_df["text"]=train_df["text"].apply(lambda x:keep_chinese_text(x))
    #test_df["text"]=test_df["text"].apply(lambda x:keep_chinese_text(x))

    train_df["text"]=train_df["text"].apply(lambda x:seg_sentence(x,stopwords))
    #test_df["text"]=test_df["text"].apply(lambda x:seg_sentence(x,stopwords))

    vocab=build_vocab(train_df["text"],True)

    # split to train and val
    train_df, test_df = train_test_split(train_df, test_size=0.01, random_state=2019)
    train_df,val_df=train_test_split(train_df,test_size=0.01,random_state=2019)

    # print("Train shape: ",train_df.shape)   # (34623, 3)
    # print("Val shape: ",val_df.shape)   # (3848, 3)

    ## Get the input values
    train_X=train_df["text"].values
    val_X=val_df["text"].values
    test_X=test_df["text"].values

    ## Get the target values
    train_y=train_df["label"].values
    val_y=val_df["label"].values
    test_y=test_df["label"].values

    np.random.seed(2019)
    trn_idx=np.random.permutation(len(train_X))
    val_idx=np.random.permutation(len(val_X))

    train_X=train_X[trn_idx]
    val_X=val_X[val_idx]
    train_y=train_y[trn_idx]
    val_y=val_y[val_idx]

    # Tokenize the sentences
    train_X=texts_to_sequences(train_X, vocab)
    val_X=texts_to_sequences(val_X,vocab)
    test_X=texts_to_sequences(test_X,vocab)
    # Pad the sentences
    train_X=pad_sequences(train_X,maxlen=maxlen)
    val_X=pad_sequences(val_X,maxlen=maxlen)
    test_X=pad_sequences(test_X,maxlen=maxlen)

    return train_df,test_df,train_X,val_X,test_X,train_y,val_y,test_y,vocab

def word2vec_model(train_df, test_df, vocab):
    count=0
    nb_words=len(vocab)
    print(nb_words)
    start=time.clock()
    all_data=pd.concat([train_df["text"],test_df["text"]])
    file_name='data/word2vec.model'
    if not os.path.exists(file_name):
        model=Word2Vec([[word for word in sentence.split()] for sentence in all_data.values],
                       size=embed_size,window=5,iter=10,workers=11,seed=2019,min_count=2)
        model.save(file_name)
    else:
        model=Word2Vec.load(file_name)
    print("add word2vec finished...")
    end=time.clock()
    print('Running time: %s Seconds' %(end-start))

    nb_words=min(max_features,len(vocab))
    embedding_word2vec_matrix=np.zeros((nb_words,embed_size))
    for word,i in vocab.items():
        if i>max_features:continue
        embedding_vector=model[word] if word in model else None
        if embedding_vector is not None:
            count+=1
            embedding_word2vec_matrix[i]=embedding_vector
        else:
            unk_vec=np.random.random(embed_size)*0.5
            unk_vec=unk_vec-unk_vec.mean()
            embedding_word2vec_matrix[i]=unk_vec
    del model;
    gc.collect()
    return embedding_word2vec_matrix

"""Attention Layer"""
class Attention(Layer):
    def __init__(self, step_dim,
                 W_regularizer=None, b_regularizer=None,
                 W_constraint=None, b_constraint=None,
                 bias=True, **kwargs):
        self.supports_masking = True
        self.init = initializers.get('glorot_uniform')

        self.W_regularizer = regularizers.get(W_regularizer)
        self.b_regularizer = regularizers.get(b_regularizer)

        self.W_constraint = constraints.get(W_constraint)
        self.b_constraint = constraints.get(b_constraint)

        self.bias = bias
        self.step_dim = step_dim
        self.features_dim = 0
        super(Attention, self).__init__(**kwargs)

    def build(self, input_shape):
        assert len(input_shape) == 3

        self.W = self.add_weight((input_shape[-1],),
                                 initializer=self.init,
                                 name='{}_W'.format(self.name),
                                 regularizer=self.W_regularizer,
                                 constraint=self.W_constraint)
        self.features_dim = input_shape[-1]

        if self.bias:
            self.b = self.add_weight((input_shape[1],),
                                     initializer='zero',
                                     name='{}_b'.format(self.name),
                                     regularizer=self.b_regularizer,
                                     constraint=self.b_constraint)
        else:
            self.b = None

        self.built = True

    def compute_mask(self, input, input_mask=None):
        return None

    def call(self, x, mask=None):
        features_dim = self.features_dim
        step_dim = self.step_dim

        eij = K.reshape(K.dot(K.reshape(x, (-1, features_dim)),
                              K.reshape(self.W, (features_dim, 1))), (-1, step_dim))

        if self.bias:
            eij += self.b

        eij = K.tanh(eij)

        a = K.exp(eij)

        if mask is not None:
            a *= K.cast(mask, K.floatx())

        a /= K.cast(K.sum(a, axis=1, keepdims=True) + K.epsilon(), K.floatx())

        a = K.expand_dims(a)
        weighted_input = x * a
        return K.sum(weighted_input, axis=1)

    def compute_output_shape(self, input_shape):
        return input_shape[0], self.features_dim

"CNN Model"
def model_cnn(embedding_matrix):
    filter_sizes=[1,2,3,5]
    num_filters=36

    inp=Input(shape=(maxlen,))
    x=Embedding(max_features,embed_size,weights=[embedding_matrix])(inp)
    x=Reshape((maxlen,embed_size,1))(x)

    maxpool_pool=[]
    for i in range(len(filter_sizes)):
        conv=Conv2D(num_filters,kernel_size=(filter_sizes[i],embed_size),
                    kernel_initializer='he_normal',activation='elu')(x)
        maxpool_pool.append(MaxPool2D(pool_size=(maxlen-filter_sizes[i]+1,1))(conv))

    z=Concatenate(axis=1)(maxpool_pool)
    z=Flatten()(z)
    z=Dropout(0.1)(z)

    outp=Dense(1,activation="sigmoid")(z)

    model=Model(inputs=inp,outputs=outp)
    model.compile(loss='binary_crossentropy',optimizer='adam',metrics=['accuracy'])
    return model

"LSTM models"
def model_lstm_attn(embedding_matrix):
    inp=Input(shape=(maxlen,))
    x=Embedding(max_features,embed_size,weights=[embedding_matrix],trainable=False)(inp)
    x=Bidirectional(LSTM(128,return_sequences=True))(x)
    x=Bidirectional(LSTM(64,return_sequences=True))(x)
    x=Attention(maxlen)(x)
    x=Dense(64,activation="relu")(x)
    x=Dense(1,activation="sigmoid")(x)
    model=Model(inputs=inp,outputs=x)
    model.compile(loss='binary_crossentropy',optimizer='adam',metrics=['accuracy'])
    return model

"""Train and predict"""
def train_pred(model,epochs=2):
    for e in range(epochs):
        model.fit(train_X,train_y,batch_size=512,epochs=1,validation_data=(val_X,val_y))
        pred_val_y=model.predict([val_X],batch_size=1024,verbose=0)
    pred_test_val=model.predict([val_X],batch_size=1024,verbose=0)
    return pred_val_y,pred_test_val

def train_pred_F1(model,epochs=2):
    for e in range(epochs):
        model.fit(train_X,train_y,batch_size=512,epochs=1,validation_data=(val_X,val_y))
        pred_val_y=model.predict([val_X],batch_size=1024,verbose=0)

        best_thresh=0.5
        best_score=0.0
        for thresh in np.arange(0.1,0.501,0.01):
            thresh=np.round(thresh,2)
            score=metrics.f1_score(val_y,(pred_val_y>thresh).astype(int))
            if score>best_score:
                best_thresh=thresh
                best_score=score
        print("Val F1 Score: {:.4f}".format(best_score))

    pred_test_y=model.predict([test_X],batch_size=1024,verbose=0)
    return pred_val_y,pred_test_y,best_score

if __name__ == '__main__':
    train_df, test_df, train_X, val_X,test_X, train_y, val_y, test_y, vocab=load_and_prec()
    embedding_word2vec_matrix=word2vec_model(train_df,test_df,vocab)
    print(embedding_word2vec_matrix.shape)  #(73915, 200)

    # model_gru_atten_3=model_gru_atten_3(embedding_word2vec_matrix)
    # model_gru_atten_3.summary()

    outputs=[]

    pred_val_y,pred_test_y,best_score=train_pred_F1(model_cnn(embedding_word2vec_matrix),epochs=1)
    outputs.append([pred_val_y,pred_test_y,best_score,'2d CNN']) # 0.912718204488778 2d CNN

    pred_val_y,pred_test_y,best_score=train_pred_F1(model_lstm_attn(embedding_word2vec_matrix),epochs=1)
    outputs.append([pred_val_y,pred_test_y,best_score,'2 LSTM x/ attention'])   # 0.9282051282051282 2 LSTM x/ attention

    outputs.sort(key=lambda x:x[2])
    weights=[i for i in range(1,len(outputs)+1)]
    weights=[float(i)/sum(weights) for i in weights]

    for output in outputs:
        print(output[2],output[3])

    from sklearn.linear_model import LinearRegression
    X=np.asarray([outputs[i][0] for i in range(len(outputs))])
    print(X.shape)  # (2, 381, 1)
    X=X[...,0] # 只取最后一维的第零个元素，是对验证集的预测值
    print(X.shape)  # (2, 381)
    reg=LinearRegression().fit(X.T,val_y)
    print(reg.score(X.T,val_y),reg.coef_)   # 0.7642986141890105 [0.17936741 0.8483659 ]

    pred_val_y=np.sum([outputs[i][0]*reg.coef_[i] for i in range(len(outputs))],axis=0)

    thresholds=[]
    for thresh in np.arange(0.1,0.501,0.01):
        thresh=np.round(thresh,2)
        res=metrics.f1_score(val_y,(pred_val_y>thresh).astype(int))
        thresholds.append([thresh,res])
        print("F1 score at threshold {0} is {1}".format(thresh,res))

    thresholds.sort(key=lambda  x:x[1],reverse=True)
    best_thresh=thresholds[0][0]
    print("Best threshold: ",best_thresh)

    pred_test_y=np.sum([outputs[i][1]*reg.coef_[i] for i in range(len(outputs))],axis=0)
    pred_test_y=(pred_test_y>best_thresh).astype(int)
    # test_df=pd.read_csv("data/test.csv",usecols=["qid"])
    out_df=pd.DataFrame({"id":test_df["id"].values})
    print(len(out_df))
    out_df['prediction']=pred_test_y
    out_df.to_csv("submission.csv",index=False)

    """score_f1 :  0.9172932330827067
    score_acc :  0.9142857142857143"""
    score_f1 = metrics.f1_score(test_y, (pred_test_y > best_thresh).astype(int))
    print("score_f1 : ",score_f1)
    score_acc = metrics.accuracy_score(test_y, (pred_test_y > best_thresh).astype(int))
    print("score_acc : ",score_acc)


