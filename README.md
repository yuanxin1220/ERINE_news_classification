# ERINE_news_classification
the java solution for the part of LinkedList in leetcode

as the change on dictory of dataset, there should be some modification 

1. line 30 in lightgbm-tfidf
    train_df=pd.read_csv('data/news_classification_dataset.csv')
    --> train_df=pd.read_csv('news_classification_dataset.csv')
    
2. line 30 in xgboost-tfidf
    train_df=pd.read_csv('data/news_classification_dataset.csv')
    --> train_df=pd.read_csv('news_classification_dataset.csv')
    
3. line 83 in blend-models-processed
    train_df=pd.read_csv('data/news_classification_dataset.csv')
    --> train_df=pd.read_csv('news_classification_dataset.csv')
