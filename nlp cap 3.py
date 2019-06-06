# coding: utf-8

# In[1]:

""" NLP capstone"""

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
get_ipython().run_line_magic('matplotlib', 'inline')
import pickle #pprint, pickle

import pandas as pd
import re
from collections import Counter
import nltk
import spacy
from nltk.corpus import stopwords

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import Normalizer
from sklearn.cluster import AffinityPropagation
from sklearn.cluster import SpectralClustering
from sklearn.cluster import MeanShift, estimate_bandwidth
from sklearn.cluster import KMeans
from sklearn.preprocessing import LabelEncoder

from nltk.stem import WordNetLemmatizer

from sklearn import metrics
#from matplotlib.backends.backend_pdf import PdfPages

import warnings
from sklearn.model_selection import train_test_split
warnings.simplefilter(action='ignore', category=FutureWarning)
warnings.simplefilter(action='ignore', category=Warning)

NEW_DATA = False
NEW_DATA = True
#import sys
#sys.stdout = open('graphs', 'w')

#pp = PdfPages('foo.pdf')
#pp.savefig(plot1)


def clean_data(text):
    text = text.lower()
    text = re.sub(r'--',' ',text)
    text = re.sub(r'_',' ',text)
    text = re.sub("[\[].,*?[\]]", "", text)
    text = re.sub('[0-9]+', '', text)
    text = ' '.join(text.split())
    return text

en_stops = set(stopwords.words('english'))
best_var = -1
best_ft_cnt=0

if NEW_DATA is  True:
    print("gen new data.....")
    file = 'articles cut 1.csv'
    file = 'articles1.csv'
    df = pd.read_csv(file,nrows=5000)
    df = df[(df['year'] == 2016) & (df['month'] < 13)] # use only 2016 data

    df.dropna(subset=['author'], how='all', inplace = True)
#        
    print ("unique author names ->", df['author'].nunique())
    LE = LabelEncoder()
    df['author_id'] = LE.fit_transform(df['author'])

    print ("articles ->",len(df))
    print ("unique author IDs ->", df['author_id'].nunique())

    s= (df.groupby(['author']).author.agg('count'))
    s.sort_values(ascending=False,inplace=True)  # get list of top 10 authors

    top_author_list=s[:10].index.tolist()# get top 10 authors

    df = df[df['author'].isin(s[:10].index.tolist())]
    print ("unique author names ->", df['author'].nunique())
    author = pd.DataFrame()
    new_cols = ['author', 'author_id','text','len']
    features = pd.DataFrame(columns=new_cols)

    for x,item in df.iterrows():
        x=nlp(clean_data(str(item['text'])))
        name = item['author']
        author_id = item['author_id']
        row = [name, author_id, x, len(x)]
        features.loc[len(features)] = row
    #print(name,author_id)

#features['text'] = features['text'].str.lower().str.replace('—', '')
#features['text'] = features['text'].str.replace('"', '')
#rint("fetures len:",len(features))

    articles = pd.DataFrame(columns = ['author','author_id','content'])
    print("len of articles:",len(articles))
    lemmatizer = WordNetLemmatizer()


    for i,j,k in zip(features.text, features.author, features.author_id):
        line = []
        temp  = str(nltk.LineTokenizer().tokenize(i)).split()
        for word in temp:
            if word not in en_stops:
                word = lemmatizer.lemmatize(word)
                line.append(word)
        articles = articles.append({'content': ' '.join(line), 'author':j,'author_id':k}, ignore_index=True)
#print("len of articles:",len(articles),"line len",len(line))


    xtrain, xtest = train_test_split(articles, test_size=0.25, random_state=0)

    vectorizer = TfidfVectorizer(max_df=0.5,min_df=2,use_idf=True, smooth_idf=True,norm=u'l2',                             stop_words='english')

    articles_tfidf=vectorizer.fit_transform(features.text)  #vectorize data
    print("features:",articles_tfidf.get_shape()[1])

#create test sets
    xtrain_tfidf, xtest_tfidf= train_test_split(articles_tfidf, test_size=0.25, random_state=0)

    xtrain_tfidf_csr = xtrain_tfidf.tocsr()#Reshape vector

    article_cnt = xtrain_tfidf_csr.shape[0]
    tfidf_by_article = [{} for _ in range(0,article_cnt)]
    terms = vectorizer.get_feature_names() #list of features

    for idx, j in zip(*xtrain_tfidf_csr.nonzero()):  # get features words and scores
        tfidf_by_article[idx][terms[j]] = xtrain_tfidf_csr[idx, j]
#print('\n\nOriginal sentence:', xtrain.iloc[index])
#print('Tf_idf vector:', tfidf_by_article[index])


#for feature_cnt in range(400,1000,100):#,150,50):#25,250,25):
    start = 600
    end = start + 500 + 1
    for feature_cnt in range(start,end,100):#,150,50):#25,250,25):
        svd= TruncatedSVD(feature_cnt)
        lsa = make_pipeline(svd, Normalizer(copy=False))
        
        xtrain_lsa = lsa.fit_transform(xtrain_tfidf)  #SVD data fit

        variance_explained=svd.explained_variance_ratio_
        variance_tot = variance_explained.sum()
        print("features -> ",feature_cnt, ", variance % ->",round(variance_tot*100,3))
        if variance_tot > best_var:
            best_ft_cnt = feature_cnt
            best_var = variance_tot
            if (variance_tot > .93):  #good enough!
                break

    content_by_component=pd.DataFrame(xtrain_lsa,index=xtrain)# what content is similar

    plt.plot([sum(variance_explained[:i]) for i in range(len(variance_explained))])
    plt.title('Variance')
    plt.xlim([0, len(variance_explained)])
    plt.xlabel('Component')
    plt.ylim([0, 1])
    plt.ylabel('Variance')
    plt.show()

# Compute document similarity using LSA components
    similarity = np.asarray(np.asmatrix(xtrain_lsa) * np.asmatrix(xtrain_lsa).T)
    sim_matrix=pd.DataFrame(similarity,index=xtrain).iloc[0:10,0:10] #take first ten sentences
#Making a plot
    ax = sns.heatmap(sim_matrix,yticklabels=range(10))
    plt.title("LSA similarity")
    plt.show()

# ##Clustering 

    svd= TruncatedSVD(best_ft_cnt) #use best feature count from SVD pipeline
    lsa = make_pipeline(svd, Normalizer(copy=False))
    xtrain_lsa = lsa.fit_transform(xtrain_tfidf)#projext training data
    print("Writing new data file....")
    fn = "NLP data"

    output = open(fn, 'wb')
    pickle.dump(xtrain_lsa, output)
    output.close()
#To read it back:
else:
    print("use old  data.....")
    fn = "NLP data"
    pkl_file = open(fn, 'rb')
    xtrain_lsa = pickle.load(pkl_file)
    pkl_file.close()

"""
# Meanshift
#bandwidth = estimate_bandwidth(X_train, quantile=0.2, n_samples=5000)
band_width = estimate_bandwidth(xtrain_lsa, quantile=0.9)#, n_samples=500)# get bandwidth
ms = MeanShift(bandwidth=band_width, bin_seeding = False)
ms.fit_predict(xtrain_lsa)
#cluster_centers_indices = ms.cluster_centers_indices_

labels = ms.labels_  # get clusters
n_clusters = len(labels)
plt.title("Mean Shift")
plt.scatter(xtrain_lsa[:,0], xtrain_lsa[:,1], c =labels )
plt.show()
#print("Silhouette Coefficient: %0.3f"
#    % metrics.silhouette_score(xtrain_lsa, labels, metric='sqeuclidean'))
#print("mean shift, cluster_centers_indices", n_clusters)
"""

SKIP=True
if SKIP is False:
    sc = SpectralClustering(n_clusters=6)#n_clusters)
    sc.fit(xtrain_lsa)
        
    predict_sc = sc.fit_predict(xtrain_lsa)# Predicted clusters
    labels = sc.labels_
    plt.title("SpectralClustering")
    plt.scatter(xtrain_lsa[:,0], xtrain_lsa[:,1], c = labels)
    plt.show()
    clusters=len(labels)
#pd.crosstab(predict_sc, c)

    print("spectral: Silhouette Coefficient: %0.3f"
      % metrics.silhouette_score(xtrain_lsa, labels, metric='sqeuclidean'))
#print(len(labels)," clusters")

#KMeans
    km = KMeans(n_clusters = 10, random_state=100)# init the model
    km.fit(xtrain_lsa)# predict training set
    labels = km.labels_
    title = "Kmeans,training"
    c=np.array(xtrain['author_id'])
    c.reshape(-1,1)
    plt.scatter(xtrain_lsa[:, 0], xtrain_lsa[:, 1], c=labels)
    plt.title(title)
#pd.crosstab(y_pred_km, c)
    plt.show()
    print("kmeans: Silhouette Coefficient: %0.3f"
      % metrics.silhouette_score(xtrain_lsa, labels, metric='sqeuclidean'))
#print(len(labels)," clusters")

 #precomputed and euclidean
af_type = 'precomputed'
af_type = 'euclidean'

af = AffinityPropagation(damping=0.95, max_iter=200, convergence_iter=15, \
                         copy=True, preference=None, affinity='euclidean',\
                         verbose=False)#, affinity=’euclidean’)
af.fit(xtrain_lsa)
cluster_centers_indices = af.cluster_centers_indices_
labels = af.labels_
cluster_centers_indices = af.cluster_centers_indices_
n_clusters = len(cluster_centers_indices)
# Extract cluster assignments for each data point.
labels = af.labels_
t = "Affinity Clustering, " + str(n_clusters) + " clusters"
plt.title(t)
plt.scatter(xtrain_lsa[:,0], xtrain_lsa[:,1], c = labels)
plt.show()
print("Silhouette Coefficient: %0.3f"
      % metrics.silhouette_score(xtrain_lsa, labels-1, metric='sqeuclidean'))
"""

# Compute Affinity Propagation
X=xtrain_lsa
af = AffinityPropagation().fit(X)
#print (af.estimator.get_params())
cluster_centers_indices = af.cluster_centers_indices_
labels = af.labels_

n_clusters_ = len(cluster_centers_indices)

print('Estimated number of clusters: %d' % n_clusters_)
#print("Homogeneity: %0.3f" % metrics.homogeneity_score(labels_true, labels))
#rint("Completeness: %0.3f" % metrics.completeness_score(labels_true, labels))
#print("V-measure: %0.3f" % metrics.v_measure_score(labels_true, labels))
#print("Adjusted Rand Index: %0.3f"
#      % metrics.adjusted_rand_score(labels_true, labels))
#print("Adjusted Mutual Information: %0.3f"
#      % metrics.adjusted_mutual_info_score(labels_true, labels,
#  
"""                                        


"""
# #############################################################################
# Plot result
import matplotlib.pyplot as plt
from itertools import cycle

plt.close('all')
plt.figure(1)
plt.clf()

colors = cycle('bgrcmykbgrcmykbgrcmykbgrcmyk')
for k, col in zip(range(n_clusters_), colors):
    class_members = labels == k
    cluster_center = X[cluster_centers_indices[k]]
    plt.plot(X[class_members, 0], X[class_members, 1], col + '.')
    plt.plot(cluster_center[0], cluster_center[1], 'o', markerfacecolor=col,
             markeredgecolor='k', markersize=14)
    for x in X[class_members]:
        plt.plot([cluster_center[0], x[0]], [cluster_center[1], x[1]], col)

plt.title('Estimated number of clusters: %d' % n_clusters_)
plt.show()

print(n_clusters," clusters")
print("affinity: Silhouette Coefficient: %0.3f"
      % metrics.silhouette_score(xtrain_lsa, labels, metric='sqeuclidean'))
#print(len(labels)," clusters")
"""