# -*- coding: utf-8 -*-
"""
Created on Sat Sep 17 04:26:02 2016

@author: naman
"""
import multiprocessing
import json
from operator import itemgetter
import os
import gensim, logging
from gensim.models.doc2vec import TaggedDocument 
from gensim.models.doc2vec import LabeledSentence
import re
import numpy as np

logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)


#def get_text(path_to_json_file):
#    with open(path_to_json_file, 'r') as fp:
#        script=json.load(fp)
#        
#        script_text = list()
#        for i in range(1, len(script)+1):
#            scene = script[str(i)] 
#            list_dialogues = scene['char_dialogues']
#            list_desc = scene['scene_descriptons_list']
#            list_scene = list_dialogues + list_desc
#            list_scene = sorted(list_scene)
#            list_scene = [l[1:] for l in list_scene ]
#            list_scene = [' '.join(l) for l in list_scene]
#            text = ' . '.join(list_scene).encode('utf-8')
#            if len(text.split()) > 10:
#                script_text.append(text)
#        return script_text


def get_text(directory, filename):
    fp=open(directory+'/'+filename, 'r')
    script = fp.readlines()
    script=[re.sub('<.*?>', '', s).rstrip('\n').rstrip('\r').strip() for s in script]
    script=[s for s in script if s!='']
    partition_size= len(script)/200
    
    if partition_size == 0:
        return list((-1,''))

    partitions=list()
    partition=''
    for i in range(0,200):        
        for j in range(i*partition_size, (i+1)*partition_size):
            partition += ' '+script[j]
        index = partition_size*(i+1)-1
        partitions.append(partition)
        tmp_line=''.join(e for e in script[index] if e.isalpha())
        if(tmp_line.isupper()):
            partition=script[index]
        else:
            partition=''
        
    if len(script) % 200 != 0:
        partition=''
        for s in script[index:]:
            partition += s +' . '
        partitions[-1] += partition
    return list(enumerate(partitions))

directory='/home/naman/SNLP/imsdb'
text_files=os.listdir(directory)
scripts_text=list()
scripts_text1=list()
for f in text_files:
    scripts_text.append((f[:-4].replace(',',''), get_text(directory,f)))
    scripts_text1.append(f[:-4])

all_scripts=list()
for movie in scripts_text:
    if movie[1][0] == -1:
        continue
    all_scripts += ([("%s_%d"% (movie[0],scene_num), scene) for scene_num, scene in movie[1]])    
    



#==============================================================================
# docLabels = [t[0] for t in all_scripts]
# docs = [t[1] for t in all_scripts]    
# 
# def remove_punc(text):
#     punc=['.',',','!','?']
#     new_text=''.join(e for e in text if e not in punc)
#     return new_text
#     
# class DocIterator(object):
# 
#     #SPLIT_SENTENCES = re.compile(u"[.!?:]\s+")  # split sentences on these characters
# 
#     def __init__(self, doc_list, labels_list):
#        self.labels_list = labels_list
#        self.doc_list = doc_list
#     def __iter__(self):
#         for idx, doc in enumerate(self.doc_list):
#             yield TaggedDocument(words=remove_punc(doc),tags=[self.labels_list[idx]])
#             
# it=DocIterator(docs, docLabels)
# 
# model = gensim.models.Doc2Vec(size=300, 
#                               window=10, 
#                               min_count=1, 
#                               workers=3,
#                               alpha=0.025, 
#                               min_alpha=0.025) # use fixed learning rate
# 
# model.build_vocab(it)
# 
# for epoch in range(10):
#     model.train(it)
#     model.alpha -= 0.002 # decrease the learning rate
#     model.min_alpha = model.alpha # fix the learning rate, no deca
#     model.train(it)
#     print "Epoch %d Completed" % epoch
# 
# model.save('/home/naman/SNLP/imsdb_doc2vec_scriptpartitions_nopunc.model')
# 
#==============================================================================

model = gensim.models.Doc2Vec.load('/home/naman/SNLP/imsdb_doc2vec_scriptpartitions.model')

centroid_vectors=list()
variance_vectors=list()
ii=-1
scripts_text2=list()
for movie,_ in scripts_text:
    ii+=1
    movie_labels=[movie+'_%d' % i for i in range(0,200)]
    try:
        vectors=model.docvecs[movie_labels]
        scripts_text2.append(scripts_text1[ii])  
    except KeyError:
        continue
    centroid=np.mean(vectors,axis=0) 
    centroid_vectors.append((movie,centroid))
    variance=np.diag(np.cov(vectors.T))
    variance_vectors.append((movie, variance))
    
#central_vectors=np.array([c[1] for c in central_vectors])   
    


import pandas as pd
cols=range(300)
var_vectors=[[v[0]]+list(v[1]) for v in variance_vectors]
mean_vectors=[[m[0]]+list(m[1]) for m in centroid_vectors]
df_m=pd.DataFrame(mean_vectors,columns=['Movie']+cols)
df_v=pd.DataFrame(var_vectors,columns=['Movie']+cols)


from sklearn.decomposition import PCA


mean_components=5
var_components=5
pca = PCA(n_components=mean_components)
pca.fit(df_m.ix[:,1:])
X = pca.transform(df_m.ix[:,1:])
dim_reduced=zip([v[0] for v in mean_vectors], X)
dim_reduced=[[v[0]]+list(v[1]) for v in dim_reduced]
df_mean=pd.DataFrame(dim_reduced,columns=['Movie']+range(mean_components))

pca = PCA(n_components=var_components)
pca.fit(df_v.ix[:,1:])
X = pca.transform(df_v.ix[:,1:])
dim_reduced=zip([v[0] for v in var_vectors], X)
dim_reduced=[[v[0]]+list(v[1]) for v in dim_reduced]
df_var=pd.DataFrame(dim_reduced,columns=['Movie']+range(var_components))


df=pd.merge(df_mean,df_var, on='Movie')
df_full=pd.merge(df_m, df_v, on='Movie')

ratings=pd.read_pickle('/home/naman/SNLP/ratings.pkl')['ratings']
ratings=list(ratings.items())
r_mean=np.mean([t[1] for t in ratings])
r_var=np.var([t[1] for t in ratings])
def label(a,m,v):
    if a>m+v:
        return 1
    elif a<m-v:
        return -1
    else:
        return 0
ratings_labels=[(r[0],label(r[1], r_mean, r_var)) for r in ratings]
df_ratings=pd.DataFrame(ratings_labels, columns=['Movie', 'Rating'])

df['Movie']=scripts_text2
df_full['Movie']=scripts_text2
df_final_full=pd.merge(df_full, df_ratings, on='Movie')



df_Shankar=pd.read_pickle('/home/naman/SNLP/char_net_final_II.pkl')
df_Shankar.columns=['Movie']+range(8)
df_Jar=pd.read_pickle('/home/naman/SNLP/emotion_binwise2.pkl')
df_Jar.columns=['Movie']+range(100,110)
df_Shankar2=pd.read_pickle('/home/naman/SNLP/topic_overlap.pkl')
df_Shankar2.columns=['Movie']+range(200,210)

df_Shankar=pd.merge(df_Shankar, df_ratings, on='Movie')
df_Shankar2=pd.merge(df_Shankar2, df_ratings, on='Movie')
df_Jar=pd.merge(df_Jar, df_ratings, on='Movie')
df_Naman=pd.merge(df, df_ratings, on='Movie')

df_final=pd.merge(df_Naman.ix[:,:-1], df_Shankar.ix[:,:-1], on='Movie')
df_final=pd.merge(df_final, df_Shankar2.ix[:,:-1], on='Movie')
df_final=pd.merge(df_final, df_Jar.ix[:,:-1], on='Movie')
df_final=pd.merge(df_final, df_ratings, on='Movie')

X_Naman=df_Naman.ix[:,1:-1]
Y_Naman=df_Naman.ix[:,-1]
X_Shankar=df_Shankar.ix[:,1:-1]
Y_Shankar=df_Shankar.ix[:,-1]
X_Shankar2=df_Shankar2.ix[:,1:-1]
Y_Shankar2=df_Shankar2.ix[:,-1]
X_Jar=df_Jar.ix[:,1:-1]
Y_Jar=df_Jar.ix[:,-1]


pd.to_pickle(df_Naman, '/home/naman/SNLP/FinalVectors_D2V.pkl')
pd.to_pickle(df_Naman.ix[:,:-1], '/home/naman/SNLP/FinalVectors_Naman.pkl')
pd.to_pickle(df_final_full, '/home/naman/SNLP/FinalVectors_Full_D2V.pkl')


X_final=df_final.ix[:,1:-1]
Y_final=df_final.ix[:,-1]


from sklearn.svm import SVC
from sklearn.cross_validation import cross_val_score
clf = SVC()
scores_Naman = cross_val_score(clf, X_Naman, Y_Naman, cv=10)
scores_Shankar = cross_val_score(clf, X_Shankar, Y_Shankar, cv=10)
scores_Shankar2 = cross_val_score(clf, X_Shankar, Y_Shankar, cv=10)
scores_Jar = cross_val_score(clf, X_Jar, Y_Jar, cv=10)
scores_final = cross_val_score(clf, X_final, Y_final, cv=10)

scores=np.vstack((scores_Shankar, scores_Shankar2, scores_Jar, scores_Naman)).T
scores=pd.DataFrame(scores, columns=['CharacterNetworks','TopicOverlap','EmotionAnalysis','Doc2Vec'])

from sklearn.cross_validation import train_test_split
#X_train, X_test, Y_train, Y_test = train_test_split(X_final, Y_final, stratify=Y_final)
clf1=SVC()
clf1.fit(X_final, Y_final)
Y_pred=clf1.predict(X_final)

from sklearn.metrics import confusion_matrix
result=confusion_matrix(Y_final, Y_pred)


from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
plt.style.use('ggplot')

tsne_model = TSNE(n_components=2, random_state=0)
#tsne_op = tsne_model.fit_transform(np.array([c[1] for c in variance_vectors]))
tsne_op = tsne_model.fit_transform(np.array(df_final_full.ix[:,1:-1]))
plt.figure(figsize=(10,10))
plt.scatter(tsne_op[:,0], tsne_op[:,1])
plt.show()





#plotting some movies document vectors for all scenes
movie= 'Terminator Salvation'
movie_labels=[movie+'_%d' % i for i in range(0,200)]
vectors=model.docvecs[movie_labels]  
tsne_model = TSNE(n_components=2, random_state=0)
tsne_op = tsne_model.fit_transform(vectors)
plt.figure(figsize=(10,10))
plt.scatter(tsne_op[:,0], tsne_op[:,1])
plt.show()

