import itertools

import numpy as np
from scipy import linalg
import matplotlib.pyplot as plt
import matplotlib as mpl
from collections import Counter
from sklearn import mixture

import pandas as pd

import cPickle

color_iter = itertools.cycle(['navy', 'c', 'cornflowerblue', 'gold',
                              'darkorange'])
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
plt.style.use('ggplot')


"""DATA INITIALIZATION"""

rating_bins = np.arange(0.5,10.5,1)

"""Original Movie Database"""
movieDict = open("/home/naman/SNLP/ratings.pkl")
movieDatabase = cPickle.load(movieDict)

"""Shankar's char_net_final_1"""
fp_1 = open("/home/naman/SNLP/char_net_final_I.pkl")
char_net_final_1 = cPickle.load(fp_1)
char_net_final_1 = char_net_final_1.rename(columns={0:'Movie'}) # To make the first column heading as Movie

# print char_net_final_1.head

"""Shankar's char_net_final_1"""
fp_2 = open("/home/naman/SNLP/char_net_final_II.pkl")
char_net_final_2 = cPickle.load(fp_2)
char_net_final_2 = char_net_final_2.rename(columns={0:'Movie'}) # To make the first column heading as Movie

# print char_net_final_2.head

"""Shankar's movie topic overlap"""
fp_3 = open("/home/naman/SNLP/topic_overlap.pkl")
topicOverlap = cPickle.load(fp_3)
topicOverlap = topicOverlap.rename(columns={0:'Movie'}) # To make the first column heading as Movie

# print topicOverlap.head


"""Jar's emotions"""


fp_4 = open("/home/naman/SNLP/emotion_binwise2.pkl")
emotionAnalysis = cPickle.load(fp_4)
emotionAnalysis = emotionAnalysis.rename(columns={0:'Movie'}) # To make the first column heading as Movie


"""Naman's doc2vec features"""
fp_5 = open("/home/naman/SNLP/FinalVectors_Doc2Vec.pkl")
doc2vec_features = cPickle.load(fp_5)
# print doc2vec_features.head

def plot_results(X, Y_, means, covariances, index, title, x_lowerLimit, x_upperLimit, y_lowerLimit, y_upperLimit):
    splot = plt.subplot(1, 1, 1)
    for i, (mean, covar, color) in enumerate(zip(
            means, covariances, color_iter)):
        v, w = linalg.eigh(covar)
        v = 2. * np.sqrt(2.) * np.sqrt(v)
        u = w[0] / linalg.norm(w[0])
        # as the DP will not use every component it has access to
        # unless it needs it, we shouldn't plot the redundant
        # components.
        if not np.any(Y_ == i):
            continue
        plt.scatter(X[Y_ == i, 0], X[Y_ == i, 1], .8, color=color)

        # Plot an ellipse to show the Gaussian component
        angle = np.arctan(u[1] / u[0])
        angle = 180. * angle / np.pi  # convert to degrees
        ell = mpl.patches.Ellipse(mean, v[0], v[1], 180. + angle, color=color)
        ell.set_clip_box(splot.bbox)
        ell.set_alpha(0.5)
        # splot.add_artist(ell)

    plt.xlim(x_lowerLimit, x_upperLimit)
    plt.ylim(y_lowerLimit, y_upperLimit)
    plt.xticks(())
    plt.yticks(())
    plt.title(title)
     plt.show()

def cluster_analysis(classScore,originalDatasetWithMovieName,predictedScoreArray):
    #genreCounter = Counter()
    clusterMovieRatingMovies = {}
    clusterMovieGenreMovies = {}
    clusterMovieGenreScore = {}
    clusterMovieRatingScore = {}
    # clusterMovieGenre['SCORE'] = {}
    # clusterMovieGenre['MOVIES'] = {}
    clusterMovieGenreScore['score']=classScore
    clusterMovieRatingScore['score']=classScore

    clusterMovieRating={}
    clusterMovieGenre={}
    for i in range(predictedScoreArray.shape[0]):
        if predictedScoreArray[i]==classScore:
            tempMovieName = originalDatasetWithMovieName.ix[i]['Movie']

            if (tempMovieName in movieDatabase['ratings'].keys()):
                #print "Checking ",tempMovieName
                tempMovieRating = movieDatabase['ratings'][tempMovieName]
                tempMovieGenre = movieDatabase['genres'][tempMovieName]
                clusterMovieRatingMovies[tempMovieName]=tempMovieRating
                clusterMovieGenreMovies[tempMovieName]=tempMovieGenre
            else:
                #print tempMovieName, "is not available in the movie ratings"
                pass
            
    genres_allMovies = clusterMovieGenreMovies.values()
    print len(clusterMovieGenreMovies)
    #genreArray = np.asarray(genres_allMovies)
    #genreArray = genreArray.flatten()
    genreArray = [item for sublist in genres_allMovies for item in sublist]
    

    #for i in range(genreArray.shape[0]):
        #genreCounter+=Counter(genreArray)

    #print genreCounter
    #print Counter(genreArray)

    ratings_allMovies = clusterMovieRatingMovies
    ratingArray = np.array(ratings_allMovies.values())
    #histogramRatings = np.digitize(ratingArray,rating_bins)

    #print "Mean, Variance = (%f,%f)" %(np.mean(ratingArray), np.var(ratingArray))
    
    #plt.
    (ratingArray, bins=10)
    #plt.show()

    clusterMovieRating['SCORE']=clusterMovieRatingScore
    clusterMovieRating['MOVIE']=clusterMovieRatingMovies

    clusterMovieGenre['SCORE']=clusterMovieGenreScore
    clusterMovieGenre['MOVIE']=clusterMovieGenreMovies

    #print clusterMovieRating
    #print "*********************\n"
    #print clusterMovieGenre


"""Loading the features to a numpy array for GMM Training"""

char_net_final_1_npArray = np.array(char_net_final_1.loc[:,[1,2,3,4,5,6,7,8]]) # Original Dataframe minus movie names
char_net_final_2_npArray = np.array(char_net_final_2.loc[:,[1,2,3,4,5,6,7,8]]) # Original Dataframe minus movie names
topicOverlap_npArray = np.array(topicOverlap.loc[:,[1,2,3,4,5,6,7,8,9,10]]) # Original Dataframe minus movie names
emotionAnalysis_npArray = np.array(emotionAnalysis.loc[:,[1,2,3,4,5,6,7,8]]) # Original Dataframe minus movie names
doc2vec_features_npArray = np.array(doc2vec_features.loc[:,['0_x','1_x','2_x','3_x','4_x','0_y','1_y','2_y','3_y','4_y']]) # Original Dataframe minus movie names

n_components = 5 # Number of Cluster

# gmm = mixture.GaussianMixture(n_components, covariance_type='full',max_iter=25000,n_init=100).fit(X)
# plot_results(X[:,2:4], gmm.predict(X), gmm.means_, gmm.covariances_, 0,
#              'Gaussian Mixture')
# print gmm.predict(X)


""" Fit a Dirichlet process Gaussian mixture using n components """

"""Fitting Shankar's char_net_final_1"""
"""
dpgmm = mixture.BayesianGaussianMixture(n_components,
                                        covariance_type='full',max_iter=25000,n_init=500).fit(char_net_final_1_npArray)

plot_results(char_net_final_1_npArray[:,2:4], dpgmm.predict(char_net_final_1_npArray), dpgmm.means_, dpgmm.covariances_, 1,
             'Bayesian GMM for char_net_final_1',0,1,0,0.1) # Visualizing feature 3 and 4

scores_char_1 = dpgmm.predict(char_net_final_1_npArray)
# print dgmm.predict(char_net_final_1_npArray)
print 'GMM Shankar done'
"""

"""Fitting Shankar's char_net_final_2"""

dpgmm_1 = mixture.BayesianGaussianMixture(n_components,
                                        covariance_type='full',max_iter=25000,n_init=500).fit(char_net_final_2_npArray)

plot_results(char_net_final_2_npArray[:,2:4], dpgmm_1.predict(char_net_final_1_npArray), dpgmm_1.means_, dpgmm_1.covariances_, 1,
             'Bayesian GMM for char_net_final_2',0,1,0,0.1) # Visualizing feature 3 and 4

scores_char_2 = dpgmm_1.predict(char_net_final_2_npArray)
# print dgmm.predict(char_net_final_2_npArray)
print 'GMM Shankar done'

"""Fitting Shankar's topic overlap"""

dpgmm_2 = mixture.BayesianGaussianMixture(n_components,
                                        covariance_type='full',max_iter=25000,n_init=500).fit(topicOverlap_npArray)

plot_results(topicOverlap_npArray[:,2:4], dpgmm_2.predict(topicOverlap_npArray), dpgmm_2.means_, dpgmm_2.covariances_, 1,
             'Bayesian GMM for topic overlap',0,1,0,0.1) # Visualizing feature 3 and 4

scores_topic = dpgmm_2.predict(topicOverlap_npArray)

# print dgmm.predict(topicOverlap_npArray)
print 'GMM Shankar2 done'

"""Fitting Jar's emotion analysis"""


dpgmm = mixture.BayesianGaussianMixture(n_components,
                                        covariance_type='full',max_iter=25000,n_init=500).fit(emotionAnalysis_npArray)

plot_results(emotionAnalysis_npArray[:,2:4], dpgmm.predict(emotionAnalysis_npArray), dpgmm.means_, dpgmm.covariances_, 1,
             'Bayesian GMM for emotion analysis',0,1,0,0.1) # Visualizing feature 3 and 4

scores_emoAnalysis = dpgmm.predict(emotionAnalysis_npArray)
# print dgmm.predict(emotionAnalysis_npArray)
print 'GMM Jar done'

"""Fitting Naman's doc2vec features"""

dpgmm_3 = mixture.BayesianGaussianMixture(n_components,
                                        covariance_type='full',max_iter=25000,n_init=500).fit(doc2vec_features_npArray)

plot_results(doc2vec_features_npArray[:,2:4], dpgmm_3.predict(doc2vec_features_npArray), dpgmm_3.means_, dpgmm_3.covariances_, 1,
             'Bayesian GMM for doc2vec',0,1,0,0.1) # Visualizing feature 3 and 4

scores_doc2vec = dpgmm_3.predict(doc2vec_features_npArray)
# print dgmm.predict(doc2vec_features_npArray)
print 'GMM Naman done'

classes = np.arange(0,n_components,1)

print "Char Network"
for i in classes:
    print "Cluster: %d" % i
    cluster_analysis(i,char_net_final_2,scores_char_2)

print "Topic Overlap"    
for i in classes:
    print "Cluster: %d" % i
    cluster_analysis(i,topicOverlap,scores_topic)
    
print "Emotion Analysis"
for i in classes:
    print "Cluster: %d" % i
    cluster_analysis(i,emotionAnalysis,scores_emoAnalysis)
    
print "Doc2Vec"
for i in classes:
    print "Cluster: %d" % i
    cluster_analysis(i,doc2vec_features,scores_doc2vec)
    # cluster_analysis(0,score_count,data)
