import reddit_forum_categories
recovery_subreddits = reddit_forum_categories.recovery_subreddits
drug_subreddits = reddit_forum_categories.drug_subreddits
import pymongo
import nltk.stem
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from statsmodels.stats.weightstats import ztest
from sklearn.model_selection import train_test_split
english_stemmer = nltk.stem.PorterStemmer()
import multiprocessing
import time
from Classification import get_db


def Match(groups, propensity, caliper):
    '''
    Inputs:
    groups = Treatment assignments.  Must be 2 groups
    propensity = Propensity scores for each observation. Propensity and groups should be in the same order (matching indices)
    caliper = Maximum difference in matched propensity scores. For now, this is a caliper on the raw
            propensity; Austin reccommends using a caliper on the logit propensity.

    Output:
    A series containing the individuals in the control group matched to the treatment group.
    Note that with caliper matching, not every treated individual may have a match.
    '''

    # Check inputs
#     print "Match function" , propensity
    if any(propensity < 0) or any(propensity >1):
        raise ValueError('Propensity scores must be between 0 and 1')
    elif not(0<caliper<1):
        raise ValueError('Caliper must be between 0 and 1')
    elif len(groups)!= len(propensity):
        raise ValueError('groups and propensity scores must be same dimension')
    elif len(groups.unique()) != 2:
        raise ValueError('wrong number of groups')


    # Code groups as 0 and 1
    groups = groups == groups.unique()[0]
    N = len(groups)
    N1 = groups.sum(); N2 = N-N1
    g1, g2 = propensity[groups == 1], (propensity[groups == 0])
    # Check if treatment groups got flipped - treatment (coded 1) should be the smaller
    if N1 > N2:
       N1, N2, g1, g2 = N2, N1, g2, g1


    # Randomly permute the smaller group to get order for matching
    morder = np.random.permutation(g1.index)
    matches = pd.Series(np.empty(N1))
    matches.index = morder
    matches[:] = np.NAN
    for m in morder:
        dist = abs(g1[m] - g2)
        if dist.min() <= caliper:
            matches[m] = dist.argmin()
    return (matches)


def mp_psm(params, key):
    X_stopwords    = params["X_stopwords"]
    stopwords_vect = params["stopwords_vect"]
    recovery_forum = params["recovery_forum"]
    project_id = params["project_id"]
    v = key
    term = v
    position = stopwords_vect.vocabulary_[v]
    
    db = get_db()
    collection = db.psm_terms

    try:
        X = X_stopwords[:,[i for i in range(len(stopwords_vect.vocabulary_)) if i != position]]
        Y = X_stopwords[:,position]
        Y = Y.toarray() > 0
        Y= np.ravel(Y)
        # grid search
        X_train, X_test, Y_train, Y_test = train_test_split(X,Y,random_state=0)
        lr = LogisticRegression(C = 0.0001)
        lr.fit(X_train, Y_train)
    #     print "log predict"
    #     lr.fit(X,np.ravel(Y))

        propensity_score = lr.predict_proba(X)
    #     print propensity_score
        propensity_score = propensity_score[:,1]
    #     print propensity_score
        df = pd.DataFrame({'treatment': Y, 'effect': np.array(recovery_forum), 'propensity_scores': propensity_score })
        df0 = df.loc[df['treatment']==False,:]
        df1 = df.loc[df['treatment']==True,:]
        caliper = 0.5
    #     print "propensity", df.propensity_scores
        stuff = Match(df.treatment, df.propensity_scores, caliper)
    #     print np.isnan(stuff).sum()
        #check for nans
        while np.isnan(stuff).sum()!=0:
            if caliper >=1:
                break
    #         print np.isnan(stuff).sum()
    #         print caliper
            caliper = caliper + 0.1
            stuff = Match(df.treatment, df.propensity_scores, caliper)

        if caliper >=1:
            return
        # ate = df0.propensity_scores.mean() - df1.propensity_scores.mean()
        treated = df.iloc[stuff.index,:]
        control = df.iloc[stuff.values.astype(np.int64),:]
        ate = treated.effect.mean() - control.effect.mean()

        ztest_values = ztest(treated.effect, control.effect, value=0, alternative='two-sided', usevar='pooled', ddof=1.0)

        # print "ate ", ate
        # print "zscore ", ztest_values[0]
        if str(ztest_values[0]) =='inf' or str(ztest_values[0])=='-inf':
            return
        term_dic={}
        term_dic["term"]=term
        term_dic["zscore"] =ztest_values[0]
        term_dic["ate"] = ate
        term_dic["project_id"] = project_id
        # print term_dic
        collection.insert_one(term_dic)
    except:
        # print "less than two classes",term
        return


def run_psm(stopwords_vect, X_stopwords, recovery_forum,project):
    recovery_forum = recovery_forum
    stopwords_vect = stopwords_vect
    X_stopwords = X_stopwords
    project_id = project.id
    keys = stopwords_vect.vocabulary_.keys()
    from contextlib import closing

    from functools import partial
    params = dict()
    params["recovery_forum"] = recovery_forum
    params["stopwords_vect"] = stopwords_vect
    params["X_stopwords"]    = X_stopwords
    params["project_id"]     = project_id

    mp_psm_wrapper = partial(mp_psm, params)

    start_time = time.time()
    with closing(multiprocessing.Pool(processes=1)) as pool:
        pool.map(mp_psm_wrapper, keys)
        pool.terminate()
    end_time = time.time()
