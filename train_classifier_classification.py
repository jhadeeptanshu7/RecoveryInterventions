import reddit_forum_categories
drug_subreddits = reddit_forum_categories.drug_subreddits
recovery_subreddits = reddit_forum_categories.recovery_subreddits
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import LinearSVC
from mlxtend.classifier import StackingClassifier
from sklearn import metrics
from sklearn.metrics import accuracy_score,confusion_matrix, classification_report
from scipy import stats
from sklearn.metrics import f1_score
import statistics as s
import operator


def harmonic_mean(auc,f1):
    numerator = 2 * auc * float(f1)
    denominator = auc + float(f1)
    hm = float(numerator)/float(denominator)
    return hm

# stopwords_vect
# posts_psa_dic
# decreased_posts_psa_dic
#n
#sclf

def best_classification_features(ate_scores,psa_features):
    all_redditors= psa_features[0] #name of the redditors
    stopwords_vect = psa_features[1]
    X_stopwords = psa_features[2]
    recovery_forum= psa_features[3]

    print X_stopwords.shape
    print "vocabulary length ", len(stopwords_vect.vocabulary_)

    bow_posts = X_stopwords
    print bow_posts.shape
    post_vect = stopwords_vect

    posts_psa_dic = ate_scores[0]
    print len(posts_psa_dic)
    sorted_posts_dic = sorted(posts_psa_dic, key=posts_psa_dic.get, reverse=True)


    decreased_posts_psa_dic = ate_scores[1]
    print len(decreased_posts_psa_dic)
    decreased_sorted_posts_dic = sorted(decreased_posts_psa_dic, key=decreased_posts_psa_dic.get)

    # top_n = [100,1000,2000,3000,4000,5000,6000,7000,8000,9000,10000,110000,12000,13000,14000,15000]
    top_n = list(range(50,max(len(posts_psa_dic),len(decreased_posts_psa_dic))+50,50))
    print top_n
    posts_matrix = bow_posts.toarray()
    max_score_dic = {}
    max_score_classifier  = {}
    max_score = 0
    for n in top_n:
        print n
        post_terms = sorted_posts_dic[0:n]
        decreased_post_terms = decreased_sorted_posts_dic[0:n]


        post_term_indices ={}
        decreased_post_term_indices ={}


        for pt in post_terms:
            post_term_indices[post_vect.vocabulary_[pt]] = posts_psa_dic[pt]

        for dpt in decreased_post_terms:
            decreased_post_term_indices[post_vect.vocabulary_[dpt]] = decreased_posts_psa_dic[dpt]

        matrix_only_posts = []

        for c,row in enumerate(posts_matrix):
            new_row_posts = []

            for i in post_term_indices:
                new_row_posts.append(row[i])


            for k in decreased_post_term_indices:
                new_row_posts.append(row[k])

            new_row_posts.append(recovery_forum[c])
            matrix_only_posts.append(new_row_posts)

        matrix = np.array(matrix_only_posts)
        X = matrix[:,:-1]

        y = matrix[:,-1]

        X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.2, stratify=y, random_state=0)
        clf1 = RandomForestClassifier(random_state=0)
        clf2 = LogisticRegression()
        clf3 = LinearSVC()
        sclf = StackingClassifier(classifiers=[clf1, clf2, clf3],
                              meta_classifier=clf1)
        print "fitting data"
        sclf.fit(X_train,y_train)
        y_pred = sclf.predict(X_test)
        print confusion_matrix(y_test,y_pred)
        print classification_report(y_test,y_pred)
        print accuracy_score(y_test, y_pred)
        y_pred_proba = sclf.predict_proba(X_test)[::,1]
        fpr, tpr, _ = metrics.roc_curve(y_test,  y_pred_proba)
        auc = metrics.roc_auc_score(y_test, y_pred_proba)
        print auc
        f1 = "%.2f"%f1_score(y_test, y_pred, average='macro')
        print f1
        score = harmonic_mean(auc,f1)
        if score >= max_score:
            max_score_dic[n] = score
            max_score = score
            max_score_classifier[n] = sclf
        print "****************************"
    print max_score_dic
    n = max(max_score_dic.iteritems(), key=operator.itemgetter(1))[0]
    print max_score_classifier[n]
    pickle_dic = {}
    pickle_dic['n'] = n
    pickle_dic['classifier'] = max_score_classifier[n]
    pickle_dic['positive_ate'] = posts_psa_dic
    pickle_dic['negative_ate'] = decreased_posts_psa_dic
    pickle_dic['post_vectorizer'] = stopwords_vect
    return pickle_dic




def run_classification(project, ate_scores, psa_features):
    project_id = project.id
    ate_scores = ate_scores
    psa_features = psa_features
    pickle_dic = best_classification_features(ate_scores,psa_features)
    return pickle_dic