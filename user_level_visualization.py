import numpy as np
import matplotlib.pyplot as plt
import pymongo
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.decomposition import LatentDirichletAllocation
import stopwords
import re
import pyLDAvis
import pyLDAvis.sklearn
from wordcloud import WordCloud
import os
from sklearn.externals import joblib
import squarify


VISUALIZATION_FOLDER = os.path.dirname(__file__) + '/visualizations/'


def user_lda_and_word_cloud(user_name,posts, recovery, project):
    user_posts_cumulative =[]
    for p in posts:
        no_url_post = re.sub(r'http\S+', '', p)
        user_posts_cumulative.append(no_url_post)
    vect = CountVectorizer(stop_words=stopwords.stopwords)
    X = vect.fit_transform(user_posts_cumulative)
    lda = LatentDirichletAllocation(n_topics=10, random_state=0,learning_method='online')
    document_topics = lda.fit_transform(X)
    sorting = np.argsort(lda.components_,axis =1)[:,::-1]
    feature_names = np.array(vect.get_feature_names())
    # mglearn.tools.print_topics(topics=range(10),feature_names=feature_names,sorting=sorting,topics_per_chunk=5,n_words=10)
    recovery_topics = pyLDAvis.sklearn.prepare(lda,X,vect,mds='mmds')

     #wordcloud
    text = (' '.join(user_posts_cumulative)).lower()
    wordcloud = WordCloud(max_words=50).generate(text)
    plt.imshow(wordcloud, interpolation='bilinear')
    plt.axis("off")


    if recovery==1:
        pyLDAvis.save_html(recovery_topics, VISUALIZATION_FOLDER + project.id + '/user_level/recovery_users/' + user_name + '/document_topics.html')
        wordcloud.to_file(VISUALIZATION_FOLDER + project.id + '/user_level/recovery_users/' + user_name + '/word_cloud.png')
    else:
        pyLDAvis.save_html(recovery_topics, VISUALIZATION_FOLDER + project.id + '/user_level/non_recovery_users/' + user_name + '/document_topics.html')
        wordcloud.to_file(VISUALIZATION_FOLDER + project.id + '/user_level/non_recovery_users/' + user_name + '/word_cloud.png')


def drug_tree_map(user_name,posts,recovery, project):
    regex = joblib.load("drug_regex.pkl")
    drug_count_dic = {}
    for p in posts:
        text = re.sub('[^0-9a-zA-Z]+', ' ', p)
        found_drugs = regex.findall(text)
        for fd in (found_drugs):
            strip_fd = fd.strip().lower()
            if strip_fd in drug_count_dic:
                drug_count_dic[strip_fd] +=1
            else:
                drug_count_dic[strip_fd] =1
    drug_counts = []
    drug_terms = []
    # print len(drug_count_dic)
    for d in drug_count_dic:
        drug_terms.append(d +" ("+ str(drug_count_dic[d]) +")")
        drug_counts.append(drug_count_dic[d])
    fig = plt.figure()
    squarify.plot(sizes=drug_counts, label=drug_terms, alpha=.7 )
    plt.axis('off')
    # plt.show()
    if recovery==1:
        plt.savefig(VISUALIZATION_FOLDER + project.id + '/user_level/recovery_users/' + user_name + '/drug_treemap.png')
    else:
        plt.savefig(VISUALIZATION_FOLDER + project.id + '/user_level/non_recovery_users/' + user_name + '/drug_treemap.png')
    # plt.savefig('treemap')


def recovery_tree_map(user_name,posts,recovery, project):
    regex = joblib.load("recovery_regex.pkl")
    recovery_count_dic = {}
    for p in posts:
        text = re.sub('[^0-9a-zA-Z]+', ' ', p)
        found_recovery_terms = regex.findall(text)
        for fr in (found_recovery_terms):
            strip_fd = fr.strip().lower()
            if strip_fd in recovery_count_dic:
                recovery_count_dic[strip_fd] +=1
            else:
                recovery_count_dic[strip_fd] =1
    drug_counts = []
    drug_terms = []
    # print len(drug_count_dic)
    for d in recovery_count_dic:
        drug_terms.append(d +" ("+ str(recovery_count_dic[d]) +")")
        drug_counts.append(recovery_count_dic[d])
    fig = plt.figure()
    squarify.plot(sizes=drug_counts, label=drug_terms, alpha=.7 )
    plt.axis('off')
    # plt.show()
    if recovery==1:
        plt.savefig(VISUALIZATION_FOLDER + project.id + '/user_level/recovery_users/' + user_name + '/recovery_treemap.png')
    else:
        plt.savefig(VISUALIZATION_FOLDER + project.id + '/user_level/non_recovery_users/' + user_name + '/recovery_treemap.png')
    # plt.savefig('treemap')

def positive_ate_treemap(user_name,posts,recovery, project):
    regex = joblib.load("positive_psm_terms.pkl")
    recovery_count_dic = {}
    for p in posts:
        text = re.sub('[^0-9a-zA-Z]+', ' ', p)
        found_recovery_terms = regex.findall(text)
        for fr in (found_recovery_terms):
            strip_fd = fr.strip().lower()
            if strip_fd in recovery_count_dic:
                recovery_count_dic[strip_fd] +=1
            else:
                recovery_count_dic[strip_fd] =1
    drug_counts = []
    drug_terms = []
    # print len(drug_count_dic)
    for d in recovery_count_dic:
        drug_terms.append(d +" ("+ str(recovery_count_dic[d]) +")")
        drug_counts.append(recovery_count_dic[d])
    fig = plt.figure()
    squarify.plot(sizes=drug_counts, label=drug_terms, alpha=.7 )
    plt.axis('off')
    # plt.show()
    if recovery==1:
        plt.savefig(VISUALIZATION_FOLDER + project.id + '/user_level/recovery_users/' + user_name + '/postive_ate_treemap.png')
    else:
        plt.savefig(VISUALIZATION_FOLDER + project.id + '/user_level/non_recovery_users/' + user_name + '/postive_ate_treemap.png')

def negative_ate_treemap(user_name,posts,recovery, project):
    regex = joblib.load("negative_psm_terms.pkl")
    recovery_count_dic = {}
    for p in posts:
        text = re.sub('[^0-9a-zA-Z]+', ' ', p)
        found_recovery_terms = regex.findall(text)
        for fr in (found_recovery_terms):
            strip_fd = fr.strip().lower()
            if strip_fd in recovery_count_dic:
                recovery_count_dic[strip_fd] +=1
            else:
                recovery_count_dic[strip_fd] =1
    drug_counts = []
    drug_terms = []
    # print len(drug_count_dic)
    for d in recovery_count_dic:
        drug_terms.append(d +" ("+ str(recovery_count_dic[d]) +")")
        drug_counts.append(recovery_count_dic[d])
    fig = plt.figure()
    squarify.plot(sizes=drug_counts, label=drug_terms, alpha=.7 )
    plt.axis('off')
    # plt.show()
    if recovery==1:
        plt.savefig(VISUALIZATION_FOLDER + project.id + '/user_level/recovery_users/' + user_name + '/negative_ate_treemap.png')
    else:
        plt.savefig(VISUALIZATION_FOLDER + project.id + '/user_level/non_recovery_users/' + user_name + '/negative_ate_treemap.png')


def user_visualization(project):
    client = pymongo.MongoClient()
    db = client.Recovery
    collection = db['drug_users']
    cursor = collection.find({"project_id": project.id}, no_cursor_timeout=True)
    for c,i in enumerate(cursor):
        user = i["user"]
        print c, user
        posts = i["posts"]
        recovery = i["recovery"]
        if recovery:
            path = VISUALIZATION_FOLDER + project.id + '/user_level/recovery_users/' + user
            os.makedirs(path)
        else:
            path = VISUALIZATION_FOLDER + project.id + '/user_level/non_recovery_users/' + user
            os.makedirs(path)
        user_lda_and_word_cloud(user,posts,recovery, project)
        drug_tree_map(user,posts,recovery, project)
        recovery_tree_map(user,posts,recovery, project)
        negative_ate_treemap(user,posts,recovery, project)
        positive_ate_treemap(user,posts,recovery, project)







