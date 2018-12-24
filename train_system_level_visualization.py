from Classification import Project
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



VISUALIZATION_FOLDER = os.path.dirname(__file__) + '/visualizations/'

#
# def recovery_non_recovery_donut(data, project):
#     fig, ax = plt.subplots(figsize=(6, 3), subplot_kw=dict(aspect="equal"))
#     data = data
#     #names of the category
#     ingredients = ["Individuals open\n to recovery", "Individuals not \nopen to recovery"]
#     colors = ['yellow','red']
#     def func(pct, allvals):
#         absolute = int(pct/100.*np.sum(allvals))
#         return "{:.1f}%\n({:d})".format(pct, absolute)
#
#     wedges, texts, autotexts = ax.pie(data, autopct=lambda pct:func(pct, data),colors=colors,
#                                       textprops=dict(color="black"),pctdistance=0.8)
#     ax.legend(wedges, ingredients,
#               loc="center left",
#               bbox_to_anchor=(1, 0, 0.5, 1))
#
#     plt.setp(autotexts, size=8, weight="bold")
#     #draw circle
#     centre_circle = plt.Circle((0,0),0.60,fc='white')
#     fig = plt.gcf()
#     fig.gca().add_artist(centre_circle)
#     # ax.set_title("Matplotlib bakery: A pie")
#     # plt.show()
#     fig.savefig(VISUALIZATION_FOLDER + project.id + '/system_level/donut.png')



def recovery_lda_and_word_cloud(project):
    client = pymongo.MongoClient()
    db = client.Recovery
    collection = db['drug_users']
    cursor = collection.find({"project_id": project.id, "recovery":1}, no_cursor_timeout=True)
    user_posts_cumulative =[]
    for c,i in enumerate(cursor):
        user = i["user"]
        print c, user
        posts = i["posts"]
        for p in posts:
            no_url_post = re.sub(r'http\S+', '', p)
            user_posts_cumulative.append(no_url_post)
    client.close()
    vect = CountVectorizer(stop_words=stopwords.stopwords)
    X = vect.fit_transform(user_posts_cumulative)
    lda = LatentDirichletAllocation(n_topics=10, random_state=0,learning_method='online')
    document_topics = lda.fit_transform(X)
    sorting = np.argsort(lda.components_,axis =1)[:,::-1]
    feature_names = np.array(vect.get_feature_names())
    # mglearn.tools.print_topics(topics=range(10),feature_names=feature_names,sorting=sorting,topics_per_chunk=5,n_words=10)
    recovery_topics = pyLDAvis.sklearn.prepare(lda,X,vect,mds='mmds')
    pyLDAvis.save_html(recovery_topics, VISUALIZATION_FOLDER + project.id + '/system_level/recovery_topics.html')


    #wordcloud
    text = (' '.join(user_posts_cumulative)).lower()
    wordcloud = WordCloud(max_words=50).generate(text)
    plt.imshow(wordcloud, interpolation='bilinear')
    plt.axis("off")
    # plt.show()
    wordcloud.to_file(VISUALIZATION_FOLDER + project.id + '/system_level/recovery_word_cloud.png')
    plt.close()




def non_recovery_lda_and_word_cloud(project):
    client = pymongo.MongoClient()
    db = client.Recovery
    collection = db['drug_users']
    cursor = collection.find({"project_id": project.id, "recovery":0}, no_cursor_timeout=True)
    user_posts_cumulative =[]
    for c,i in enumerate(cursor):
        user = i["user"]
        print c, user
        posts = i["posts"]
        for p in posts:
            no_url_post = re.sub(r'http\S+', '', p)
            user_posts_cumulative.append(no_url_post)
    client.close()
    vect = CountVectorizer(stop_words=stopwords.stopwords)
    X = vect.fit_transform(user_posts_cumulative)
    lda = LatentDirichletAllocation(n_topics=10,random_state=0,learning_method='online')
    document_topics = lda.fit_transform(X)
    sorting = np.argsort(lda.components_,axis =1)[:,::-1]
    feature_names = np.array(vect.get_feature_names())

    # mglearn.tools.print_topics(topics=range(10),feature_names=feature_names,sorting=sorting,topics_per_chunk=5,n_words=10)
    non_recovery_topics = pyLDAvis.sklearn.prepare(lda,X,vect,mds='mmds')
    pyLDAvis.save_html(non_recovery_topics, VISUALIZATION_FOLDER + project.id + '/system_level/non_recovery_topics.html')

    #wordcloud
    text = (' '.join(user_posts_cumulative)).lower()
    wordcloud = WordCloud(max_words=50).generate(text)
    plt.imshow(wordcloud, interpolation='bilinear')
    plt.axis("off")
    # plt.show()
    wordcloud.to_file(VISUALIZATION_FOLDER + project.id + '/system_level/non_recovery_word_cloud.png')
    plt.close()

# def main():
#     folder = "/Users/jhadeeptanshu/RecoveryInterventions/train_uploads/"
#     project = Project("5bfa2995473c8923db51e0b2", folder, "5bf8c2da473c89cfb14d63d2")
#     recovery_lda_and_word_cloud(project)
#     non_recovery_lda_and_word_cloud(project)
#
#
# if __name__ == "__main__":
#     main()