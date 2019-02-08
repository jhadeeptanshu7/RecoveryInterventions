from optparse import OptionParser

from Classification import Project
from sklearn.feature_extraction.text import CountVectorizer
import os
import pymongo
import propensity_score_matching
import train_classifier_classification
from Classification import VISUALIZATION_FOLDER
from sklearn.externals import joblib


def creating_user_post_and_recovery_matrix(project):
    client = pymongo.MongoClient()
    db = client.Recovery
    collection = db['drug_users']
    cursor = collection.find({"project_id":project.id}, no_cursor_timeout=True)

    recovery=[]
    all_redditors=[]
    user_posts_cumulative = []


    for i in cursor:
        redditor = i["user"]
        all_redditors.append(redditor)
        user_recovery = i['recovery']
        posts = i['posts']
        single_user_posts = " "
        for p in posts:
            single_user_posts = single_user_posts + ' ' + p

        user_posts_cumulative.append(single_user_posts)
        recovery.append(user_recovery)

    stopwords_vect = CountVectorizer(min_df=5, stop_words="english",ngram_range=(1,2)).fit(user_posts_cumulative)
    X_stopwords = stopwords_vect.transform(user_posts_cumulative)
    print X_stopwords.shape
    print "vocabulary length ", len(stopwords_vect.vocabulary_)
    return [all_redditors, stopwords_vect, X_stopwords, recovery]


def insert_data_mongodb(folder, project):

#     client = pymongo.MongoClient()
#     db = client.Recovery

    db = pymongo.MongoClient("mongodb://recovery:interventions@localhost:27017/recoveryi?authMechanism=SCRAM-SHA-256").recoveryi
    collection = db['drug_users']

    for dir in os.listdir(folder):
        if dir[0] == ".":
            continue

        for sub_dir in os.listdir(folder+"/"+dir):
            if sub_dir[0]==".":
                continue
            user_dic = {}
            user_dic["project_id"] = project.id
            user_dic["user"] = sub_dir
            sub_folder_path = folder+"/"+dir+"/"+sub_dir
            posts = []
            for text_file in os.listdir(sub_folder_path):
                # print text_file
                text_file_path = sub_folder_path + "/" + text_file
                if text_file.endswith(".txt"):
                    # print text_file
                    file_object = open(text_file_path, "r")
                    post_content = file_object.read()
                    posts.append(post_content)
            user_dic["posts"] = posts
            if dir =='recovery':
                user_dic['recovery']=1
            if dir =='non_recovery':
                user_dic['recovery']=0
            collection.insert_one(user_dic)

def sorting_terms_by_ate(project):
    project_id = project.id
    client = pymongo.MongoClient()
    db = client.Recovery
    collection = db["psm_terms"]
    cursor = collection.find({"project_id":project_id}, no_cursor_timeout=True)

    ate_increased_rates_of_transfer = {}
    ate_decreased_rates_of_transfer = {}

    for i in cursor:
        term = i["term"]
        ate =  i["ate"]
        zscore = i["zscore"]

        if str(zscore) == 'inf':
            continue
        if str(zscore) =='-inf':
            continue
        if ate >0:
            ate_increased_rates_of_transfer[term] = ate
        if ate <0:
            ate_decreased_rates_of_transfer[term] = ate
    return [ate_increased_rates_of_transfer,ate_decreased_rates_of_transfer]


def create_visualization_folders(project):
    folder_path_1 = os.path.dirname(__file__) + '/visualizations/'  + project.id + '/system_level'
    folder_path_2 = os.path.dirname(__file__) + '/visualizations/'  + project.id + '/user_level'
    if not os.path.isdir(folder_path_1):
        os.makedirs(folder_path_1)

    if not os.path.isdir(folder_path_2):
        os.makedirs(folder_path_2)
    client = pymongo.MongoClient()
    db = client.Recovery
    collection = db['drug_users']
    cursor = collection.find({"project_id":project.id},no_cursor_timeout=True)
    for i in cursor:
        user = i["user"]
        recovery =i["recovery"]
        if recovery==1:
            folder_path = os.path.dirname(__file__) + '/visualizations/' + project.id + '/user_level/recovery_users/' + user
            if not os.path.isdir(folder_path):
                os.makedirs(folder_path)
        if recovery==0:
            folder_path = os.path.dirname(__file__) + '/visualizations/' + project.id + '/user_level/non_recovery_users/' + user
            if not os.path.isdir(folder_path):
                os.makedirs(folder_path)


def create_output_files(output_dict, output_folder):
    if not os.path.exists(output_folder):
        os.mkdir(output_folder)

    for key in output_dict.keys():
        joblib.dump(output_dict[key], os.path.join(output_folder, key + ".pkl"))


def run(folder, project):
    # print folder
    # print project.id
    # print os.listdir(folder)
    insert_data_mongodb(folder,project)
    psa_features = creating_user_post_and_recovery_matrix(project)
    #psa_features[1] = stopwords_vect, psa_features[2] = X_stopwords, psa_features[3] = recovery
    propensity_score_matching.run_psm(psa_features[1],psa_features[2],psa_features[3],project)
    increased_decreased_ate = sorting_terms_by_ate(project)
    output_dict = train_classifier_classification.run_classification(project,increased_decreased_ate,psa_features)

    output_folder = os.path.join(VISUALIZATION_FOLDER, project.id)
    create_output_files(output_dict, output_folder)

    # create_visualization_folders(project)
    # system_level_visualization(project)
    # user_level_visualization(project)


def main():
    parser = OptionParser()
    parser.add_option('-f', '--folder', dest='folder', help="Folder name", type=str)
    parser.add_option('-p', '--project', dest='project', help="Project id", type=str)
    parser.add_option('-t', '--job_type', dest='job_type', help="Job type", type=str)

    (options, args) = parser.parse_args()
    print options.folder
    print options.project
    run(options.folder, Project(options.project, options.folder, options.job_type, ""))


# Project(options.project, options.folder, options.job_type, options.user_email)

#
# def main():
#     folder = "train_uploads/"
#     project = Project("5bfa2995473c8923db51e0b2", folder,"TRAIN","NA")
#     run(folder, project)


if __name__ == "__main__":
    main()
