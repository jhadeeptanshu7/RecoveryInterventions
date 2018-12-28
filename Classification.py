import datetime
import logging
import zipfile
import os

import sys
from sklearn.externals import joblib
import pymongo
import re
import numpy as np
import visualizations
import user_level_visualization
import pymongo
from bson.objectid import ObjectId
from optparse import OptionParser
from Helpers import send_email, BODY


VISUALIZATION_FOLDER = os.path.dirname(__file__) + '/visualizations/'
logging.basicConfig(filename='classification.log', level=logging.DEBUG)
OUTPUT_FOLDER = os.path.dirname(__file__) + '/output'
UPLOAD_FOLDER = os.path.dirname(__file__) + '/run_uploads/'

sys.path.append(os.path.dirname(__file__) + "/sentiment_analysis/")

client = pymongo.MongoClient()
db = client.Recovery


class Project:
    def __init__(self, project_id, filename, job_type):
        self.id = str(project_id)
        self.filename = filename
        self.job_type = job_type


def fileHandler(project_id):

    project = db.projects.find_one({'_id': ObjectId(project_id)})

    if not project:
        return

    project = Project(project['_id'], project['file'], project['job_type'])
    file = project.filename

    logging.info(str(datetime.datetime.now()) + ': ' + file)
    zip_ref = zipfile.ZipFile(UPLOAD_FOLDER + file, 'r')
    extracted = zip_ref.namelist()
    # print extracted[0]
    zip_ref.extractall(OUTPUT_FOLDER)
    zip_ref.close()

    if project.job_type == "CLASSIFY":
        print "CLASSIFY"
        os.system("python /Users/jhadeeptanshu/RecoveryInterventions/Classification.py -f " + str(extracted[0]) + " -p " + str(project.id) + " -t " + project.job_type)
    elif project.job_type == "TRAIN":
        print "TRAIN"
        os.system("python /Users/jhadeeptanshu/RecoveryInterventions/TrainClassifier.py -f " + str(extracted[0]) + " -p " + str(project.id) + " -t " + project.job_type)

    db['projects'].find_one_and_update({"project_id": project_id},
                                 {"$set": {"job_status": "1"}})
    # user = db.users.find_one({'_id': ObjectId(project.user_email)})

    # send_email(project.user_email, BODY % ("User", project.id))


def unzip_folder(input_file):
    zip_ref = zipfile.ZipFile(input_file, 'r')
    extracted = zip_ref.namelist()
    zip_ref.extractall(OUTPUT_FOLDER)
    zip_ref.close()

    return extracted[0]



def insert_data_mongodb(folder, project):
    # print folder.split("/")

    collection = db['drug_users']

    for sub_folder in os.listdir(folder):
        if sub_folder[0]==".":
            continue
        user_dic = {}
        user_dic["project_id"] = project.id
        user_dic["user"] = sub_folder
        print sub_folder
        sub_folder_path = folder+"/"+sub_folder
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
        collection.insert_one(user_dic)


def classification(project):
    # client = pymongo.MongoClient()
    # db = client.Recovery
    collection = db['drug_users']
    cursor = collection.find({"project_id": project.id},no_cursor_timeout=True)
    all_redditors=[] #name of the redditors
    # each element in the list has all the posts of a particular user. upc[0] = all posts of user0, upc[1] = all posts of user1
    user_posts_cumulative =[]
    for c,i in enumerate(cursor):
        user = i["user"]
        print c, user
        all_redditors.append(user)
        posts = i["posts"]
        single_user_post = ""
        for p in posts:
            no_url_post = re.sub(r'http\S+', '', p)
            single_user_post = single_user_post+ " " + no_url_post
        user_posts_cumulative.append(single_user_post)

    #post vectorizer  pickle
    post_vect = joblib.load('min_df_4_posts_vect.pkl')
    bow_posts = post_vect.transform(user_posts_cumulative)
    posts_matrix = bow_posts.toarray()
    print bow_posts.shape

    # +ve ate score pickle
    posts_psa_dic = joblib.load('ate_posts_increased_rates_of_transfer.pkl')
    sorted_posts_dic = sorted(posts_psa_dic, key=posts_psa_dic.get, reverse=True)

    # -ve ate score pickle
    decreased_posts_psa_dic = joblib.load('ate_posts_decreased_rates_of_transfer.pkl')
    decreased_sorted_posts_dic = sorted(decreased_posts_psa_dic, key=decreased_posts_psa_dic.get)

    n=4000
    post_terms = sorted_posts_dic[0:n]
    decreased_post_terms = decreased_sorted_posts_dic[0:n]
    post_term_indices ={}
    decreased_post_term_indices ={}
    for pt in post_terms:
        post_term_indices[post_vect.vocabulary_[pt]] = posts_psa_dic[pt]

    for dpt in decreased_post_terms:
        decreased_post_term_indices[post_vect.vocabulary_[dpt]] = decreased_posts_psa_dic[dpt]

    matrix = []
    for c,row in enumerate(posts_matrix):
        new_row=[]
        for i in post_term_indices:
            new_row.append(row[i])

        for k in decreased_post_term_indices:
            new_row.append(row[k])

        matrix.append(new_row)

    matrix = np.array(matrix)

    #classifier pickle
    sclf = joblib.load('sclf.pkl')

    y_pred = sclf.predict(matrix)
    # print y_pred
    for c,user in enumerate(all_redditors):
        user_dic = collection.find_one({"user":user, "project_id": project.id})
        user_dic["recovery"] = y_pred[c]
        collection.save(user_dic)
    client.close()
    return [all_redditors,y_pred]


def run_batch_classification(folder, project):
    # if not os.path.exists(VISUALIZATION_FOLDER + "/" + project.id + "/system_level"):
    #     os.makedirs(VISUALIZATION_FOLDER + "/" + project.id + "/system_level")
    folder = OUTPUT_FOLDER + "/" + folder

    insert_data_mongodb(folder, project)

    # gets all_redditors, y_pred
    classification_results = classification(project)
    # visualizations.recovery_non_recovery_donut([sum(classification_results[1]),
    #                                             len(classification_results[1])-sum(classification_results[1])],
    #                                             project)
    # visualizations.recovery_lda_and_word_cloud(project)
    # visualizations.non_recovery_lda_and_word_cloud(project)
    # user_level_visualization.user_visualization(project)

    base_input_folder = folder + "/"
    base_output_folder = VISUALIZATION_FOLDER + str(project.id) + "/"

    for user_folder in os.listdir(folder):
        if user_folder[0] == ".":
            continue

        input_folder = base_input_folder + user_folder
        output_folder = base_output_folder + user_folder + "/"

        if not os.path.exists(output_folder):
            os.makedirs(output_folder)

        run_single_classification(input_folder, output_folder)


def run_single_classification(input_folder, output_folder):
    # sentiment_analysis.main(input_folder, output_folder)
    os.system("cd /Users/jhadeeptanshu/RecoveryInterventions/sentiment_analysis && python sentiment_analysis.py -i %s -o %s"
              %(input_folder, output_folder))


def train_classification(folder, project):
    pass


def main():
    parser = OptionParser()
    parser.add_option('-f', '--folder', dest='folder', help="Folder name", type=str)
    parser.add_option('-p', '--project', dest='project', help="Project id", type=str)
    parser.add_option('-t', '--job_type', dest='job_type', help="Job type", type=str)

    (options, args) = parser.parse_args()
    run_batch_classification(options.folder, Project(options.project, options.folder, options.job_type))


if __name__ == "__main__":
    # folder_name = "aj_ds"
    # folder_name = "1500_copy_dataset"
    # from TestModels import Project
    # doWork("aj_ds", Project("5bfa2995473c8923db51e0b2", "aj_ds.zip", "5bf8c2da473c89cfb14d63d2"))
    # fileHandler("5bfa126b473c8916fdd955b6")

    main()
    # input_folder = "/Users/jhadeeptanshu/RecoveryInterventions/run_uploads/user_data"
    # output_folder = "/Users/jhadeeptanshu/RecoveryInterventions/visualizations/1/"
    # run_trained_classification_single(input_folder, output_folder)
