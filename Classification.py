import logging
import zipfile
import os
import sys
from sklearn.externals import joblib
import re
import numpy as np
import pymongo
from bson.objectid import ObjectId
from optparse import OptionParser


VISUALIZATION_FOLDER = os.path.join(os.path.dirname(__file__), 'visualizations')
logging.basicConfig(filename='classification.log', level=logging.DEBUG)
OUTPUT_FOLDER = os.path.join(os.path.dirname(__file__), 'output')
UPLOAD_FOLDER = os.path.join(os.path.dirname(__file__), '/run_uploads/')

sys.path.append(os.path.dirname(__file__) + "/sentiment_analysis/")

environment = "DEV"
# environment = "PROD"


def get_db():
    if environment == "DEV":
        client = pymongo.MongoClient('mongodb://localhost:27017/Recovery')
        db = client.recoveryi
    else:
        client = pymongo.MongoClient('mongodb://recovery:interventions@localhost:27017/recoveryi?authMechanism=SCRAM-SHA-256')
        db = client.Recovery
    return db


class Project:
    def __init__(self, project_id, filename, job_type, classifier):
        self.id = str(project_id)
        self.filename = filename
        self.job_type = job_type
        self.classifier = classifier


def fileHandler(project_id):

    db = get_db()
    project = db.projects.find_one({'_id': ObjectId(project_id)})

    if not project:
        return

    project = Project(project['_id'], project['file'], project['job_type'], project['classifier'])
    file_name = project.filename
    zip_ref = zipfile.ZipFile(os.path.join(UPLOAD_FOLDER, file_name), 'r')
    extracted = zip_ref.namelist()
    zip_ref.extractall(OUTPUT_FOLDER)
    zip_ref.close()

    input_folder = os.path.join(OUTPUT_FOLDER, file_name).replace(".zip", "")
    os.rename(os.path.join(OUTPUT_FOLDER, str(extracted[0]))[:-1], input_folder)

    if project.job_type == "CLASSIFY":
        print "CLASSIFY"
        if not project.classifier:
            os.system("python Classification.py -f " + input_folder + " -p " + str(project_id) + " -t " + project.job_type)
        else:
            os.system("python Classification.py -f " + input_folder + " -p " + str(project_id) + " -t " + project.job_type + " -c " + project.classifier)

    elif project.job_type == "TRAIN":
        print "TRAIN"
        os.system("python TrainClassifier.py -f " + input_folder + " -p " + str(project_id) + " -t " + project.job_type)
    
    db['projects'].find_one_and_update({"project_id": project_id},
                                 {"$set": {"job_status": "1"}})


def unzip_folder(input_file):
    zip_ref = zipfile.ZipFile(input_file, 'r')
    extracted = zip_ref.namelist()
    zip_ref.extractall(OUTPUT_FOLDER)
    zip_ref.close()

    return extracted[0]


def insert_data_mongodb(folder, project):
    db = get_db()

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


def classification(project, classifier_folder):
    db = get_db()
    #post vectorizer  pickle
    post_vectorizer_pickle = os.path.join(os.path.dirname(__file__), 'min_df_4_posts_vect.pkl')
    # +ve ate score pickle
    positive_ate_pickle = os.path.join(os.path.dirname(__file__), 'ate_posts_increased_rates_of_transfer.pkl')
    # -ve ate score pickle
    negative_ate_pickle = os.path.join(os.path.dirname(__file__), 'ate_posts_decreased_rates_of_transfer.pkl')
    n = 4000
    #classifier pickle
    sclf_pickle = os.path.join(os.path.dirname(__file__), 'sclf.pkl')

    if classifier_folder and classifier_folder != "na":
        post_vectorizer_pickle = os.path.join(classifier_folder, "post_vectorizer.pkl")
        positive_ate_pickle = os.path.join(classifier_folder, "positive_ate.pkl")
        negative_ate_pickle = os.path.join(classifier_folder, "negative_ate.pkl")
        n = joblib.load(os.path.join(classifier_folder, 'n.pkl'))
        sclf_pickle = os.path.join(classifier_folder, "classifier.pkl")

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

    post_vect = joblib.load(post_vectorizer_pickle)

    bow_posts = post_vect.transform(user_posts_cumulative)
    posts_matrix = bow_posts.toarray()
    print bow_posts.shape

    posts_psa_dic = joblib.load(positive_ate_pickle)
    sorted_posts_dic = sorted(posts_psa_dic, key=posts_psa_dic.get, reverse=True)

    decreased_posts_psa_dic = joblib.load(negative_ate_pickle)
    decreased_sorted_posts_dic = sorted(decreased_posts_psa_dic, key=decreased_posts_psa_dic.get)


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

    sclf = joblib.load(sclf_pickle)
    y_pred = sclf.predict(matrix)
    # print y_pred
    for c,user in enumerate(all_redditors):
        user_dic = collection.find_one({"user":user, "project_id": project.id})
        user_dic["recovery"] = y_pred[c]
        collection.save(user_dic)
    return [all_redditors,y_pred]


def run_batch_classification(folder, project):
    # if not os.path.exists(VISUALIZATION_FOLDER + "/" + project.id + "/system_level"):
    #     os.makedirs(VISUALIZATION_FOLDER + "/" + project.id + "/system_level")
    insert_data_mongodb(folder, project)

    # gets all_redditors, y_pred
    classification_results = classification(project, project.classifier)

    # visualizations.recovery_non_recovery_donut([sum(classification_results[1]),
    #                                             len(classification_results[1])-sum(classification_results[1])],
    #                                             project)
    # visualizations.recovery_lda_and_word_cloud(project)
    # visualizations.non_recovery_lda_and_word_cloud(project)
    # user_level_visualization.user_visualization(project)

    base_input_folder = folder
    base_output_folder = os.path.join(VISUALIZATION_FOLDER, str(project.id))

    if not os.path.exists(base_output_folder):
        os.mkdir(base_output_folder)

    write_classification_result_file(classification_results, base_output_folder)

    for c,user_folder in enumerate(os.listdir(folder)):
        print c, "****************************************"
        if user_folder[0] == ".":
            continue

        input_folder = os.path.join(base_input_folder, user_folder)
        output_folder = os.path.join(base_output_folder, user_folder)

        if not os.path.exists(output_folder):
            os.mkdir(output_folder)

        output_folder = os.path.join(os.path.abspath("RecoveryIntervention.py").replace("RecoveryIntervention.py", ""),
                                     output_folder)
        run_sentiment_analysis(input_folder, output_folder, project)


def run_sentiment_analysis(input_folder, output_folder, project):
    # print project.classifier
    sentiment_analysis_folder = os.path.join(os.path.dirname(__file__), "sentiment_analysis")
    # insert_data_mongodb(input_folder, project)

    # for user_folder in os.listdir(input_folder):
    #     if user_folder[0] != ".":
    #         input_folder = os.path.join(input_folder, user_folder)
    #         break

    # classification_results = classification(project, project.classifier)

    os.system("cd %s && python sentiment_analysis.py -i %s -o %s -c %s, -n %d" %
              (sentiment_analysis_folder, input_folder, output_folder, project.classifier, -1))


def run_single_classification(input_folder, output_folder, project):
    insert_data_mongodb(input_folder, project)
    classification_results = classification(project, project.classifier)

    write_classification_result_file(classification_results, output_folder)

    for user_folder in os.listdir(input_folder):
        if user_folder[0] != ".":
            input_folder = os.path.join(input_folder, user_folder)
            break

    run_sentiment_analysis(input_folder, output_folder, project)


def write_classification_result_file(classification_result, output_folder):
    with open(os.path.join(output_folder, 'recovery_intervention_result.txt'), 'a') as fp:
        for i in range(len(classification_result[0])):
            fp.write(str(classification_result[0][i]) + " " + str(classification_result[1][i]) + "\n")
            fp.flush()


def get_classification_result(input_file):

    with open(os.path.join(input_file, "recovery_intervention_result.txt")) as fp:
        return fp.readline().strip().split(" ")


def modify_number_of_topics_helper(project_id, n):
    sentiment_analysis_folder = os.path.join(os.path.dirname(__file__), "sentiment_analysis")
    db = get_db()
    project = db.projects.find_one({'_id': ObjectId(project_id)})
    output_folder = os.path.join(VISUALIZATION_FOLDER, str(project['_id']))

    input_folder = os.path.join(OUTPUT_FOLDER,  project['file'].replace(".zip", ""))
    for user_folder in os.listdir(input_folder):
        if user_folder[0] != ".":
            input_folder = os.path.join(input_folder, user_folder)
            break

    os.system("cd %s && python sentiment_analysis.py -i %s -o %s -n %d" %
              (sentiment_analysis_folder, input_folder, output_folder, n))


def main():
    parser = OptionParser()
    parser.add_option('-f', '--folder', dest='folder', help="Folder name", type=str)
    parser.add_option('-p', '--project', dest='project', help="Project id", type=str)
    parser.add_option('-t', '--job_type', dest='job_type', help="Job type", type=str)
    parser.add_option('-c', '--classifier', dest='classifier', help="Classifier", type=str)

    (options, args) = parser.parse_args()
    classifier_folder = "na"
    if options.classifier:
        classifier_folder = options.classifier
    run_batch_classification(options.folder, Project(options.project, options.folder, options.job_type, classifier_folder))


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
