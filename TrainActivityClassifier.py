from sklearn.externals import joblib
from sklearn.model_selection import train_test_split, GridSearchCV
import numpy as np
from sklearn.linear_model import LogisticRegression
import os
from optparse import OptionParser

def load_input_folder():
    input_folder = '/Users/jhadeeptanshu/plos_one_classifications/activity/train_classifier_input_folder'
    return input_folder


def create_subreddit_dic(input_folder):
    print input_folder
    subreddit_dic = {}
    for dir in os.listdir(input_folder):
        if dir[0] == ".":
            continue
        # print dir
        for sub_dir in os.listdir(input_folder+"/"+dir):
            if sub_dir[0]==".":
                continue
            # print sub_dir
            sub_folder_path = input_folder+"/"+dir+"/"+sub_dir
            for text_file in os.listdir(sub_folder_path):
                # print text_file
                text_file_path = sub_folder_path + "/" + text_file
                if text_file.endswith(".txt"):
                    # print text_file
                    file_object = open(text_file_path, "r")
                    post_content = file_object.readlines()
                    for p in post_content:
                        if p.strip() in subreddit_dic:
                            continue
                        else:
                            subreddit_dic[p.strip()] = len(subreddit_dic)
                            # print subreddit_dic
    return subreddit_dic


def create_subreddit_activity_matrix(input_folder,subreddit_dic):
    subreddit_dic = subreddit_dic
    row_len = len(subreddit_dic)
    matrix = []
    for dir in os.listdir(input_folder):
        if dir[0] == ".":
            continue
        # print dir
        for sub_dir in os.listdir(input_folder+"/"+dir):
            if sub_dir[0]==".":
                continue
            print sub_dir
            redditor_row = [0] * row_len
            sub_folder_path = input_folder+"/"+dir+"/"+sub_dir
            for text_file in os.listdir(sub_folder_path):
                # print text_file
                text_file_path = sub_folder_path + "/" + text_file
                if text_file.endswith(".txt"):
                    # print text_file
                    file_object = open(text_file_path, "r")
                    post_content = file_object.readlines()
                    for p in post_content:
                        redditor_row[subreddit_dic[p.strip()]] = 1
            if dir =='recovery':
                redditor_row.append(1)
            else:
                redditor_row.append(0)
            matrix.append(redditor_row)
    return matrix

def build_model(subreddit_activity_matrix):

    matrix = np.array(subreddit_activity_matrix)

    X = matrix[:,:-1]

    y = matrix[:,-1]

    X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.2, stratify=y, random_state=0)
    weights = np.linspace(0.05, 0.95, 20)
    params = {"C":np.logspace(-3,3,7),
              "penalty":["l1","l2"],
            'class_weight': [{0: x, 1: 1.0-x} for x in weights]
        }
    xgb = LogisticRegression(max_iter=600)
    clf = GridSearchCV(xgb, params, n_jobs=-1,cv=5,scoring='f1')
    print ("fitting data")
    clf.fit(X,y)
    return clf


def load_output_folder():
    op_folder = '/Users/jhadeeptanshu/plos_one_classifications/activity/train_classifier_op_folder'
    return op_folder

def save_model(op_folder,subreddit_dic,clf):
    print op_folder
    subreddit_dic_path = os.path.join(op_folder,"bibm_subreddit_dic.pkl")
    rf_clf_path = os.path.join(op_folder,"subreddit_activity_classifier.pkl")
    joblib.dump(subreddit_dic,subreddit_dic_path)
    joblib.dump(clf,rf_clf_path)

def main():
    parser = OptionParser()
    parser.add_option('-i', '--input', dest='input_folder', help="input folder", type=str)
    parser.add_option('-o', '--output', dest='output_folder', help="output folder", type=str)

    (options, args) = parser.parse_args()

    input_folder = options.input_folder
    op_folder = options.output_folder

    subreddit_dic = create_subreddit_dic(input_folder)
    print subreddit_dic
    subreddit_activity_matrix = create_subreddit_activity_matrix(input_folder,subreddit_dic)
    clf = build_model(subreddit_activity_matrix)
    save_model(op_folder,subreddit_dic,clf)


    # print subreddit_dic
    # print len(subreddit_dic)

    # for sub_folder in os.listdir(input_folder):
    #     if sub_folder[0]==".":
    #         continue
    #     user = sub_folder
    #     print user, subreddit_activity
    #     op_folder = load_output_folder()

if __name__=='__main__':
    main()

# db = get_db()
#     collection = db['drug_users']

    # for dir in os.listdir(folder):
    #     if dir[0] == ".":
    #         continue
    #
    #     for sub_dir in os.listdir(folder+"/"+dir):
    #         if sub_dir[0]==".":
    #             continue
    #         user_dic = {}
    #         user_dic["project_id"] = project.id
    #         user_dic["user"] = sub_dir
    #         sub_folder_path = folder+"/"+dir+"/"+sub_dir
    #         posts = []
    #         for text_file in os.listdir(sub_folder_path):
    #             # print text_file
    #             text_file_path = sub_folder_path + "/" + text_file
    #             if text_file.endswith(".txt"):
    #                 # print text_file
    #                 file_object = open(text_file_path, "r")
    #                 post_content = file_object.read()
    #                 posts.append(post_content)
    #         user_dic["posts"] = posts
    #         if dir =='recovery':
    #             user_dic['recovery']=1
    #         if dir =='non_recovery':
    #             user_dic['recovery']=0
    #         collection.insert_one(user_dic)