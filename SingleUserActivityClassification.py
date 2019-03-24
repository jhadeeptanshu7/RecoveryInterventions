from sklearn.externals import joblib
import os
import reddit_forum_categories
from bokeh.models import ColumnDataSource
from bokeh.plotting import figure, output_file, save
from optparse import OptionParser

recovery_subreddits = reddit_forum_categories.recovery_subreddits
drug_subreddits = reddit_forum_categories.drug_subreddits


def load_input_folder():
    input_folder = '/Users/jhadeeptanshu/plos_one_classifications/activity/single_user_input_folder'
    return input_folder

# def loading_subreddit_activity_text_file():

def get_user_subreddit_activity(input_folder):
    subreddit_activity=[]
    for sub_folder in os.listdir(input_folder):
        if sub_folder[0]==".":
            continue
        # print sub_folder
        user = sub_folder
        sub_folder_path = os.path.join(input_folder,sub_folder)
        for text_file in os.listdir(sub_folder_path):
            # print text_file
            if text_file.endswith(".txt"):
                text_file_path = sub_folder_path = os.path.join(sub_folder_path,text_file)
                file_object = open(text_file_path, "r")
                # print file_object
                post_content = file_object.readlines()
                for p in post_content:
                    subreddit_activity.append(p.strip())
    return user, subreddit_activity


def load_pickles(folder):
    subreddit_activity_dic = joblib.load(os.path.join(folder, "bibm_subreddit_dic.pkl"))
    rf_classifier = joblib.load(os.path.join(folder, 'subreddit_activity_classifier.pkl'))
    return subreddit_activity_dic, rf_classifier


def run_single_user_classification(subreddit_activity,subreddit_dic,rf_clf):
    # print subreddit_activity
    redditor_row = [0] * 16181
    for sub in subreddit_activity:
        if sub in recovery_subreddits:
            return 1
        if sub in subreddit_dic:
            redditor_row[subreddit_dic[sub]] = 1
        else:
            continue

    redditor_row = [redditor_row]
    prediction = rf_clf.predict(redditor_row)
    return prediction

def load_output_folder():
    op_folder = '/Users/jhadeeptanshu/plos_one_classifications/activity/single_user_op_folder'
    return op_folder


def write_result(op_folder,user,classification_result):
    print op_folder
    user_op_folder = os.path.join(op_folder,user)
    try:
        os.mkdir(user_op_folder)
    except:
        print "folder exists"
    file_path = os.path.join(user_op_folder,"prediction.txt")
    file = open(file_path,'w')
    if classification_result ==1:
        file.write(user + " is open to a drug addiction recovery intervention.")
    else:
        file.write(user + " is not open to a drug addiction recovery intervention.")

def find_width(sub_terms):
    # print len(sub_terms)
    if len(sub_terms) == 1:
        # print "len 1"
        width = 0.015
    elif len(sub_terms) < 4:
        # print "len 1"
        width = 0.05
    elif 10 < len(sub_terms) < 12:
        width = 0.2
    elif len(sub_terms) < 10:
        width = 0.09
    else:
        width = 0.1
    return width

def subreddit_histogram(subreddit_activity,user,op_folder):
    user_op_folder = os.path.join(op_folder,user)


    subreddit_count_dic = {}

    for sub in subreddit_activity:
        if sub in subreddit_count_dic:
            subreddit_count_dic[sub] +=1
        else:
            subreddit_count_dic[sub] =1
    sorted_keys = sorted(subreddit_count_dic, key=subreddit_count_dic.__getitem__, reverse=True)
    sub_counts = []
    sub_terms = []
    color = []
    types = []
    for d in sorted_keys:
        # print d
        sub_terms.append(d)
        sub_counts.append(subreddit_count_dic[d])
        if d in drug_subreddits:
            color.append('red')
            types.append("Drug Subreddit")
        elif d in recovery_subreddits:
            color.append('blue')
            types.append("Recovery Subreddit")
        else:
            color.append('black')
            types.append("Other")

    # print color
    # types = ["Drug Subreddit","Recovery Subreddit", "Other Subreddit"]
    source = ColumnDataSource(data=dict(sub_terms=sub_terms, sub_counts=sub_counts, color=color,types=types))


    p = figure(x_range=sub_terms, plot_height=300, plot_width=1100, title="Subreddit post frequency")

    width = find_width(sub_terms)
    p.vbar(x='sub_terms', top='sub_counts', width=width,color = 'color',legend = 'types',source=source)
    p.xgrid.grid_line_color = None
    p.y_range.start = 0
    p.x_range.range_padding = 0.1
    p.xaxis.major_label_orientation = 1

    output_file(os.path.join(user_op_folder, "subreddit_bar_chart.html"))
    save(p)


def main():
    parser = OptionParser()
    parser.add_option('-i', '--input', dest='input_folder', help="input folder", type=str)
    parser.add_option('-o', '--output', dest='output_folder', help="output folder", type=str)
    parser.add_option('-c', '--classifier', dest='classifier', help="classifier folder", type=str)

    (options, args) = parser.parse_args()

    input_folder = options.input_folder
    op_folder = options.output_folder
    subreddit_dic, rf_clf = load_pickles(options.classifier)

    user, subreddit_activity = get_user_subreddit_activity(input_folder)
    classification_result = run_single_user_classification(subreddit_activity, subreddit_dic, rf_clf)
    write_result(op_folder, user, classification_result)
    subreddit_histogram(subreddit_activity, user, op_folder)


if __name__ == '__main__':
    main()
