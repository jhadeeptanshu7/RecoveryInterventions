import sys
import os
sys.path.append('./../')
sys.path.append(os.path.dirname(__file__))

from optparse import OptionParser
from time import sleep
import afinn_sentiment_scores
import folium
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.decomposition import LatentDirichletAllocation
import stopwords
import re
import pyLDAvis
import pyLDAvis.sklearn

from sklearn.externals import joblib
from bokeh.plotting import figure, output_file, save
from bokeh.models import ColumnDataSource
from bokeh.models.tools import HoverTool
import pandas as pd
from bokeh.models.annotations import Span
from numpy import pi
from bokeh.transform import cumsum
from bokeh.layouts import gridplot
from bokeh.transform import dodge
from bokeh.core.properties import value
import en_core_web_sm

nlp = en_core_web_sm.load()
from collections import defaultdict
from geopy.geocoders import Nominatim
from geopy.exc import GeocoderTimedOut

geolocator = Nominatim(user_agent="project-geolocation-1")
from Classification import get_db
import nltk


def create_post_dic(folder):
    post_dic = {}
    for text_file in os.listdir(folder):
        if text_file[0] == ".":
            continue
        if text_file.endswith(".txt"):
            key = text_file.split(".")[0]
            text_file_path = folder + "/" + text_file
            file_object = open(text_file_path, "r")
            post_content = file_object.read()
            post_dic[key] = post_content

    return post_dic


def sentiment_score_of_each_post(post_dic):
    afinn_sentiment_dic = afinn_sentiment_scores.affinn_sentiment_scores
    regex = joblib.load("sentiment_regex.pkl")
    post_sentiment_score_dic = {}
    for p in post_dic:
        sentiment_term_count_dic = {}
        text = re.sub('[^0-9a-zA-Z]+', ' ', post_dic[p])
        found_terms = regex.findall(text)
        for fd in (found_terms):
            strip_fd = fd.strip().lower()
            if strip_fd in sentiment_term_count_dic:
                sentiment_term_count_dic[strip_fd] += 1
            else:
                sentiment_term_count_dic[strip_fd] = 1
        sentiment_score = 0
        for term in sentiment_term_count_dic:
            # print sentiment_term_count_dic[term], afinn_sentiment_dic[term]
            sentiment_score += sentiment_term_count_dic[term] * afinn_sentiment_dic[term]
        try:
            post_sentiment_score_dic[p] = float(sentiment_score) / float(len(found_terms))
        except:
            post_sentiment_score_dic[p] = 0
    return post_sentiment_score_dic


def sentiment_line_plot(sentiment_data_frame, output_folder):
    source = ColumnDataSource(sentiment_data_frame)

    p = figure(x_axis_type="datetime", plot_width=1000, plot_height=400, title="Temporal Sentiment Scores")
    p.xaxis[0].formatter.days = '%m-%d-%Y'
    p.xgrid.grid_line_color = None
    p.ygrid.grid_line_alpha = 0.5
    p.xaxis.axis_label = 'Time'
    p.yaxis.axis_label = 'Sentiment Value'
    p.y_range.end = 6
    p.y_range.start = -6
    p.yaxis.ticker = [-5, -4, -3, -2, -1, 0, 1, 2, 3, 4, 5]
    upper = Span(location=5, dimension='width', line_color='green', line_width=0.5)
    p.add_layout(upper)

    lower = Span(location=-5, dimension='width', line_color='red', line_width=0.5)
    p.add_layout(lower)

    p.line('date', 'sentiment_score', source=source, line_width=1, line_color='black', line_alpha=0.1)
    # p.circle('date', 'sentiment_score', source=source, fill_color="orange", size=8)
    p.circle('date', 'sentiment_score', source=source, fill_color='color', size='size', line_color='black',
             legend='legend_value')

    hover = HoverTool()
    hover.tooltips = [
        ('Date', '@str_date'),
        ('Sentiment Value', '@sentiment_score'),
        ('Post', '@post')
    ]

    p.add_tools(hover)

    output_file(os.path.join(output_folder, "sentiment_analysis.html"))
    save(p)


def converting_data_to_pandas_df(sentiment_score, post_dic):
    list_of_dictionaries = []
    for s in sentiment_score:
        dic = {}
        # print s
        dic['date'] = s
        dic['sentiment_score'] = sentiment_score[s]
        dic['post'] = post_dic[s]
        dic['str_date'] = s
        if sentiment_score[s] == 0:
            dic['color'] = 'grey'
            dic['size'] = 8
            dic['legend_value'] = 'Neutral'
        if sentiment_score[s] > 0:
            dic['legend_value'] = 'Positive Sentiment'
            dic['color'] = 'green'
            if 0 < sentiment_score[s] <= 1:
                dic['size'] = 8
            if 1 < sentiment_score[s] <= 2:
                dic['size'] = 10
            if 2 < sentiment_score[s] <= 3:
                dic['size'] = 12
            if 3 < sentiment_score[s] <= 4:
                dic['size'] = 14
            if 4 < sentiment_score[s] <= 5:
                dic['size'] = 16
        if sentiment_score[s] < 0:
            dic['legend_value'] = 'Negative Sentiment'
            dic['color'] = 'red'
            if -1 <= sentiment_score[s] < 0:
                dic['size'] = 8
            if -2 <= sentiment_score[s] < -1:
                dic['size'] = 10
            if -3 <= sentiment_score[s] < -2:
                dic['size'] = 12
            if -4 <= sentiment_score[s] < -3:
                dic['size'] = 14
            if -5 <= sentiment_score[s] < -4:
                dic['size'] = 16
        list_of_dictionaries.append(dic)
    df = pd.DataFrame(list_of_dictionaries)
    df['date'] = pd.to_datetime(df['date'], format='%m-%d-%Y')
    df.sort_values('date', inplace=True)
    return df


def geolocation_based_sentiment_analysis(post_dic):
    pass


def user_posts_topic_modeling(post_dic, n, output_folder):
    user_posts_cumulative = []
    for p in post_dic:
        no_url_post = re.sub(r'http\S+', '', post_dic[p])
        user_posts_cumulative.append(no_url_post)
    vect = CountVectorizer(stop_words=stopwords.stopwords)
    X = vect.fit_transform(user_posts_cumulative)
    lda = LatentDirichletAllocation(n_topics=n, random_state=0, learning_method='online')
    document_topics = lda.fit_transform(X)
    sorting = np.argsort(lda.components_, axis=1)[:, ::-1]
    feature_names = np.array(vect.get_feature_names())
    # mglearn.tools.print_topics(topics=range(10),feature_names=feature_names,sorting=sorting,topics_per_chunk=5,n_words=10)
    topics = pyLDAvis.sklearn.prepare(lda, X, vect, mds='mmds')

    pyLDAvis.save_html(topics, os.path.join(output_folder, 'document_topics.html'))


def drug_bar_chart(post_dic, output_folder):
    exactMatch = joblib.load("drugs_exactMatch.pkl")
    drug_count_dic = {}
    for p in post_dic:
        text = re.sub('[^0-9a-zA-Z]+', ' ', post_dic[p])
        found_drugs = exactMatch.findall(text)
        for fd in (found_drugs):
            strip_fd = fd.strip().lower()
            if strip_fd in drug_count_dic:
                drug_count_dic[strip_fd] += 1
            else:
                drug_count_dic[strip_fd] = 1

    sorted_keys = sorted(drug_count_dic, key=drug_count_dic.__getitem__, reverse=True)
    drug_counts = []
    drug_terms = []
    for d in sorted_keys:
        drug_terms.append(d)
        drug_counts.append(drug_count_dic[d])

    p = figure(x_range=drug_terms, plot_height=300, plot_width=1100, title="Drug Counts")

    p.vbar(x=drug_terms, top=drug_counts, width=0.5)
    p.xgrid.grid_line_color = None
    p.y_range.start = 0
    p.x_range.range_padding = 0.1
    p.xaxis.major_label_orientation = 1

    output_file(os.path.join(output_folder, "drug_bar_chart.html"))
    save(p)


def stacked_drug_bar_chart(post_dic, output_folder):
    exactMatch = joblib.load("drugs_exactMatch.pkl")
    drug_count_dic = {}
    for p in post_dic:
        text = re.sub('[^0-9a-zA-Z]+', ' ', post_dic[p])
        found_drugs = exactMatch.findall(text)
        for fd in (found_drugs):
            strip_fd = fd.strip().lower()
            if strip_fd in drug_count_dic:
                drug_count_dic[strip_fd] += 1
            else:
                drug_count_dic[strip_fd] = 1

    sorted_keys = sorted(drug_count_dic, key=drug_count_dic.__getitem__, reverse=True)
    drug_value_dic = joblib.load('drug_value_dic.pkl')
    drug_counts = []
    drug_terms = []
    all_user_list = []
    recovery_user_list = []
    non_recovery_user_list = []
    user_type = ['Current User', 'All Users', 'Recovery Users', 'Non Recovery Users']
    for d in sorted_keys:
        drug_terms.append(d)
        drug_counts.append(drug_count_dic[d])
        all_user_list.append(drug_value_dic['all_users'][d][0])
        if d in drug_value_dic['recovery_users']:
            recovery_user_list.append(drug_value_dic['recovery_users'][d][0])
        else:
            recovery_user_list.append(0)

        if d in drug_value_dic['non_recovery_users']:
            non_recovery_user_list.append(drug_value_dic['non_recovery_users'][d][0])
        else:
            non_recovery_user_list.append(0)

    data = {'drug_terms': drug_terms, 'Current User': drug_counts, 'All Users': all_user_list,
            'Non Recovery Users': non_recovery_user_list, 'Recovery Users': recovery_user_list}
    source = ColumnDataSource(data=data)

    p = figure(x_range=drug_terms, y_range=(0, 10), plot_height=500, plot_width=2000, title="User Drug Count",
               toolbar_location=None, tools="")

    p.vbar(x=dodge('drug_terms', -0.20, range=p.x_range), top='Current User', width=0.2, source=source,
           color="black", legend=value("Current User"))

    p.vbar(x=dodge('drug_terms', -0.0, range=p.x_range), top='All Users', width=0.2, source=source,
           color="green", legend=value("All Users"))

    p.vbar(x=dodge('drug_terms', 0.20, range=p.x_range), top='Non Recovery Users', width=0.2, source=source,
           color="red", legend=value("Non Recovery Users"))

    p.vbar(x=dodge('drug_terms', 0.40, range=p.x_range), top='Recovery Users', width=0.2, source=source,
           color="blue", legend=value("Recovery Users"))

    p.x_range.range_padding = 0.01
    p.xgrid.grid_line_color = None
    p.legend.location = "top_center"
    p.legend.orientation = "horizontal"
    p.xaxis.major_label_orientation = 1
    output_file(os.path.join(output_folder, "stacked_drug_bar_chart.html"))
    save(p)


def stacked_recovery_bar_chart(post_dic, output_folder):
    exactMatch = joblib.load("recovery_exactMatch.pkl")
    drug_count_dic = {}
    for p in post_dic:
        text = re.sub('[^0-9a-zA-Z]+', ' ', post_dic[p])
        found_drugs = exactMatch.findall(text)
        for fd in (found_drugs):
            strip_fd = fd.strip().lower()
            if strip_fd in drug_count_dic:
                drug_count_dic[strip_fd] += 1
            else:
                drug_count_dic[strip_fd] = 1

    sorted_keys = sorted(drug_count_dic, key=drug_count_dic.__getitem__, reverse=True)
    drug_value_dic = joblib.load('recovery_value_dic.pkl')
    drug_counts = []
    drug_terms = []
    all_user_list = []
    recovery_user_list = []
    non_recovery_user_list = []
    user_type = ['Current User', 'All Users', 'Recovery Users', 'Non Recovery Users']
    for d in sorted_keys:
        drug_terms.append(d)
        drug_counts.append(drug_count_dic[d])
        all_user_list.append(drug_value_dic['all_users'][d][0])
        if d in drug_value_dic['recovery_users']:
            recovery_user_list.append(drug_value_dic['recovery_users'][d][0])
        else:
            recovery_user_list.append(0)

        if d in drug_value_dic['non_recovery_users']:
            non_recovery_user_list.append(drug_value_dic['non_recovery_users'][d][0])
        else:
            non_recovery_user_list.append(0)

    data = {'drug_terms': drug_terms, 'Current User': drug_counts, 'All Users': all_user_list,
            'Non Recovery Users': non_recovery_user_list, 'Recovery Users': recovery_user_list}
    source = ColumnDataSource(data=data)

    p = figure(x_range=drug_terms, y_range=(0, 10), plot_height=500, plot_width=2000, title="User Recovery Term Count",
               toolbar_location=None, tools="")

    p.vbar(x=dodge('drug_terms', -0.20, range=p.x_range), top='Current User', width=0.2, source=source,
           color="black", legend=value("Current User"))

    p.vbar(x=dodge('drug_terms', -0.0, range=p.x_range), top='All Users', width=0.2, source=source,
           color="green", legend=value("All Users"))

    p.vbar(x=dodge('drug_terms', 0.20, range=p.x_range), top='Non Recovery Users', width=0.2, source=source,
           color="red", legend=value("Non Recovery Users"))

    p.vbar(x=dodge('drug_terms', 0.40, range=p.x_range), top='Recovery Users', width=0.2, source=source,
           color="blue", legend=value("Recovery Users"))

    p.x_range.range_padding = 0.01
    p.xgrid.grid_line_color = None
    p.legend.location = "top_center"
    p.legend.orientation = "horizontal"
    p.xaxis.major_label_orientation = 1
    output_file(os.path.join(output_folder, "stacked_recovery_bar_chart.html"))
    save(p)


def drug_terms_and_no_of_drug_terms_in_a_post(post):
    exactMatch = joblib.load("drugs_exactMatch.pkl")
    drug_count_dic = {}
    text = re.sub('[^0-9a-zA-Z]+', ' ', post)
    found_drugs = exactMatch.findall(text)
    for fd in (found_drugs):
        strip_fd = fd.strip().lower()
        if strip_fd in drug_count_dic:
            drug_count_dic[strip_fd] += 1
        else:
            drug_count_dic[strip_fd] = 1
    return (drug_count_dic, sum(drug_count_dic.values()))


def creating_drug_pandas_df(post_dic):
    list_of_dictionaries = []
    for p in post_dic:
        dic = {}
        dic['date'] = p
        dic['post'] = post_dic[p]
        dic['str_date'] = p
        func_return = drug_terms_and_no_of_drug_terms_in_a_post(post_dic[p])
        dic['drug_terms'] = str(func_return[0])
        dic['drug_count'] = func_return[1]
        # dic['drug_terms'],dic['drug_count']= drug_terms_and_no_of_drug_terms_in_a_post(post_dic[p])[0],drug_terms_and_no_of_drug_terms_in_a_post(post_dic[p])[0]
        list_of_dictionaries.append(dic)
    df = pd.DataFrame(list_of_dictionaries)
    df['date'] = pd.to_datetime(df['date'], format='%m-%d-%Y')
    df.sort_values('date', inplace=True)
    return df


def temporal_drug_use(drugs_data_frame, output_folder):
    # output_file("temporal_drug_count.html")

    source = ColumnDataSource(drugs_data_frame)

    p = figure(x_axis_type="datetime", plot_width=1000, plot_height=400, title="Temporal Drug Use")
    p.xaxis[0].formatter.days = '%m-%d-%Y'
    p.xgrid.grid_line_color = None
    p.ygrid.grid_line_alpha = 0.5
    p.xaxis.axis_label = 'Time'
    p.yaxis.axis_label = 'Drug Count'
    #
    #
    p.line('date', 'drug_count', source=source, line_width=1, line_color='black', line_alpha=0.1)
    # # p.circle('date', 'sentiment_score', source=source, fill_color="orange", size=8)
    p.circle('date', 'drug_count', source=source, fill_color='orange', size=12, line_color='black')

    hover = HoverTool()
    hover.tooltips = [
        ('Date', '@str_date'),
        ('Drug Count', '@drug_count'),
        ('Drug Terms', '@drug_terms'),
    ]

    p.add_tools(hover)

    output_file(os.path.join(output_folder, "temporal_drug_count.html"))

    save(p)


def geolocation_based_drug_use_analysis(post_dic):
    pass


def recovery_bar_chart(post_dic, output_folder):
    exactMatch = joblib.load("recovery_exactMatch.pkl")
    drug_count_dic = {}
    for p in post_dic:
        text = re.sub('[^0-9a-zA-Z]+', ' ', post_dic[p])
        found_drugs = exactMatch.findall(text)
        for fd in (found_drugs):
            strip_fd = fd.strip().lower()
            if strip_fd in drug_count_dic:
                drug_count_dic[strip_fd] += 1
            else:
                drug_count_dic[strip_fd] = 1

    sorted_keys = sorted(drug_count_dic, key=drug_count_dic.__getitem__, reverse=True)
    drug_counts = []
    drug_terms = []
    for d in sorted_keys:
        drug_terms.append(d)
        drug_counts.append(drug_count_dic[d])

    p = figure(x_range=drug_terms, plot_height=300, plot_width=1100, title="Recovery Terms Count")

    p.vbar(x=drug_terms, top=drug_counts, width=0.5)
    p.xgrid.grid_line_color = None
    p.y_range.start = 0
    p.x_range.range_padding = 0.1
    p.xaxis.major_label_orientation = 1

    output_file(os.path.join(output_folder, "recovery_bar_chart.html"))
    save(p)


def recovery_terms_and_no_of_drug_terms_in_a_post(post):
    exactMatch = joblib.load("recovery_exactMatch.pkl")
    drug_count_dic = {}
    text = re.sub('[^0-9a-zA-Z]+', ' ', post)
    found_drugs = exactMatch.findall(text)
    for fd in (found_drugs):
        strip_fd = fd.strip().lower()
        if strip_fd in drug_count_dic:
            drug_count_dic[strip_fd] += 1
        else:
            drug_count_dic[strip_fd] = 1
    return (drug_count_dic, sum(drug_count_dic.values()))


def creating_recovery_pandas_df(post_dic):
    list_of_dictionaries = []
    for p in post_dic:
        dic = {}
        dic['date'] = p
        dic['post'] = post_dic[p]
        dic['str_date'] = p
        func_return = recovery_terms_and_no_of_drug_terms_in_a_post(post_dic[p])
        dic['drug_terms'] = str(func_return[0])
        dic['drug_count'] = func_return[1]
        # dic['drug_terms'],dic['drug_count']= drug_terms_and_no_of_drug_terms_in_a_post(post_dic[p])[0],drug_terms_and_no_of_drug_terms_in_a_post(post_dic[p])[0]
        list_of_dictionaries.append(dic)
    df = pd.DataFrame(list_of_dictionaries)
    df['date'] = pd.to_datetime(df['date'], format='%m-%d-%Y')
    df.sort_values('date', inplace=True)
    return df


def temporal_recovery_use(recovery_data_frame, output_folder):
    source = ColumnDataSource(recovery_data_frame)

    p = figure(x_axis_type="datetime", plot_width=1000, plot_height=400, title="Temporal Recovery Term Use")
    p.xaxis[0].formatter.days = '%m-%d-%Y'
    p.xgrid.grid_line_color = None
    p.ygrid.grid_line_alpha = 0.5
    p.xaxis.axis_label = 'Time'
    p.yaxis.axis_label = 'Recovery Term  Count'
    #
    #
    p.line('date', 'drug_count', source=source, line_width=1, line_color='black', line_alpha=0.1)
    # # p.circle('date', 'sentiment_score', source=source, fill_color="orange", size=8)
    p.circle('date', 'drug_count', source=source, fill_color='orange', size=12, line_color='black')

    hover = HoverTool()
    hover.tooltips = [
        ('Date', '@str_date'),
        ('Recovery Term Count', '@drug_count'),
        ('Recovery Terms', '@drug_terms'),
    ]

    p.add_tools(hover)

    output_file(os.path.join(output_folder, "temporal_recovery_count.html"))

    save(p)


def geolocation_based_recovery_use_analysis(post_dic):
    pass


def no_of_positive_negative_neutral_posts(sentiment_data_frame, output_folder, user_name):
    output_file(os.path.join(output_folder, "no_of_positive_negative_neutral_posts.html"))
    sentiment_scores = sentiment_data_frame["sentiment_score"].tolist()
    no_of_positive_posts = sum(ss > 0 for ss in sentiment_scores)
    no_of_negative_posts = sum(ss < 0 for ss in sentiment_scores)
    no_of_neutral_posts = sum(ss == 0 for ss in sentiment_scores)
    # post_values = {"% Positive Posts":}
    denominator = no_of_negative_posts + no_of_neutral_posts + no_of_positive_posts
    x = {
        'Percentage of Positive Posts': (float(no_of_positive_posts) / float(denominator)) * 100,
        'Percentage of Negative Posts': (float(no_of_negative_posts) / float(denominator)) * 100,
        'Percentage of Neutral Posts': (float(no_of_neutral_posts) / float(denominator)) * 100
    }

    data = pd.Series(x).reset_index(name='value').rename(columns={'index': 'sentiment'})
    data['angle'] = data['value'] / data['value'].sum() * 2 * pi
    data['color'] = ['red', 'grey', 'green']

    # ajinkya
    p = figure(plot_height=350, title=user_name + "                   Average Sentiment Score = " + str(
        round(float(sum(sentiment_scores) / float(len(sentiment_scores))), 2)), toolbar_location=None,
               tools="hover", tooltips="@sentiment: @value", x_range=(-0.5, 1.0))

    p.wedge(x=0, y=1, radius=0.4,
            start_angle=cumsum('angle', include_zero=True), end_angle=cumsum('angle'),
            line_color="white", fill_color='color', legend='sentiment', source=data)

    p.axis.axis_label = None
    p.axis.visible = False
    p.grid.grid_line_color = None

    db_sentiment_scores = joblib.load('db_sentiment_scores.pkl')

    average_user_sentiment_score = db_sentiment_scores['average_user_sentiment_score']
    average_no_positive_posts = db_sentiment_scores['average_positive_posts']
    average_no_negative_posts = db_sentiment_scores['average_negative_posts']
    average_no_of_positive_posts = db_sentiment_scores['average_neutral_posts']
    denominator = average_no_positive_posts + average_no_negative_posts + average_no_of_positive_posts
    x = {
        'Percentage of Positive Posts': (float(average_no_positive_posts) / float(denominator)) * 100,
        'Percentage of Negative Posts': (float(average_no_negative_posts) / float(denominator)) * 100,
        'Percentage of Neutral Posts': (float(average_no_of_positive_posts) / float(denominator)) * 100
    }

    all_users_data = pd.Series(x).reset_index(name='value').rename(columns={'index': 'country'})
    all_users_data['angle'] = all_users_data['value'] / all_users_data['value'].sum() * 2 * pi
    all_users_data['color'] = ['red', 'grey', 'green']

    all_users_p = figure(plot_height=350, title="All Users                   Average Sentiment Score = " + str(
        round(average_user_sentiment_score, 2)), toolbar_location=None,
                         tools="hover", tooltips="@country: @value", x_range=(-0.5, 1.0))

    all_users_p.wedge(x=0, y=1, radius=0.4,
                      start_angle=cumsum('angle', include_zero=True), end_angle=cumsum('angle'),
                      line_color="white", fill_color='color', legend='country', source=all_users_data)

    all_users_p.axis.axis_label = None
    all_users_p.axis.visible = False
    all_users_p.grid.grid_line_color = None

    recovery_average_user_sentiment_score = db_sentiment_scores['recovery_average_user_sentiment_score']
    recovery_average_no_positive_posts = db_sentiment_scores['recovery_average_positive_posts']
    recovery_average_no_negative_posts = db_sentiment_scores['recovery_average_negative_posts']
    recovery_average_no_of_neutral_posts = db_sentiment_scores['recovery_average_neutral_posts']
    denominator = recovery_average_no_positive_posts + recovery_average_no_negative_posts + recovery_average_no_of_neutral_posts
    x = {
        'Percentage of Positive Posts': (float(recovery_average_no_positive_posts) / float(denominator)) * 100,
        'Percentage of Negative Posts': (float(recovery_average_no_negative_posts) / float(denominator)) * 100,
        'Percentage of Neutral Posts': (float(recovery_average_no_of_neutral_posts) / float(denominator)) * 100
    }

    recovery_users_data = pd.Series(x).reset_index(name='value').rename(columns={'index': 'country'})
    recovery_users_data['angle'] = recovery_users_data['value'] / recovery_users_data['value'].sum() * 2 * pi
    recovery_users_data['color'] = ['red', 'grey', 'green']

    recovery_users_p = figure(plot_height=350,
                              title="Recovery Users                   Average Sentiment Score = " + str(
                                  round(recovery_average_user_sentiment_score, 2)), toolbar_location=None,
                              tools="hover", tooltips="@country: @value", x_range=(-0.5, 1.0))

    recovery_users_p.wedge(x=0, y=1, radius=0.4,
                           start_angle=cumsum('angle', include_zero=True), end_angle=cumsum('angle'),
                           line_color="white", fill_color='color', legend='country', source=recovery_users_data)

    recovery_users_p.axis.axis_label = None
    recovery_users_p.axis.visible = False
    recovery_users_p.grid.grid_line_color = None

    non_recovery_average_user_sentiment_score = db_sentiment_scores['non_recovery_average_user_sentiment_score']
    non_recovery_average_no_positive_posts = db_sentiment_scores['non_recovery_average_positive_posts']
    non_recovery_average_no_negative_posts = db_sentiment_scores['non_recovery_average_negative_posts']
    non_recovery_average_no_of_neutral_posts = db_sentiment_scores['non_recovery_average_neutral_posts']
    denominator = non_recovery_average_no_positive_posts + non_recovery_average_no_negative_posts + non_recovery_average_no_of_neutral_posts
    x = {
        'Percentage of Positive Posts': (float(non_recovery_average_no_positive_posts) / float(denominator)) * 100,
        'Percentage of Negative Posts': (float(non_recovery_average_no_negative_posts) / float(denominator)) * 100,
        'Percentage of Neutral Posts': (float(non_recovery_average_no_of_neutral_posts) / float(denominator)) * 100
    }

    non_recovery_users_data = pd.Series(x).reset_index(name='value').rename(columns={'index': 'country'})
    non_recovery_users_data['angle'] = non_recovery_users_data['value'] / non_recovery_users_data[
        'value'].sum() * 2 * pi
    non_recovery_users_data['color'] = ['red', 'grey', 'green']

    non_recovery_users_p = figure(plot_height=350,
                                  title="Non Recovery Users                   Average Sentiment Score = " + str(
                                      round(non_recovery_average_user_sentiment_score, 2)), toolbar_location=None,
                                  tools="hover", tooltips="@country: @value", x_range=(-0.5, 1.0))

    non_recovery_users_p.wedge(x=0, y=1, radius=0.4,
                               start_angle=cumsum('angle', include_zero=True), end_angle=cumsum('angle'),
                               line_color="white", fill_color='color', legend='country', source=non_recovery_users_data)

    non_recovery_users_p.axis.axis_label = None
    non_recovery_users_p.axis.visible = False
    non_recovery_users_p.grid.grid_line_color = None

    save(gridplot([[p, all_users_p], [recovery_users_p, non_recovery_users_p, None]]))


def entity_extraction(post_dic):
    user_entity_dic = {}
    for p in post_dic:
        u = unicode(post_dic[p], "utf-8")
        doc = nlp(u)
        interested_entities = ['PERSON', 'ORG', 'NORP', 'FAC', 'LOC', 'LAW', 'GPE']
        # print [(X.text, X.label_) for X in doc.ents]
        for ent in doc.ents:
            # print ent.label_
            if str(ent.label_) in interested_entities:
                if ent.label_ in user_entity_dic:
                    user_entity_dic[ent.label_].append(ent.text)
                else:
                    user_entity_dic[ent.label_] = [ent.text]

    print user_entity_dic


def find_location_terms_in_text(post_dic):
    loc_df = pd.read_csv('world-cities_csv.csv')

    list_of_cities = loc_df['name'].unique().tolist()
    list_of_countires = loc_df['country'].unique().tolist()
    list_of_states = loc_df['subcountry'].unique().tolist()

    #     lat_long_city_df = pd.read_csv('worldcitiespop.txt', sep=",")
    #     country_names_df = pd.read_csv('countrynames.csv')
    #     country_position_df = pd.read_csv('countrypositions.csv')

    country_post_dic = defaultdict(list)
    state_post_dic = defaultdict(list)
    city_post_dic = defaultdict(list)

    for p in post_dic:
        post_unicode = unicode(post_dic[p], "utf-8")
        doc = nlp(post_unicode)
        valid_ner = ['GPE']
        for X in doc.ents:
            if X.label_ in valid_ner:
                loc = str(X.text.encode("utf-8"))
                try:
                    if loc in list_of_countires:
                        country_post_dic[loc].append({p: post_dic[p]})

                    elif loc in list_of_states:
                        state_post_dic[loc].append({p: post_dic[p]})

                    elif loc in list_of_cities:
                        city_post_dic[loc].append({p: post_dic[p]})

                        # print country

                except Exception as e:
                    pass

    # print country_post_dic.keys()
    # print state_post_dic.keys()
    # print city_post_dic.keys()
    return country_post_dic, state_post_dic, city_post_dic


def single_post_sentiment_score(post):
    afinn_sentiment_dic = afinn_sentiment_scores.affinn_sentiment_scores
    regex = joblib.load("sentiment_regex.pkl")
    sentiment_term_count_dic = {}
    text = re.sub('[^0-9a-zA-Z]+', ' ', post)
    found_terms = regex.findall(text)
    for fd in (found_terms):
        strip_fd = fd.strip().lower()
        if strip_fd in sentiment_term_count_dic:
            sentiment_term_count_dic[strip_fd] += 1
        else:
            sentiment_term_count_dic[strip_fd] = 1
    sentiment_score = 0
    for term in sentiment_term_count_dic:
        # print sentiment_term_count_dic[term], afinn_sentiment_dic[term]
        sentiment_score += sentiment_term_count_dic[term] * afinn_sentiment_dic[term]
    try:
        normalized_sentiment_score = float(sentiment_score) / float(len(found_terms))
    except:
        normalized_sentiment_score = 0

    # print normalized_sentiment_score
    return normalized_sentiment_score


def location_sentiment_score(country_post_dic, state_post_dic, city_post_dic):
    country_post_dic = country_post_dic
    state_post_dic = state_post_dic
    city_post_dic = city_post_dic

    country_sentiment_scores_dic = {}
    for country in country_post_dic:
        # print country
        country_posts = []
        for post in country_post_dic[country]:
            country_posts.append(post.values()[0])
        unique_posts = list(set(country_posts))

        country_sentiment_score = 0

        for up in unique_posts:
            country_sentiment_score += single_post_sentiment_score(up)

        country_sentiment_scores_dic[country] = float(country_sentiment_score) / float(len(unique_posts))

    state_sentiment_scores_dic = {}
    for state in state_post_dic:
        # print state
        state_posts = []
        for post in state_post_dic[state]:
            # print post
            state_posts.append(post.values()[0])
        unique_posts = list(set(state_posts))

        state_sentiment_score = 0
        for up in unique_posts:
            state_sentiment_score += single_post_sentiment_score(up)

        state_sentiment_scores_dic[state] = float(state_sentiment_score) / float(len(unique_posts))

    city_sentiment_scores_dic = {}
    for city in city_post_dic:
        # print city
        city_posts = []
        for post in city_post_dic[city]:
            city_posts.append(post.values()[0])
        unique_posts = list(set(city_posts))

        city_sentiment_score = 0
        for up in unique_posts:
            city_sentiment_score += single_post_sentiment_score(up)

        city_sentiment_scores_dic[city] = float(city_sentiment_score) / float(len(unique_posts))

    return country_sentiment_scores_dic, state_sentiment_scores_dic, city_sentiment_scores_dic


def single_post_drug_terms(post):
    exactMatch = joblib.load("drugs_exactMatch.pkl")
    drug_count_dic = {}
    text = re.sub('[^0-9a-zA-Z]+', ' ', post)
    found_drugs = exactMatch.findall(text)
    for fd in (found_drugs):
        strip_fd = fd.strip().lower()
        if strip_fd in drug_count_dic:
            drug_count_dic[strip_fd] += 1
        else:
            drug_count_dic[strip_fd] = 1
    return drug_count_dic


def single_post_recovery_terms(post):
    exactMatch = joblib.load("recovery_exactMatch.pkl")
    drug_count_dic = {}
    text = re.sub('[^0-9a-zA-Z]+', ' ', post)
    found_drugs = exactMatch.findall(text)
    for fd in (found_drugs):
        strip_fd = fd.strip().lower()
        if strip_fd in drug_count_dic:
            drug_count_dic[strip_fd] += 1
        else:
            drug_count_dic[strip_fd] = 1
    return drug_count_dic


def location_drug_recovery_terms(country_post_dic, state_post_dic, city_post_dic):
    country_post_dic = country_post_dic
    state_post_dic = state_post_dic
    city_post_dic = city_post_dic

    country_drug_terms_dic = {}
    country_recovery_terms_dic = {}
    for country in country_post_dic:
        # print country
        country_posts = []
        for post in country_post_dic[country]:
            country_posts.append(post.values()[0])
        unique_posts = list(set(country_posts))

        country_drug_terms = []
        country_recovery_terms = []
        for up in unique_posts:
            country_drug_terms.append(single_post_drug_terms(up))
            country_recovery_terms.append(single_post_recovery_terms(up))

        drug_temp_dic = defaultdict(int)
        for dic in country_drug_terms:
            for key in dic:
                drug_temp_dic[key] += dic[key]

        country_drug_terms_dic[country] = drug_temp_dic

        recovery_temp_dic = defaultdict(int)
        for dic in country_recovery_terms:
            for key in dic:
                recovery_temp_dic[key] += dic[key]

        country_recovery_terms_dic[country] = recovery_temp_dic

    state_drug_terms_dic = {}
    state_recovery_terms_dic = {}
    for state in state_post_dic:
        # print state
        state_posts = []
        for post in state_post_dic[state]:
            state_posts.append(post.values()[0])
        unique_posts = list(set(state_posts))

        state_drug_terms = []
        state_recovery_terms = []
        for up in unique_posts:
            state_drug_terms.append(single_post_drug_terms(up))
            state_recovery_terms.append(single_post_recovery_terms(up))

        drug_temp_dic = defaultdict(int)

        for dic in state_drug_terms:
            for key in dic:
                drug_temp_dic[key] += dic[key]
        state_drug_terms_dic[state] = drug_temp_dic

        recovery_temp_dic = defaultdict(int)

        for dic in state_recovery_terms:
            for key in dic:
                recovery_temp_dic[key] += dic[key]
        state_recovery_terms_dic[state] = recovery_temp_dic

    city_drug_terms_dic = {}
    city_recovery_terms_dic = {}
    for city in city_post_dic:
        city_posts = []
        for post in city_post_dic[city]:
            city_posts.append(post.values()[0])
        unique_posts = list(set(city_posts))

        city_drug_terms = []
        city_recovery_terms = []
        for up in unique_posts:
            city_drug_terms.append(single_post_drug_terms(up))
            city_recovery_terms.append(single_post_drug_terms(up))

        drug_temp_dic = defaultdict(int)

        for dic in city_drug_terms:
            for key in dic:
                drug_temp_dic[key] += dic[key]

        city_drug_terms_dic[city] = drug_temp_dic

        recovery_temp_dic = defaultdict(int)

        for dic in city_recovery_terms:
            for key in dic:
                recovery_temp_dic[key] += dic[key]

        city_recovery_terms_dic[city] = recovery_temp_dic

    return country_drug_terms_dic, country_recovery_terms_dic, state_drug_terms_dic, state_recovery_terms_dic, city_drug_terms_dic, city_recovery_terms_dic


def state_lat_long(state, country, country_short_name, lat_long_city_df, loc_df):
    # print lat_long_city_df.shape
    lat_long_city_df = lat_long_city_df[lat_long_city_df['Population'] > 200000]
    # print lat_long_city_df.shape
    cities = loc_df.loc[(loc_df['country'] == country) & (loc_df['subcountry'] == state)]['name'].unique().tolist()
    latitudes = []
    longitudes = []

    for c in cities:
        lat = lat_long_city_df.loc[
            (lat_long_city_df['City'] == c.lower()) & (lat_long_city_df['Country'] == country_short_name)]
        long = lat_long_city_df.loc[
            (lat_long_city_df['City'] == c.lower()) & (lat_long_city_df['Country'] == country_short_name)]
        if not lat.empty:
            latitudes.append(lat.loc[lat['Population'].idxmax()]['Latitude'])
            longitudes.append(long.loc[long['Population'].idxmax()]['Longitude'])

    # print latitudes
    # print longitudes
    return [np.mean(latitudes), np.mean(longitudes)]


def modify_number_of_topics(input_folder, output_folder, n):
    print 'Number of topics', n
    post_dic = create_post_dic(input_folder)
    user_posts_topic_modeling(post_dic, n, output_folder)


def primary_location(post_dic):
    posts = post_dic.values()
    pattern1 = "(\s|\.|^|\W)(i|we) live (in|at)"
    pattern2 = "(\s|\.|^|\W)(I am|I'm|we are|we're|(\s|\.|^|\W)Im) living (in|at)"
    pattern3 = "(\s|\.|^|\W)(I|we) have been living (in|at)"
    patterns = [pattern1, pattern2, pattern3]

    db = get_db()
    collection = db.location_term_to_raw_address

    location_dictionary = {}
    for p in posts:
        locations_in_posts = []
        sentence_split = nltk.sent_tokenize(p.decode('utf-8'))
        for s in sentence_split:
            if bool(re.search("|".join(patterns), s, re.IGNORECASE)):
                # post_unicode = unicode(s, "utf-8")
                post_unicode = s
                doc = nlp(post_unicode)
                valid_ner = ['GPE']
                locations_in_posts = []
                for X in doc.ents:
                    if X.label_ in valid_ner:
                        try:
                            loc = str(X.text.encode("utf-8"))
                            print loc
                            print s

                            loc_lower = loc.lower()
                            print "loc_lower", loc_lower

                            loc_info = collection.find_one({"location": str(loc_lower)})





                            if loc_info == None:
                                print "INSIDE NONE"
                                try:
                                    location = geolocator.geocode(loc_lower, addressdetails=True, timeout=10)
                                    if location == None:
                                        continue
                                    print location.raw['address']
                                    if location == None:
                                        continue
                                    locations_in_posts.append(location.raw['address'])
                                    collection.insert({"location": loc_lower, "address": location.raw['address']})
                                except Exception as e:
                                    print e
                            else:
                                locations_in_posts.append(loc_info['address'])
                        except Exception as e:
                            print e

                            # continue
        if len(locations_in_posts) > 0:
            print len(locations_in_posts)
            loc_idx = -1
            loc_len = 0
            for c, l in enumerate(locations_in_posts):
                loc_len_temp = len(l)
                if loc_len_temp > loc_len:
                    loc_len = loc_len_temp
                    loc_idx = c
            location_dictionary[c] = locations_in_posts[loc_idx]
            print location_dictionary
    if len(location_dictionary) > 1:
        loc_dic_key = 0
        loc_dic_key_len = 0
        for ld in location_dictionary:
            print location_dictionary[ld]
            ld_len = len(location_dictionary[ld])

            if ld_len > loc_dic_key_len:
                loc_dic_key_len = ld_len
                loc_dic_key = ld
        user_primary_location = location_dictionary[loc_dic_key]
    elif len(location_dictionary) == 0:
        return None
    else:
        print "location_dictionary", len(location_dictionary)
        print location_dictionary
        user_primary_location = location_dictionary[location_dictionary.keys()[0]]

    return user_primary_location


def user_age(post_dic):
    # I a,/'m/im 21 years/year old.
    pattern1 = "(I am|I'm|(\s|\.|^|\W)Im) (\d+) (years|year) old"

    # I am/'m/im male, 21 years/year old.
    # I am/'m/im a 21 years/year old guy.
    pattern2 = "(I am|I'm|(\s|\.|^|\W)Im) (([a-z]*)|([a-z]*)[^a-zA-Z\d\s]) (\d+) (years|year) old"

    # I 'll be/ will be/ ill be turning 21.
    pattern3 = "(i'll be|i will be|(\s|\.|^|\W)ill be) turning \d+"

    # I 'll / will/ ill turn 21.
    pattern4 = "(i'll turn|i will turn|(\s|\.|^|\W)ill turn) \d+"

    # I am a baseball player and am 21 years old.
    pattern5 = "(((?<!I))(\s|\.|^|\W)am) (([a-z]*)|([a-z]*)[^a-zA-Z\d\s]) (\d+) (years|year) old"

    patterns_years_old = [pattern1, pattern2]
    patterns_turning = [pattern3, pattern4]
    patterns_am = pattern5
    posts = post_dic.values()
    ages = []

    for p in posts:
        sentence_split = nltk.sent_tokenize(p.lower().decode('utf-8'))
        for s in sentence_split:
            if bool(re.search("|".join(patterns_years_old), s, re.IGNORECASE)):
                print s
                token_split = s.split()
                indexes = [i for i, x in enumerate(token_split) if x == 'years']
                if len(indexes) > 0:
                    for idx in indexes:
                        if re.sub(r'\W+', '', token_split[idx + 1].lower()) == "old":
                            age = token_split[idx - 1]
                            try:
                                age = int(re.sub(r'\W+', '', age))
                            except:
                                continue
                            print age
                            ages.append(age)
                            print ages
                else:
                    indexes = [i for i, x in enumerate(token_split) if x == 'year']
                    if len(indexes) > 0:
                        for idx in indexes:
                            if re.sub(r'\W+', '', token_split[idx + 1].lower()) == "old":
                                age = token_split[idx - 1]
                                try:
                                    age = int(re.sub(r'\W+', '', age))
                                except:
                                    continue
                                print age
                                ages.append(age)
                                print ages


            elif bool(re.search("|".join(patterns_turning), s, re.IGNORECASE)):
                print s
                token_split = s.split()
                indexes = [i for i, x in enumerate(token_split) if x == 'turning']
                if len(indexes) > 0:
                    for idx in indexes:
                        age = token_split[idx + 1]
                        age = int(re.sub(r'\W+', '', age))
                        print age
                        ages.append(age)
                        print ages


                else:
                    indexes = [i for i, x in enumerate(token_split) if x == 'turn']
                    for idx in indexes:
                        age = token_split[idx + 1]
                        age = int(re.sub(r'\W+', '', age))
                        print age
                        ages.append(age)
                        print ages

            elif bool(re.search(patterns_am, s, re.IGNORECASE)):
                print s
                token_split = s.split()
                indexes = [i for i, x in enumerate(token_split) if x == 'years']
                if len(indexes) > 0:
                    for idx in indexes:
                        if re.sub(r'\W+', '', token_split[idx + 1].lower()) == "old":
                            age = token_split[idx - 1]
                            age = int(re.sub(r'\W+', '', age))
                            print age
                            ages.append(age)
                            print ages
                else:
                    indexes = [i for i, x in enumerate(token_split) if x == 'year']
                    if len(indexes) > 0:
                        for idx in indexes:
                            if re.sub(r'\W+', '', token_split[idx + 1].lower()) == "old":
                                age = token_split[idx - 1]
                                age = int(re.sub(r'\W+', '', age))
                                print age
                                ages.append(age)
                                print ages
    if len(ages) > 0:
        user_age_identified = max(ages)
        if user_age_identified < 13:
            user_age_identified = None
        if len(ages) > 1:
            print ages
            print user_age_identified
            print True
    else:
        user_age_identified = None

    return user_age_identified


def map_visualization(location_matrix, output_folder):
    # loc_df = pd.DataFrame(location_matrix)
    # ['location','sentiment_score','drug_terms','recovery_terms','lat','long']
    loc_df = pd.DataFrame.from_records(location_matrix,
                                       columns=['location', 'sentiment_score', 'drug_terms', 'recovery_terms', 'lat',
                                                'long', 'primary_location'])
    # loc_df.loc[ (loc_df.sentiment_score < 0), 'color'] = 'red'
    # loc_df.loc[ (loc_df.sentiment_score > 0),'color'] = 'green'

    folium_map = folium.Map(tiles="CartoDB dark_matter")

    for row in loc_df.values.tolist():
        # print row
        color = 'grey'
        if row[1] > 0:
            color = 'green'
        elif row[1] < 0:
            color = 'red'

        if row[6] == 1:
            folium.Marker(location=[row[4], row[5]], popup=row[0] + "\n" + "sentiment score =" + str(row[1]) + "\n"
                                                           + "drug_terms: " + str(
                dict(row[2])) + "\n" + "recovery_terms: " + str(dict(row[3])),
                          icon=folium.Icon(icon='home', color=color)).add_to(folium_map)
        else:
            folium.Marker(location=[row[4], row[5]], popup=row[0] + "\n" + "sentiment score =" + str(row[1]) + "\n"
                                                           + "drug_terms: " + str(
                dict(row[2])) + "\n" + "recovery_terms: " + str(dict(row[3])), icon=folium.Icon(color=color)).add_to(
                folium_map)

    # folium_map.save("my_map.html")
    #  output_file(os.path.join(output_folder, "temporal_recovery_count.html"))
    folium_map.save(os.path.join(output_folder, "my_map.html"))


def plot_geo_data_2(country_post_dic, state_post_dic, city_post_dic, location_sentiment_values, location_drug_values,
                    location_recovery_values, user_primary_location, output_folder):
    db = get_db()

    pl_dic = {}
    print "user PRIMARY LOCATION", user_primary_location
    if user_primary_location is not None:
        if 'city' in user_primary_location:
            user_pl = user_primary_location['city']
            pl_dic['city'] = user_pl
        elif 'state' in user_primary_location:
            user_pl = user_primary_location['state']
            pl_dic['state'] = user_pl
        elif 'country' in user_primary_location:
            user_pl = user_primary_location['country']
            pl_dic['country'] = user_pl

    print "PL DIC", pl_dic

    city_collection = db.cities
    cities = city_collection.distinct("city")
    print cities

    state_collection = db.states
    states = state_collection.distinct("state")
    print states

    country_collection = db.countries
    countries = country_collection.distinct("country")
    print countries
    # print country_post_dic
    # print state_post_dic
    # print city_post_dic
    # print location_sentiment_values
    # print location_drug_values
    # print location_recovery_values


    locations = country_post_dic.keys() + state_post_dic.keys() + city_post_dic.keys()
    location_matrix = []

    for l in locations:
        primary_loc_exists = False
        print "here", l

        if l in cities:
            print "city found", l
            city_info = city_collection.find_one({"city": l})
            lat = city_info['latitude']
            long = city_info['longitude']

            if 'city' in pl_dic:
                if pl_dic['city'] == l:
                    print "PRIMARY LOCATION ", primary_loc_exists
                    primary_loc_exists = True
                    print "PRIMARY LOCATION ", primary_loc_exists


        elif l in states:
            print "state found", l
            state_info = state_collection.find_one({"state": l})
            lat = state_info['latitude']
            long = state_info['longitude']
            if 'state' in pl_dic:
                if pl_dic['state'] == l:
                    primary_loc_exists = True

        elif l in countries:
            print "country found", l
            country_info = country_collection.find_one({"country": l})
            lat = country_info['latitude']
            long = country_info['longitude']
            if 'country' in pl_dic:
                if pl_dic['country'] == l:
                    primary_loc_exists = True
        else:
            print "geopy"
            try:
                location = geolocator.geocode(l, addressdetails=True, timeout=10)
                if location == None:
                    continue
                print location.raw['address']
                if location == None:
                    continue
                raw_address = location.raw['address']
                print raw_address
                if 'city' in raw_address:
                    if pl_dic['city'] == raw_address['city']:
                        primary_loc_exists = True
                elif 'state' in raw_address:
                    if pl_dic['state'] == raw_address['state']:
                        primary_loc_exists = True
                elif 'country' in raw_address:
                    if pl_dic['country'] == raw_address['country']:
                        primary_loc_exists = True
                lat_long = geolocator.geocode(l, exactly_one=True)
                if lat_long == None:
                    continue
                else:
                    lat = lat_long.latitude
                    long = lat_long.longitude
            except:
                continue

        row = []
        row.append(l)
        row.append(location_sentiment_values[l])
        row.append(location_drug_values[l])
        row.append(location_recovery_values[l])
        row.append(lat)
        row.append(long)

        if primary_loc_exists:
            row.append(1)

        else:
            row.append(0)
        location_matrix.append(row)

    map_visualization(location_matrix, output_folder)


def main(folder, output_folder, classifier_folder):
    post_dic = create_post_dic(folder)
    user_name = folder.split(os.path.sep)[-1]

    # SENTIMENT ANALYSIS VISUALIZATION

    sentiment_score_dic = sentiment_score_of_each_post(post_dic)
    sentiment_data_frame = converting_data_to_pandas_df(sentiment_score_dic, post_dic)
    sentiment_line_plot(sentiment_data_frame, output_folder)
    no_of_positive_negative_neutral_posts(sentiment_data_frame, output_folder, user_name)
    geolocation_based_sentiment_analysis(post_dic)

    # TOPIC MODELING VISUALIZATION

    n = 10
    user_posts_topic_modeling(post_dic, n, output_folder)

    # DRUG TERM VISUALIZATION

    # stacked_drug_bar_chart(post_dic, output_folder)
    drug_bar_chart(post_dic, output_folder)
    drugs_data_frame = creating_drug_pandas_df(post_dic)
    temporal_drug_use(drugs_data_frame, output_folder)

    # RECOVERY TERM VISUALIZATION

    recovery_bar_chart(post_dic, output_folder)
    recovery_data_frame = creating_recovery_pandas_df(post_dic)
    temporal_recovery_use(recovery_data_frame, output_folder)
    # stacked_recovery_bar_chart(post_dic, output_folder)


    # Geolocation Visualization
    user_primary_location = primary_location(post_dic)
    print "USER PRIMARY LOCATION IS ", user_primary_location
    print output_folder

    if user_primary_location is not None:
        print "entering none if not none"
        filename = os.path.join(output_folder, "user_location.txt")
        with open(filename, "w") as fp:
            if "city" in user_primary_location:
                fp.write("City : " + user_primary_location['city'].encode('utf-8'))
                fp.write('\n')
            if "state" in user_primary_location:
                fp.write("State : " + user_primary_location['state'].encode('utf-8'))
                fp.write('\n')
            if "country" in user_primary_location:
                fp.write("Country : " + user_primary_location['country'].encode('utf-8'))
                fp.write('\n')

    country_post_dic, state_post_dic, city_post_dic = find_location_terms_in_text(post_dic)
    country_sentiment_scores_dic, state_sentiment_scores_dic, city_sentiment_scores_dic = location_sentiment_score(
        country_post_dic, state_post_dic, city_post_dic)
    country_drug_terms_dic, country_recovery_terms_dic, state_drug_terms_dic, state_recovery_terms_dic, city_drug_terms_dic, city_recovery_terms_dic = location_drug_recovery_terms(
        country_post_dic, state_post_dic, city_post_dic)
    location_sentiment_values = [country_sentiment_scores_dic, state_sentiment_scores_dic, city_sentiment_scores_dic]
    location_sentiment_values = dict(country_sentiment_scores_dic, **state_sentiment_scores_dic)
    location_sentiment_values = dict(location_sentiment_values, **city_sentiment_scores_dic)
    location_drug_recovery_values = [country_drug_terms_dic, country_recovery_terms_dic, state_drug_terms_dic,
                                     state_recovery_terms_dic, city_drug_terms_dic, city_recovery_terms_dic]
    location_drug_values = dict(country_drug_terms_dic, **state_drug_terms_dic)
    location_drug_values = dict(location_drug_values, **city_drug_terms_dic)
    location_recovery_values = dict(country_recovery_terms_dic, **state_recovery_terms_dic)
    location_recovery_values = dict(location_recovery_values, **city_recovery_terms_dic)
    plot_geo_data_2(country_post_dic, state_post_dic, city_post_dic, location_sentiment_values, location_drug_values,
                    location_recovery_values, user_primary_location, output_folder)

    # age identification
    drug_user_age = user_age(post_dic)
    print "USER AGE IS ", drug_user_age
    if user_age is not None and (drug_user_age >= 13) and (drug_user_age <= 80):
        filename = os.path.join(output_folder, "user_age.txt")
        with open(filename, "w") as fp:
            fp.write(str(drug_user_age))
            fp.write('\n')


def run():
    parser = OptionParser()
    parser.add_option('-i', '--input', dest='input_folder', help="input folder", type=str)
    parser.add_option('-o', '--output', dest='output_folder', help="output folder", type=str)
    parser.add_option('-c', '--classifier', dest='classifier', help="classifier folder", type=str)
    parser.add_option('-n', '--topics', dest='number_of_topics', help="Number of Topics", type=int)

    (options, args) = parser.parse_args()

    if options.number_of_topics != -1:
        modify_number_of_topics(options.input_folder, options.output_folder, options.number_of_topics)
        return

    classifier_folder = ""
    if options.classifier and options.classifier != "na":
        classifier_folder = options.classifier

    main(options.input_folder, options.output_folder, classifier_folder)


if __name__ == "__main__":
    # folder = "/Users/jhadeeptanshu/RecoveryInterventions/run_uploads/user_data"
    # output_folder = "../visualizations/1/"
    # main(folder, output_folder)

    run()
