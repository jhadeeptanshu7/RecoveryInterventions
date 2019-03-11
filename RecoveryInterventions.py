import json
import os
import shutil

import datetime
import zipfile

from flask import Flask, render_template, request, send_from_directory, make_response, redirect, url_for
from flask_mongoengine import MongoEngine

from rq import Queue
from worker import conn
from Classification import fileHandler, unzip_folder, run_single_classification, OUTPUT_FOLDER, \
    get_classification_result, modify_number_of_topics_helper, get_user_age, get_user_location


CLASSIFY = 'CLASSIFY'
TRAIN = 'TRAIN'
q = Queue(connection=conn, default_timeout=300000)

app = Flask(__name__)

app.environment = "DEV"
# app.environment = "PROD"

if app.environment == "DEV":
    app.config['MONGODB_SETTINGS'] = {'host': 'mongodb://localhost:27017/Recovery'}
else:
    app.config['MONGODB_SETTINGS'] = {'host':"mongodb://recovery:interventions@localhost:27017/recoveryi?authMechanism=SCRAM-SHA-256"}


VISUALIZATION_FOLDER = os.path.join(app.root_path, 'visualizations')
UPLOAD_FOLDER = os.path.join(app.root_path, 'run_uploads')

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['SECRET_KEY'] = 'deeptanshu'

db = MongoEngine(app)


class DrugUser(db.Document):
    meta       = {'collection': 'drug_users'}
    user       = db.StringField()
    project_id = db.StringField()
    recovery   = db.StringField()
    posts      = db.StringField()


class Project(db.Document):
    meta = {'collection': 'projects'}
    file = db.StringField()
    job_id = db.StringField()
    job_type = db.StringField()
    job_status = db.StringField()
    classifier = db.StringField()


@app.route('/')
def hello_world():
    return render_template("sample.html")


@app.route('/run-single-reddit')
def run_reddit_single():
    return render_template("run_pretrained_single.html")


@app.route('/run-single-reddit', methods=['POST'])
def run_single_user_reddit_classifier():
    # filename = os.path.join(app.config['UPLOAD_FOLDER'], request.form.get("filename"))
    file_name = request.form.get("filename")

    classifier_name = request.form.get("classifierName")
    classifier_folder = os.path.join(app.root_path, "reddit_classifier")

    if classifier_name:
        zip_ref = zipfile.ZipFile(os.path.join(UPLOAD_FOLDER, classifier_name), 'r')
        extracted = zip_ref.namelist()
        zip_ref.extractall(os.path.join(OUTPUT_FOLDER, classifier_name.replace(".zip", "")))
        zip_ref.close()

        classifier_folder = os.path.join(OUTPUT_FOLDER, classifier_name).replace(".zip", "")

    project = Project(file_name, "-1", CLASSIFY, "-1", classifier_folder).save()

    file_path = os.path.join(UPLOAD_FOLDER, file_name)
    unzipped_folder_name = unzip_folder(file_path)
    input_folder = os.path.join(OUTPUT_FOLDER, file_name).replace(".zip", "")
    os.rename(os.path.join(OUTPUT_FOLDER, unzipped_folder_name), input_folder)

    output_folder = os.path.join(VISUALIZATION_FOLDER, str(project['id']))

    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    run_single_classification(input_folder, output_folder, project)

    return json.dumps(project.id)


@app.route('/run-single-twitter')
def run_twitter_single():
    return render_template("run_pretrained_single.html")


@app.route('/run-single-twitter', methods=['POST'])
def run_single_user_twitter_classifier():
    # filename = os.path.join(app.config['UPLOAD_FOLDER'], request.form.get("filename"))
    file_name = request.form.get("filename")

    classifier_name = request.form.get("classifierName")
    classifier_folder = os.path.join(app.root_path, "twitter_classifier")

    if classifier_name:
        zip_ref = zipfile.ZipFile(os.path.join(UPLOAD_FOLDER, classifier_name), 'r')
        extracted = zip_ref.namelist()
        zip_ref.extractall(os.path.join(OUTPUT_FOLDER, classifier_name.replace(".zip", "")))
        zip_ref.close()

        classifier_folder = os.path.join(OUTPUT_FOLDER, classifier_name).replace(".zip", "")

    project = Project(file_name, "-1", CLASSIFY, "-1", classifier_folder).save()

    file_path = os.path.join(UPLOAD_FOLDER, file_name)
    unzipped_folder_name = unzip_folder(file_path)
    input_folder = os.path.join(OUTPUT_FOLDER, file_name).replace(".zip", "")
    os.rename(os.path.join(OUTPUT_FOLDER, unzipped_folder_name), input_folder)

    output_folder = os.path.join(VISUALIZATION_FOLDER, str(project['id']))

    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    run_single_classification(input_folder, output_folder, project)

    return json.dumps(project.id)



# GET Batch existing
@app.route('/run-batch-reddit')
def run_reddit_batch():
    return render_template("run_pretrained_batch.html")


@app.route('/run-batch-reddit', methods=['POST'])
def reddit_classifier_job():
    # filename = os.path.join(app.config['UPLOAD_FOLDER'], request.form.get("filename"))
    filename = request.form.get("filename")
    classifier_name = request.form.get("classifierName")
    classifier_folder = os.path.join(app.root_path, "reddit_classifier")
    if classifier_name:
        zip_ref = zipfile.ZipFile(os.path.join(UPLOAD_FOLDER, classifier_name), 'r')
        zip_ref.extractall(os.path.join(OUTPUT_FOLDER, classifier_name.replace(".zip", "")))
        zip_ref.close()

        classifier_folder = os.path.join(OUTPUT_FOLDER, classifier_name).replace(".zip", "")

    project = Project(filename, '-1', CLASSIFY, "-1", classifier_folder).save()

    job = q.enqueue_call(func=fileHandler, args=(project.id,), result_ttl=5000, timeout=300000)
    job_id = str(job.get_id())

    Project.objects(id=project.id).update_one(set__job_id=job_id)

    return render_template("run_pretrained_batch.html",
                           submission_successful=True,
                           project_id=project.id,
                           job_id=job_id)


# GET Batch existing
@app.route('/run-batch-twitter')
def run_twitter_batch():
    return render_template("run_pretrained_batch.html")


@app.route('/run-batch-twitter', methods=['POST'])
def twitter_classifier_job():
    # filename = os.path.join(app.config['UPLOAD_FOLDER'], request.form.get("filename"))
    filename = request.form.get("filename")
    classifier_name = request.form.get("classifierName")
    classifier_folder = os.path.join(app.root_path, "twitter_classifier")
    if classifier_name:
        zip_ref = zipfile.ZipFile(os.path.join(UPLOAD_FOLDER, classifier_name), 'r')
        zip_ref.extractall(os.path.join(OUTPUT_FOLDER, classifier_name.replace(".zip", "")))
        zip_ref.close()

        classifier_folder = os.path.join(OUTPUT_FOLDER, classifier_name).replace(".zip", "")

    project = Project(filename, '-1', CLASSIFY, "-1", classifier_folder).save()

    job = q.enqueue_call(func=fileHandler, args=(project.id,), result_ttl=5000, timeout=300000)
    job_id = str(job.get_id())

    Project.objects(id=project.id).update_one(set__job_id=job_id)

    return render_template("run_pretrained_batch.html",
                           submission_successful=True,
                           project_id=project.id,
                           job_id=job_id)


@app.route('/train')
def train_classifier():
    return render_template("train_classifier.html")


@app.route('/train', methods=['POST'])
def train_classifier_job():
    # filename = os.path.join(app.config['UPLOAD_FOLDER'], request.form.get("filename"))
    filename = request.form.get("filename")
    classifier_folder = ""
    project = Project(filename, '-1', TRAIN, "-1", classifier_folder).save()

    job = q.enqueue_call(func=fileHandler, args=(project.id,), result_ttl=5000)
    job_id = str(job.get_id())

    Project.objects(id=project.id).update_one(set__job_id=job_id)

    return render_template("train_classifier.html",
                           submission_successful=True,
                           project_id=project.id,
                           job_id=job_id)


@app.route('/download/<project_id>')
def get_zip_file(project_id):
    if not os.path.exists(os.path.join(VISUALIZATION_FOLDER, project_id) + ".zip"):
        shutil.make_archive(os.path.join(VISUALIZATION_FOLDER, project_id), 'zip', os.path.join(VISUALIZATION_FOLDER, project_id))

    return send_from_directory(VISUALIZATION_FOLDER, project_id + ".zip")


@app.route('/results')
def get_results():
    return render_template('download.html')


@app.route('/results/<project>')
def get_visualization(project):
    root_visualization_folder = os.path.join(VISUALIZATION_FOLDER, project)

    return render_template("visualization_v2.html", project_id=project,
                           recovery_intervention=get_classification_result(root_visualization_folder),
                           user_location=get_user_location(root_visualization_folder),
                           user_age=get_user_age(root_visualization_folder))


@app.route('/results', methods=['POST'])
def post_results():
    project_id = request.form.get("project_id")
    if not project_id:
        return render_template('download.html')

    elif not os.path.exists(os.path.join(VISUALIZATION_FOLDER, project_id)):
        return render_template('download.html', project_id=project_id)

    elif not os.path.exists(os.path.join(VISUALIZATION_FOLDER, project_id) + ".zip"):
        shutil.make_archive(os.path.join(VISUALIZATION_FOLDER, project_id), 'zip', os.path.join(VISUALIZATION_FOLDER, project_id))

    return send_from_directory(VISUALIZATION_FOLDER, project_id + ".zip")


@app.route('/visualizations/<project>/<filename>')
def visualization_file(project, filename):
    return send_from_directory(os.path.join(VISUALIZATION_FOLDER, project), filename, cache_timeout=0)


@app.route('/upload-data', methods=['POST'])
def upload():
    file = request.files['file']
    save_path = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
    current_chunk = int(request.form['dzchunkindex'])
    # If the file already exists it's ok if we are appending to it,
    # but not if it's new file that would overwrite the existing one
    if os.path.exists(save_path) and current_chunk == 0:
        # 400 and 500s will tell dropzone that an error occurred and show an error
        return make_response(('File already exists', 400))

    try:
        with open(save_path, 'ab') as f:
            f.seek(int(request.form['dzchunkbyteoffset']))
            f.write(file.stream.read())
    except OSError:
        # log.exception will include the traceback so we can see what's wrong
        print ('Could not write to file')
        return make_response(("Not sure why,"
                              " but we couldn't write the file to disk", 500))

    total_chunks = int(request.form['dztotalchunkcount'])

    if current_chunk + 1 == total_chunks:
        # This was the last chunk, the file should be complete and the size we expect
        if os.path.getsize(save_path) != int(request.form['dztotalfilesize']):
            print ("File {file.filename} was completed, " +
                      "but has a size mismatch." +
                      "Was {os.path.getsize(save_path)} but we" +
                      " expected {request.form['dztotalfilesize']} ")
            return make_response(('Size mismatch', 500))

    else:
        print('Chunk {current_chunk + 1} of {total_chunks} ' + 'for file {file.filename} complete')

    new_name = str(datetime.datetime.now().time()).replace("/", "-") + ".zip"
    renamed_path = os.path.join(app.config['UPLOAD_FOLDER'], new_name)
    os.rename(save_path, renamed_path)
    return make_response((new_name, 200))


@app.route('/upload-classifier', methods=['POST'])
def upload_classifier():
    file = request.files['file']

    save_path = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
    current_chunk = int(request.form['dzchunkindex'])

    # If the file already exists it's ok if we are appending to it,
    # but not if it's new file that would overwrite the existing one
    if os.path.exists(save_path) and current_chunk == 0:
        # 400 and 500s will tell dropzone that an error occurred and show an error
        return make_response(('File already exists', 400))

    try:
        with open(save_path, 'ab') as f:
            f.seek(int(request.form['dzchunkbyteoffset']))
            f.write(file.stream.read())
    except OSError:
        # log.exception will include the traceback so we can see what's wrong
        print ('Could not write to file')
        return make_response(("Not sure why,"
                              " but we couldn't write the file to disk", 500))

    total_chunks = int(request.form['dztotalchunkcount'])

    if current_chunk + 1 == total_chunks:
        # This was the last chunk, the file should be complete and the size we expect
        if os.path.getsize(save_path) != int(request.form['dztotalfilesize']):
            print ("File {file.filename} was completed, " +
                      "but has a size mismatch." +
                      "Was {os.path.getsize(save_path)} but we" +
                      " expected {request.form['dztotalfilesize']} ")
            return make_response(('Size mismatch', 500))

    else:
        print('Chunk {current_chunk + 1} of {total_chunks} ' + 'for file {file.filename} complete')

    new_name = str(datetime.datetime.now().time()).replace("/", "-") + ".zip"
    renamed_path = os.path.join(app.config['UPLOAD_FOLDER'], new_name)
    os.rename(save_path, renamed_path)
    return make_response((new_name, 200))


@app.route('/contact')
def get_contact():
    return render_template('contact.html')


@app.route('/citation')
def get_citation():
    return render_template('citation.html')


@app.route('/tutorial')
def get_tutorial():
    return render_template('tutorial.html')


@app.route('/results/<project>', methods=['POST'])
def modify_number_of_topics(project):
    number_of_topics = int(request.form.get('numberOfTopic'))
    if number_of_topics > 0:
        modify_number_of_topics_helper(project, number_of_topics)

    root_visualization_folder = os.path.join(VISUALIZATION_FOLDER, project)

    return render_template("visualization_v2.html", project_id=project,
                           recovery_intervention=get_classification_result(root_visualization_folder),
                           user_location=get_user_location(root_visualization_folder),
                           user_age=get_user_age(root_visualization_folder))


if __name__ == '__main__':
    app.run(threaded=True, debug=True)
