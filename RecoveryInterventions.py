import os
import shutil

import datetime
import zipfile

from flask import Flask, render_template, request, send_from_directory, make_response
from flask_mongoengine import MongoEngine
from flask_assets import Environment, Bundle

from rq import Queue
from worker import conn
from Classification import fileHandler, unzip_folder, run_single_classification, OUTPUT_FOLDER


CLASSIFY = 'CLASSIFY'
TRAIN = 'TRAIN'
q = Queue(connection=conn, default_timeout=300000)

VISUALIZATION_FOLDER = os.path.dirname(__file__) + '/visualizations/'

UPLOAD_FOLDER = os.path.dirname(__file__) + '/run_uploads'
# ALLOWED_EXTENSIONS = set(['txt', 'pdf', 'png', 'jpg', 'jpeg', 'gif','m'])

app = Flask(__name__)

assets     = Environment(app)
assets.url = app.static_url_path
scss       = Bundle('index.scss', filters='pyscss', output='all.css')

app.config['MONGODB_SETTINGS'] = {
    'host': 'mongodb://localhost:27017/Recovery'
}
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


# GET Single existing
@app.route('/run-single')
def run_single():
    return render_template("run_pretrained_single.html")


@app.route('/run-single', methods=['POST'])
def run_single_user_classifier():
    # filename = os.path.join(app.config['UPLOAD_FOLDER'], request.form.get("filename"))
    file_name = request.form.get("filename")

    classifier_name = request.form.get("classifierName")
    classifier_folder = "na"

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

    output_folder = VISUALIZATION_FOLDER + str(project['id']) + "/"

    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    run_single_classification(input_folder, output_folder, project)

    return render_template("run_pretrained_single.html", project_id=project.id)


# GET Batch existing
@app.route('/run-batch')
def run_batch():
    return render_template("run_pretrained_batch.html")


@app.route('/run-batch', methods=['POST'])
def classifier_job():
    # filename = os.path.join(app.config['UPLOAD_FOLDER'], request.form.get("filename"))
    filename = request.form.get("filename")
    classifier_name = request.form.get("classifierName")
    classifier_folder = ""
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
    if not os.path.exists(VISUALIZATION_FOLDER + project_id + ".zip"):
        shutil.make_archive(VISUALIZATION_FOLDER + project_id, 'zip', VISUALIZATION_FOLDER + project_id)

    return send_from_directory(VISUALIZATION_FOLDER, project_id + ".zip")


@app.route('/results')
def get_results():
    return render_template('download.html')


@app.route('/results/<project>')
def get_visualization(project):
    return render_template("visualization.html", project_id=project)


@app.route('/results', methods=['POST'])
def post_results():
    project_id = request.form.get("project_id")
    if not project_id:
        return render_template('download.html')

    elif not os.path.exists(VISUALIZATION_FOLDER + project_id):
        return render_template('download.html', project_id=project_id)

    elif not os.path.exists(VISUALIZATION_FOLDER + project_id + ".zip"):
        shutil.make_archive(VISUALIZATION_FOLDER + project_id, 'zip', VISUALIZATION_FOLDER + project_id)

    return send_from_directory(VISUALIZATION_FOLDER, project_id + ".zip")


@app.route('/visualizations/<project>/<filename>')
def visualization_file(project, filename):
    return send_from_directory(VISUALIZATION_FOLDER + project, filename)


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


if __name__ == '__main__':
    app.run(threaded=True, debug=True)
