import os
import shutil
import zipfile

from flask import Flask,flash, render_template, request, redirect, url_for, send_from_directory, make_response, send_file
from werkzeug.utils import secure_filename
from flask_login import LoginManager, login_user, UserMixin, current_user, login_required, logout_user
from flask_mongoengine import MongoEngine
from rq import Queue
from worker import conn
import logging
from Classification import fileHandler, unzip_folder, run_trained_classification_single, OUTPUT_FOLDER
from Helpers import zipdir

CLASSIFY = 'CLASSIFY'
TRAIN = 'TRAIN'
q = Queue(connection=conn, default_timeout=300000)

VISUALIZATION_FOLDER = os.path.dirname(__file__) + '/visualizations/'

UPLOAD_FOLDER = os.path.dirname(__file__) + '/run_uploads'
# ALLOWED_EXTENSIONS = set(['txt', 'pdf', 'png', 'jpg', 'jpeg', 'gif','m'])

app = Flask(__name__)

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
    # name = db.StringField()
    file = db.StringField()
    job_id = db.StringField()
    job_type = db.StringField()
    job_status = db.StringField()
    user_email = db.StringField()


@app.route('/')
def hello_world():

    return render_template("sample.html")


# GET Single existing
@app.route('/run_single')
def run_single():
    return render_template("run_pretrained_single.html")


# GET Batch existing
@app.route('/run_batch')
def run_batch():
    return render_template("run_pretrained_batch.html")


@app.route('/train')
def train_classifier():
    return render_template("train_classifier.html")


@app.route('/uploadfile', methods=['POST'])
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

            print ('File {file.filename} has been uploaded successfully')
    else:
        print('Chunk {current_chunk + 1} of {total_chunks} ' + 'for file {file.filename} complete')

    return make_response(("Chunk upload successful", 200))


@app.route('/run', methods=['POST'])
def classifier_job():
    project_name = request.form.get("project_name")
    # filename = os.path.join(app.config['UPLOAD_FOLDER'], request.form.get("filename"))
    filename = request.form.get("filename")

    project = Project(filename, '-1', CLASSIFY, "-1", "N/A").save()

    job = q.enqueue_call(func=fileHandler, args=(project.id,), result_ttl=5000, timeout=300000)
    job_id = str(job.get_id())

    Project.objects(id=project.id).update_one(set__job_id=job_id)

    return render_template("run_pretrained_classification.html",
                           submission_successful=True,
                           project_id=project.id,
                           job_id=job_id)


@app.route('/train', methods=['POST'])
def train_classifier_job():
    project_name = request.form.get("project_name")
    # filename = os.path.join(app.config['UPLOAD_FOLDER'], request.form.get("filename"))
    filename = request.form.get("filename")

    project = Project(filename, '-1', TRAIN, "-1", "N/A").save()

    job = q.enqueue_call(func=fileHandler, args=(project.id,), result_ttl=5000)
    job_id = str(job.get_id())

    Project.objects(id=project.id).update_one(set__job_id=job_id)

    return render_template("train_classifier.html",
                           submission_successful=True,
                           project_id=project.id,
                           job_id=job_id)


@app.route('/project')
def getProjects():

    projects = Project.objects(user_id=str(current_user.id))
    # projects = [p for p in projects if p.job_status == "1"]
    return render_template("projects.html", projects=projects)


@app.route('/project/<project_id>')
def getProject(project_id):

    project = Project.objects(id=str(project_id))[0]
    return render_template("results_system.html", project=project)


@app.route('/project/<project_id>/system-level/<filename>')
def getSysImg(project_id, filename):

    dir = VISUALIZATION_FOLDER + project_id + '/system_level/'

    return send_from_directory(dir, filename)


@app.route('/search/<project_id>/<user>')
def searchUser(project_id, user):

    user = DrugUser.objects(user=str(user), project_id=str(project_id)).first()
    project = Project.objects(id=str(project_id))[0]

    path = "user_level"
    path = path + "/recovery_users/" if user["recovery"] else path + "/non_recovery_users/"
    path += user["user"]

    return render_template("search.html", project=project,
                           path=path)


@app.route('/search/<project_id>/user_level/<recovery>/<user_name>/<filename>')
def getUsrImg(project_id, user_name, recovery, filename):

    dir = VISUALIZATION_FOLDER + project_id + '/user_level/' + recovery + '/' + user_name + "/"

    return send_from_directory(dir, filename)


@app.route('/download/<project_id>')
def get_zip_file(project_id):
    if not os.path.exists(VISUALIZATION_FOLDER + project_id + ".zip"):
        shutil.make_archive(VISUALIZATION_FOLDER + project_id, 'zip', VISUALIZATION_FOLDER + project_id)

    return send_from_directory(VISUALIZATION_FOLDER, project_id + ".zip")


@app.route('/results')
def get_results():
    return render_template('download.html')


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

# @app.route('/run', methods=['POST'])
# def upload():
#     if request.method == 'POST':
#         # check if the post request has the file part
#         if 'file' not in request.files:
#             flash('No file part')
#             return redirect(request.url)
#         file = request.files['file']
#         # if user does not select file, browser also
#         # submit an empty part without filename
#         if file.filename == '':
#             flash('No selected file')
#             return redirect(request.url)
#         if file:
#             filename = secure_filename(file.filename)
#             file.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))
#
#             # Worker entry
#             job = q.enqueue_call(
#                 func=mytest, args=(filename,), result_ttl=5000
#             )
#
#             print(job.get_id())
#
#             return redirect('/run')
#
#     return redirect('/run')


# @app.route("/results/<job_key>", methods=['GET'])
# def get_results(job_key):
#
#     job = Job.fetch(job_key, connection=conn)
#
#     if job.is_finished:
#         result = Result.query.filter_by(id=job.result).first()
#         results = sorted(
#             result.result_no_stop_words.items(),
#             key=operator.itemgetter(1),
#             reverse=True
#         )[:10]
#         return jsonify(results)
#     else:
#         return "Nay!", 202


@app.route('/uploads/<filename>')
def uploaded_file(filename):
    return send_from_directory(app.config['UPLOAD_FOLDER'],
                               filename)


@app.route('/run_single', methods=['POST'])
def run_single_user_classifier():
    # filename = os.path.join(app.config['UPLOAD_FOLDER'], request.form.get("filename"))
    filename = request.form.get("filename")
    project = Project(filename, "-1", CLASSIFY, "-1", "N/A").save()

    input_folder = OUTPUT_FOLDER + "/" + unzip_folder(UPLOAD_FOLDER + "/" + filename)
    output_folder = VISUALIZATION_FOLDER + str(project['id']) + "/"

    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    run_trained_classification_single(input_folder, output_folder)

    return render_template("run_pretrained_single.html", project_id=project.id)


@app.route('/visualizations/<project>/<filename>')
def visualization_file(project, filename):
    return send_from_directory(VISUALIZATION_FOLDER + project, filename)


@app.route('/download')
def download_project():
    return render_template("download.html")


if __name__ == '__main__':
    app.run(threaded=True, debug=True)
