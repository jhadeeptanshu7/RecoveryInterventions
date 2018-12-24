import datetime
import logging
import zipfile
import os
import pymongo
from bson.objectid import ObjectId

from RecoveryInterventions import CLASSIFY,TRAIN

VISUALIZATION_FOLDER = os.path.dirname(__file__) + '/visualizations/'
logging.basicConfig(filename='classification.log', level=logging.DEBUG)
OUTPUT_FOLDER = os.path.dirname(__file__) + '/output'
UPLOAD_FOLDER = os.path.dirname(__file__) + '/run_uploads/'




client = pymongo.MongoClient()
db = client.Recovery

def fileHandler(project_id):

    project = db.projects.find_one({'_id': ObjectId(project_id)})

    if not project:
        return

    project = Project(project['_id'], project['file'], project['user_id'])
    file = project.filename

    logging.info(str(datetime.datetime.now()) + ': ' + file)
    zip_ref = zipfile.ZipFile(UPLOAD_FOLDER + file, 'r')
    extracted = zip_ref.namelist()
    # print extracted[0]
    zip_ref.extractall(OUTPUT_FOLDER)
    zip_ref.close()

    if project.job_type == CLASSIFY:
        os.system("python /Users/jhadeeptanshu/RecoveryInterventions/Classification.py -f " + str(extracted[0]) + " -p " + str(project.id) + " -u " + project.user_id)
    elif project.job_type == TRAIN:
        os.system("python /Users/jhadeeptanshu/RecoveryInterventions/TrainClassifier.py -f " + str(extracted[0]) + " -p " + str(project.id) + " -u " + project.user_id)
    # doWork(extracted[0], project)