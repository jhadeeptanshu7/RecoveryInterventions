# /Users/jhadeeptanshu/RecoveryInterventions/train_uploads/
#
# 5bfa2995473c8923db51e0b2
#
# 5bf8c2da473c89cfb14d63d2
#
# python TrainClassifier.py -f /Users/jhadeeptanshu/RecoveryInterventions/train_uploads/ -p bfa2995473c8923db51e0b2 -u bf8c2da473c89cfb14d63d2
#
#
# a = "hello %s, %s"
# print a % ("aji", "asd")
import os
import zipfile


def zipdir(path, ziph):
    for root, dirs, files in os.walk(path):
        for file in files:
            ziph.write(os.path.join(root, file))
if __name__ == '__main__':
    if not os.path.isfile("visualizations/5bfa2fd3473c8927971c0c74.zip"):
        zipf = zipfile.ZipFile('visualizations/5bfa2fd3473c8927971c0c74.zip', 'w', zipfile.ZIP_DEFLATED)
        zipdir("visualizations/5bfa2fd3473c8927971c0c74/", zipf)
        zipf.close()
        print "Zip created"
    else:
        print "Already created"

