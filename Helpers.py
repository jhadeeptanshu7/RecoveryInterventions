import os
import smtplib
import zipfile

FROM =  "recoveryinterventions123@gmail.com"
SUBJECT = "RecoveryIntervention Project Done"
PASSWORD = "kunalbhai"
BODY = "Hi %s, \n Project %s is done."


def send_email(recipient, body):

    TO = recipient if isinstance(recipient, list) else [recipient]
    TEXT = body

    # Prepare actual message
    message = """From: %s\nTo: %s\nSubject: %s\n\n%s
    """ % (FROM, ", ".join(TO), SUBJECT, TEXT)
    try:
        server = smtplib.SMTP("smtp.gmail.com", 587)
        server.ehlo()
        server.starttls()
        server.login(FROM, PASSWORD)
        server.sendmail(FROM, TO, message)
        server.close()
        print 'successfully sent the mail'

    except:
        print "failed to send mail"


def zipdir(path, ziph):
    for root, dirs, files in os.walk(path):
        for file in files:
            ziph.write(os.path.join(root, file))

if __name__ == "__main__":

    recipient = "djha@mail.sfsu.edu"
    subject = "RecoveryIntervention Project Done"
    body = BODY % ("DJ", "12345678912345678234567asdasd")
    send_email(recipient, body)
