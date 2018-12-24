from collections import Counter
from sklearn.externals import joblib
import pymongo
import re
client = pymongo.MongoClient()
db = client.Recovery
regex = joblib.load("drug_regex.pkl")

recovery_user_count_dic={}
cursor = db["1500_copy_dataset"].find({"recovery":1},no_cursor_timeout=True)
recovery_users = cursor.count()
print recovery_users
for c,i in enumerate(cursor):
    # if c>10:
    #     break
    user = i['user']
    print c, i["user"]
    posts = i["posts"]
    for p in posts:
        text = re.sub('[^0-9a-zA-Z]+', ' ', p)
        found_drugs = regex.findall(text)
        for fd in (found_drugs):
            strip_fd = fd.strip().lower()
            if strip_fd in recovery_user_count_dic:
                recovery_user_count_dic[strip_fd].add(user)
            else:
                recovery_user_count_dic[strip_fd] ={user}

recovery_user_count_dic_frequencies={}
for u in recovery_user_count_dic:
    recovery_user_count_dic_frequencies[u] = len(recovery_user_count_dic[u])
counted_drug_terms_recovery = Counter(recovery_user_count_dic_frequencies)
print counted_drug_terms_recovery.most_common()
# for d in counted_drug_terms_recovery.most_common():
#     print d[0], float(d[1]/recovery_users)



non_recovery_user_count_dic={}
cursor = db["1500_copy_dataset"].find({"recovery":0},no_cursor_timeout=True)
non_recovery_users = cursor.count()
print non_recovery_users
for c,i in enumerate(cursor):
    # if c>10:
    #     break
    user = i['user']
    print c, i["user"]
    posts = i["posts"]
    for p in posts:
        text = re.sub('[^0-9a-zA-Z]+', ' ', p)
        found_drugs = regex.findall(text)
        for fd in (found_drugs):
            strip_fd = fd.strip().lower()
            if strip_fd in non_recovery_user_count_dic:
                non_recovery_user_count_dic[strip_fd].add(user)
            else:
                non_recovery_user_count_dic[strip_fd] ={user}

non_recovery_user_count_dic_frequencies={}
for u in non_recovery_user_count_dic:
    non_recovery_user_count_dic_frequencies[u] = len(non_recovery_user_count_dic[u])
counted_drug_terms_drugs = Counter(non_recovery_user_count_dic_frequencies)
print counted_drug_terms_drugs.most_common()

print "RECOVERY"
for d in counted_drug_terms_recovery.most_common(10):
    print d[0],d[1], float(d[1])/float(recovery_users)
print "**************************************"
print "NON RECOVERY"
for d in counted_drug_terms_drugs.most_common(10):
    print d[0], float(d[1])/float(non_recovery_users)