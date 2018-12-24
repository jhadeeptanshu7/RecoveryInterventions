import pymongo
import re
from sklearn.externals import joblib
client = pymongo.MongoClient()
db = client.reddit
collection = db.drug_slang
collection3 = db.user_drug_stats
cursor = collection.find()

drugs = []
drug_class_dic={}
for i in cursor:
    for k in i:
        if k=='' or k == '_id':
            continue
        for d in i[k]:
            drugs.append(d)
            drug_class_dic[d] = k
            # print drug_class_dic
exactMatch = re.compile(r'\b%s\b' % '\\b|\\b'.join(drugs), flags=re.IGNORECASE)
joblib.dump(exactMatch,'drugs_exacMatch.pkl')


# reg_drugs = []
# one_char_drugs=[]
# for d in drugs:
#     # if len(d) == 1:
#     #     continue
#     #     one_char_drugs.append(d)
#     # print "\b"+d+"\b"
#     reg_drugs.append(" "+d+" ")
#     # reg_drugs.append(d+ "\b")
#     reg_drugs.append(d+" ")
#     reg_drugs.append(" "+d)
#
# regex_text = "|".join(reg_drugs)
# print regex_text
# regex = re.compile(regex_text,flags=re.I )
# print regex
# joblib.dump(regex,"drug_regex.pkl")