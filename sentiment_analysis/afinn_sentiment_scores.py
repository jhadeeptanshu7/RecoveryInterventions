import re
from sklearn.externals import joblib

affinn_sentiment_scores = {}

F = open("AFINN-111.txt","r")
for l in F.readlines():
    affinn_sentiment_scores[l.split("\t")[0]] = int(l.split("\t")[1])


reg_sentiment_terms = []

for rt in affinn_sentiment_scores.keys():
    reg_sentiment_terms.append(" "+rt+" ")
    reg_sentiment_terms.append(" "+rt)
    reg_sentiment_terms.append(rt+" ")
# print reg_drugs
# print len(one_char_drugs)
# for d in one_char_drugs:
#     print d
regex_text = "|".join(reg_sentiment_terms)
# print regex_text
regex = re.compile(regex_text,flags=re.I )
# print regex
joblib.dump(regex,"sentiment_regex.pkl")


