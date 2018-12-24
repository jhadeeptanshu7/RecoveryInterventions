drug_subreddits = ["cocaine","Drugs","trees","researchchemicals",
                   "Nootropics","trees","DrugNerds","RCSources","tripreports",
                   "opiates","dxm","LSD","meth","microdosing","StackAdvice",
                   "DrugCombos","dissociatives","TeenageDrugs","drugscience"
                   "Stims","MDMA","benzodiazepines","kratom","electronic_cigarette",
                   "phenibut","TripSit","KratomKorner","AskDrugNerds","Tianeptine",
                   "glassine","Petioles","eldertrees","randomactsofkratom","vaporents",
                   "electronic_cigarette","1P_LSD","Shroomers","Nbome","Psychedelics",
                   "Vyvanse","afinil","DMT","PsychonautReadingClub","pregabalin",
                   "MagicMushrooms","PsychedelicReddit","Ecstasy","DopeFiend",
                   "PsilocybinMushrooms","SEXONDRUGS","highdeas","DankNation",
                   "uktrees","sporetraders","TripTales","eurodrugsocial",
                   "TODispensaries","Triptongue","StonerProTips","PsychedelicMedicine",
                   "Tramadol","Deschloroketamine","Methoxetamine","NovelDissos",
                   "DPH","onadderall","drunk","OutlandishAlcoholics",
                   "CodeineCowboys","drugscience","Deliriants","erowid","DrugCombos","HeroinHighway",
                   "dissociatives","KetKatPack","Ceretropic","PEDs","CBD",
                   "treedibles","GCMSresults","researchchemsamerica","ecigclassifieds",
                   "nootropic_deals","RChemsTradingPost","mescaline","StonerPhilosophy",
                   "Ayahuasca","drugsmart","ObscureDrugs","drugscirclejerk",
                   "TeenageDrugs","1p_LSD_Sources","druggardening","hydroponics",
                   "TheDrugDare","DippingTobacco","blotter","PsychedelicStudies",
                   "ModafinilCat","PoppyTea","mdmatherapy","RationalPsychonaut",
                   "mdmatherapy","RationalPsychonaut","psychopharmacology",
                   "StratteraRx","dnp","AskStims","Eticyclidone","drugswtf",
                   "alcohol","Ephenidine","Cigarettes","tryptonaut",
                   "LSA","opiatescirclejerk","saplings","CannabisExtracts",
                   "Etizolam","modafinil_talk",
                   "shrooms","Psychonaut","askdrugs","AddyUp"]

recovery_subreddits = ["OpiatesRecovery","AlAnon","REDDITORSINRECOVERY",
                    "Methadone","suboxone","secularsobriety","dryalcoholics",
                    "AtheistTwelveSteppers","alcoholism","Sober","Ibogaine",
                    "BaclofenForAlcoholism","NarcoticsAnonymous",
                    "stopdrinking","leaves","stopspeeding","buprenorphine",
                    "addiction","unspun","AABeyondBelief","AlcoholStopsNow",
                    "benzorecovery","quittingkratom","ResearchRecovery",
                    "cripplingalcoholism","Opiatewithdrawal","alcoholicsanonymous",
                    "stopsmoking","naranon","stopdrinking"]

drug_acquisition = ["ResearchMarkets","ResearchVendors","DreamMarketplace"
                      "DarkNetMarkets","DarkNetMarketsNoobs","DNMParanoia"]

# print len(drug_subreddits)
# print len(recovery_subreddits)
#
# import pymongo
# client = pymongo.MongoClient()
# db = client.reddit
# collection =  db.subreddit_description_type
# cursor = collection.find({"subreddit_type":1},no_cursor_timeout=True)
#
# for i in cursor:
#     if i["subreddit"] in drug_subreddits:
#         lposts = len(i["posts"])
#         if lposts < 3:
#             print i["subreddit"]