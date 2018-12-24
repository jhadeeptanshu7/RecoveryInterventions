import pandas as pd
import unidecode

loc_df = pd.read_csv('/Users/jhadeeptanshu/WebAppVisualizations/sentiment_analysis/world-cities_csv.csv')
states = loc_df['subcountry'].unique()

city_df = pd.read_csv('worldcitiespop.txt', sep=",")
print city_df.head()


for s in states:
    print s
    accent_cities = (loc_df.loc[loc_df['subcountry'] == s]['name']).tolist()
    unaccented_cities = []
    for ac in accent_cities:
        try:
            ua_city = unidecode.unidecode(ac)
            unaccented_cities.append(ua_city)
        except:
            pass

    for city in accent_cities:
        print city
        print (city_df.loc[city_df['AccentCity'] == city])
        print (city_df.loc[city_df['AccentCity'] == city])

