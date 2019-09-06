import matplotlib
matplotlib.use('Agg')

import osmnx as ox
import pandas as pd
import numpy as np
# %matplotlib inline
import pickle

df = pd.read_csv('data/worldcities.csv')
print(len(df))

df1 = df
# df1 = df[df.index <= 10]
print(len(df1))

distance = 3000 #3km
lat_diff = (distance/1000)/ 111
print(lat_diff)
lon_diff = (distance/1000)/ 111 
print(lon_diff)

def draw(lat,lon,fname,distance):
        G = ox.graph_from_point((lat, lon), distance=distance, network_type='all_private')
        img = ox.plot_graph(G,show = False,save=True,close =True, filename = fname, node_size=0)
        del G
        
stat = []

for index,row in df1.iterrows():
    fname = str(index)+"_" + row.country + "_" + row.city
    print(fname)
    try:
        G = ox.graph_from_point((row.lat, row.lng), distance=distance, network_type='all_private')
        basic_stats = ox.basic_stats(G)
        stat.append([index, basic_stats])
        
        """
        draw(row.lat + lat_diff, row.lng, fname +"_"+ str(1), distance)        
        draw(row.lat + lat_diff, row.lng - lon_diff, fname +"_"+ str(2), distance)        
        draw(row.lat + lat_diff, row.lng + lon_diff, fname +"_"+ str(3), distance)        
        draw(row.lat, row.lng - lon_diff, fname +"_"+ str(4), distance)        
        draw(row.lat, row.lng + lon_diff, fname +"_"+ str(5), distance)        
        draw(row.lat - lat_diff, row.lng - lon_diff, fname +"_"+ str(6), distance)        
        draw(row.lat - lat_diff, row.lng, fname +"_"+ str(7), distance)        
        draw(row.lat - lat_diff, row.lng + lon_diff, fname +"_"+ str(8), distance)  
        """
        
        del G
    except Exception as e:
        print(e)
        
f = open('data/stat.pkl','wb')
pickle.dump(stat,f)
f.close