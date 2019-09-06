import numpy as np
import glob
import pandas as pd 

# print(glob.glob("images/*"))
list_img_original = glob.glob("images_original/*")
list_img = glob.glob("images/*")

print(len(list_img_original), len(list_img))

from PIL import Image

pix = 256
x =[]
# fname = "Aberdeen.png"

df = pd.DataFrame(index=[], columns=['idx', 'city'])
 
for index, fname in enumerate(list_img_original):
    # print(".", end=' ')
    x_temp =np.array(Image.open(fname).convert('L').resize((pix, pix)))
    x.append(x_temp)
    city = fname.lstrip("images_original/").rstrip(".png")
    print("\r{:}:{:}".format(index,city), end="")
    series = pd.Series([index, city], index=df.columns)
    df = df.append(series,ignore_index=True)

for index, fname in enumerate(list_img):
    # print(".", end=' ')
    x_temp =np.array(Image.open(fname).convert('L').resize((pix, pix)))
    x.append(x_temp)

x = np.asarray(x)
x.shape
np.save('road_network_' + str(pix) + '.npy',x)

df.to_csv('city_index_combination_' + str(pix) + '.csv',header=True, index=False)

