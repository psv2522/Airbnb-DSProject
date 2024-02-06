import pickle
import pandas as pd
import numpy as np
import nltk
import re

def write():
    data = pd.read_csv("./listings.csv")
    data = data[['amenities','neighbourhood_cleansed','neighbourhood_group_cleansed','price']]
    data['price'] = data['price'].str.replace("$","",regex=True)
    data['price'] = data['price'].str.replace(",","",regex=True)
    data['price'] = pd.to_numeric(data['price'])
    data = data.head(2000)
    amenitiesDF = data[['amenities','price',]]
    amenitiesDFTopper = amenitiesDF.sort_values('price',ascending=[0])
    amenitiesDFtop=amenitiesDFTopper.head(2000)
    allemenities = ''
    for index,row in amenitiesDFtop.iterrows():
        p = re.sub('[^a-zA-Z]+',' ', row['amenities'])
        allemenities+=p
    allemenities_data= nltk.word_tokenize(allemenities)
    filtered_data=[word for word in allemenities_data if word not in stopwords.words('english')] 
    wnl = nltk.WordNetLemmatizer() 
    allemenities_data=[wnl.lemmatize(data) for data in filtered_data]
    allemenities_words=' '.join(allemenities_data)
    x = []
    y = []
    for i in allemenities_data:
      if(i not in x):
        x.append(i)
        y.append(allemenities_data.count(i))
    new_y, new_x = zip(*sorted(zip(y,x),reverse=True))
    file = open('items_1.txt','w')
    file_2 = open('items_2.txt','w')
    for i in range(len(new_x)):
        x = new_x[i]
        y = new_y[i]
        file.write(x+"\n")
        file_2.write(str(y)+"\n")
    file.close()
    file_2.close()
	
def transform(a1):
        data = pd.read_csv("./listings.csv")
        data = data[['amenities','neighbourhood_cleansed','neighbourhood_group_cleansed','price']]
        data['price'] = data['price'].str.replace("$","",regex=True)
        data['price'] = data['price'].str.replace(",","",regex=True)
        data['price'] = pd.to_numeric(data['price'])

        data = data.head(20000)
        
        a = a1[0]
        a = a.replace(", "," ")
        a = a.replace("\"","")
        l = a[1:(len(a)-1)].split(" ")
        c = 0
        my_file = open("items_1.txt", "r")
        my_file_2 = open("items_2.txt", "r")
        data2 = my_file_2.read()
        data1 = my_file.read()
        new_x = data1.replace('\n', ' ').split(" ")
        new_x = new_x[0:len(new_x)-1]
        new_yz = data2.replace('\n', ' ').split(" ") 
        new_y = []
        for k in range(len(new_yz)-1):
          new_y.append(int(new_yz[k]))
        for i in range(len(l)):
          if l[i] in new_x:
            c += new_y[new_x.index(l[i])]
          else:
            c += 0
        
        l1 = round(c/len(l))

        arr1 = data['neighbourhood_group_cleansed'].unique()
        a = a1[2]
        l3 = (np.where(arr1 == a)[0][0]+1)

        arr = data['neighbourhood_cleansed'].unique()
        a = a1[1]
        b = a1[2]
        l2 = (np.where(arr == a)[0][0]+1+(50*(np.where(arr1 == b)[0][0])))

        return [l1,l2,l3]

def load_model(modelfile):
	loaded_model = pickle.load(open(modelfile, 'rb'))
	return loaded_model

#["First aid kit", "Microwave", "Stove", "Coffee maker", "Long term stays allowed", "Carbon monoxide alarm", "TV with standard cable", "Essentials", "Oven", "Hot water", "Outlet covers", "Heating", "Smoke alarm", "Fire extinguisher", "Free street parking", "Keypad", "Free parking on premises", "Extra pillows and blankets", "Cleaning before checkout", "Dishes and silverware", "Hair dryer", "Cable TV", "Refrigerator", "Bed linens", "Cooking basics", "Hangers", "Luggage dropoff allowed", "Iron", "Elevator", "Shampoo", "Kitchen", "Dryer", "Washer", "Wifi"]
#["Stove", "Coffee maker", "Long term stays allowed", "Carbon monoxide alarm", "Essentials", "Oven", "Dedicated workspace", "Hot water", "Heating", "Smoke alarm", "Fire extinguisher", "Free street parking", "Keypad", "Air conditioning", "Extra pillows and blankets", "Cleaning before checkout", "Paid parking off premises", "Dishes and silverware", "Baking sheet", "Hair dryer", "Bathtub", "Refrigerator", "TV", "Bed linens", "Cooking basics", "Hangers", "Luggage dropoff allowed", "Iron", "Ethernet connection", "Kitchen", "Wifi"]

# a = transform(['["Stove", "Coffee maker", "Long term stays allowed", "Carbon monoxide alarm", "Essentials", "Oven", "Dedicated workspace", "Hot water", "Heating", "Smoke alarm", "Fire extinguisher", "Free street parking", "Keypad", "Air conditioning", "Extra pillows and blankets", "Cleaning before checkout", "Paid parking off premises", "Dishes and silverware", "Baking sheet", "Hair dryer", "Bathtub", "Refrigerator", "TV", "Bed linens", "Cooking basics", "Hangers", "Luggage dropoff allowed", "Iron", "Ethernet connection", "Kitchen", "Wifi"]'
# ,"Midtown","Manhattan"])
# a = np.array(a).reshape(1,-1)
# loaded_model = load_model('model.pkl')
# loadded_model_2 = load_model('LR.pkl')
# prediction_2 = loadded_model_2.predict(a)
# prediction = loaded_model.predict(a)
# print(prediction.item(),round(prediction_2.item()))  