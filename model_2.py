import pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn import model_selection
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn import multioutput
from sklearn.multioutput import MultiOutputRegressor
from sklearn.svm import SVR
import nltk
from nltk.corpus import stopwords
nltk.download('stopwords')
nltk.download('punkt')
nltk.download('wordnet')
nltk.download('omw-1.4')
import re

class Prediction():
    def __init__(self) -> None:
        data = pd.read_csv("./listings.csv")
        data = data[['amenities','neighbourhood_cleansed','neighbourhood_group_cleansed','price']]
        data['price'] = data['price'].str.replace("$","",regex=True)
        data['price'] = data['price'].str.replace(",","",regex=True)
        data['price'] = pd.to_numeric(data['price'])

        data = data.head(20000)
        amenitiesDF = data[['amenities','price',]]
        amenitiesDFTopper = amenitiesDF.sort_values('price',ascending=[0])
        amenitiesDFtop=amenitiesDFTopper.head(20000)
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
        l1 = []
        for j in range(20000):  
          a = data.loc[j].at['amenities']
          a = a.replace(", "," ")
          a = a.replace("\"","")
          l = a[1:(len(a)-1)].split(" ")
          c = 0
          for i in range(len(l)):
            if l[i] in new_x:
              c += new_y[new_x.index(l[i])]
            else:
              c += 0
          l1.append(round(c/len(l)))

        l3 = []
        arr1 = data['neighbourhood_group_cleansed'].unique()
        for i in range(20000):
          a = data.loc[i].at['neighbourhood_group_cleansed']
          l3.append(np.where(arr1 == a)[0][0]+1)

        l2 = []
        arr = data['neighbourhood_cleansed'].unique()
        for i in range(20000):
          a = data.loc[i].at['neighbourhood_cleansed']
          b = data.loc[i].at['neighbourhood_group_cleansed']
          l2.append(np.where(arr == a)[0][0]+1+(50*(np.where(arr1 == b)[0][0])))
        
        data['amenities_score'] = l1
        data['n_no'] = l2
        data['gn_no'] = l3

        data_f = data[['amenities_score','n_no','gn_no','price']]
        train1 = data_f.drop(['price'],axis=1)
        test1 = data_f[['price']] 

        X_train1, X_test1, y_train1, y_test1 = train_test_split(train1, test1, test_size=0.2, random_state=4)

        self.regr2 = LinearRegression()
        self.regr2.fit(X_train1.values,y_train1)
        self.save_model(self.regr2,'LR.pkl')

    def predict(self,a1,a2,s):
        Xn1 = [[a1,a2,s]]
        pred = self.regr2.predict(Xn1)
        return round(pred[0][0])

    def save_model(self,model,filename):
        pickle.dump(model, open(filename, 'wb'))

def transform(a1):
   pass

i = Prediction()
print(i.predict(1120,1,1))

def load_model(modelfile):
	loaded_model = pickle.load(open(modelfile, 'rb'))
	return loaded_model

single_pred = [1120,1,1]
single_pred = np.array(single_pred).reshape(1,-1)
loaded_model = load_model('model.pkl')
prediction = loaded_model.predict(single_pred)
#print(prediction.item())            