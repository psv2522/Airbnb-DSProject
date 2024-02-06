import streamlit as st 
import pandas as pd
import numpy as np
import os
import pickle
import warnings
from soln import transform

st.set_page_config(page_title="AirBnB Reccomendation System", page_icon="🌿", layout='centered', initial_sidebar_state="collapsed")

def load_model(modelfile):
	loaded_model = pickle.load(open(modelfile, 'rb'))
	return loaded_model

def main():
    # title
    html_temp = """
    <div>
    <h1 style="color:#0066b2;text-align:left;"> AirBnB Reccomendation System  </h1>
    </div>
    """
    st.markdown(html_temp, unsafe_allow_html=True)

    col1,col2  = st.columns([2,2])
    
    with col1: 
        with st.expander(" ℹ️ Information", expanded=True):
            st.write("""
            Airbnb is an online marketplace that enables people to rent out their homes or apartments to travellers. It was founded in 2008 and is headquartered in San Francisco, California. The platform allows property owners to list their properties for short-term rental and allows travellers to book accommodations for their trips. The platform includes a wide range of properties, from budget-friendly apartments to luxury villas, and is available in over 220 countries and regions. 
            As a traveller, several questions prop up while booking an accommodation like the amenities present, safety and more important than anything, the price of the accommodation. Even as a host of an accommodation, this information can help to service the clients better about what amenities are the most important to the travellers and hence design their listing appropriately.
            """)
        '''
        ## How does it work ❓ 
        Add all the parameters and the machine learning model will predict the most suitable price for your listing based on various parameters.
        '''


    with col2:
        st.subheader(" Find out the Most Suitable Price for your Listing")
        temp = st.text_input("Ammenities")
        N = st.text_input("Neighbourhood")
        N2 = st.text_input("Neighbourhood Group")
        
        if st.button('Predict'):
            feature_list = [temp, N, N2]
            feature_list = transform(feature_list)
            single_pred = np.array(feature_list).reshape(1,-1)
            loaded_model = load_model('LR.pkl')
            prediction = loaded_model.predict(single_pred)
            col1.write('''
		    ## Results 🔍 
		    ''')
            col1.success(f"${round(prediction.item())} is recommended as a Price for your Listing.")
      #code for html ☘️ 🌾 🌳 👨‍🌾  🍃

    hide_menu_style = """
    <style>
    #MainMenu {visibility: hidden;}
    </style>
    """

hide_menu_style = """
        <style>
        #MainMenu {visibility: hidden;}
        </style>
        """
st.markdown(hide_menu_style, unsafe_allow_html=True)

if __name__ == '__main__':
	main()