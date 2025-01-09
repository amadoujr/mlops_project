import streamlit as st
import requests



## titre 
st.header("Analyse pertinance d'un film en se basant sur le feedback d'un utilisateur")

## texte à entrer
user_input = st.text_area('saisie le texte à analyser: ')

## prediction 

if st.button('analyse le text'):
    if user_input.strip() == "":
        st.warning("saisie un texte valide.")
    else:
        try:
            api_url = "http://webapp:8000/predict"
            payload = {"reviews": [user_input]}
            response = requests.post(api_url, json=payload)

            # Check if the request was successful
            if response.status_code == 200:
                result = response.json()
                sentiment = result.get("sentiments", ["Unknown"])[0]
                st.success(f"The sentiment of the text is: **{sentiment}**")
            else:
                st.error(f"Error from the API: {response.status_code} - {response.text}")
        except Exception as e:
            st.error(f"An error occurred: {e}")
