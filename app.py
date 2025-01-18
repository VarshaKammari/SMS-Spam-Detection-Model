import nltk
nltk.download('punkt')
nltk.download('punkt_tab')
nltk.download('stopwords')

import streamlit as st
import pickle 
import string
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer

ps = PorterStemmer()


def transform_text(text):
    text = text.lower()
    text = nltk.word_tokenize(text)

    y = []
    for i in text:
        if i.isalnum():
            y.append(i)

    text = y[:]
    y.clear()

    for i in text:
        if i not in stopwords.words('english') and i not in string.punctuation:
            y.append(i)

    text = y[:]
    y.clear()

    for i in text:
        y.append(ps.stem(i))

    return " ".join(y)


tk = pickle.load(open("vectorizer.pkl", 'rb'))
model = pickle.load(open("model.pkl", 'rb'))


    
st.sidebar.title("Dashboard")
app_mode = st.sidebar.selectbox("Select Page",["Home","Email Classification"])

if(app_mode=="Home"):
     st.header("SMS Spam Detection Model")
     st.markdown("""
    Welcome to the SMS Spam Detection Model! 🔍
    
    Our mission is to help in identifying spam mail efficiently. Type the message of email, and our system will analyze it to detect any signs of spam. Together, let's classify spam and not spam messages!

    ### How It Works
    1. **Type your message:** Go to the **Email Classification** page and type the content of the mail.
    2. **Analysis:** Our system will process the text using advanced algorithms to identify Spam and Not Spam.
    3. **Results:** View the results and recommendations for further action.

    ### Why Choose Us?
    - **Accuracy:** Our system utilizes state-of-the-art machine learning techniques for accurate spam classificatio.
    - **User-Friendly:** Simple and intuitive interface for seamless user experience.
    - **Fast and Efficient:** Receive results in seconds, allowing for quick decision-making.

    ### Get Started
    Click on the **Email Classification** page in the sidebar to upload an image and experience the power of our Spam mail classification System!

    
    """)

elif(app_mode=="Email Classification"):
    st.title("SMS Spam Detection Model")
    st.write("*Made by Varsha Kammari, Supported by Edunet Foundation*")
    input_sms = st.text_input("Enter the SMS")

    if st.button('Predict'):

        # 1. preprocess
        transformed_sms = transform_text(input_sms)
        # 2. vectorize
        vector_input = tk.transform([transformed_sms])
        # 3. predict
        result = model.predict(vector_input)[0]
        # 4. Display
        if result == 1:
            st.header("Spam")
        else:
            st.header("Not Spam")
