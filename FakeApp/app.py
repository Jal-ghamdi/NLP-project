import streamlit as st
import pandas as pd
import numpy as np
import joblib
import altair as alt
pipe_lr=joblib.load(open("models/fake_classifier_pipe_lr_aug_2022.pkl","rb"))
def predict_fake(doc):
    results=pipe_lr.predict([doc])
    return results[0]
def get_proba(doc):
    results=pipe_lr.predict_proba([doc])
    return results
def main():
    st.title("Fake News Detection App")
    menu=["Home","Manage","About"]
    choice=st.sidebar.selectbox("Menu",menu)
    if choice=="Home":
        st.subheader("Home---Fake Content Detection")
        with st.form(key='fake_clf_form'):
            text=st.text_area("Type Your Text Here")
            submit_text=st.form_submit_button(label='Predict')
        if submit_text:
            col1,col2=st.columns(2)
            predictions=predict_fake(text)
            probability=get_proba(text)
            with col1:
                st.success("Original Text")
                st.write(text)
                st.success("Prediction")
                st.write(predictions)
                st.write("Confidence:{}".format(np.max(probability)))

            with col2:
                st.success("The Prediction Probability")
                #st.write(probability)
                proba_df=pd.DataFrame(probability,columns=pipe_lr.classes_)
                #st.write(proba_df.T)
                proba_df=proba_df.T.reset_index()
                proba_df.columns=['Label','Probability']
                fig = alt.Chart(proba_df).mark_bar().encode(x='Label',y='Probability',color='Label')
                st.altair_chart(fig,use_container_width=True)
    elif choice=="Manage":
        st.subheader("Manage")
    else:
        st.subheader("About")

if __name__ == '__main__':
    main()
