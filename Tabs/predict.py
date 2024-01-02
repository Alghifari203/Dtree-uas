import streamlit as st 
from web_function import predict

def app(df, x, y):

    st.title("Halaman prediksi")

    col1, col2, col3 =st.columns(3)

    with col1:
        sex = st.text_input ('input Nilai sex')
    with col1:
        cp  = st.text_input ('input Nilai cp')
    with col1:
        trestbps = st.text_input ('input Nilai trestbps')
    with col1:
        chol = st.text_input ('input Nilai chol')
    with col2:
        fbs = st.text_input ('input Nilai fbs')
    with col2:
        restecg = st.text_input ('input Nilai restecg')
    with col2:
        thalach = st.text_input ('input Nilai thalach')
    with col3:
        exang = st.text_input ('input Nilai exang')
    with col3:
        oldpeak = st.text_input ('input Nilai oldpeak')
    with col3:
        slope = st.text_input ('input Nilai slope')
    with col3:
        ca = st.text_input ('input Nilai ca')
    with col3:
        thal = st.text_input ('input Nilai thal')

    features =[sex,cp,trestbps,chol,fbs,restecg,thalach,exang,oldpeak,slope,ca,thal]

    if st.button("prediksi"):
        prediction, score = predict(x,y,features)
        score = score
        st.info("prediksi sukses...")

        if (prediction == 1):
            st.warning("orang tersebut rentan kenak stroke")
        else:
            st.success("orang tersebut relatif aman")
        
        st.write("model yang digunakan memiliki tingkat akurasi", (score*100),"%")