# Laporan Proyek Machine Learning
### Nama : M. Fadhil Alghifari
### Nim : 211351078
### Kelas : IF Pagi B

## Domain Proyek
Proyek ini memiliki tujuan untuk mengembangkan sistem prediksi yang dapat menilai potensi risiko penyakit stroke pada individu. Dengan menggunakan metode analisis data dan penerapan machine learning, proyek ini akan memanfaatkan data historis dan variabel-variabel terkait untuk merancang model prediktif yang dapat memberikan estimasi tingkat risiko terjadinya penyakit stroke pada seseorang. Hasil prediksi ini diharapkan dapat memberikan pandangan lebih jauh mengenai potensi risiko kesehatan pada individu yang bersangkutan

## Business Understanding 
Dalam konteks prediksi penyakit stroke, pemahaman bisnis difokuskan pada identifikasi faktor-faktor yang dapat mempengaruhi kemungkinan seseorang mengalami penyakit stroke. Data historis dan tren kesehatan dapat memberikan wawasan yang mendalam tentang faktor-faktor risiko dan mengapa seseorang mungkin rentan terhadap penyakit stroke. Beberapa faktor yang mungkin berdampak pada prediksi ini melibatkan riwayat kesehatan keluarga, gaya hidup, tekanan darah, kadar kolesterol, dan riwayat merokok. Selain itu, faktor-faktor seperti aktivitas fisik, pola makan, dan kebiasaan hidup sehari-hari juga dapat menjadi pertimbangan penting dalam memprediksi potensi risiko penyakit stroke. Dengan pemahaman mendalam ini, proyek ini bertujuan untuk membangun model prediktif yang dapat secara akurat mengidentifikasi individu yang berisiko tinggi mengalami penyakit stroke, sehingga memungkinkan adopsi tindakan pencegahan dan intervensi yang lebih tepat waktu.

### Problem Statements
Latar belakang masalah:
- Rumah sakit menghadapi kesulitan untuk memproyeksikan apakah pasien baru memiliki potensi risiko penyakit stroke atau tidak.
- Rumah sakit kesulitan untuk merancang strategi baru dalam menyusun pelayanan kesehatan terkait prediksi penyakit stroke untuk pasien-pasien potensial.

### Goals
Tujuan dari pernyataan masalah :
- Mengidentifikasi bahwa rumah sakit mengalami kesulitan dalam memprediksi risiko penyakit stroke pada pasien baru, menyoroti ketidakpastian dalam penilaian risiko tersebut.
- Menyoroti pentingnya pemutakhiran dan perbaikan dalam pencatatan data kesehatan untuk meningkatkan kualitas dan akurasi model prediktif

## Data Understanding

dataset yang ada diambil langsung dari kagglen 

### [*Hypertension prediction data*](https://www.kaggle.com/datasets/prosperchuks/health-dataset)

#### Mengimport dataset dari kaggle :

``` Python
from google.colab import files
files.upload()
```

``` Python
!mkdir -p ~/.kaggle
!cp kaggle.json ~/.kaggle/
!chmod 600 ~/.kaggle/kaggle.json
!ls ~/.kaggle
```
Mendownload dataset

``` Python
!kaggle datasets download -d prosperchuks/health-dataset
```
Mengekstrak dataset

``` Python
!mkdir Hypertension
!unzip health-dataset.zip -d Hypertension
!ls Hypertension
```
## Mengimport library

``` Python
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import seaborn as sns
import plotly.express as px
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.tree import DecisionTreeClassifier
from sklearn import tree
from sklearn.tree import plot_tree
```

## Data Discovery
Membaca data csv
``` Python
df = pd.read_csv('/content/Hypertension/hypertension_data.csv')
df.head()
```
![image1](head.png)

Melihat Type data pada setiap kolom
``` Python
df.info() 
```
![image1](info.png)

menghilangkan nilai yang tidak dibutuhkan
``` Python
df = df.drop(['age'], axis=1)
```
Melihat sample
sebelum 
``` Python
df.sample()
```
![image1](sample1.png)

sesudah hilang nilai
``` Python
df.sample()
```
![image1](sample2.png)

### Variabel-variabel pada Travel Insurance Prediction Data adalah sebagai berikut:

- Sex : Menunjukkan jenis kelamin pasien (Dtype : float64)
- Cp : menggambarkan jenis nyeri dada yang dialami pasien (Dtype : Int64) 
- Trestbps : Menunjukkan tekanan darah istirahat pasien (Dtype : Object) 
- Chol : Menunjukkan tingkat kolesterol dalam darah pasien(Dtype : object) 
- Fbs  : menunjukkan apakah kadar gula darah puasa melebihi batas tertentu atau tidak. (Dtype : Int64)
- Restcg  : menggambarkan hasil elektrokardiografi pada istirahat. (Dtype : Int64)
- thalach (Detak Jantung Maksimal): Menunjukkan detak jantung maksimal yang dicapai oleh pasien selama uji stres (Dtype : Int64)
- exang (Angina yang Diinduksi Latihan) : Variabel biner yang menunjukkan keberadaan angina yang diinduksi (Dtype : int64) 
- oldpeak  : Menunjukkan depresi segmen ST relatif terhadap istirahat yang diinduksi oleh latihan (Dtype : float64) 
- Tslope  : Menunjukkan kemiringan segmen ST latihan relatif terhadap istirahat. (Dtype : int64)
- ca  : Menunjukkan jumlah saluran utama yang diwarnai oleh fluoroskopi. (Dtype : int64)
- thal  : MMenunjukkan jenis thalassemia yang dialami pasien (Dtype : int64)
- target  :Variabel target yang menunjukkan apakah pasien mengalami masalah kesehatan tertentu (misalnya, penyakit jantung)  (Dtype : int64)

## EDA

menampilkan status pasien pada setiap nilai yang ada 
``` Python
df.hist(figsize=(20,20))
plt.show()
```
![image1](eda1.png)

menampilkan status target 
``` Python
print(df['target'].value_counts())
_ = sns.countplot(x='target', data=df)
```
![image1](eda2.png)

menampilkan representasi tingkat detak jantung pasien 
``` Python
print(df['thalach'].value_counts())
_ = sns.countplot(x='thalach', data=df)
```
![image1](eda3.png)

menampilkan jumlah data berdasarkan physiq health
``` Python
fig, ax = plt.subplots(6, 3, figsize=(15, 30))
i = 0
for col in df.columns:
    if col=='MentHlth' or col=='PhysHlth':
        sns.kdeplot(data=df, x=col, hue='target', ax=ax[i//3, i%3])
    else:
            sns.histplot(data=df, x=col, hue='target', ax=ax[i//3, i%3])
    i+=1
plt.show()
```
![image1](eda4.png)

menampilkan cholesterol pasien 
``` Python
print(df['chol'].value_counts())
_ = sns.countplot(x='chol', data=df)
```
![image1](eda5.png)

## Data Preprocessing
Sebelum data di modeling kita lakukan proses data agar data tersebut lebih matang untuk di pakai.

lakukan pengubahan bentuk nilai
``` Python
from collections.abc import Collection
numerical = []
catgcols = []

for col in df.columns:
  if df[col].dtype=="float64":
    numerical.append(col)
  else:
      catgcols.append(col)

for col in df.columns:
  if col in numerical:
    df[col].fillna(df[col].median(), inplace=True)
  else:
      df[col].fillna(df[col].mode()[0], inplace=True)
```

panggil kembali nilai numerik dan objek 

``` Python
numerical
```
``` Python
catgcols
```
buatlah tranforms data 

``` Python
df['target'].value_counts()
```
``` Python
ind_col = [col for col in df.columns if col !='target']
dep_col = 'target'
```
``` Python
df[dep_col].value_counts()
```
ubah data yang telah di tranforms 
``` Python
le = LabelEncoder()

for col in catgcols:
    df[col] = le.fit_transform(df[col])
```

``` Python
df['target'] = le.fit_transform(df['target'])
```
``` Python
x = df[ind_col]
y = df[dep_col]
```

setelah diubah data yg telah diambil , kemudian save data kedalam bentuk csv 
``` Python
df.to_csv('Hypertension-prediction.csv')
```

## Modeling
setelah melakukan preprocessing yang cukup panjang terakhir kita menambahkan model

``` Python
dtc = DecisionTreeClassifier(
     ccp_alpha=0.0, class_weight=None, criterion='entropy',
     max_depth=4, max_features=None, max_leaf_nodes=None,
     min_impurity_decrease=0.0, min_samples_leaf=1,
     min_samples_split=2, min_weight_fraction_leaf=0.0,
     random_state=42, splitter='best')

model = dtc.fit(x_train, y_train)
dtc_acc = accuracy_score(y_test, dtc.predict(x_test))

print(f"akurasi data training = {accuracy_score(y_train, dtc.predict(x_train))}")
print(f"akurasi data testing = {dtc_acc}\n")

print(f"confusion matrix : \n{confusion_matrix(y_test, dtc.predict(x_test))}\n")
confusion = confusion_matrix(y_test, dtc.predict(x_test))
tn, fp, fn, tp = confusion.ravel()
print(f"classification report : \n {classification_report(y_test, dtc.predict(x_test))}")
```

``` Python
input_data = (3, 31, 64, 1, 0, 49, 0, 2.3, 0, 0, 1, 1)
input_data_as_numpy_array = np.array(input_data)
input_data_reshape = input_data_as_numpy_array.reshape(1,-1)
prediction = model.predict(input_data_reshape)
print(prediction)
if (prediction[0]==0):
  print('Pasien tidak mempunyai hipertensi')

else:
  print('pasien mempunyai hipertensi')
```

## Visualisasi Hasil Algoritma

``` Python
fig = plt.figure(figsize=(25,20))
_ = tree.plot_tree(model,
                    feature_names=ind_col,
                   class_names=['0','1'],
                   filled = True)
```
![image1](visual.png)

## Deployment
![image](app.png)


[linkStreamlit](https://dtree-uas.streamlit.app/)







