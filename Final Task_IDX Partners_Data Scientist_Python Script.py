#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
pd.set_option('display.max_columns', None)

import matplotlib.pyplot as plt
import seaborn as sns
import missingno as msno


# ## Import Dataset

# In[2]:


df = pd.read_csv('loan_data_2007_2014.csv', dtype={"desc": object})
# df = pd.read_csv('loan_data_2007_2014.csv')


# In[3]:


df


# ## Data Cleaning

# In[4]:


df.info()


# In[5]:


df[(df['loan_status'] == "Current")]


# In[6]:


df['loan_status'].unique()


# Dataset memiliki total keseluruhan 466.285 baris dan 75 kolom. Kolom-kolom tersebut terdiri dari 22 kolom kategorikal dan 53 kolom numerik.
# 
# Perlu diperhatikan bahwa dataset tidak memiliki kolom target. Berdasarkan penjelasan yang diberikan mengenai project ini, hal yang hendak dilakukan adalah untuk mengembangkan sebuah model Supervised Learning (dataset memiliki target). 
# 
# Sesuai dengan apa yang dijelaskan pada **Hint** pengerjaan project untuk memberikan label sendiri untuk setiap data, berupa pinjaman yang berisiko atau tidak aman ('BAD': 1) dan pinjaman yang baik atau aman ('GOOD': 0) berdasarkan ciri-ciri tertentu.
# 
# 
# Kolom yang dapat digunakan untuk menentukan resiko suatu pinjaman adalah kolom `loan_status`, di mana pinjaman dengan status `Fully Paid`, `Current`, dan `Does not meet the credit policy. Status:Fully Paid` adalah pinjaman yang baik ('GOOD': 0). Kemudian pinjaman dengan status `Charged Off`, `Default`, `Late (31-120 days)`, dan `Does not meet the credit policy. Status:Charged Off` adalah pinjaman yang beresiko ('BAD': 1). Sedangkan pinjaman dengan status `In Grace Period` dan `Late (16-30 days)` adalah pinjaman yang harus dilakukan pengecekan lebih lanjut terhadap riwayat tunggakannya, dikarenakan pinjaman ini sudah terlambat melakukan pembayaran, namun masih dalam batas wajar (pengampunan).

# #### Pengecekan Lebih Lanjut Status Loan

# In[7]:


# Visualisasi grade untuk setiap status loan
sns.set(rc ={'figure.figsize':(10, 5)})
def loan_grade(name):
    data = df[df['loan_status'] == name]
    data_grade = data.groupby(['grade']).size().reset_index(name='count')
    
    sns.barplot(x='grade', y='count', data=data_grade, hue='grade', palette='Set2')
    plt.title(f'{name} Loans with Respect to Grades')
    plt.xlabel('Grade')
    plt.ylabel('Count')
    plt.show()

loan_grade('Charged Off')
loan_grade('Default')
loan_grade('Late (31-120 days)')
loan_grade('Does not meet the credit policy. Status:Charged Off')
loan_grade('Current')
loan_grade('Fully Paid')


# Dari visualisasi pesebaran data pinjaman terhadap Grade-nya, terlihat bahwa banyak pinjaman yang masuk kategori 'BAD' memiliki grade C dan D, sedangkan pinjaman yang termasuk kategori 'GOOD' cenderung memiliki grade B dan C.
# 
# Hal ini dapat dijadikan pertimbangan lebih lanjut untuk menentukan kategori data pinjaman yang berstatus `In Grace Period` dan `Late (16-30 days)`. Pinjaman tersebut yang memiliki grade D ke bawah akan dikategorikan sebagai pinjaman 'BAD' dikarenakan indikasi riwayat pinjaman yang buruk.

# In[8]:


# membuat fungsi untuk assign kolom target
def assign_target(stat, grade):
    good_stat = ['Fully Paid', 'Current', 'Does not meet the credit policy. Status:Fully Paid']
    bad_stat = ['Charged Off', 'Default', 'Late (31-120 days)', 'Does not meet the credit policy. Status:Charged Off']
    
    if (stat in good_stat):
        return int('0')
    elif (stat in bad_stat):
        return int('1')
    else:
        good_grade = ['A', 'B', 'C']
        bad_grade = ['D', 'E', 'F', 'G']
        
        if grade in bad_grade:
            return int('1')
        elif grade in good_grade:
            return int('0')
        else:
            return np.nan


# In[9]:


# salin dataframe ke dataframe baru
df_target = df.copy()

# vectorized fungsi assign target
assign_vect = np.vectorize(assign_target)

# pass argument ke fungsi dan tampung hasil pada dataframe baru
df_target['target'] = assign_vect(df_target['loan_status'], df['grade'])


# In[10]:


df_target


# In[11]:


df_target.info()


# #### Penghapusan Kolom-Kolom Tertentu

# Beberapa kolom dirasa tidak memiliki pengaruh yang signifikan terhadap pemodelan dan analisa data. Penghapusan kolom-kolom ini ditujukan untuk dapat melakukan analisa yang lebih efektif dan hasil yang lebih akurat.
# 
# Selain itu ada beberapa kolom yang memiliki nilai yang tidak dapat memberikan insight, seperti kolom `id` yang merupakan sebuah kode penanda unik untuk setiap data serta kolom-kolom seperti `policy_code` dan `application_type` yang hanya memiliki satu value untuk semua data. Kemudian ada kolom-kolom yang berisi data string, salah satu contohnya adalah `desc`.

# In[13]:


# Penghapusan kolom-kolom dengan data null (tidak ada data sama sekali)
df_target = df_target.loc[:,df_target.notna().any(axis=0)]

# menghapus kolom-kolom yang tidak relevan atau tidak digunakan
del_cols = ['Unnamed: 0', 'id', 'member_id', 'url', 'desc', 'emp_title', 'title', 
            'zip_code', 'earliest_cr_line', 'collection_recovery_fee', 'issue_d',
            'last_pymnt_d', 'next_pymnt_d', 'last_credit_pull_d', 
            'application_type', 'policy_code']

# drop columns
df_target = df_target.drop(del_cols, axis=1)


# In[14]:


df_target.info()


# ### Handling Missing Values

# In[15]:


# pengecekan missing values menggunakan library missingno
msno.bar(df_target)


# In[16]:


df_target.isna().sum()


# Kolom-kolom yang memiliki missing values lebih besar dari 50% dihapus karena tidak memungkinkan untuk melakukan imputasi terhadap data-data yang hilang tersebut. Hal ini dapat mempengaruhi analisa data dan pemodelan 

# In[17]:


# menghapus kolom dengan missing values lebih dari 50%
for col in df_target.columns:
    if ((df_target[col].isna().sum()/len(df_target))*100) >= 50:
        df_target = df_target.drop([col], axis=1)


# Lakukan pengecekan lebih lanjut terhadap data-data yang missing lainnya.

# In[18]:


msno.heatmap(df_target)


# In[19]:


msno.dendrogram(df_target)


# Dari pengecekan lebih lanjut, terlihat ada banyak data hilang yang terindikasi MCAR (Missing Completely at Random). Untuk meng-handle missing value dilakukan imputasi di beberapa kolom serta penghapusan missing data khusus untuk kolom-kolom `tot_coll_amt`, `tot_cur_bal`, dan `total_rev_hi_lim` di karenakan memiliki lebih dari 70.000 MCAR missing data.

# In[20]:


# menghapus baris data missing dari kolom 'tot_coll_amt', 'tot_cur_bal', 'total_rev_hi_lim'
df_target = df_target.dropna(axis=0, subset=['tot_coll_amt', 'tot_cur_bal', 'total_rev_hi_lim'])

df_target.isna().sum()


# Dari hasil penghapusan tiga kolom tersebut, terlihat mayoritas data missing ikut terhapus. Ini berarti bahwa kebanyakan dari data missing pada kolom lain, memiliki data missing juga pada kolom-kolom `tot_coll_amt`, `tot_cur_bal`, dan `total_rev_hi_lim`.
# 
# Untuk kolom `emp_length` akan dilakukan imputasi KNN, sedangkan kolom `revol_util` akan dilakukan imputasi mean / median.

# ##### Imputasi KNN kolom `emp_length`

# In[21]:


# ambil hanya kolom-kolom numerik dan kolom `emp_length` saja
numerical_cols = [cname for cname in df_target.columns
                 if df_target[cname].dtype in ["int32", "int64", "float64"]]

numerical_cols.append('emp_length')

df_with_null = df_target[numerical_cols]

# import library yang dibutuhkan
from sklearn.impute import KNNImputer
from sklearn.preprocessing import MinMaxScaler

mm = MinMaxScaler()

def find_category_mappings(df, variable):
    return {k: i for i, k in enumerate(df[variable].dropna().unique(), 0)}

def integer_encode(df , variable, ordinal_mapping):
    df[variable] = df[variable].map(ordinal_mapping)

mappin = dict()
def imputation(df1 , cols):
    df = df1.copy()
    #Encoding dict & Removing nan    
    #mappin = dict()
    for variable in cols:
        mappings = find_category_mappings(df, variable)
        mappin[variable] = mappings

    #Apply mapping
    for variable in cols:
        integer_encode(df, variable, mappin[variable])  

    #Minmaxscaler and KNN imputation 
    sca = mm.fit_transform(df)
    knn_imputer = KNNImputer()
    knn = knn_imputer.fit_transform(sca)
    df.iloc[:,:] = mm.inverse_transform(knn)
    for i in df.columns : 
        df[i] = round(df[i]).astype('int')

    #Inverse transform
    for i in cols:
        inv_map = {v: k for k, v in mappin[i].items()}
        df[i] = df[i].map(inv_map)
    return df[cols]


#Imputation
df_imput = imputation(df_with_null, ['emp_length'])


# In[22]:


# gabungkan kolom `emp_length` yang telah diimputasi dengan dataframe utama
df_target = df_target.drop(columns=['emp_length'])
df_target['emp_length'] = df_imput


# In[23]:


# imputasi kolom `revol_util` dengan mean
df_target['revol_util'] = df_target['revol_util'].fillna(df_target['revol_util'].mean())


# ### Checking Anomaly Outliers

# In[25]:


numerical_cols = [cname for cname in df_target.columns
                 if df_target[cname].dtype in ["int64", "float64"]]

# visualisasikan pesebaran data setiap kolom numerik
plt.figure(figsize=(20, 85))
for i, col in enumerate(list(numerical_cols), 1):
    plt.subplot(15, 2, i)
    sns.boxplot(x=col, data=df_target[df_target.columns[:-1]])
    plt.title("Pesebaran Data " + col)


# Dari hasil pengecekan data anomali, terdapat beberapa data di beberapa kolom yang memiliki jarak sangat ekstrim. Data-data yang berjarak sangat ekstrim ini dianggap sebagai anomali yang tidak dapat dijelaskan penyebabnya dan dihapus agar tidak menyebabkan kesalahan saat analisis 

# In[26]:


# menghapus data-data anomali
df_target = df_target.drop(df_target[(df_target['annual_inc'] > 2000000) |
                                  (df_target['revol_bal'] > 800000) |
                                  (df_target['revol_util'] > 200) |
                                  (df_target['tot_coll_amt'] > 1000000) |
                                  (df_target['tot_cur_bal'] > 3000000) |
                                  (df_target['total_rev_hi_lim'] > 1000000)].index)


# ## Exploratory Data Analysis

# ### Pesebaran Data Target

# In[27]:


# melihat pesebaran data menggunakan Countplot
fig = plt.figure()
ax = sns.countplot(x=df_target['target'], hue=df_target['target'], palette='Set2')
ax.bar_label(ax.containers[0])

plt.title("Pesebaran Data Pinjaman")
plt.xlabel("0 = 'Good' dan 1 = 'Bad'")
for i in ax.containers:
    ax.bar_label(i,)
    
plt.show()


# Pesebaran data target sangat tidak seimbang (imbalance data). Hanya terdapat sekitar 10% data yang merupakan pinjaman bermasalah (Bad Loans). Hal ini menjadi catatan penting dalam pemilihan metrik evaluasi saat melakukan modelling.

# ### Analisa Data Kolom-Kolom Kategorikal

# In[28]:


# fungsi untuk mempermudah visualisasi kolom kategorikal
def vis_cat(cat_column, title_count, title_stacked, label):
    fig = plt.figure(figsize=(12, 5))
    
    ax = fig.add_subplot(121)
    sns.countplot(data=df_target, x=cat_column, hue='target', palette = 'Set2')
    ax.set_title(title_count)
    plt.xlabel(label)
    plt.ylabel("Count")
    for i in ax.containers:
        ax.bar_label(i,)
    
    ax2 = fig.add_subplot(122)
    pd.crosstab(df_target[cat_column], df_target['target']).apply(lambda x: x*100/x.sum(), axis = 1).plot(kind = 'bar', stacked = True, ax=ax2, rot=0)
    ax2.set_title(title_stacked)
    plt.xlabel(label)
    plt.ylabel("Persentase")

    fig.tight_layout(pad=2)


# In[29]:


# pesebaran data Target dan perbandingannya dengan lama pembayaran
col = 'term'
title_c = 'Pesebaran Target berdasarkan Lama Pembayaran'
title_b = 'Persentase Target berdasarkan Lama Pembayaran'
label = 'Term'

vis_cat(col, title_c, title_b, label)


# Lama waktu pembayaran terbanyak adalah 36 bulan, namun persentase pinjaman yang terindikasi jelek lebih besar pada jenis pinjaman 60 bulan.

# In[30]:


# pesebaran data Target dan perbandingannya dengan riwayat grade peminjam
col = 'grade'
title_c = 'Pesebaran Target berdasarkan Grade Peminjam'
title_b = 'Persentase Target berdasarkan Grade Peminjam'
label = 'Grade'

vis_cat(col, title_c, title_b, label)


# Mayoritas pinjaman memiliki riwayat grade B dan C. Sementara terdapat pola di mana, semakin rendah grade peminjam, maka akan semakin besar resiko peminjaman tersebut (semakin BAD).

# In[31]:


# pesebaran data Target dan perbandingannya dengan Sub Grade
fig = plt.figure(figsize=(25, 5))
ax = sns.countplot(data=df_target, x='sub_grade', hue='target', palette = 'Set2')
ax.set_title('Pesebaran Sub Grade')
plt.xlabel('Sub Grade')
plt.ylabel("Count")
for i in ax.containers:
    ax.bar_label(i,)

ax2 = pd.crosstab(df_target['sub_grade'], df_target['target']).apply(lambda x: x*100/x.sum(), axis = 1).plot(kind = 'bar', stacked = True, rot=0, figsize=(15, 5))
ax2.set_title("Persentase Target berdasarkan Sub Grade")
plt.xlabel("Sub Grade")
plt.ylabel("Persentase")


# Pesebaran data pinjaman mayoritas memiliki Sub-Grade B3 dan B4. Sub-Grade dengan pinjaman beresiko terbanyak adalah D1. Pola yang ditunjukkan selaras dengan pola Grade, di mana semakin rendah Sub-Grade maka akan semakin besar resiko pinjaman tersebut.

# In[32]:


# pesebaran data Target dan perbandingannya dengan kepemilikan rumah
col = 'home_ownership'
title_c = 'Pesebaran Target berdasarkan Kepemilikan Rumah'
title_b = 'Persentase Target berdasarkan Kepemilikan Rumah'
label = 'Kepemilikan Rumah'

vis_cat(col, title_c, title_b, label)


# Mayoritas data kepemilikan rumah adalah Kredit Rumah, dan pinjaman beresiko paling banyak merupakan tempat tinggal mengontrak. Persentase pinjaman berisiko paling tinggi ada pada peminjam yang tidak memiliki rumah dan status penghuni yang lain, namun datanya masih cukup sedikit. Mereka yang tempat tinggal mengontrak persentase pinjaman berisiko nya lebih tinggi.

# In[33]:


# pesebaran data Target dan perbandingannya dengan status verifikasi grade
col = 'verification_status'
title_c = 'Pesebaran Target berdasarkan Status Verifikasi Grade'
title_b = 'Persentase Target berdasarkan Status Verifikasi Grade'
label = 'Status Verifikasi Grade'

vis_cat(col, title_c, title_b, label)


# Pesebaran terbanyak ada pada jenis peminjam yang riwayatnya ter-verified, namun perbedaannya dengan yang lain tidak terlalu jauh. Persentase pinjaman dengan resiko buruk juga terdapat pada verified.

# In[34]:


# pesebaran data Target dan perbandingannya dengan rencana pembayaran
col = 'pymnt_plan'
title_c = 'Pesebaran Target berdasarkan Rencana Pembayaran'
title_b = 'Persentase Target berdasarkan Rencana Pembayaran'
label = 'Rencana Pembayaran'

vis_cat(col, title_c, title_b, label)


# Uniknya, hampir seluruh pinjaman tidak menempatkan rencana pembayaran terhadap kredit mereka.

# In[35]:


# pesebaran data Target dan perbandingannya dengan Tujuan Pinjaman
fig = plt.figure(figsize=(25, 5))
ax = sns.countplot(data=df_target, x='purpose', hue='target', palette = 'Set2')
ax.set_title('Pesebaran Tujuan Pinjaman')
plt.xlabel('Tujuan Pinjaman')
plt.ylabel("Count")
for i in ax.containers:
    ax.bar_label(i,)

ax2 = pd.crosstab(df_target['purpose'], df_target['target']).apply(lambda x: x*100/x.sum(), axis = 1).plot(kind = 'bar', stacked = True, rot=0, figsize=(18, 5))
ax2.set_title("Persentase Target berdasarkan Tujuan Pinjaman")
plt.xticks(fontsize=8)
plt.xlabel("Tujuan Pinjaman")
plt.ylabel("Persentase")


# Kebanyakan pinjaman ditujukan untuk pembayaran hutang dan kartu kredit. Persentase tertinggi pinjaman beresiko adalah pinjaman yang ditujukan untuk memulai bisnis kecil, pindahan, kesehatan, dan pernikahan.

# In[36]:


# pesebaran data Target dan perbandingannya dengan lokasi alamat
fig = plt.figure(figsize=(25, 8))
ax = sns.countplot(data=df_target, x='addr_state', hue='target', palette = 'Set2')
ax.set_title('Pesebaran Lokasi Alamat')
plt.xlabel('Lokasi Alamat')
plt.ylabel("Count")
for i in ax.containers:
    ax.bar_label(i,)

ax2 = pd.crosstab(df_target['addr_state'], df_target['target']).apply(lambda x: x*100/x.sum(), axis = 1).plot(kind = 'bar', stacked = True, rot=0, figsize=(18, 5))
ax2.set_title("Persentase Target berdasarkan Lokasi Alamat")
plt.xlabel("Lokasi Alamat")
plt.ylabel("Persentase")


# Mayoritas peminjam berasal dari negara bagian California dan disusul oleh New York, Texas, dan Florida. Persentase pinjaman beresiko terbesar ada pada negara bagian Nevada, Hawaii, dan Alaska. Sedangkan untuk persentase terendah ada pada District of Columbia dan Vermont.

# In[37]:


# pesebaran data Target dan perbandingannya dengan status awal pinjaman
col = 'initial_list_status'
title_c = 'Pesebaran Target berdasarkan Status Awal Pinjaman'
title_b = 'Persentase Target berdasarkan Status Awal Pinjaman'
label = 'Status Awal Pinjaman'

vis_cat(col, title_c, title_b, label)


# Status Awal Pinjaman mayoritas adalah Fractional, meskipun tidak jauh berbeda dengan status Whole. Persentase kredit beresiko juga tidak terlalu berbeda.

# In[38]:


# pesebaran data Target dan perbandingannya dengan lama bekerja
fig = plt.figure(figsize=(13, 5))
ax = sns.countplot(data=df_target, x='emp_length', hue='target', palette = 'Set2')
ax.set_title('Pesebaran Lama Bekerja')
plt.xlabel('Lama Bekerja')
plt.ylabel("Count")
for i in ax.containers:
    ax.bar_label(i,)

ax2 = pd.crosstab(df_target['emp_length'], df_target['target']).apply(lambda x: x*100/x.sum(), axis = 1).plot(kind = 'bar', stacked = True, rot=0, figsize=(13, 5))
ax2.set_title("Persentase Target berdasarkan Lama Bekerja")
plt.xlabel("Lama Bekerja")
plt.ylabel("Persentase")


# Peminjam terbanyak adalah mereka yang sudah bekerja lebih dari 10 tahun, kemudian ada mereka yang bekerja 2 tahun. Persentase pinjaman beresiko terbesar ada pada mereka yang baru bekerja kurang dari setahun, meskipun besaran persentase nya tidak terlalu berbeda dengan yang lain.

# ### Analisa Data Kolom-Kolom Numerikal

# In[39]:


# cek kolom dengan tipe data numerikal
numerical_cols = [cname for cname in df_target.columns
                 if df_target[cname].dtype in ["int64", "float64"]]

#numerical_cols.remove('target')

df_target[numerical_cols].head(10)


# In[40]:


# Analisa statistik deskriptif untuk setiap kolom numerik
df_target[numerical_cols].describe()


# In[41]:


# tampilkan nilai correlation setiap kolom numerik
df_target[numerical_cols].corr()


# In[42]:


# visualisasi correlation setiap kolom numerik menggunakan heatmap
plt.figure(figsize=(23, 13))

heatmap = sns.heatmap(df_target[numerical_cols].corr(), vmin=-1, vmax=1, annot=True, mask=np.triu(df_target[numerical_cols].corr(), k=1))
heatmap.set_title('Correlation Heatmap');


# Korelasi positif banyak terjadi pada data-data yang tentang jumlah pinjaman yang diberikan dan kaitannya dengan hal-hal yang berhubungan dengan pembayarannya. Selain itu, juga terdapat korelasi positif yang tidak terlalu signifikan terhadap jumlah penghasilan peminjam dengan jumlah pinjaman yang diajukan atau yang diterima.

# In[43]:


# visualisasikan pesebaran data setiap kolom numerik menggunakan boxplot
plt.figure(figsize=(20, 85))
for i, col in enumerate(list(numerical_cols), 1):
    plt.subplot(15, 2, i)
    sns.boxplot(x=col, data=df_target[numerical_cols])
    plt.title("Pesebaran Data " + col)


# Untuk data-data yang berkaitan tentang jumlah pinjaman yang diberikan kepada peminjam, memiliki pesebaran yang cukup normal. Begitu juga terhadap pesebaran data DTI dan suku bunga pinjaman.
# 
# Untuk mayoritas data memiliki pola right-skewed. Terutama pada data-data yang berkaitan dengan total saldo kredit yang dimiliki oleh peminjam yang berkorelasi positif dengan pesebaran data pendapatan peminjam.
# 
# Selain itu, banyak data yang right-skewed dikarenakan mayoritas data bernilai 0. Hal ini dipengaruhi oleh variabel-variabel penentu suatu pinjaman itu beresiko atau tidak. Contohnya adalah jumlah penagihan yang dilakukan dalam 12 bulan terakhir; dikarenakan mayoritas data (> 90%) adalah pinjaman yang tidak beresiko/GOOD, maka banyak dari pinjaman tersebut yang tidak perlu ditagih.

# In[44]:


# visualisasikan pesebaran data setiap kolom numerik terhadap TARGET dengan kdeplot
plt.figure(figsize=(25, 90))
for i, col in enumerate(list(numerical_cols), 1):
    plt.subplot(15, 2, i)
    sns.kdeplot(x=col, data=df_target, hue='target', fill=True)
    plt.title("Pesebaran Data " + col)


# Hasil visualisasi pesebaran data numerik berdasarkan data TARGET hampir keseluruhan menunjukkan pola yang sama; semakin meningkat pesebaran data pada suatu rentang nilai, maka data TARGET juga akan mengikuti kenaikan tersebut.
# 
# Namun, terdapat pola unik pada tingkat suku bunga 5-10% memiliki perbandingan pinjaman beresiko yang cukup rendah, meskipun terjadi penigkatan jumlah pinjaman pada rate bunga di-range tersebut.

# ## Modelling

# ### Feature Encoding

# In[45]:


data = df_target.copy()


# In[46]:


# Hapus kolom loan_status untuk menghindari bias saat pemodelan
data = data.drop(columns=['loan_status'])


# In[47]:


# ordinal encoding
from sklearn.preprocessing import OrdinalEncoder

# create all of the categories into lists
term_cat = [' 36 months', ' 60 months']
grad_cat = ['A', 'B', 'C', 'D', 'E', 'F', 'G']
subgrade_cat = ['A1', 'A2', 'A3', 'A4', 'A5',
                 'B1', 'B2', 'B3', 'B4', 'B5',
                 'C1', 'C2', 'C3', 'C4', 'C5',
                 'D1', 'D2', 'D3', 'D4', 'D5',
                 'E1', 'E2', 'E3', 'E4', 'E5',
                 'F1', 'F2', 'F3', 'F4', 'F5',
                 'G1', 'G2', 'G3', 'G4', 'G5',]
emp_cat = ['< 1 year', '1 year', '2 years', '3 years', '4 years', 
           '5 years', '6 years', '7 years', '8 years', '9 years', '10+ years']

# Instantiate the encoder, all of these lists go in one big categories list:
encoder = OrdinalEncoder(categories=[term_cat, grad_cat, subgrade_cat, emp_cat])

data[['term', 'grade', 'sub_grade', 'emp_length']] = encoder.fit_transform(data[['term', 'grade', 'sub_grade', 'emp_length']])


# In[48]:


# label encoding
from sklearn.preprocessing import LabelEncoder

# Fit dan transform kolom-kolom yang hendak di Label Encoding
data['addr_state'] = LabelEncoder().fit_transform(data['addr_state'])


# In[49]:


# One Hot Encoding features yang tersisa
data = pd.get_dummies(data, dtype=int)


# ### Train Test Split

# In[50]:


# Define variabel X dan y
X = data.drop(columns=['target'])
y = data['target']


# In[51]:


from sklearn.model_selection import train_test_split

# split data menjadi training dan testing set dengan perbadingan 80:20
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


# ### Logistic Regression

# In[52]:


from sklearn.linear_model import LogisticRegression

# LogisticRegression menggunakan semua features
# membuat objek LogisticRegression
lreg = LogisticRegression()

# train model LogisticRegression
lreg.fit(X_train, y_train)

# hasil prediksi model
lreg_pred = lreg.predict(X_test)

# Evaluasi model Logistic Regression menggunakan beberapa metrik berbeda
from sklearn.metrics import accuracy_score, precision_score, recall_score, roc_auc_score, confusion_matrix
print(accuracy_score(y_test, lreg_pred))
print(precision_score(y_test, lreg_pred, average='macro'))
print(recall_score(y_test, lreg_pred, average='macro'))
print(roc_auc_score(y_test, lreg.predict_proba(X_test)[:, 1]))
print(confusion_matrix(y_test, lreg_pred))


# ### Random Forest

# In[53]:


from sklearn.ensemble import RandomForestClassifier

classifier_rf = RandomForestClassifier(random_state=42, n_jobs=-1, max_depth=5,
                                       n_estimators=100, oob_score=True)

classifier_rf.fit(X_train, y_train)

# Hasil prediksi
rfclass_pred = classifier_rf.predict(X_test)

# Evaluasi Random Forest
print(accuracy_score(y_test, rfclass_pred))
print(precision_score(y_test, rfclass_pred, average='macro'))
print(recall_score(y_test, rfclass_pred, average='macro'))
print(roc_auc_score(y_test, classifier_rf.predict_proba(X_test)[:, 1]))
print(confusion_matrix(y_test, rfclass_pred))


# ### LightGBM

# In[54]:


# rename the columns' name first
import re

X_train = X_train.rename(columns = lambda x:re.sub('[^A-Za-z0-9_]+', '', x))
X_test = X_test.rename(columns = lambda x:re.sub('[^A-Za-z0-9_]+', '', x))


# In[55]:


# build the lightgbm model
import lightgbm as lgb

clf = lgb.LGBMClassifier()

clf.fit(X_train, y_train)

# Hasil prediksi
clf_pred = clf.predict(X_test)

# Evaluasi LightGBM
print(accuracy_score(y_test, clf_pred))
print(precision_score(y_test, clf_pred, average='macro'))
print(recall_score(y_test, clf_pred, average='macro'))
print(roc_auc_score(y_test, clf.predict_proba(X_test)[:, 1]))
print(confusion_matrix(y_test, clf_pred))


# ### CatBoost

# In[56]:


from catboost import CatBoostClassifier

cbc = CatBoostClassifier(logging_level='Silent')

cbc.fit(X_train, y_train)

cbc_pred = cbc.predict(X_test)

# Evaluasi CatBoost
print(accuracy_score(y_test, cbc_pred))
print(precision_score(y_test, cbc_pred, average='macro'))
print(recall_score(y_test, cbc_pred, average='macro'))
print(roc_auc_score(y_test, cbc.predict_proba(X_test)[:, 1]))
print(confusion_matrix(y_test, cbc_pred))


# ### XGBoost

# In[57]:


import xgboost as xgb

xgb_classifier = xgb.XGBClassifier()

xgb_classifier.fit(X_train, y_train)

xgb_pred = xgb_classifier.predict(X_test)

# Evaluasi XGBoost
print(accuracy_score(y_test, xgb_pred))
print(precision_score(y_test, xgb_pred, average='macro'))
print(recall_score(y_test, xgb_pred, average='macro'))
print(roc_auc_score(y_test, xgb_classifier.predict_proba(X_test)[:, 1]))
print(confusion_matrix(y_test, xgb_pred))


# ### CatBoost Hyperparameter Tuning

# In[58]:


# set parameter
cb_params = {'depth':[3,1,2,6,4,5,7,8,9,10],
          'iterations':[250,100,150,300,200],
          'learning_rate':[0.03,0.001,0.02,0.1,0.2,0.3], 
          'l2_leaf_reg':[3,1,5,10,100,25],
          'border_count':[32,5,10,20,50,100,200]}


# In[59]:


from sklearn.model_selection import RandomizedSearchCV

# Random search of parameters, using 2 fold cross validation, 
# search across 100 different combinations, and use all available cores
cbc_random = RandomizedSearchCV(estimator=cbc, param_distributions=cb_params,
                              n_iter = 100, cv=5, verbose=3, random_state=42, n_jobs=-1,
                              return_train_score=True)


# In[60]:


# fit random search model
cbc_random.fit(X_train, y_train)

# tampilkan parameter terbaik
cbc_random.best_params_


# In[61]:


# ambil parameter terbaik
cbc_best = cbc_random.best_estimator_

# traing model CatBoostRegressor dengan parameter pilihan menggunakan training sets
cbc_best.fit(X_train, y_train)

# hasil prediksi model new CBR
pred_new = cbc_best.predict(X_test)

# Evaluasi CatBoost
print(accuracy_score(y_test, pred_new))
print(precision_score(y_test, pred_new, average='macro'))
print(recall_score(y_test, pred_new, average='macro'))
print(roc_auc_score(y_test, cbc_best.predict_proba(X_test)[:, 1]))
print(confusion_matrix(y_test, pred_new))


# ## ~ END OF FILE ~
