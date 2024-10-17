import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt 
import seaborn as sns
import plotly.express as px 
import warnings as w 
w.filterwarnings('ignore')

df = pd.read_csv("D:/Users/sefa.erkan/Desktop/Kira/antalya_kiralik_ev.csv")
df.head(5)

df.drop("Unnamed: 0",axis=1,inplace=True)

df.describe()

df.describe(include='object')

df.info()

df.duplicated().sum() # Tekrar eden satıları bulduk

df.drop_duplicates(inplace=True) # Tekrar eden 2 satırı sildik

df.nunique() # her sütündan kac tane unique deger var

def colm(df): # Kategorik, Nümerik, Kardinal sütünları bulmak
    cat_cols = [col for col in df.columns if str(df[col].dtypes) in ['category','object','bool']]
    num_but_cat = [col for col in df.columns if df[col].nunique() < 10 and df[col].dtypes in ['int','float']]
    cat_but_car = [col for col in df.columns if df[col].nunique() > 15 and str(df[col].dtypes) in ['category','object']]
    cat_cols = cat_cols + num_but_cat
    cat_cols = [col for col in cat_cols if col not in cat_but_car]
    num_cols = [col for col in df.columns if df[col].dtypes in ['int','float']]
    num_cols = [col for col in num_cols if col not in cat_cols]

    print(f'Kategorik : {cat_cols} \n Nümerik : {num_cols} \n Kardinal : {cat_but_car}')
    return cat_cols,num_cols,cat_but_car

cat_cols,num_cols,car_cols = colm(df)

df['bina_yas'] = df['bina_yas'].replace({
    '0': 0,
    '1-5 arası': 3,
    '5-10 arası': 7.5,
    '11-15 arası': 13,
    '16-20 arası': 18,
    '21-25 arası': 23,
    '26-30 arası': 28,
    '31 ve üzeri': 35
}).astype(int) # Bina yaşını sayısal değerlere cevirdim.

df['net_brut_orani'] = df['net_alan_m2'] / df['brut_alan_m2'] #Kullanılabilirlik

df['isitma_asansor'] = df['isitma_turu'] + '_' + df['asansor'].astype(str) 

df['isitma_otopark'] = df['isitma_turu'] + '_' + df['otopark'].astype(str)

df['aidat_depozito'] = (df['aidat']*12) + df['depozito']

df.drop(['aidat','depozito'],axis=1,inplace=True) # Aidat ve Depozito sutunlarını sildik

df.groupby('mahalle').agg({'fiyat':'mean','bina_yas':'mean','aidat_depozito':'mean'}).sort_values(by='fiyat',ascending=False)
# Mahalleye göre fiyat, bina yaşı ve aidat-depozito ortalamalar

df[df['sahibi']==1].groupby('mahalle')['sahibi'].count().sort_values(ascending=False) 
# Mahalleye göre ev sahibi sayıları

df.groupby(cat_cols).agg({'fiyat':'mean'}).sort_values(by='fiyat',ascending=False).reset_index()[0:15]
# Kategorik sütünları kullanarak fiyat neden yüksek onu öğrenemye çalışıyoruz 

px.box(df,x='fiyat') # aykırı değerler için

outliers = df[['fiyat']].quantile(q=.99) # üst sınır belirliyoruz
outliers

df_non_outliers=df[df['fiyat']<outliers[0]] # üst sınırdan yüksek olanları veri setinden çıkartıyoruz

px.box(df_non_outliers,x='fiyat') #  88775.0 değenrinden yüksek değerler atıldı 

cat_cols,num_cols,car_cols = colm(df_non_outliers)

df_non_outliers.groupby(cat_cols).agg({'fiyat':'mean'}).sort_values(by='fiyat',ascending=False).reset_index()[0:15]
# 88775.0'den yüksekfiyatı bulunan sütünlar yeni veri setinde bulunmuyor

df_non_outliers.groupby('mahalle').agg({'fiyat':'mean','bina_yas':'mean','aidat_depozito':'mean'}).sort_values(by='fiyat',ascending=False)
# aykırı değer bulunan mahallelerin ortalama fiyatı düşmüş.

mahalle_vc=df_non_outliers['mahalle'].value_counts()
other = mahalle_vc[mahalle_vc<=10]
df_non_outliers['mahalle'] = df_non_outliers['mahalle'].apply(lambda x: 'other' if x in other else x)
# çok fazla ve korelasyonu düşük sütün çıkmaması için vc 10 dan düşük olanları other(diğer) olarak değiştiriyoruz

df_non_outliers['dairenin_bulundugu_kat'] = df_non_outliers['dairenin_bulundugu_kat'].apply(lambda x:0 if (x=='Giriş Katı' or x=='Bahçe Katı' or x=='Yüksek Giriş' or x=='Zemin Kat') else(-1 if (x=='Bodrum Kat' or x=='Giriş Altı Kot 2') else(-5 if (x=='Çatı Katı' or x=='Villa Tipi') else x))).astype('int')
# Giriş katı yada ona benzor olanrı 0 olarak değiştirdik
# Diğer değerleride mantıklı şekilde doldurduk

from sklearn.preprocessing import LabelEncoder # Kategorik verileri sayısallaştırma
le = LabelEncoder()
df_non_outliers['mahalle'] = le.fit_transform(df_non_outliers['mahalle'])
# Aynı mahalleler aynı değer
df_non_outliers['dairenin_bulundugu_kat'] = le.fit_transform(df_non_outliers['dairenin_bulundugu_kat'])
# Aynı kat aynı değer

cat_cols.append('isitma_asansor')
cat_cols.append('isitma_otopark')

dummies_df = pd.get_dummies(df_non_outliers[cat_cols],drop_first=True)
# Kategorik sütünları dönüştürme

# Dönüştürdüğümüz sütünları veri setinden çıkartıyoruz
for col in cat_cols:
    df_non_outliers.drop(columns=col, axis=1, inplace=True) 
df_non_outliers = pd.concat([df_non_outliers,dummies_df],axis=1)
# Dönüştürdüğümüz değerler ile veri setini birleştiriyoruz

df_non_outliers.corr()['fiyat'].sort_values(ascending=False,key=abs)[1:25] 
# Fiyat ile korelasyonu en yüksek olan sütunlar

from sklearn.model_selection import train_test_split

X = df_non_outliers.drop('fiyat',axis=1)
y = df_non_outliers['fiyat']
X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=.2,random_state=43)

## Model eğitimi
from sklearn.model_selection import cross_val_score
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import SVR
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor
from catboost import CatBoostRegressor

def compare_models(X_train,X_test,y_train, y_test):
    
    models = [
        LinearRegression(),
        Ridge(),
        Lasso(),
        DecisionTreeRegressor(),
        RandomForestRegressor(),
        SVR(),
        XGBRegressor(),
        LGBMRegressor(verbose=-1),
    ]
    
    results = []
    for model in models:
        model_name = model.__class__.__name__

        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        rmse = mean_squared_error(y_test, y_pred, squared=False)

        r2 = r2_score(y_test, y_pred)
        scores = cross_val_score(model, X_train, y_train, cv=5, scoring='neg_mean_squared_error')
        rmse_cv = np.sqrt(-scores.mean())
        print(f"{model_name} = RMSE: {rmse}, RMSEcv : {rmse_cv}, r2 Score: {r2}")

compare_models(X_train,X_test,y_train, y_test)       

model = CatBoostRegressor(verbose=False)
model.fit(X_train, y_train,verbose=False)
y_pred = model.predict(X_test)
rmse = mean_squared_error(y_test, y_pred, squared=False)
r2Score = r2_score(y_test, y_pred)
print(f"CatBoostRegressor RMSE: {rmse} \n CatBoostRegressor r2: {r2Score}")

plt.figure(figsize=(10, 6))
plt.scatter(y_test, y_pred, color='navy', alpha=0.3, label='Tahmin Değerler')
plt.scatter(y_test, y_test, color='yellow', alpha=0.4, label='Gerçek Değerler')
plt.xlabel('Gerçek Değerler')
plt.ylabel('Tahmin Değerler')
plt.title('Gerçek vs Tahmin')
plt.legend()
plt.show()