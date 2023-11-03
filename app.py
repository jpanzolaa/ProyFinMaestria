import streamlit as st
#Librerias de python
import datetime
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import plotly.express as px
#Librerias para abrir la serializacion de archivos
import pickle as pk
import os

# Libreria de Maquina de aprendizaje
from sklearn import ensemble, tree, linear_model
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import r2_score, mean_squared_error
from sklearn.utils import shuffle
import warnings
from scipy import stats
from scipy.stats import norm, skew

warnings.filterwarnings('ignore')

# datos de Entrenamiento
train_path = "data/train.csv"
test_path = "data/test.csv"
#real = "data/RealEstate.csv" <- Eliminar
head = 'images/img01.jpg'

@st.cache_data
def load_train_data(train_path):
    return pd.read_csv(train_path)

@st.cache_data
def load_test_data(test_path):
    return pd.read_csv(test_path)

@st.cache_data # <- Eliminar
def load_data(real):
    return pd.read_csv(real)


def save_data(value,res):# <- Eliminar
    file = 'db.csv'
    if not os.path.exists(file):
        with open(file,'w') as f:
            f.write("OverallQual,GrLivArea,GarageCars,GarageArea,TotalBsmtSF,fstFlrSF,FullBath,TotRmsAbvGrd,YearBuilt,YearRemodAdd,GarageYrBlt,MasVnrArea,Fireplaces,BsmtFinSF1,Result\n")
    with open(file,'a') as f:
        data = f"{OverallQual},{GrLivArea},{GarageCars},{GarageArea},{TotalBsmtSF},{fstFlrSF},{FullBath},{TotRmsAbvGrd},{YearBuilt},{YearRemodAdd},{GarageYrBlt},{MasVnrArea},{Fireplaces},{BsmtFinSF1},{res}\n"        
        f.write(data)

st.sidebar.image(head, caption="Proyecto prediccion del precio de vivienda",use_column_width=True)




st.title("Analisis del precio de vivienda")

menu=["Precidccion Vivienda","Visualizacion datos","Acerca de","Modo Visual"]
choices=st.sidebar.selectbox("Menu Bar",menu)

if choices=='Precidccion Vivienda':
    st.subheader("Precidccion Vivienda")
    OverallQual=st.selectbox("Seleccione la calidad general (10 es \"Muy excelente\" y 1 es \"muy pobre\")",(10,9,8,7,6,5,4,3,2,1))
    GrLivArea= st.number_input("Ingrese el area de la sala (en pies cuadrados)",value=0,min_value=0,format='%d')
    GarageArea=st.number_input("Ingrese el área del garaje (en pies cuadrados)",value=0.0,format='%f',step=1.0)
    GarageCars=st.number_input("Número de cupos (autos por garaje)",min_value=1.0,max_value=10.0,step=1.0,format='%f')
    TotalBsmtSF=st.number_input("Ingrese el área del sótano (en pies cuadrados)",value=0.0,format='%f',step=1.0)
    fstFlrSF=st.number_input("Ingrese el área del primer piso (en pies cuadrados)",value=0,format='%d')
    FullBath=st.number_input("Ingrese el número de baños",min_value=1,max_value=10,format='%d')
    TotRmsAbvGrd=st.number_input("Ingrese el número de habitaciones",min_value=1,max_value=10,format='%d')
    years=tuple([i for i in range(1872,2011)])
    YearBuilt=st.selectbox("Seleccione la calidad general (10 es \"Muy excelente\" y 1 es \"muy pobre\")",years)
    remyears=tuple([i for i in range(1950,2011)])
    YearRemodAdd=st.selectbox("Seleccione la fecha de remodelación (igual que la fecha de construcción si no hay remodelaciones ni adiciones)",remyears)
    garyears=tuple([i for i in range(1872,2011)])
    garyears=tuple(map(float,garyears))
    GarageYrBlt=st.selectbox("Seleccione el año en que se construyó el garaje)",garyears)
    MasVnrArea=st.number_input("Área de revestimiento de mampostería (en pies cuadrados)",value=0.0,format='%f',step=1.0)
    Fireplaces=st.number_input("Seleccione el número de chimeneas",min_value=1,max_value=10,format='%d')
    BsmtFinSF1=st.number_input("Ingrese al área terminada del sótano (en pies cuadrados)",value=0,format='%d')
    submit = st.button('Predecir')
    if submit:
        st.success("Prediccion realizada")
        value=[OverallQual,GrLivArea,GarageCars,GarageArea,TotalBsmtSF,fstFlrSF,FullBath,TotRmsAbvGrd,YearBuilt,YearRemodAdd,GarageYrBlt,MasVnrArea,Fireplaces,BsmtFinSF1]
        df=pd.DataFrame(value).transpose()
        # st.dataframe(df)
        model=pk.load(open('model & scaler/rfrmodel.pkl','rb'))
        scaler=pk.load(open('model & scaler/scale.pkl','rb'))
        scaler.transform(df)
        ans=int(model.predict(df)) * 5
        st.subheader(f"The price is {ans} (INR) ")
        save_data(value,ans)

if choices=='Visualizacion datos':
    st.subheader("Datos")
    st.info("ampliar para ver los datos claramente")
    if os.path.exists("db.csv"):
        data = pd.read_csv('db.csv')
        st.write(data)
    else:
        st.error("Pruebe alguna predicción y los datos estarán disponibles aquí.")
if choices=='Acerca de':
    st.subheader("Acerca del proyecto")
    info='''
        El valor de una casa es simplemente más que la ubicación y los metros cuadrados. Al igual que las características que componen a una persona, una persona educada querría saber todos los aspectos que dan valor a una casa.

         Aprovecharemos todas las variables de características disponibles para usar y las usaremos para analizar y predecir los precios de la vivienda.

         Vamos a dividir todo en pasos lógicos que nos permitan garantizar los datos más limpios y realistas para que nuestro modelo pueda realizar predicciones precisas.

         - Cargar datos y paquetes
         - Análisis de la Variable de Prueba (Precio de Venta)
         - Análisis multivariable
         - Imputar datos faltantes y limpiar datos
         - Transformación de características/Ingeniería
         - Modelado y Predicciones
    '''
    st.markdown(info,unsafe_allow_html=True)

if choices=='Modo Visual':
    st.subheader("Data Visualization")      

    train_data = load_train_data(train_path)
    test_data = load_test_data(test_path)


    if st.checkbox("view dataset colum description"):
        st.subheader('displaying the column wise stats for the dataset')
        st.write(train_data.columns)
        st.write(train_data.describe())

    st.subheader('Correlation b/w dataset columns')
    #numeric_cols = train_data.select_dtypes(include=[np.number])
    #corrmatrix = train_data.select_dtypes(include=[np.number]).corr()
    numeric_cols = train_data.select_dtypes(include=[np.number])
    corrmatrix = numeric_cols.corr()
    f,ax = plt.subplots(figsize=(20,9))
    sns.heatmap(corrmatrix,vmax = .8, annot=True)
    st.pyplot(f)

    st.subheader("Características más correlacionadas")
    #top_corr = train_data.corr()
    #top_corr_feat = corrmatrix.index[abs(corrmatrix['SalePrice'])>.5]
    top_corr = train_data.select_dtypes(include=[np.number])
    top_corr_feat = numeric_cols.corr()
    plt.figure(figsize=(10,10))
    sns.heatmap(train_data[top_corr_feat].corr(), annot=True, cmap="RdYlGn")
    st.pyplot(f)

    st.subheader("Comparación de la calidad general con el precio de venta")
    #sns.barplot(train_data.OverallQual, train_data.SalePrice)
    sns.barplot(x='OverallQual', y='SalePrice', data=train_data, ax=ax)
    st.pyplot(f)
    st.set_option('deprecation.showPyplotGlobalUse', False)

    st.subheader("Visualización de diagramas de pares para describir la correlación fácilmente")
    sns.set()
    cols = ['SalePrice', 'OverallQual', 'GrLivArea', 'GarageCars', 'TotalBsmtSF', 'FullBath', 'YearBuilt']
    sns.pairplot(train_data[cols], size=2.5)
    st.pyplot()

    st.subheader("Análisis de la columna Precio de venta en el conjunto de datos")
    sns.distplot(train_data['SalePrice'] , fit=norm)# Get the fitted parameters used by the function
    (mu, sigma) = norm.fit(train_data['SalePrice'])
    st.write( '\n mu = {:.2f} and sigma = {:.2f}\n'.format(mu, sigma))

    plt.legend(['Normal dist. ($\mu=$ {:.2f} and $\sigma=$ {:.2f} )'.format(mu, sigma)], loc='best')
    plt.ylabel('Frequency')
    plt.title('Distribucion SalePrice')
    st.pyplot(f)

    fig = plt.figure(figsize=(10,10))
    res = stats.probplot(train_data['SalePrice'], plot=plt,)
    st.pyplot(f)