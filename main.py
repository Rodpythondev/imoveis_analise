

import streamlit as st
import pandas as pd
import numpy as np
import folium as fol
from streamlit_folium import folium_static
from folium.plugins import MarkerCluster
import plotly.express as px
from datetime import datetime

st.set_page_config(layout='wide')

def set_feature( data ):
  # ------------------------
  #  Adicionando novos atributos
  # ------------------------
  # Preço por metro quadrado
  data['size_m2'] = data['sqft_living'].apply(lambda x: x * 0.092903)
  data['price_m2'] = data['price'] / data['size_m2']

  return data


def overview_data(data):
  # ------------------------
  #  Data Overview
  # 1.2- Escolher uma ou mais variáveis para visualizar.
  # ------------------------

  # Filtros

  # campos_default = ['id', 'price', 'zipcode', 'size_m2', 'price_m2']

  f_atributos = st.sidebar.multiselect('Selecione as colunas', data.columns.tolist())
  f_zipcode = st.sidebar.multiselect('Selecione o zipcode', data['zipcode'].sort_values(kind='mergesort').unique())

  if f_zipcode != [] and f_atributos != []:
      data = data.loc[data['zipcode'].isin(f_zipcode), f_atributos]

  elif f_zipcode != [] and f_atributos == []:
      data = data.loc[data['zipcode'].isin(f_zipcode), :]

  elif f_zipcode == [] and f_atributos != []:
      data = data.loc[:, f_atributos]
      
  else:
      data = data.copy()

  if st.checkbox('visualizar dados crus'):
      st.markdown(f"<span style='color: darkgreen;font-size: 15px;'>Amostra dos Dados Crus</span>",
                  unsafe_allow_html=True)
      st.dataframe(data.sample(20))

  c1, c2 = st.beta_columns((1.2, 1))

  # ------------------------
  # 1.3- Observar o número total de imóveis, a média de preço, a média da sala de estar e também a média do preço por metro
  #     quadrado em cada um dos códigos postais.
  # ------------------------

  # Contagem por código postal
  df1 = data[['id', 'zipcode']].groupby('zipcode').count().reset_index()
  # Média de preço por código postal
  df2 = data[['price', 'zipcode']].groupby('zipcode').mean().reset_index()
  # Média do tamanho por código postal
  df3 = data[['size_m2', 'zipcode']].groupby('zipcode').mean().reset_index()
  # Média do preço por metro quadrado por código postal
  df4 = data[['price_m2', 'zipcode']].groupby('zipcode').mean().reset_index()

  # merge
  m1 = pd.merge(df1, df2, on='zipcode', how='inner')
  m2 = pd.merge(m1, df3, on='zipcode', how='inner')
  dfmerged = pd.merge(m2, df4, on='zipcode', how='inner')
  dfmerged.columns = ['zipcode', 'total_houses', 'mean_price', 'mean_size', 'mean_price_m2']

  c1.markdown("<span style='color:CadetBlue;font-size: 15px; font-weight: bold'>Valores médios</span>",
              unsafe_allow_html=True)
  c1.dataframe(dfmerged, height=580)

  # ------------------------
  # 1.4- Analisar cada uma das colunas de um modo mais descrito.
  # ------------------------

  # Estatísticas descritivas
  # ------------------------
  num_atributos = data.select_dtypes(include=['int64', 'float64'])
  media = pd.DataFrame(num_atributos.apply(np.mean))
  mediana = pd.DataFrame(num_atributos.apply(np.median))
  std = pd.DataFrame(num_atributos.apply(np.std))
  max_ = pd.DataFrame(num_atributos.apply(np.max))
  min_ = pd.DataFrame(num_atributos.apply(np.min))

  dfestatistica = pd.concat([max_, min_, media, mediana, std], axis=1).reset_index()
  dfestatistica.columns = ['atributos', 'max', 'min', 'media', 'mediana', 'desvio_padrao']

  c2.markdown(f"<span style='color: CadetBlue;font-size: 15px; font-weight: bold'>Estatística Descritiva</span>",
                  unsafe_allow_html=True)
  c2.dataframe(dfestatistica, height=580)

  return None



def portfolio_density(data, geofile):
  # ------------------------
  # 1.5- Um mapa com a densidade de portfólio por região e também densidade de preço
  # ------------------------

  #  Densidade de Portfolio
  # ------------------------
  st.title('Overview por Região')

  c1, c2 = st.beta_columns((1, 1))
  c1.header('Densidade de Portfolio')

  df = data.sample(10)

  # Base Map - Folium
  densidade_map = fol.Map(location=[data['latitude'].mean(),
                                   data['longitude'].mean()],
                         default_zoom_start=15)
  marker_cluster = MarkerCluster().add_to(densidade_map)
  for name, row in df.iterrows():
      fol.Marker( [row['latitude'], row['longitude']],
                 popup='Sold R${0} on: {1}. Features: {2} sqft, {3} bedrooms, {4} bathrooms, year built: {5}'.format(row['price'],
                                                                                                                      row['date'],
                                                                                                                     row['sqft_living'],
                                                                                                                     row['bedrooms'],
                                                                                                                     row['bathrooms'],
                                                                                                                     row['yr_built'])).add_to(marker_cluster)
  with c1:
      folium_static(densidade_map)

  #  Densidade de Preço
  # ------------------------
  # Mapa de preço por região
  c2.header('Densidade de Preço')

  df = data[['price', 'zipcode']].groupby('zipcode').mean().reset_index()
  df.columns = ['zip', 'price']
  # df = df.sample(10)

  regiao_map = fol.Map(location=[data['latitude'].mean(),
                                   data['longitude'].mean()],
                         default_zoom_start=15)


  geofile = geofile[geofile['ZIP'].isin( df['zip'].tolist() )]

  regiao_map.choropleth(data=df, geo_data=geofile, columns=['zip', 'price'], key_on='feature.properties.ZIP',
                        fill_color='YlOrRd',
                        fill_opacity=0.7,
                        line_opacity=0.2,
                        legend_name='MEDIA PREÇO')
  with c2:
      folium_static(regiao_map)

  
  return None



def commercial_distribution(data):
  # ------------------------
  #  Distribuicao dos imoveis por categorias comerciais
  # ------------------------
  st.sidebar.title('Opções Comerciais')
  st.title('Atributos Comerciais')

  #  Media de Preço por Ano
  # ------------------------
  st.header('Média de preços por Ano de Construção')

  # Filtros
  min_year_built = int(data['yr_built'].min())
  max_year_built = int(data['yr_built'].max())
  st.sidebar.write('Selecione o Ano de Construção Máximo')
  f_yr_built = st.sidebar.slider('Ano de Construção', min_year_built, max_year_built, max_year_built)

  # Selecao de dados
  df = data.loc[data['yr_built'] <= f_yr_built]
  df = df[['yr_built', 'price']].groupby('yr_built').mean().reset_index()

  # Plot
  fig = px.line(df, x='yr_built', y='price')
  st.plotly_chart(fig, use_container_width=True)

  #  Media de Preço por Dia
  # ------------------------
  st.header('Média de preços por Dia')

  # Filtros
  #  st.write(data['date'][0])
  try:
    min_date = datetime.strptime(str(data['date'].min()), '%Y-%m-%d %H:%M:%S')
    max_date = datetime.strptime(str(data['date'].max()), '%Y-%m-%d %H:%M:%S')
  except ValueError:
    min_date = datetime.strptime(str(data['date'].min()), '%Y-%m-%d')
    max_date = datetime.strptime(str(data['date'].max()), '%Y-%m-%d')
    
  st.sidebar.write('Selecione a Data Máxima')
  f_date = st.sidebar.slider('Data', min_date, max_date, max_date)

  # Selecao de dados
  
  data['date'] = pd.to_datetime(data['date'])
  df = data.loc[data['date'] < f_date]
  df = df[['date', 'price']].groupby('date').mean().reset_index()

  # Plot
  fig = px.line(df, x='date', y='price')
  st.plotly_chart(fig, use_container_width=True)

  #  Histograma de Distribuição do Preço
  # ------------------------
  st.header('Distribuição de preço')
  st.sidebar.subheader('Selecione o Preço Máximo')

  # Filtro
  min_price = int(data['price'].min())
  max_price = int(data['price'].max())
  mean_price = int(data['price'].mean())

  f_price = st.sidebar.slider('Preço', min_price, max_price, mean_price)
  df = data.loc[data['price'] < f_price]

  # Plot
  fig = px.histogram(df, x='price', nbins=50)
  st.plotly_chart(fig, use_container_width=True)

  return None



def attributes_distribution(data):
  # ------------------------
  #  Distribuicao dos imoveis por categorias físicas
  # ------------------------
  st.sidebar.title('Opções de Atributos')
  st.title('Atributos das Casas')

  # Filtros
  f_bedrooms = st.sidebar.selectbox('Nº max. de quartos', sorted(set(data['bedrooms'].unique())))
  f_bathrooms = st.sidebar.selectbox('Nº max. de banheiros', sorted(set(data['bathrooms'].unique())))
  f_floors = st.sidebar.selectbox('Nº max. de andares', sorted(set(data['floors'].unique())))
  f_waterview = st.sidebar.checkbox('Somente casas com vista para água')

  c1, c2 = st.beta_columns(2)

  #  Casas por quartos
  c1.header('Distribuição de casas por quartos')
  df = data.loc[data['bedrooms'] <= f_bedrooms]
  fig = px.histogram(df, x='bedrooms', nbins=10)
  c1.plotly_chart(fig, use_container_width=True)

  #  Casas por banheiros
  c2.header('Distribuição de casas por banheiros')
  df = data.loc[data['bathrooms'] <= f_bathrooms]
  fig = px.histogram(df, x='bathrooms', nbins=10)
  c2.plotly_chart(fig, use_container_width=True)


  c1, c2 = st.beta_columns(2)

  #  Casas por andares
  c1.header('Distribuição de casas por andares')
  df = data.loc[data['floors'] <= f_floors]
  fig = px.histogram(df, x='floors', nbins=10)
  c1.plotly_chart(fig, use_container_width=True)

  #  Casas por vista para agua
  if f_waterview:
    df = data[data['waterfront'] == 1]
  else:
    df = data.copy()

  c2.header('Distribuição de casas de acordo com a vista')
  fig = px.histogram(df, x='waterfront', nbins=10)
  c2.plotly_chart(fig, use_container_width=True)



if __name__ == '__main__':
  
  # Cabeçalho
  st.title('Aula #06 - Tipos de Visualização de Dados')
  st.subheader('House Rocket Company')
  st.markdown(f"<span style='color: blue;font-size: 20px;font-weight: bold;'>Welcome to House Rocket Data Analisys</span>", unsafe_allow_html=True)

  # ETL
  # Data Extraction
  dfhouses = fd.load_data(data_url='dados/originais/kc_house_data.csv.zip', date_f='date')
  geourl = 'https://opendata.arcgis.com/datasets/83fc2e72903343aabff6de8cb445b81c_2.geojson'
  geofile = fd.load_geofile(geourl)

  # Transformation
  dfhouses = set_feature(dfhouses)

  overview_data(dfhouses)

  portfolio_density(dfhouses, geofile)

  commercial_distribution(dfhouses)

  attributes_distribution(dfhouses)
