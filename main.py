import folium
import geopandas
import numpy as np
import pandas as pd
import streamlit as st
from folium.plugins import MarkerCluster
from streamlit_folium import folium_static
import seaborn as sns
import plotly.express as px
from datetime import datetime
st.set_page_config(layout='wide')
st.title('House Rocket Company')
st.markdown('Welcome to House Rocket Data Analysis')
st.header('Load data')

@st.cache(allow_output_mutation=True)
def get_data(path):
    df = pd.read_csv(path)
    df['date'] = pd.to_datetime(df['date'])
    return df


@st.cache(allow_output_mutation=True)
def get_geofile(web):
    geo = geopandas.read_file(web)

    return geo
def  set_feature(data):
    # add new feature
    data['price_m2'] = data['price'] / data['sqft_lot']
    return data

def overview(data):
    f_atribulto = st.sidebar.multiselect('Enter columns', data.columns)

    f_zipcode = st.sidebar.multiselect('enter zipcode', data['zipcode'].unique())

    if (f_atribulto != []) & (f_zipcode != []):
        data = data.loc[data['zipcode'].isin(f_zipcode), f_atribulto]
    elif (f_atribulto == []) & (f_zipcode != []):
        data = data.loc[data['zipcode'].isin(f_zipcode), :]
    elif (f_atribulto != []) & (f_zipcode == []):
        data = data.loc[:, f_atribulto]
    else:
        data = data.copy()
    st.dataframe(data)
    c1, c2 = st.columns(2)
    df1 = data[['id', 'zipcode']].groupby('zipcode').count().reset_index()
    # merge
    df2 = data[['price', 'zipcode']].groupby('zipcode').mean().reset_index()
    df3 = data[['sqft_living', 'zipcode']].groupby('zipcode').mean().reset_index()
    df4 = data[['price_m2', 'zipcode']].groupby('zipcode').mean().reset_index()
    m1 = pd.merge(df1, df2, on='zipcode', how='inner')
    m2 = pd.merge(m1, df3, on='zipcode', how='inner')
    df = pd.merge(m2, df4, on='zipcode', how='inner')
    df.columns = ['zipcode', 'total houses', 'price', 'sqft living', 'price/m2']
    with c1:
        st.dataframe(df)

    # statistic descriptive
    num_attributes = data.select_dtypes(include=['int64', 'float64'])
    media = pd.DataFrame(num_attributes.apply(np.mean))
    mediana = pd.DataFrame(num_attributes.apply(np.median))
    std = pd.DataFrame(num_attributes.apply(np.std))
    max_ = pd.DataFrame(num_attributes.apply(np.max))
    min_ = pd.DataFrame(num_attributes.apply(np.min))

    df1 = pd.concat([max_, min_, media, mediana, std], axis=1).reset_index()
    df1.columns = ['attributes', 'max', 'min', 'mean', 'median', 'std']
    with c2:
        st.dataframe(df1)
    return None

def portfolio_density(data, geofile):
    # ======================
    # densidade de protifolio
    # ==================
    st.title('Region Overview')
    c1, c2 = st.columns((1, 1))
    c1.header('Portfolio Density')

    df = data

    # Base Map - Folium
    density_map = folium.Map(location=[data['lat'].mean(),
                                       data['long'].mean()],
                             default_zoom_start=15)

    marker_cluster = MarkerCluster().add_to(density_map)
    for name, row in df.iterrows():
        folium.Marker([row['lat'], row['long']],
                      popup='Sold R${0} on: {1}. Features: {2} sqft, {3} bedrooms, {4} bathrooms, year built: {5}'.format(
                          row['price'],
                          row['date'],
                          row['sqft_living'],
                          row['bedrooms'],
                          row['bathrooms'],
                          row['yr_built'])).add_to(marker_cluster)


    folium_static(density_map)



    df = data[['price', 'zipcode']].groupby('zipcode').mean().reset_index()
    df.columns = ['ZIP', 'PRICE']



    geofile = geofile[geofile['ZIP'].isin(df['zip'].tolist())]

    region_price_map = folium.Map(location=[data['lat'].mean(),
                                            data['long'].mean()],
                                  default_zoom_start=15)
    st.header('Price Density')
    region_price_map.choropleth(data=df,
                                geo_data=geofile,
                                columns=['ZIP', 'PRICE'],
                                key_on='feature.properties.ZIP',
                                fill_color='YlGn',
                                fill_opacity=0.7,
                                line_opacity=0.4,
                                legend_name='AVG PRICE')


    folium_static(region_price_map)
    return None

def commercial(data):
    # ==============================================
    # Distribuicao dos imoveis por categorias comerciais

    st.sidebar.title('Commercial Options')
    # Filter yr built

    min_yr_built = int(data['yr_built'].min())
    max_yr_built = int(data['yr_built'].max())
    mean_yr_built = int(data['yr_built'].mean())

    st.sidebar.subheader('Select Max Year Built')
    f_yr_built = st.sidebar.slider('Year Built', min_yr_built, max_yr_built, mean_yr_built)
    df = data.loc[data['yr_built'] < f_yr_built]

    st.title('Commercial Attributes')
    # Avg Price per year
    st.header('Avg Price per Year')
    df = df[['yr_built', 'price']].groupby('yr_built').mean().reset_index()
    fig = px.line(df, x='yr_built', y='price')
    st.plotly_chart(fig, use_container_width=True)
    data['date'] = pd.to_datetime(data['date']).dt.strftime('%Y-%m-%d')
    # Filter day built
    st.sidebar.subheader('Select Max Day Built')
    min_date = datetime.strptime(data['date'].min(), '%Y-%m-%d')
    max_date = datetime.strptime(data['date'].max(), '%Y-%m-%d')
    f_date = st.sidebar.slider('Date', min_date, max_date, min_date)
    data['date'] = pd.to_datetime(data['date'])
    df = data.loc[data['date'] < f_date]

    # Avg Price per Day
    st.header('Avg Price per Day')
    df = df[['date', 'price']].groupby('date').mean().reset_index()
    fig = px.line(df, x='date', y='price')
    st.plotly_chart(fig, use_container_width=True)

    # ----------------------
    # Histograma
    st.header('Price Distribution')
    st.sidebar.subheader('Select Max Price')
    min_price = int(data['price'].min())
    max_price = int(data['price'].max())
    avg_price = int(data['price'].mean())
    f_price = st.sidebar.slider('Price', min_price, max_price, avg_price)
    df = data.loc[data['price'] < f_price]
    fig = px.histogram(df, x='price', nbins=50)
    st.plotly_chart(fig, use_container_width=True)
    return None

def attributesr_distribution(data):
    # Districuicao dos imoveis por catergoria fisica
    # filters
    st.sidebar.title('Attributes Options')
    f_bedrooms = st.sidebar.selectbox('Max number of bedrooms', sorted(set(data['bedrooms'].unique())))
    f_bathrooms = st.sidebar.selectbox('Max number of bathrooms', sorted(set(data['bathrooms'].unique())))
    st.title('House Attributes')
    # house por bedrooms
    df = data[data['bedrooms'] < f_bedrooms]
    fig = px.histogram(df, x='bedrooms', nbins=19)
    st.plotly_chart(fig, use_container_width=True)

    # house por bathrooms
    df = data[data['bathrooms'] < f_bathrooms]
    fig = px.histogram(df, x='bathrooms', nbins=19)
    st.plotly_chart(fig, use_container_width=True)

    # filters
    f_floors = st.sidebar.selectbox('Max number of floors', sorted(set(data['floors'].unique())))
    f_waterfront = st.sidebar.checkbox('Only Houses with Water View')

    # house por floors
    df = data[data['floors'] < f_floors]
    fig = px.histogram(df, x='floors', nbins=19)
    st.plotly_chart(fig, use_container_width=True)

    # house por water view
    if f_waterfront:
        df = data[data['waterfront'] == 1]
    else:
        df = data.copy()
    fig = px.histogram(df, x='waterfront', nbins=19)
    st.plotly_chart(fig, use_container_width=True)
    return None

if  __name__ == '__main__':
    # ETL
    # data extration
    url = 'https://gis.marlborough.govt.nz/server/rest/services/OpenData/OpenData2/MapServer/12/query?outFields=*&where=1%3D1&f=geojson'
    geofile = get_geofile(url)
    data = get_data('kc_house_data.csv')
    # transformation
    data = set_feature(data)
    overview(data)
    portfolio_density(data, geofile)
    commercial(data)
    attributesr_distribution(data)
    # loading













