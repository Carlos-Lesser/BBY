import streamlit as st
import pandas as pd
import plotly.express as px
from fbprophet import Prophet
import plotly.io as pio
pio.renderers.default = 'iframe'


# Set page title
st.set_page_config(page_title='Sellthrough Data Analysis', page_icon=':bar_chart:')

# Upload data
st.subheader('Upload Data')
file = st.file_uploader('Choose a CSV file', type='csv')

# Analyze data
if file is not None:
    # Read in data
    sellthrough_data = pd.read_csv(file)

    # Display raw data
    st.subheader('Raw Data')
    st.write(sellthrough_data)

    # Get unique stores
    stores = sellthrough_data['Store #'].unique()

    # Filter data by store
    store = st.selectbox('Select a store', stores)
    filtered_data = sellthrough_data[sellthrough_data['Store #'] == store]

    # Display filtered data
    st.subheader('Filtered Data')
    st.write(filtered_data)

    # Get unique buyer SKUs
    buyer_skus = filtered_data['Buyer SKU'].unique()

    # Filter data by buyer SKU
    buyer_sku = st.selectbox('Select a buyer SKU', buyer_skus)
    filtered_data = filtered_data[filtered_data['Buyer SKU'] == buyer_sku]

    # Display filtered data
    st.subheader('Filtered Data')
    st.write(filtered_data)

    # Create line chart of sellthrough data
    fig = px.line(filtered_data, x='End Date', y='Qty Sold', title=f'Sellthrough Data for {buyer_sku} at Store #{store}')
    st.plotly_chart(fig)

    # Create bar chart of top selling items
    top_selling_items = filtered_data.groupby('Item Description')['Qty Sold'].sum().sort_values(ascending=False).head(10)
    fig = px.bar(top_selling_items, x=top_selling_items.index, y='Qty Sold', title=f'Top Selling Items for {buyer_sku} at Store #{store}')
    st.plotly_chart(fig)

    # Create pie chart of sellthrough by UPC/EAN
    sellthrough_by_upc = filtered_data.groupby('UPC/EAN')['Qty Sold'].sum().sort_values(ascending=False).reset_index()
    fig = px.pie(sellthrough_by_upc, values='Qty Sold', names='UPC/EAN', title=f'Sellthrough by UPC/EAN for {buyer_sku} at Store #{store}')
    st.plotly_chart(fig)

    # Create forecast of sellthrough data
    forecast_data = filtered_data[['End Date', 'Qty Sold']]
    forecast_data.columns = ['ds', 'y']

    # Create and fit Prophet model
    m = Prophet()
    m.fit(forecast_data)

    # Create future dataframe
    future = m.make_future_dataframe(periods=30)

    # Make predictions
    forecast = m.predict(future)

    # Create line chart of forecast
    fig = px.line(forecast, x='ds', y='yhat', title=f'Forecast of Sellthrough Data for {buyer_sku} at Store #{store}')
    fig.add_scatter(x=forecast['ds'], y=forecast['yhat_upper'], mode='lines', name='Upper Bound')
    fig.add_scatter(x=forecast['ds'], y=forecast['yhat_lower'], mode='lines', name='Lower Bound')
    st.plotly_chart(fig)
    
    
