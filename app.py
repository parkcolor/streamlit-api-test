import streamlit as st
import yfinance as yf
import matplotlib.pyplot as plt
from datetime import datetime
import pandas as pd
import numpy as np

# api 호출을 위한 라이브러리 임포트
import requests

# 프로펫 라이브러리 임포트
from fbprophet import Prophet


# 야후 금융에서 주식정보를 제공하는 라이브러리 yfinance 이용
# 주식정보를 불러오고 차트 그리는거 합니다.

# 해당 주식에 대한 트윗글들을 불러올 수 있는 API가 있다
# stocktwits.com 에서 제공하는 restful API를 호출해서 데이터 가져오는 실습

def main():
    st.header('Online Stock Price Ticker')

    #yfinance 실행
    symbol = st.text_input('심볼입력') # 주식종목 이름
    data = yf.Ticker(symbol) # 원하는 종목의 데이터를 불러온다

    today = datetime.now().date().isoformat()
    print(today)

    df = data.history(start= '2010-06-01', end = today)

    st.dataframe(df)

    st.subheader('종가')
    
    st.line_chart(df['Close'])

    st.subheader('거래량')

    st.line_chart(df['Volume'])

    # data.info # 회사정보
    # data.calender #어닝 정보
    # data.major_holder #대주주
    # data.institutional_holders #거래은행
    # data.recommendations #추천은행
    div_df = data.dividends #배당금

    st.dataframe(div_df.resample('Y').sum())

    new_df = div_df.reset_index()
    new_df['Year'] = new_df['Date'].dt.year

    st.dataframe(new_df)

    fig = plt.figure()
    plt.bar(new_df['Year'], new_df['Dividends'])
    st.pyplot(fig)

    # 여러 주식 데이터를 한번에 보여주기

    favorites = ['msft','tsla','nvda','aapl','amzn'] 

    f_df = pd.DataFrame()

    for stock in favorites :
        f_df[stock] = yf.Ticker(stock).history(start='2010-01-01', end= today)['Close']

    st.dataframe(f_df)

    st.line_chart(f_df)

    # stock트윗의 API를 호출한다
    res = requests.get('https://api.stocktwits.com/api/2/streams/symbol/{}.json'.format(symbol))

    #json 형식이므로 .json(활용)
    res_data = res.json()

    # 파이썬의 딕셔너리와 리스트 조합으로 활용가능
    # st.write(res_data)

    # for message in res_data['messages'] :

    #     col1, col2 = st.beta_columns([1,4])

    #     with col1 :
    #         st.image(message['user']['avatar_url'])
    #     with col2 :
    #         st.write('유저 이름 : ' + message['user']['username'])
    #         st.write('트윗내용 : ' + message['body'])
    #         st.write('트윗시간 : ' + message['created_at'])

    
    # 프로펫으로 주식 예측하기
    p_df = df.reset_index()
    p_df.rename(columns = {'Date' : 'ds', 'Close' :'y'}, inplace = True)

    # st.dataframe(p_df)

    # 예측 가능
    m = Prophet()
    m.fit(p_df)

    future = m.make_future_dataframe(periods=365)
    forecast = m.predict(future)

    fig1 = m.plot(forecast)
    st.pyplot(fig1)

    fig2 = m.plot_components(forecast)
    st.pyplot(fig2)

    st.dataframe(forecast)










if __name__ == '__main__':
    main()