# import packages
import pandas as pd
import requests
from bs4 import BeautifulSoup


# Function to call SPX data
def get_sp500_tickers():
    url = 'https://en.wikipedia.org/wiki/List_of_S%26P_500_companies'
    response = requests.get(url)
    soup = BeautifulSoup(response.text, 'html.parser')
    table = soup.find('table', {'id': 'constituents'})
    df = pd.read_html(str(table))[0]
    tickers = df['Symbol'].tolist()
    tickers = [ticker.replace('.', '-') for ticker in tickers]
    return tickers


# Function to call NDX data
def get_nasdaq100_tickers():
    url = 'https://en.wikipedia.org/wiki/NASDAQ-100'
    response = requests.get(url)
    soup = BeautifulSoup(response.text, 'html.parser')
    table = soup.find('table', {'id': 'constituents'})
    df = pd.read_html(str(table))[0]
    tickers = df['Ticker'].tolist()
    tickers = [ticker.replace('.', '-') for ticker in tickers]
    return tickers
