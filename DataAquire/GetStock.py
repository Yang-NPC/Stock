import yfinance as yf
import pandas as pd
import requests 
from bs4 import BeautifulSoup 
import csv
import os

start_date = '2014-07-01' 
end_date = '2015-06-01'


#open the file
with open('MW-NIFTY-50-26-Oct-2023.csv', mode='r') as file:
    reader = csv.reader(file)
    first_column = [row[0] for row in reader if row]  

#get the stock name and paresing array 
stock_name = first_column[first_column.index('TATASTEEL'):] 
for i in range(len(stock_name)):
    #make sure there is .NS at the end for NIFTY-50
    stock_name[i] = stock_name[i] + '.NS'
    
print(stock_name)

save_dir = 'DataAquire/StockData'
if not os.path.exists(save_dir):
    os.makedirs(save_dir)  # Create the directory if it doesn't exist

#write the data that download from yfinance to the cvs file
for ticker_symbol in stock_name:
    stock_data = yf.download(ticker_symbol, start=start_date, end=end_date)
    print(f"Stock Data for {ticker_symbol}:")
    print(stock_data)
    print("\n" + "-"*50 + "\n")
    csv_filename = f'{ticker_symbol}_{start_date}_to_{end_date}.csv'
    csv_filepath = os.path.join(save_dir, csv_filename)
    print(csv_filepath)
    stock_data.to_csv(csv_filepath)
    




    