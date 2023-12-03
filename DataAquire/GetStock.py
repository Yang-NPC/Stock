import yfinance as yf
import pandas as pd
import requests 
from bs4 import BeautifulSoup 
import csv
import os
import datetime

#unix to human readable
def InttoStringTime(unix_timestamp):
    human_readable_date = datetime.datetime.fromtimestamp(unix_timestamp)
    return human_readable_date

#human readable to unix
def StringtoIntTime(human_readable_date):
    start_date_time_obj = datetime.datetime.strptime(human_readable_date, '%Y-%m-%d %H:%M:%S')
    unix_date = int(start_date_time_obj.timestamp())
    return unix_date

def getMinuteWise():
    start_date = "2023-11-01 00:00:00"
    end_date = "2023-12-03 00:00:00"
    day_7 = 7*24*60*60
    day_29 = 29*24*60*60

    unix_end_date = StringtoIntTime(end_date)
    unix_start_date = unix_end_date - day_29
    
    with open('MW-NIFTY-50-26-Oct-2023.csv', mode='r') as file:
            reader = csv.reader(file)
            first_column = [row[0] for row in reader if row]  

    while unix_start_date < unix_end_date: 
        unix_after_seven_date = unix_start_date + day_7

        

        #get the stock name and paresing array 
        stock_name = first_column[first_column.index('TATASTEEL'):] 
        for i in range(len(stock_name)):
            #make sure there is .NS at the end for NIFTY-50
            stock_name[i] = stock_name[i] + '.NS'
    
        print(stock_name)

        save_dir = 'StockData/TimeWise'
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)  # Create the directory if it doesn't exist

        #write the data that download from yfinance to the cvs file
        for ticker_symbol in stock_name:
            stock_data = yf.download(ticker_symbol, start=unix_start_date, end=unix_after_seven_date, interval='1m')
            print(f"Stock Data for {ticker_symbol}:")
            print(stock_data)
            print("\n" + "-"*50 + "\n")
            csv_filename = f'{ticker_symbol}.csv'
            csv_filepath = os.path.join(save_dir, csv_filename)
            print(csv_filepath)
            stock_data.to_csv(csv_filepath,mode='a')

        unix_start_date = unix_start_date + day_7
    




def getAllStock():
    start_date = '2014-07-01' 
    end_date = '2015-06-02'
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

    save_dir = 'StockData'
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
        

getMinuteWise()




    