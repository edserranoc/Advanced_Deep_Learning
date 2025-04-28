import numpy as np
import pandas as pd
from typing import List, Tuple
import os

import yfinance  as yf
from sklearn.preprocessing import MinMaxScaler
from torch.utils.data import DataLoader


class GenerateData:
    @staticmethod
    def download_crypto_data(cryptos:List = ['BTC-USD', 'ETH-USD', 'BNB-USD', 'XRP-USD', 'ADA-USD', 'SOL-USD', 'DOGE-USD'], 
                             period:str = '1mo',
                             interval:str = '1h', 
                             folder_name:str = 'data', 
                             file_name:str = 'crypto_data')-> pd.DataFrame:
        """Download data from Yahoo Finance API
        
        :param cryptos: List of cryptocurrencies to download
        :type cryptos: List, optional
        :param period: Period of data to download, defaults to '1mo'
        :type period: str, optional
        :param interval: Interval of data to download, defaults to '1d'
        :type interval: str, optional
        :param folder_name: Folder to save data, defaults to 'data'
        :type folder_name: str, optional
        :param file_name: File name to save data, defaults to 'crypto_data'
        :type file_name: str, optional
        
        :return: Data downloaded
        :rtype: pd.DataFrame
        """
        
        # Create the directory if it doesn't exist
        if not os.path.exists(folder_name):
            os.makedirs(folder_name)
        
        file_path = os.path.join(folder_name, f'{file_name}.csv')
        
        if os.path.exists(file_path):
            print(f"Loading existing data from {file_path}")
            return pd.read_csv(file_path, index_col=0)
        
        data = {}
        for crypto in cryptos:
            crypto_data = yf.download(crypto, period=period, interval=interval)
            crypto_data['Volume'] = crypto_data['Volume'].astype(float)
            crypto_data['Asset'] = crypto
            data[crypto] = crypto_data[['Open', 'High', 'Low', 'Close', 'Volume','Asset']]

        df = pd.concat(data, axis=0)
        
        df.to_csv(file_path)
        print(f"Data saved in {file_path}")
        return df
    
    
    @staticmethod
    def download_exchange_and_oil_data(currencies: List = ['EUR','JPY', 'GBP', 'AUD', 'CAD','HKD', 'MXN','COP'],
                                       oil_ticker: str = 'CL=F',
                                       period: str = '1mo',
                                       interval: str = '1h',
                                       folder_name: str = 'data',
                                       file_name: str = 'exchange_oil_data')-> pd.DataFrame:
        """Download currency exchange and oil price data from Yahoo Finance API
        
        :param currencies: List of currencies to download
        :type currencies: List, optional
        :param oil_ticker: Ticker for oil price, defaults to 'CL=F'
        :type oil_ticker: str, optional
        :param period: Period of data to download, defaults to '1mo'
        :type period: str, optional
        :param interval: Interval of data to download, defaults to '1d'
        :type interval: str, optional
        :param folder_name: Folder to save data, defaults to 'data'
        :type folder_name: str, optional
        :param file_name: File name to save data, defaults to 'exchange_oil_data'
        :type file_name: str, optional
        
        :return: Data downloaded
        :rtype: pd.DataFrame
        """
        
        # Create the directory if it doesn't exist
        if not os.path.exists(folder_name):
            os.makedirs(folder_name)
        
        file_path = os.path.join(folder_name, f'{file_name}.csv')
        
        # Check if the file already exists
        if os.path.exists(file_path):
            print(f"Loading existing data from {file_path}")
            return pd.read_csv(file_path, index_col=0)
        
        data = {}

        # Download currency exchange data (against USD)
        for currency in currencies:
            ticker = f"{currency}=X"  # Yahoo Finance format for exchange rates
            currency_data = yf.download(ticker, period=period, interval=interval)
            currency_data['Volume'] = currency_data['Volume'].astype(float)
            currency_data['Asset'] = currency
            data[currency] = currency_data[['Open', 'High', 'Low', 'Close', 'Volume', 'Asset']]
            

        # Download oil price data
        oil_data = yf.download(oil_ticker, period=period, interval=interval)
        oil_data['Volume'] = oil_data['Volume'].astype(float)
        oil_data['Asset'] = 'Oil'
        data['Oil'] = oil_data[['Open', 'High', 'Low', 'Close', 'Volume', 'Asset']]
        
        # Combine all data into a single DataFrame
        combined_data = pd.concat(data, axis=0)

        # Save the combined data to a CSV file
        combined_data.to_csv(file_path)
        print(f"Data saved to {file_path}")
        return combined_data
    
    @staticmethod
    def normalize_data(data_: pd.DataFrame,type_normalization: str ="MinMaxScaler")-> pd.DataFrame:
        """Normalize data
        
        :param data: Data to normalize
        :type data: pd.DataFrame
        :param type_normalization: Type of normalization to apply, defaults to "MinMaxScaler"
        :type type_normalization: str, optional
        """
        data = data_.copy(deep=True)
        
        Assets = data['Asset'].unique()
        
        if type_normalization == "MinMaxScaler":
            scalers = {}
            for asset in Assets:
                scaler = MinMaxScaler()
                data.loc[data['Asset']==asset,['Open', 'High', 'Low', 'Close', 'Volume']] = scaler.fit_transform(data.loc[data['Asset']==asset,['Open', 'High', 'Low', 'Close', 'Volume']])
                scalers[asset] = scaler
            return data, scalers
        
        elif type_normalization == "ZScore":
            stats = {}
            for asset in Assets:
                means = data.loc[data['Asset']==asset,['Open', 'High', 'Low', 'Close', 'Volume']].mean()
                stds = data.loc[data['Asset']==asset,['Open', 'High', 'Low', 'Close', 'Volume']].std()
                data.loc[data['Asset']==asset,['Open', 'High', 'Low', 'Close', 'Volume']] = (data.loc[data['Asset']==asset,['Open', 'High', 'Low', 'Close', 'Volume']] - means) / stds
                stats[asset] = {'mean': means, 'std': stds}
            return data, stats
        
        elif type_normalization == "Log":
            for asset in Assets:
                data.loc[data['Asset']==asset,['Open', 'High', 'Low', 'Close', 'Volume']] = np.log1p(data.loc[data['Asset']==asset,['Open', 'High', 'Low', 'Close', 'Volume']])
            return data
        
        else:
            raise ValueError("Invalid type of normalization")
    
    @staticmethod
    def denormalize_close_value(normalized_value: pd.DataFrame,
                                       ticker:str,
                                       stats: dict=None,
                                       scalers:dict =None)-> pd.DataFrame:
        
        """Denormalize the Close value of the data using ZScore normalization
        
        :param data: Data to denormalize
        :type data: pd.DataFrame
        :param stats: Dictionary with the mean and standard deviation of each asset
        :type stats: dict, optional
        :param scalers: Dictionary with the MinMaxScaler of each asset
        :type scalers: dict, optional
        
        :return: Denormalized data
        :rtype: pd.DataFrame
        """
        
        if stats is not None:
            mean_close = stats[ticker]['mean']['Close']
            std_close  = stats[ticker]['std']['Close']
            return normalized_value * std_close + mean_close
        elif scalers is not None:
            place_holder = np.zeros((normalized_value.shape[0],5))
            place_holder[:,3]= normalized_value
            return scalers[ticker].inverse_transform(place_holder)[:,3]
        else:
            return np.expm1(normalized_value)
        
        
    @staticmethod
    def split_test_train(data: pd.DataFrame, train_perc: float = 0.8,t:int = 5)-> Tuple[pd.DataFrame, pd.DataFrame]:
        """Split data into train and test sets
        
        :param data: Data to split
        :type data: pd.DataFrame
        :param test_size: Size of the test set, defaults to 0.2
        :type test_size: float, optional
        
        :return: Train and test sets
        :rtype: Tuple[pd.DataFrame, pd.DataFrame]
        """
        
        data_ = data.copy(deep=True)
        
        grouped = data_.groupby('Asset')
        train = []
        test = []

        for ticker, df in grouped:
            series = df[['Open', 'High', 'Low', 'Close', 'Volume']].values
            l = len(series)
            train_size = int(train_perc * l)

            for i in range(l - 100):
                input_seq = series[i:i + (100 - t)]  # Últimos 100-t pasos como entrada
                target_seq = series[i + (100 - t):i + 100, 3]  # Solo el precio de cierre ('Close') como salida

                if i < train_size:
                    train.append((input_seq, target_seq))
                else:
                    test.append((input_seq, target_seq))
        
        
        return train, test
                          
    @staticmethod
    def DataLoader_data(data: pd.DataFrame, 
                        batch_size: int = 32, 
                        shuffle: bool = True, 
                        drop_last=True)-> DataLoader:
        """DataLoader for training and testing
        
        :param data: Data to load
        :type data: pd.DataFrame
        :param batch_size: Batch size, defaults to 32
        :type batch_size: int, optional
        :param shuffle: Shuffle the data, defaults to True
        :type shuffle: bool, optional
        :param drop_last: Drop the last batch if it's smaller than batch_size, defaults to True
        :type drop_last: bool, optional
        
        :return: DataLoader
        :rtype: DataLoader
        """
        
        return DataLoader(data, batch_size=batch_size, shuffle=shuffle, drop_last=drop_last)
    
    @classmethod
    def generate_data(cls, cryptos:List = ['BTC-USD', 'ETH-USD', 'BNB-USD', 'XRP-USD', 'ADA-USD', 'SOL-USD', 'DOGE-USD'], 
                      currencies: List = ['EUR','JPY', 'GBP', 'AUD', 'CAD','HKD', 'MXN','COP'],
                      oil_ticker: str = 'CL=F',
                      period:str = '1mo',
                      interval:str = '1h', 
                      folder_name:str = 'data', 
                      file_name:str = 'exchange_oil_data',
                      type_normalization:str = "MinMaxScaler",
                      test_size: float = 0.2,
                      batch_size: int = 32,
                      shuffle: bool = True,
                      drop_last: bool = True)-> Tuple[DataLoader, DataLoader, DataLoader, DataLoader]:
        """Generate data for training and testing
        
        :param cryptos: List of cryptocurrencies to download
        :type cryptos: List, optional
        :param currencies: List of currencies to download
        :type currencies: List, optional
        :param oil_ticker: Ticker for oil price, defaults to 'CL=F'
        :type oil_ticker: str, optional
        :param period: Period of data to download, defaults to '1mo'
        :type period: str, optional
        :param interval: Interval of data to download, defaults to '1d'
        :type interval: str, optional
        :param folder_name: Folder to save data, defaults to 'data'
        :type folder_name: str, optional
        :param file_name: File name to save data, defaults to 'exchange_oil_data'
        :type file_name: str, optional
        :param type_normalization: Type of normalization to apply, defaults to "MinMaxScaler"
        :type type_normalization: str, optional
        :param test_size: Size of the test set, defaults to 0.2
        :type test_size: float, optional
        :param batch_size: Batch size, defaults to 32
        :type batch_size: int, optional
        :param shuffle: Shuffle the data, defaults to True
        :type shuffle: bool, optional
        :param drop_last: Drop the last batch if it's smaller than batch_size, defaults to True
        :type drop_last: bool, optional
        
        :return: DataLoaders for training and testing
        :rtype: Tuple[DataLoader, DataLoader, DataLoader, DataLoader]
        """
        
        # Download cryptocurrency data
        crypto_data = cls.download_crypto_data(cryptos, period, interval, folder_name, file_name)
        exchange_data = cls.download_exchange_and_oil_data(currencies, oil_ticker, period, interval, folder_name, file_name)
        
        # Normalize data
        crypto_data, scalers_cd = cls.normalize_data(crypto_data, type_normalization)
        exchange_data, scalers_ed = cls.normalize_data(exchange_data, type_normalization)
        
        # Split data into train and test sets
        train_data_crypto, test_data_crypto = cls.split_test_train(crypto_data, test_size)
        train_data_exchange, test_data_exchange = cls.split_test_train(exchange_data, test_size)
        
        # Create DataLoaders
        train_loader_crypto = cls.DataLoader_data(train_data_crypto, batch_size, shuffle, drop_last)
        test_loader_crypto = cls.DataLoader_data(test_data_crypto, batch_size, shuffle, drop_last)
        
        train_loader_exchange = cls.DataLoader_data(train_data_exchange, batch_size, shuffle, drop_last)
        test_loader_exchange = cls.DataLoader_data(test_data_exchange, batch_size, shuffle, drop_last)
        
        return train_loader_crypto, test_loader_crypto, train_loader_exchange, test_loader_exchange, scalers_cd, scalers_ed

if __name__ == "__main__":
    # Download cryptocurrency data
    crypto_data = GenerateData.download_crypto_data()
    
    # Download exchange rates and oil price data
    exchange_oil_data = GenerateData.download_exchange_and_oil_data()
    