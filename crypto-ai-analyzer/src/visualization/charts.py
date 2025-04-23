from matplotlib import pyplot as plt
import pandas as pd

def plot_price_history(data: pd.DataFrame, title: str = "Price History", xlabel: str = "Date", ylabel: str = "Price"):
    plt.figure(figsize=(14, 7))
    plt.plot(data['date'], data['price'], label='Price', color='blue')
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.legend()
    plt.grid()
    plt.show()

def plot_moving_average(data: pd.DataFrame, window: int, title: str = "Moving Average"):
    data['moving_average'] = data['price'].rolling(window=window).mean()
    plt.figure(figsize=(14, 7))
    plt.plot(data['date'], data['price'], label='Price', color='blue')
    plt.plot(data['date'], data['moving_average'], label=f'Moving Average ({window})', color='orange')
    plt.title(title)
    plt.xlabel("Date")
    plt.ylabel("Price")
    plt.legend()
    plt.grid()
    plt.show()

def plot_rsi(data: pd.DataFrame, title: str = "Relative Strength Index (RSI)"):
    plt.figure(figsize=(14, 7))
    plt.plot(data['date'], data['rsi'], label='RSI', color='purple')
    plt.axhline(70, linestyle='--', alpha=0.5, color='red')
    plt.axhline(30, linestyle='--', alpha=0.5, color='green')
    plt.title(title)
    plt.xlabel("Date")
    plt.ylabel("RSI")
    plt.legend()
    plt.grid()
    plt.show()