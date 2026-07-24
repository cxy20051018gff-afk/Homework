"""
Simple News Trading System for Bitcoin
Reads local CSV files: bitcoin_price.csv (columns: Date, Close)
and news.csv (columns: Date, Text).
If files not found, downloads price data from Yahoo Finance and creates synthetic news.
Runs backtest, prints annualized profit and comparison with Buy&Hold.
"""

import pandas as pd
import numpy as np
from textblob import TextBlob
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

# ------------------ 1. Data Loading ------------------
def load_price_data(price_file='bitcoin_price.csv'):
    """Load price data from local CSV or download if missing."""
    try:
        df = pd.read_csv(price_file, parse_dates=['Date'])
        # Ensure required columns
        if 'Close' not in df.columns:
            # try to find a close-like column
            close_candidates = [col for col in df.columns if 'close' in col.lower()]
            if close_candidates:
                df.rename(columns={close_candidates[0]: 'Close'}, inplace=True)
            else:
                raise ValueError("No 'Close' column found in price file.")
        # set Date as index
        df.set_index('Date', inplace=True)
        df.sort_index(inplace=True)
        return df['Close']
    except FileNotFoundError:
        print("Price file not found. Downloading Bitcoin price from Yahoo Finance...")
        import yfinance as yf
        end = datetime.now()
        start = end - timedelta(days=400)  # at least one year
        ticker = yf.Ticker('BTC-USD')
        hist = ticker.history(start=start, end=end)
        if hist.empty:
            raise RuntimeError("Failed to download price data.")
        price_series = hist['Close']
        # save for future use
        price_series.to_csv(price_file, header=['Close'])
        print(f"Downloaded and saved to {price_file}")
        return price_series

def load_news_data(news_file='news.csv'):
    """Load news data from local CSV or generate synthetic if missing."""
    try:
        df = pd.read_csv(news_file, parse_dates=['Date'])
        if 'Text' not in df.columns:
            # try to find a text column
            text_candidates = [col for col in df.columns if 'text' in col.lower() or 'content' in col.lower() or 'title' in col.lower()]
            if text_candidates:
                df.rename(columns={text_candidates[0]: 'Text'}, inplace=True)
            else:
                raise ValueError("No text column found in news file.")
        df.set_index('Date', inplace=True)
        df.sort_index(inplace=True)
        return df['Text']
    except FileNotFoundError:
        print("News file not found. Generating synthetic news based on price movements...")
        # Generate news from price changes
        price = load_price_data()
        # Create news for each day with random sentiment correlated with daily return
        dates = price.index
        texts = []
        for i in range(1, len(dates)):
            ret = (price.iloc[i] - price.iloc[i-1]) / price.iloc[i-1]
            # positive return -> positive news, negative return -> negative news
            sentiment_score = np.clip(ret * 20, -1, 1)  # scale
            if sentiment_score > 0.3:
                text = "Bitcoin surges on strong demand and positive market sentiment."
            elif sentiment_score > 0.1:
                text = "Bitcoin shows moderate gains amid cautious optimism."
            elif sentiment_score < -0.3:
                text = "Bitcoin plummets as regulatory concerns weigh on investors."
            elif sentiment_score < -0.1:
                text = "Bitcoin declines slightly in a risk-off environment."
            else:
                text = "Bitcoin trades sideways with low volatility."
            # add some randomness
            if np.random.rand() < 0.2:
                text = "Uncertainty looms as Bitcoin faces mixed signals."
            texts.append(text)
        # create series with dates (skip first day)
        news_series = pd.Series(texts, index=dates[1:], name='Text')
        # save
        news_series.to_csv(news_file, header=['Text'])
        print(f"Generated synthetic news and saved to {news_file}")
        return news_series

# ------------------ 2. Sentiment Analysis ------------------
def get_sentiment(text):
    """Return polarity score from -1 (negative) to +1 (positive)."""
    return TextBlob(text).sentiment.polarity

# ------------------ 3. Trading Signal ------------------
def generate_signals(news_series, price_series, threshold=0.1):
    """
    For each day with news, compute average sentiment and generate signal:
    Buy if sentiment > threshold, Sell if sentiment < -threshold, else Hold.
    Returns DataFrame with signals.
    """
    # Align news with price dates (only days present in both)
    common_dates = news_series.index.intersection(price_series.index)
    if len(common_dates) == 0:
        raise ValueError("No overlapping dates between news and price data.")

    # Group by date (in case multiple news per day) and average sentiment
    daily_news = news_series.loc[common_dates]
    daily_sentiment = daily_news.groupby(daily_news.index).apply(lambda x: x.apply(get_sentiment).mean())

    # Generate signals
    signals = pd.Series(index=daily_sentiment.index, dtype=object)
    signals[daily_sentiment > threshold] = 'BUY'
    signals[daily_sentiment < -threshold] = 'SELL'
    signals[(daily_sentiment >= -threshold) & (daily_sentiment <= threshold)] = 'HOLD'
    return signals

# ------------------ 4. Backtest ------------------
def backtest(price_series, signals, initial_cash=10000):
    """
    Simulate trading: start with cash only.
    On BUY: invest all cash into Bitcoin.
    On SELL: sell all Bitcoin holdings.
    HOLD: do nothing.
    Returns final value and daily portfolio values.
    """
    # Create a full date range from price_series
    full_dates = price_series.index
    # Merge signals into full timeline (forward fill? Actually we only act on signal days)
    # We'll process day by day
    cash = initial_cash
    holdings = 0.0  # amount of BTC
    portfolio = []  # list of (date, total_value)

    # Get price at each date
    for date in full_dates:
        price = price_series.loc[date]
        # Check if there is a signal for this date
        if date in signals.index:
            signal = signals.loc[date]
            if signal == 'BUY' and cash > 0:
                # buy with all cash
                holdings = cash / price
                cash = 0.0
            elif signal == 'SELL' and holdings > 0:
                # sell all
                cash = holdings * price
                holdings = 0.0
            # HOLD: do nothing
        # Calculate total value at this date
        total_value = cash + holdings * price
        portfolio.append((date, total_value))

    portfolio_df = pd.DataFrame(portfolio, columns=['Date', 'Portfolio_Value'])
    portfolio_df.set_index('Date', inplace=True)
    final_value = portfolio_df.iloc[-1]['Portfolio_Value']
    return final_value, portfolio_df

# ------------------ 5. Performance Metrics ------------------
def calculate_metrics(start_value, end_value, start_date, end_date):
    """Calculate total return and annualized return."""
    total_return = (end_value - start_value) / start_value
    days = (end_date - start_date).days
    if days > 0:
        annualized = (1 + total_return) ** (365 / days) - 1
    else:
        annualized = total_return
    return total_return, annualized

# ------------------ Main ------------------
def main():
    # Load data
    price = load_price_data('bitcoin_price.csv')
    news = load_news_data('news.csv')

    # Ensure price and news have same timezone naive (already parsed)
    # Generate signals
    signals = generate_signals(news, price, threshold=0.1)
    print(f"Generated {len(signals)} trading signals.")
    print("Signal counts:\n", signals.value_counts())

    # Backtest strategy
    initial_cash = 10000.0
    final_value, portfolio = backtest(price, signals, initial_cash)

    # Buy&Hold: start with all cash, buy at first day, hold till end
    start_date = price.index.min()
    end_date = price.index.max()
    first_price = price.loc[start_date]
    last_price = price.loc[end_date]
    bh_final = (initial_cash / first_price) * last_price  # assuming buy at first price with all cash
    bh_total_return = (bh_final - initial_cash) / initial_cash
    bh_annualized = (1 + bh_total_return) ** (365 / (end_date - start_date).days) - 1

    # Strategy metrics
    strat_total_return = (final_value - initial_cash) / initial_cash
    strat_annualized = (1 + strat_total_return) ** (365 / (end_date - start_date).days) - 1

    # Print results
    print("\n" + "="*50)
    print(f"Backtest period: {start_date.date()} to {end_date.date()} ({(end_date - start_date).days} days)")
    print(f"Initial capital: ${initial_cash:,.2f}")
    print(f"Strategy final value: ${final_value:,.2f}")
    print(f"Strategy total return: {strat_total_return:.2%}")
    print(f"Strategy annualized return: {strat_annualized:.2%}")
    print("\nBuy&Hold final value: ${bh_final:,.2f}")
    print(f"Buy&Hold total return: {bh_total_return:.2%}")
    print(f"Buy&Hold annualized return: {bh_annualized:.2%}")
    print(f"\nStrategy vs Buy&Hold: {strat_annualized - bh_annualized:.2%} annualized difference")
    print("="*50)

if __name__ == "__main__":
    main()