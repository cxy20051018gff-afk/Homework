import pandas as pd
import numpy as np
import akshare as ak
from datetime import datetime, timedelta
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
import time
import warnings

warnings.filterwarnings('ignore')


# ---------- 1. 下载价格历史（使用 akshare，国内稳定） ----------
def fetch_price_data(symbol="600519", start_date="2025-07-23", end_date="2026-07-23"):
    """
    使用 akshare 获取 A 股日线数据（后复权）
    symbol: 股票代码，如 '600519' 贵州茅台
    """
    print(f"📥 正在从 akshare 下载 {symbol} 从 {start_date} 到 {end_date} 的价格数据...")
    try:
        # 获取日线历史数据
        df = ak.stock_zh_a_hist(
            symbol=symbol,
            period="daily",
            start_date=start_date,
            end_date=end_date,
            adjust="qfq"  # 前复权，保持价格连续性
        )
        if df.empty:
            raise ValueError("未获取到数据，请检查股票代码或日期范围")

        # 重命名列以符合统一格式
        df = df.rename(columns={
            '日期': 'Date',
            '开盘': 'Open',
            '收盘': 'Close',
            '最高': 'High',
            '最低': 'Low',
            '成交量': 'Volume'
        })
        df['Date'] = pd.to_datetime(df['Date'])
        df = df.set_index('Date')
        # 只保留所需列
        df = df[['Open', 'High', 'Low', 'Close', 'Volume']]
        print(f"✅ 价格数据下载完成，共 {len(df)} 个交易日")
        return df
    except Exception as e:
        print(f"❌ 价格数据下载失败: {e}")
        # 如果 akshare 失败，生成模拟价格数据（仅演示）
        print("🔄 生成模拟价格数据用于演示...")
        return generate_mock_price(start_date, end_date)


def generate_mock_price(start_date, end_date):
    """生成模拟价格数据（当真实数据获取失败时使用）"""
    dates = pd.date_range(start=start_date, end=end_date, freq='B')  # 仅工作日
    np.random.seed(42)
    n = len(dates)
    price = 100.0
    prices = []
    for _ in range(n):
        change = np.random.normal(0, 0.02)
        price = price * (1 + change)
        prices.append(price)
    df = pd.DataFrame({
        'Open': prices,
        'High': [p * (1 + abs(np.random.normal(0, 0.01))) for p in prices],
        'Low': [p * (1 - abs(np.random.normal(0, 0.01))) for p in prices],
        'Close': [p * (1 + np.random.normal(0, 0.005)) for p in prices],
        'Volume': np.random.randint(1000, 10000, n)
    }, index=dates)
    print(f"✅ 生成模拟价格 {len(df)} 条")
    return df


# ---------- 2. 下载官方新闻源（尝试 akshare，失败则模拟） ----------
def fetch_news(symbol="600519", max_days=365):
    """
    使用 akshare 获取股票相关新闻（东方财富源）
    由于免费接口通常只返回最近几十条，我们将获取到的与模拟新闻合并以覆盖一年
    """
    print(f"📰 正在从 akshare 获取 {symbol} 相关新闻...")
    all_news = []
    try:
        # 获取新闻列表（东方财富）
        news_df = ak.stock_news_em(symbol=symbol)
        if not news_df.empty:
            # 转换为统一格式
            for _, row in news_df.iterrows():
                pub_time = pd.to_datetime(row['发布时间'])
                # 只保留一年内的新闻
                if pub_time >= datetime.now() - timedelta(days=max_days):
                    all_news.append({
                        'datetime': pub_time,
                        'title': row['新闻标题'],
                        'body': row.get('新闻内容', '') or row['新闻标题'],  # 部分接口无正文
                        'source': '东方财富'
                    })
            print(f"   获取到 {len(all_news)} 条东方财富新闻")
    except Exception as e:
        print(f"⚠️  akshare 新闻获取失败: {e}")

    # 如果新闻数量太少（少于50条），用模拟新闻填充以覆盖整个周期
    if len(all_news) < 50:
        print(f"   真实新闻不足，补充模拟新闻以覆盖一年时间...")
        mock_news = generate_mock_news(symbol, max_days)
        all_news.extend(mock_news)

    df_news = pd.DataFrame(all_news)
    if df_news.empty:
        print("❌ 未能获取任何新闻，生成模拟新闻")
        df_news = pd.DataFrame(generate_mock_news(symbol, max_days))

    df_news = df_news.drop_duplicates(subset=['title', 'datetime'])
    df_news = df_news.sort_values('datetime')
    print(
        f"✅ 新闻数据准备完成，共 {len(df_news)} 条（时间范围: {df_news['datetime'].min()} 至 {df_news['datetime'].max()}）")
    return df_news


def generate_mock_news(symbol, max_days):
    """生成模拟新闻（情感随机）"""
    start_date = datetime.now() - timedelta(days=max_days)
    dates = pd.date_range(start=start_date, end=datetime.now(), freq='D')
    np.random.seed(123)
    news_list = []
    for d in dates:
        # 每天随机生成 0~3 条
        for _ in range(np.random.randint(0, 4)):
            score = np.random.uniform(-1, 1)
            sentiment = "positive" if score > 0.1 else "negative" if score < -0.1 else "neutral"
            news_list.append({
                'datetime': d + pd.Timedelta(hours=np.random.randint(0, 23)),
                'title': f"{symbol} 相关新闻 - {sentiment}",
                'body': f"模拟新闻内容：今日市场情绪 {sentiment}。",
                'source': '模拟'
            })
    return news_list


# ---------- 3. 情感分析（VADER） ----------
def analyze_sentiment(news_df):
    print("🧠 正在进行情感分析（VADER）...")
    analyzer = SentimentIntensityAnalyzer()

    def get_compound(text):
        if pd.isna(text) or text == '':
            return 0.0
        text = str(text)[:3000]
        return analyzer.polarity_scores(text)['compound']

    news_df['full_text'] = news_df['title'].fillna('') + " " + news_df['body'].fillna('')
    news_df['sentiment'] = news_df['full_text'].apply(get_compound)

    # 按天聚合
    news_df['date'] = news_df['datetime'].dt.date
    daily_sentiment = news_df.groupby('date')['sentiment'].mean().reset_index()
    daily_sentiment['date'] = pd.to_datetime(daily_sentiment['date'])
    daily_sentiment = daily_sentiment.rename(columns={'sentiment': 'avg_sentiment'})
    print(f"✅ 情感分析完成，覆盖 {len(daily_sentiment)} 个交易日")
    return daily_sentiment


# ---------- 4. 回测 ----------
def backtest(price_df, daily_sentiment, initial_capital=10000.0):
    df = price_df.copy()
    df['date'] = df.index.date
    df['date'] = pd.to_datetime(df['date'])

    df = df.merge(daily_sentiment, on='date', how='left')
    df['avg_sentiment'] = df['avg_sentiment'].fillna(0)

    df['raw_signal'] = 0
    df.loc[df['avg_sentiment'] > 0.1, 'raw_signal'] = 1
    df.loc[df['avg_sentiment'] < -0.1, 'raw_signal'] = -1
    df['signal'] = df['raw_signal'].shift(1).fillna(0)  # 次日执行

    cash = initial_capital
    position = 0.0
    trade_log = []

    for i in range(1, len(df)):
        row = df.iloc[i]
        price = row['Open']
        date = row['date']
        signal = row['signal']
        if signal == 1 and cash > 0:
            position = cash / price
            cash = 0.0
            trade_log.append({'date': date, 'action': 'BUY', 'price': price})
        elif signal == -1 and position > 0:
            cash = position * price
            position = 0.0
            trade_log.append({'date': date, 'action': 'SELL', 'price': price})

    final_price = df.iloc[-1]['Close']
    final_value = cash + position * final_price
    total_days = (df.iloc[-1]['date'] - df.iloc[0]['date']).days
    if total_days <= 0:
        total_days = 365
    total_return = (final_value / initial_capital) - 1
    annual_return = (1 + total_return) ** (365 / total_days) - 1

    bh_final_price = df.iloc[-1]['Close']
    bh_initial_price = df.iloc[0]['Open']
    bh_total_return = (bh_final_price / bh_initial_price) - 1
    bh_annual_return = (1 + bh_total_return) ** (365 / total_days) - 1

    print("\n" + "=" * 50)
    print("📊 回测结果")
    print("=" * 50)
    print(f"起始资金: ${initial_capital:,.2f}")
    print(f"最终资产: ${final_value:,.2f}")
    print(f"策略总收益率: {total_return * 100:.2f}%")
    print(f"策略年化收益率: {annual_return * 100:.2f}%")
    print("-" * 50)
    print(f"Buy&Hold 初始价格: ${bh_initial_price:,.2f}")
    print(f"Buy&Hold 最终价格: ${bh_final_price:,.2f}")
    print(f"Buy&Hold 总收益率: {bh_total_return * 100:.2f}%")
    print(f"Buy&Hold 年化收益率: {bh_annual_return * 100:.2f}%")
    print("-" * 50)
    print(f"📈 策略 vs B&H: {(annual_return - bh_annual_return) * 100:.2f}% 年化差额")
    print(f"交易次数: {len(trade_log)} 次")
    print("=" * 50)
    return df, trade_log, annual_return, bh_annual_return


# ---------- 主程序 ----------
def main():
    print("🚀 新闻交易系统启动（使用 akshare 数据源）")
    print("-" * 40)

    # 1. 价格数据（可自行更改股票代码，如 '000001' 上证指数）
    price_df = fetch_price_data(symbol="000858",
                                start_date="2025-07-23",
                                end_date="2026-07-23")

    # 2. 新闻数据
    news_df = fetch_news(symbol="000858", max_days=365)

    # 3. 情感分析
    daily_sentiment = analyze_sentiment(news_df)

    # 4. 回测
    df_result, trade_log, strategy_ret, bh_ret = backtest(price_df, daily_sentiment)

    if trade_log:
        print("\n📝 最近 5 笔交易记录:")
        for log in trade_log[-5:]:
            print(f"  {log['date'].date()} - {log['action']} @ ${log['price']:.2f}")

    print("\n✅ 作业全部完成！")


if __name__ == "__main__":
    main()