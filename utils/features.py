import numpy as np
import pandas as pd
import warnings
warnings.filterwarnings('ignore')

def create_robust_features(data):
    """Create ALL technical features with improved NaN handling."""
    df = data.copy()
    
    print("Creating ULTIMATE comprehensive technical features...")
    
    # Import scipy for advanced analysis
    try:
        from scipy.signal import argrelextrema, find_peaks, hilbert
        from scipy import stats
        scipy_available = True
    except ImportError:
        print("Warning: scipy not available. Some advanced features will be simplified.")
        scipy_available = False
    
    # Try to import ta-lib
    try:
        import talib
        talib_available = True
        print("TA-Lib available - using advanced indicators")
    except ImportError:
        print("Warning: TA-Lib not available. Using alternative implementations.")
        talib_available = False
    
    # Store original length
    original_len = len(df)
    
    # ==================== BASIC FEATURES ====================
    df['price_change'] = df['Close'].pct_change()
    df['log_return'] = np.log(df['Close'] / df['Close'].shift(1))
    df['high_low_ratio'] = df['High'] / df['Low'].replace(0, np.nan)
    df['open_close_ratio'] = df['Open'] / df['Close'].replace(0, np.nan)
    df['volume_price_ratio'] = df['Volume'] / df['Close'].replace(0, np.nan)
    df['typical_price'] = (df['High'] + df['Low'] + df['Close']) / 3
    df['weighted_close'] = (df['High'] + df['Low'] + 2 * df['Close']) / 4
    df['median_price'] = (df['High'] + df['Low']) / 2
    df['average_price'] = (df['Open'] + df['High'] + df['Low'] + df['Close']) / 4
    
    # Safe price position calculation
    hl_range = df['High'] - df['Low']
    df['price_position'] = np.where(hl_range > 0, 
                                (df['Close'] - df['Low']) / hl_range, 
                                0.5)
    
    # ==================== PRICE STATISTICS ====================
    for window in [2, 3, 4, 5, 6, 7, 8, 9, 10, 12, 14, 16, 18, 20, 24, 30, 36, 42, 48, 50, 60, 72, 84, 96, 100, 120, 144, 168, 200]:
        df[f'price_mean_{window}'] = df['Close'].rolling(window=window, min_periods=1).mean()
        df[f'price_std_{window}'] = df['Close'].rolling(window=window, min_periods=1).std()
        df[f'price_min_{window}'] = df['Close'].rolling(window=window, min_periods=1).min()
        df[f'price_max_{window}'] = df['Close'].rolling(window=window, min_periods=1).max()
        df[f'price_median_{window}'] = df['Close'].rolling(window=window, min_periods=1).median()
        df[f'price_skew_{window}'] = df['Close'].rolling(window=window, min_periods=1).skew()
        df[f'price_kurt_{window}'] = df['Close'].rolling(window=window, min_periods=1).kurt()
        df[f'price_quantile_25_{window}'] = df['Close'].rolling(window=window, min_periods=1).quantile(0.25)
        df[f'price_quantile_75_{window}'] = df['Close'].rolling(window=window, min_periods=1).quantile(0.75)
        df[f'price_iqr_{window}'] = df[f'price_quantile_75_{window}'] - df[f'price_quantile_25_{window}']
    
    # ==================== ALL MOVING AVERAGES ====================
    
    # 1. Simple Moving Average (SMA)
    for window in [2, 3, 4, 5, 6, 7, 8, 9, 10, 12, 14, 16, 18, 20, 24, 26, 30, 35, 40, 45, 50, 55, 60, 65, 70, 75, 80, 85, 90, 95, 100, 110, 120, 130, 140, 150, 160, 170, 180, 190, 200, 250, 300, 365, 400, 500]:
        df[f'SMA_{window}'] = df['Close'].rolling(window=window, min_periods=1).mean()
        df[f'SMA_{window}_ratio'] = df['Close'] / df[f'SMA_{window}'].replace(0, np.nan)
        df[f'SMA_{window}_distance'] = df['Close'] - df[f'SMA_{window}']
        df[f'SMA_{window}_slope'] = df[f'SMA_{window}'].diff()
    
    # 2. Exponential Moving Average (EMA)
    for span in [2, 3, 4, 5, 6, 7, 8, 9, 10, 12, 13, 14, 16, 18, 20, 21, 24, 26, 30, 34, 40, 45, 50, 55, 60, 65, 70, 75, 80, 85, 90, 95, 100, 120, 140, 160, 180, 200, 250]:
        df[f'EMA_{span}'] = df['Close'].ewm(span=span, adjust=False).mean()
        df[f'EMA_{span}_ratio'] = df['Close'] / df[f'EMA_{span}'].replace(0, np.nan)
        df[f'EMA_{span}_distance'] = df['Close'] - df[f'EMA_{span}']
        df[f'EMA_{span}_slope'] = df[f'EMA_{span}'].diff()
    
    # 3. Weighted Moving Average (WMA)
    def calculate_wma(prices, window):
        weights = np.arange(1, window + 1)
        wma = prices.rolling(window=window, min_periods=1).apply(
            lambda x: np.dot(x, weights[-len(x):]) / weights[-len(x):].sum() if len(x) > 0 else x.mean()
        )
        return wma
    
    for window in [3, 4, 5, 6, 7, 8, 9, 10, 12, 14, 16, 18, 20, 24, 30, 40, 50, 60, 70, 80, 90, 100, 120, 150, 200]:
        df[f'WMA_{window}'] = calculate_wma(df['Close'], window)
        df[f'WMA_{window}_ratio'] = df['Close'] / df[f'WMA_{window}'].replace(0, np.nan)
        df[f'WMA_{window}_distance'] = df['Close'] - df[f'WMA_{window}']
    
    # 4. Hull Moving Average (HMA)
    def calculate_hma(prices, period):
        half_period = int(period / 2)
        sqrt_period = int(np.sqrt(period))
        
        wma_half = calculate_wma(prices, half_period)
        wma_full = calculate_wma(prices, period)
        
        raw_hma = 2 * wma_half - wma_full
        hma = calculate_wma(raw_hma, sqrt_period)
        
        return hma
    
    for period in [4, 6, 8, 9, 10, 12, 14, 16, 20, 24, 30, 40, 50, 60, 80, 100, 120, 150, 200]:
        if period >= 4:  # HMA needs at least period 4
            df[f'HMA_{period}'] = calculate_hma(df['Close'], period)
            df[f'HMA_{period}_ratio'] = df['Close'] / df[f'HMA_{period}'].replace(0, np.nan)
            df[f'HMA_{period}_distance'] = df['Close'] - df[f'HMA_{period}']
    
    # 5. TEMA (Triple Exponential Moving Average)
    def calculate_tema(prices, period):
        ema1 = prices.ewm(span=period, adjust=False).mean()
        ema2 = ema1.ewm(span=period, adjust=False).mean()
        ema3 = ema2.ewm(span=period, adjust=False).mean()
        tema = 3 * ema1 - 3 * ema2 + ema3
        return tema
    
    for period in [5, 8, 10, 12, 14, 16, 20, 24, 30, 40, 50, 60, 80, 100]:
        df[f'TEMA_{period}'] = calculate_tema(df['Close'], period)
        df[f'TEMA_{period}_ratio'] = df['Close'] / df[f'TEMA_{period}'].replace(0, np.nan)
    
    # 6. DEMA (Double Exponential Moving Average) - Enhanced
    for period in [5, 8, 10, 12, 14, 16, 20, 24, 30, 40, 50, 60, 80, 100, 120, 150, 200]:
        ema = df['Close'].ewm(span=period, adjust=False).mean()
        ema_of_ema = ema.ewm(span=period, adjust=False).mean()
        df[f'DEMA_{period}'] = 2 * ema - ema_of_ema
        df[f'DEMA_{period}_ratio'] = df['Close'] / df[f'DEMA_{period}'].replace(0, np.nan)
        df[f'DEMA_{period}_distance'] = df['Close'] - df[f'DEMA_{period}']
    
    # 7. T3 Moving Average
    def calculate_t3(prices, period, vfactor=0.7):
        ema1 = prices.ewm(span=period, adjust=False).mean()
        ema2 = ema1.ewm(span=period, adjust=False).mean()
        ema3 = ema2.ewm(span=period, adjust=False).mean()
        ema4 = ema3.ewm(span=period, adjust=False).mean()
        ema5 = ema4.ewm(span=period, adjust=False).mean()
        ema6 = ema5.ewm(span=period, adjust=False).mean()
        
        c1 = -vfactor * vfactor * vfactor
        c2 = 3 * vfactor * vfactor + 3 * vfactor * vfactor * vfactor
        c3 = -6 * vfactor * vfactor - 3 * vfactor - 3 * vfactor * vfactor * vfactor
        c4 = 1 + 3 * vfactor + vfactor * vfactor * vfactor + 3 * vfactor * vfactor
        
        t3 = c1 * ema6 + c2 * ema5 + c3 * ema4 + c4 * ema3
        return t3
    
    for period in [5, 8, 10, 14, 20, 30, 50]:
        df[f'T3_{period}'] = calculate_t3(df['Close'], period)
        df[f'T3_{period}_ratio'] = df['Close'] / df[f'T3_{period}'].replace(0, np.nan)
    
    # 8. KAMA (Kaufman Adaptive Moving Average)
    def calculate_kama(prices, period=10, fast_ema=2, slow_ema=30):
        # Efficiency Ratio
        change = abs(prices.diff(period))
        volatility = prices.diff().abs().rolling(window=period).sum()
        er = change / volatility.replace(0, np.nan)
        er = er.fillna(0.5)
        
        # Smoothing constants
        fast_sc = 2 / (fast_ema + 1)
        slow_sc = 2 / (slow_ema + 1)
        
        # Variable smoothing
        sc = er * (fast_sc - slow_sc) + slow_sc
        sc = sc ** 2
        
        # KAMA calculation
        kama = prices.copy()
        for i in range(period, len(prices)):
            kama.iloc[i] = kama.iloc[i-1] + sc.iloc[i] * (prices.iloc[i] - kama.iloc[i-1])
            
        return kama
    
    for period in [5, 10, 15, 20, 30]:
        df[f'KAMA_{period}'] = calculate_kama(df['Close'], period)
        df[f'KAMA_{period}_ratio'] = df['Close'] / df[f'KAMA_{period}'].replace(0, np.nan)
    
    # 9. ZLEMA (Zero Lag Exponential Moving Average)
    def calculate_zlema(prices, period):
        lag = int((period - 1) / 2)
        ema_data = prices + (prices - prices.shift(lag))
        zlema = ema_data.ewm(span=period, adjust=False).mean()
        return zlema
    
    for period in [10, 14, 20, 30, 50]:
        df[f'ZLEMA_{period}'] = calculate_zlema(df['Close'], period)
        df[f'ZLEMA_{period}_ratio'] = df['Close'] / df[f'ZLEMA_{period}'].replace(0, np.nan)
    
    # 10. McGinley Dynamic
    def calculate_mcginley(prices, period=14):
        md = prices.iloc[0]
        md_values = [md]
        
        for i in range(1, len(prices)):
            if md > 0:
                md = md + (prices.iloc[i] - md) / (period * (prices.iloc[i] / md) ** 4)
            else:
                md = prices.iloc[i]
            md_values.append(md)
        
        return pd.Series(md_values, index=prices.index)
    
    for period in [10, 14, 20, 30]:
        df[f'McGinley_{period}'] = calculate_mcginley(df['Close'], period)
        df[f'McGinley_{period}_ratio'] = df['Close'] / df[f'McGinley_{period}'].replace(0, np.nan)
    
    # 11. VWMA (Volume Weighted Moving Average)
    for period in [10, 14, 20, 30, 50, 100]:
        vwma = (df['Close'] * df['Volume']).rolling(window=period).sum() / df['Volume'].rolling(window=period).sum()
        df[f'VWMA_{period}'] = vwma
        df[f'VWMA_{period}_ratio'] = df['Close'] / vwma.replace(0, np.nan)
    
    # 12. Arnaud Legoux Moving Average (ALMA)
    def calculate_alma(prices, period=9, offset=0.85, sigma=6):
        m = offset * (period - 1)
        s = period / sigma
        
        alma = prices.rolling(window=period).apply(
            lambda x: np.sum(
                x * np.exp(-((np.arange(len(x)) - m) ** 2) / (2 * s ** 2))
            ) / np.sum(
                np.exp(-((np.arange(len(x)) - m) ** 2) / (2 * s ** 2))
            ) if len(x) == period else x.mean()
        )
        
        return alma
    
    for period in [9, 14, 20, 30]:
        df[f'ALMA_{period}'] = calculate_alma(df['Close'], period)
        df[f'ALMA_{period}_ratio'] = df['Close'] / df[f'ALMA_{period}'].replace(0, np.nan)
    
    # 13. Least Squares Moving Average (LSMA)
    def calculate_lsma(prices, period):
        def linear_regression(y):
            if len(y) < 2:
                return y.mean()
            x = np.arange(len(y))
            slope, intercept = np.polyfit(x, y, 1)
            return slope * (len(y) - 1) + intercept
        
        lsma = prices.rolling(window=period).apply(linear_regression)
        return lsma
    
    for period in [10, 14, 20, 30, 50]:
        df[f'LSMA_{period}'] = calculate_lsma(df['Close'], period)
        df[f'LSMA_{period}_ratio'] = df['Close'] / df[f'LSMA_{period}'].replace(0, np.nan)
    
    # 14. Ehlers Filter
    def calculate_ehlers_filter(prices, period=10):
        alpha = (np.cos(2 * np.pi / period) + np.sin(2 * np.pi / period) - 1) / np.cos(2 * np.pi / period)
        
        ef = prices.copy()
        for i in range(2, len(prices)):
            ef.iloc[i] = alpha * (prices.iloc[i] + prices.iloc[i-1]) / 2 + (1 - alpha) * ef.iloc[i-1]
        
        return ef
    
    for period in [10, 20, 30]:
        df[f'Ehlers_{period}'] = calculate_ehlers_filter(df['Close'], period)
        df[f'Ehlers_{period}_ratio'] = df['Close'] / df[f'Ehlers_{period}'].replace(0, np.nan)
    
    # 15. Fractal Adaptive Moving Average (FRAMA)
    def calculate_frama(prices, period=16):
        def fractal_dimension(data):
            n = len(data)
            if n < 2:
                return 1.5
            
            max_val = np.max(data)
            min_val = np.min(data)
            
            if max_val == min_val:
                return 1.5
                
            hl = (max_val - min_val) / n
            
            return 2 - np.log(hl) / np.log(2)
        
        frama = prices.copy()
        
        for i in range(period, len(prices)):
            window = prices.iloc[i-period:i]
            fd = fractal_dimension(window.values)
            alpha = np.exp(-4.6 * (fd - 1))
            
            frama.iloc[i] = alpha * prices.iloc[i] + (1 - alpha) * frama.iloc[i-1]
        
        return frama
    
    df['FRAMA_16'] = calculate_frama(df['Close'])
    df['FRAMA_16_ratio'] = df['Close'] / df['FRAMA_16'].replace(0, np.nan)
    
    # ==================== ENHANCED MOMENTUM INDICATORS ====================
    
    # 1. RSI (Multiple Variations)
    def calculate_rsi(prices, window=14):
        delta = prices.diff()
        gain = delta.where(delta > 0, 0)
        loss = -delta.where(delta < 0, 0)
        avg_gain = gain.rolling(window=window, min_periods=1).mean()
        avg_loss = loss.rolling(window=window, min_periods=1).mean()
        rs = avg_gain / avg_loss.replace(0, np.nan)
        rsi = 100 - (100 / (1 + rs))
        return rsi.fillna(50)
    
    for period in [2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 24, 28, 30, 35, 40, 50]:
        df[f'RSI_{period}'] = calculate_rsi(df['Close'], window=period)
        df[f'RSI_{period}_ma'] = df[f'RSI_{period}'].rolling(window=9).mean()
        df[f'RSI_{period}_std'] = df[f'RSI_{period}'].rolling(window=9).std()
    
    # 2. MACD (Multiple Combinations)
    macd_params = [(5,34,5), (12,26,9), (8,16,9), (3,10,16), (5,13,8), (21,50,21), (34,90,34)]
    for fast, slow, signal in macd_params:
        macd_line = df[f'EMA_{fast}'] - df[f'EMA_{slow}']
        signal_line = macd_line.ewm(span=signal, adjust=False).mean()
        df[f'MACD_{fast}_{slow}_{signal}'] = macd_line
        df[f'MACD_{fast}_{slow}_{signal}_signal'] = signal_line
        df[f'MACD_{fast}_{slow}_{signal}_histogram'] = macd_line - signal_line
        df[f'MACD_{fast}_{slow}_{signal}_cross'] = np.where(macd_line > signal_line, 1, -1)
    
    # 3. Stochastic Oscillator (All Variations)
    def calculate_stochastic(high, low, close, period=14, smooth_k=3, smooth_d=3):
        low_min = low.rolling(window=period, min_periods=1).min()
        high_max = high.rolling(window=period, min_periods=1).max()
        hl_range = high_max - low_min
        
        k_percent = np.where(hl_range > 0, 100 * ((close - low_min) / hl_range), 50)
        k_percent = pd.Series(k_percent).rolling(window=smooth_k, min_periods=1).mean()
        d_percent = k_percent.rolling(window=smooth_d, min_periods=1).mean()
        
        return k_percent, d_percent
    
    for period in [5, 8, 9, 12, 14, 18, 21, 24]:
        for smooth in [3, 5]:
            k, d = calculate_stochastic(df['High'], df['Low'], df['Close'], period, smooth, smooth)
            df[f'stoch_k_{period}_{smooth}'] = k
            df[f'stoch_d_{period}_{smooth}'] = d
            df[f'stoch_cross_{period}_{smooth}'] = np.where(k > d, 1, -1)
    
    # 4. Stochastic RSI (Multiple Periods)
    for rsi_period in [14, 21, 28]:
        for stoch_period in [14, 21]:
            rsi = df[f'RSI_{rsi_period}']
            rsi_min = rsi.rolling(window=stoch_period, min_periods=1).min()
            rsi_max = rsi.rolling(window=stoch_period, min_periods=1).max()
            rsi_range = rsi_max - rsi_min
            df[f'stoch_rsi_{rsi_period}_{stoch_period}'] = np.where(rsi_range > 0,
                                                (rsi - rsi_min) / rsi_range,
                                                0.5)
    
    # 5. Ultimate Oscillator
    def calculate_ultimate_oscillator(df, period1=7, period2=14, period3=28):
        bp = df['Close'] - np.minimum(df['Low'], df['Close'].shift(1))
        tr = np.maximum(df['High'], df['Close'].shift(1)) - np.minimum(df['Low'], df['Close'].shift(1))
        
        avg1 = bp.rolling(period1).sum() / tr.rolling(period1).sum()
        avg2 = bp.rolling(period2).sum() / tr.rolling(period2).sum()
        avg3 = bp.rolling(period3).sum() / tr.rolling(period3).sum()
        
        uo = 100 * ((4 * avg1 + 2 * avg2 + avg3) / 7)
        return uo
    
    df['ultimate_oscillator'] = calculate_ultimate_oscillator(df)
    df['ultimate_oscillator_5_10_20'] = calculate_ultimate_oscillator(df, 5, 10, 20)
    df['ultimate_oscillator_10_20_40'] = calculate_ultimate_oscillator(df, 10, 20, 40)
    
    # 6. Williams %R (Multiple Periods)
    for period in [5, 7, 9, 10, 14, 20, 28, 50]:
        highest_high = df['High'].rolling(window=period, min_periods=1).max()
        lowest_low = df['Low'].rolling(window=period, min_periods=1).min()
        hl_range = highest_high - lowest_low
        df[f'williams_r_{period}'] = np.where(hl_range > 0,
                                            -100 * ((highest_high - df['Close']) / hl_range),
                                            -50)
    
    # 7. Commodity Channel Index (CCI) - Extended
    def calculate_cci(df, period=20):
        typical_price = (df['High'] + df['Low'] + df['Close']) / 3
        sma_tp = typical_price.rolling(window=period, min_periods=1).mean()
        mad = typical_price.rolling(window=period, min_periods=1).apply(
            lambda x: np.mean(np.abs(x - x.mean())) if len(x) > 0 else 1
        )
        cci = (typical_price - sma_tp) / (0.015 * mad.replace(0, np.nan))
        return cci.fillna(0)
    
    for period in [5, 8, 10, 12, 14, 18, 20, 24, 28, 35, 40, 50]:
        df[f'CCI_{period}'] = calculate_cci(df, period)
    
    # 8. Money Flow Index (MFI) - Extended
    def calculate_mfi(df, period=14):
        typical_price = (df['High'] + df['Low'] + df['Close']) / 3
        money_flow = typical_price * df['Volume']
        
        positive_flow = np.where(typical_price > typical_price.shift(), money_flow, 0)
        negative_flow = np.where(typical_price < typical_price.shift(), money_flow, 0)
        
        positive_mf = pd.Series(positive_flow).rolling(window=period, min_periods=1).sum()
        negative_mf = pd.Series(negative_flow).rolling(window=period, min_periods=1).sum()
        
        mfi = 100 - (100 / (1 + positive_mf / negative_mf.replace(0, np.nan)))
        return mfi.fillna(50)
    
    for period in [5, 8, 10, 14, 20, 28]:
        df[f'MFI_{period}'] = calculate_mfi(df, period)
    
    # 9. Rate of Change (ROC) - Extended
    for period in [1, 2, 3, 5, 8, 10, 12, 15, 20, 25, 30, 40, 50, 60, 100]:
        shifted_close = df['Close'].shift(period)
        df[f'ROC_{period}'] = ((df['Close'] - shifted_close) / shifted_close.replace(0, np.nan)) * 100
        df[f'ROC_{period}_ma'] = df[f'ROC_{period}'].rolling(window=10).mean()
    
    # 10. Momentum
    for period in [1, 3, 5, 10, 12, 15, 20, 30, 50]:
        df[f'momentum_{period}'] = df['Close'] - df['Close'].shift(period)
        df[f'momentum_{period}_ma'] = df[f'momentum_{period}'].rolling(window=10).mean()
    
    # 11. Trix
    def calculate_trix(prices, period=14):
        ema1 = prices.ewm(span=period, adjust=False).mean()
        ema2 = ema1.ewm(span=period, adjust=False).mean()
        ema3 = ema2.ewm(span=period, adjust=False).mean()
        trix = ema3.pct_change() * 10000
        return trix
    
    for period in [9, 14, 20, 30]:
        df[f'TRIX_{period}'] = calculate_trix(df['Close'], period)
        df[f'TRIX_{period}_signal'] = df[f'TRIX_{period}'].ewm(span=9, adjust=False).mean()
    
    # 12. Know Sure Thing (KST)
    def calculate_kst(prices):
        roc1 = ((prices - prices.shift(10)) / prices.shift(10)) * 100
        roc2 = ((prices - prices.shift(15)) / prices.shift(15)) * 100
        roc3 = ((prices - prices.shift(20)) / prices.shift(20)) * 100
        roc4 = ((prices - prices.shift(30)) / prices.shift(30)) * 100
        
        kst = (roc1.rolling(10).mean() * 1 +
               roc2.rolling(10).mean() * 2 +
               roc3.rolling(10).mean() * 3 +
               roc4.rolling(15).mean() * 4)
        
        signal = kst.rolling(9).mean()
        
        return kst, signal
    
    df['KST'], df['KST_signal'] = calculate_kst(df['Close'])
    df['KST_diff'] = df['KST'] - df['KST_signal']
    
    # 13. Coppock Curve
    def calculate_coppock(prices, wma_period=10, roc_short=11, roc_long=14):
        roc_short_val = ((prices - prices.shift(roc_short)) / prices.shift(roc_short)) * 100
        roc_long_val = ((prices - prices.shift(roc_long)) / prices.shift(roc_long)) * 100
        
        roc_sum = roc_short_val + roc_long_val
        coppock = calculate_wma(roc_sum, wma_period)
        
        return coppock
    
    df['coppock_curve'] = calculate_coppock(df['Close'])
    df['coppock_curve_5_10_15'] = calculate_coppock(df['Close'], 5, 10, 15)
    
    # ==================== VOLATILITY INDICATORS ====================
    
    # 1. Average True Range (ATR) - Extended
    def calculate_atr(df, period):
        high_low = df['High'] - df['Low']
        high_close = np.abs(df['High'] - df['Close'].shift())
        low_close = np.abs(df['Low'] - df['Close'].shift())
        ranges = pd.concat([high_low, high_close, low_close], axis=1)
        true_range = ranges.max(axis=1)
        return true_range.rolling(window=period, min_periods=1).mean()
    
    for period in [2, 5, 7, 10, 14, 20, 21, 28, 35, 42, 50]:
        df[f'ATR_{period}'] = calculate_atr(df, period)
        df[f'ATR_{period}_ratio'] = df[f'ATR_{period}'] / df['Close'].replace(0, np.nan)
        df[f'ATR_{period}_ma'] = df[f'ATR_{period}'].rolling(window=10).mean()
    
    # 2. Bollinger Bands (Multiple Std Devs)
    for window in [5, 10, 15, 20, 25, 30, 40, 50]:
        for num_std in [1, 1.5, 2, 2.5, 3]:
            bb_mean = df['Close'].rolling(window=window, min_periods=1).mean()
            bb_std = df['Close'].rolling(window=window, min_periods=1).std()
            df[f'BB_upper_{window}_{num_std}'] = bb_mean + (bb_std * num_std)
            df[f'BB_lower_{window}_{num_std}'] = bb_mean - (bb_std * num_std)
            df[f'BB_width_{window}_{num_std}'] = df[f'BB_upper_{window}_{num_std}'] - df[f'BB_lower_{window}_{num_std}']
            bb_width = df[f'BB_width_{window}_{num_std}'].replace(0, np.nan)
            df[f'BB_position_{window}_{num_std}'] = (df['Close'] - df[f'BB_lower_{window}_{num_std}']) / bb_width
            df[f'BB_bandwidth_{window}_{num_std}'] = bb_width / bb_mean.replace(0, np.nan)
    
    # 3. Keltner Channels (Multiple ATR Multipliers)
    def calculate_keltner_channels(df, ema_period=20, atr_period=10, multiplier=2):
        ema = df['Close'].ewm(span=ema_period, adjust=False).mean()
        atr = calculate_atr(df, atr_period)
        
        upper = ema + multiplier * atr
        lower = ema - multiplier * atr
        
        return ema, upper, lower
    
    for ema_period in [10, 20, 30, 50]:
        for multiplier in [1, 1.5, 2, 2.5, 3]:
            kc_ema, kc_upper, kc_lower = calculate_keltner_channels(df, ema_period=ema_period, multiplier=multiplier)
            df[f'KC_middle_{ema_period}_{multiplier}'] = kc_ema
            df[f'KC_upper_{ema_period}_{multiplier}'] = kc_upper
            df[f'KC_lower_{ema_period}_{multiplier}'] = kc_lower
            df[f'KC_width_{ema_period}_{multiplier}'] = kc_upper - kc_lower
            df[f'KC_position_{ema_period}_{multiplier}'] = (df['Close'] - kc_lower) / (kc_upper - kc_lower).replace(0, np.nan)
    
    # 4. Donchian Channels (Extended Periods)
    for period in [5, 10, 20, 30, 40, 50, 55, 60, 100]:
        df[f'donchian_high_{period}'] = df['High'].rolling(window=period, min_periods=1).max()
        df[f'donchian_low_{period}'] = df['Low'].rolling(window=period, min_periods=1).min()
        df[f'donchian_mid_{period}'] = (df[f'donchian_high_{period}'] + df[f'donchian_low_{period}']) / 2
        df[f'donchian_width_{period}'] = df[f'donchian_high_{period}'] - df[f'donchian_low_{period}']
        width = df[f'donchian_width_{period}'].replace(0, np.nan)
        df[f'donchian_position_{period}'] = (df['Close'] - df[f'donchian_low_{period}']) / width
    
    # 5. Chandelier Exit
    def calculate_chandelier_exit(df, period=22, multiplier=3):
        highest = df['High'].rolling(window=period).max()
        lowest = df['Low'].rolling(window=period).min()
        atr = calculate_atr(df, period)
        
        long_exit = highest - multiplier * atr
        short_exit = lowest + multiplier * atr
        
        return long_exit, short_exit
    
    for period in [14, 22, 30]:
        for mult in [2, 3, 4]:
            long_exit, short_exit = calculate_chandelier_exit(df, period, mult)
            df[f'chandelier_long_{period}_{mult}'] = long_exit
            df[f'chandelier_short_{period}_{mult}'] = short_exit
    
    # 6. Historical Volatility
    for period in [10, 20, 30, 50, 100]:
        returns = df['Close'].pct_change()
        df[f'hist_volatility_{period}'] = returns.rolling(window=period).std() * np.sqrt(252)
    
    # 7. Garman-Klass Volatility
    def calculate_garman_klass(df, period=30):
        log_hl = np.log(df['High'] / df['Low'])
        log_co = np.log(df['Close'] / df['Open'])
        
        rs = 0.5 * log_hl**2 - (2*np.log(2) - 1) * log_co**2
        
        return np.sqrt(rs.rolling(window=period).mean() * 252)
    
    for period in [10, 20, 30]:
        df[f'garman_klass_{period}'] = calculate_garman_klass(df, period)
    
    # 8. Parkinson Volatility
    def calculate_parkinson(df, period=30):
        log_hl = np.log(df['High'] / df['Low'])
        
        return np.sqrt(log_hl**2 / (4 * np.log(2)) * 252 / period)
    
    for period in [10, 20, 30]:
        df[f'parkinson_{period}'] = calculate_parkinson(df, period)
    
    # ==================== VOLUME INDICATORS ====================
    
    # 1. On Balance Volume (OBV) - Enhanced
    df['OBV'] = (np.sign(df['Close'].diff()) * df['Volume']).fillna(0).cumsum()
    for period in [5, 10, 20, 50]:
        df[f'OBV_EMA_{period}'] = df['OBV'].ewm(span=period, adjust=False).mean()
        df[f'OBV_signal_{period}'] = df['OBV'] - df[f'OBV_EMA_{period}']
        df[f'OBV_ratio_{period}'] = df['OBV'] / df[f'OBV_EMA_{period}'].replace(0, np.nan)
    
    # 2. Volume Weighted Average Price (VWAP) - Multiple Periods
    cumulative_pv = (df['Volume'] * df['typical_price']).cumsum()
    cumulative_volume = df['Volume'].cumsum()
    df['VWAP'] = cumulative_pv / cumulative_volume.replace(0, np.nan)
    df['VWAP_ratio'] = df['Close'] / df['VWAP'].replace(0, np.nan)
    
    # Rolling VWAP
    for period in [5, 10, 20, 30, 50, 100, 200]:
        rolling_pv = (df['Volume'] * df['typical_price']).rolling(window=period, min_periods=1).sum()
        rolling_volume = df['Volume'].rolling(window=period, min_periods=1).sum()
        df[f'VWAP_{period}'] = rolling_pv / rolling_volume.replace(0, np.nan)
        df[f'VWAP_{period}_ratio'] = df['Close'] / df[f'VWAP_{period}'].replace(0, np.nan)
        df[f'VWAP_{period}_distance'] = df['Close'] - df[f'VWAP_{period}']
    
    # 3. Accumulation/Distribution Line - Enhanced
    def calculate_ad_line(df):
        clv = ((df['Close'] - df['Low']) - (df['High'] - df['Close'])) / (df['High'] - df['Low']).replace(0, np.nan)
        clv = clv.fillna(0)
        ad = (clv * df['Volume']).cumsum()
        return ad
    
    df['AD_line'] = calculate_ad_line(df)
    for period in [5, 10, 20, 50]:
        df[f'AD_line_ema_{period}'] = df['AD_line'].ewm(span=period, adjust=False).mean()
        df[f'AD_divergence_{period}'] = df['AD_line'] - df[f'AD_line_ema_{period}']
    
    # 4. Chaikin Money Flow (CMF) - Multiple Periods
    def calculate_cmf(df, period=20):
        clv = ((df['Close'] - df['Low']) - (df['High'] - df['Close'])) / (df['High'] - df['Low']).replace(0, np.nan)
        mfv = clv * df['Volume']
        cmf = mfv.rolling(window=period, min_periods=1).sum() / df['Volume'].rolling(window=period, min_periods=1).sum().replace(0, np.nan)
        return cmf.fillna(0)
    
    for period in [5, 10, 20, 30]:
        df[f'CMF_{period}'] = calculate_cmf(df, period)
    
    # 5. Price Volume Trend (PVT) - Enhanced
    def calculate_pvt(df):
        price_change = df['Close'].pct_change()
        pvt = (price_change * df['Volume']).fillna(0).cumsum()
        return pvt
    
    df['PVT'] = calculate_pvt(df)
    for period in [5, 10, 20]:
        df[f'PVT_signal_{period}'] = df['PVT'].ewm(span=period, adjust=False).mean()
        df[f'PVT_divergence_{period}'] = df['PVT'] - df[f'PVT_signal_{period}']
    
    # 6. Ease of Movement (EOM)
    def calculate_eom(df, period=14):
        distance_moved = df['median_price'].diff()
        emv = distance_moved / (df['Volume'] / df['High'] - df['Low']).replace(0, np.nan)
        eom = emv.rolling(window=period).mean()
        return eom
    
    for period in [10, 14, 20]:
        df[f'EOM_{period}'] = calculate_eom(df, period)
    
    # 7. Force Index
    def calculate_force_index(df, period=13):
        fi = df['Close'].diff() * df['Volume']
        return fi.ewm(span=period, adjust=False).mean()
    
    for period in [2, 13, 20]:
        df[f'force_index_{period}'] = calculate_force_index(df, period)
    
    # 8. Volume Rate of Change
    for period in [5, 10, 20, 30]:
        df[f'volume_roc_{period}'] = df['Volume'].pct_change(period) * 100
    
    # 9. Klinger Oscillator
    def calculate_klinger(df, fast=34, slow=55, signal=13):
        trend = np.where(df['typical_price'] > df['typical_price'].shift(), 1, -1)
        dm = df['High'] - df['Low']
        cm = dm.cumsum()
        
        volume_force = df['Volume'] * trend * dm / cm.replace(0, np.nan)
        
        kvo = volume_force.ewm(span=fast, adjust=False).mean() - volume_force.ewm(span=slow, adjust=False).mean()
        signal_line = kvo.ewm(span=signal, adjust=False).mean()
        
        return kvo, signal_line
    
    df['klinger'], df['klinger_signal'] = calculate_klinger(df)
    df['klinger_diff'] = df['klinger'] - df['klinger_signal']
    
    # ==================== TREND INDICATORS ====================
    
    # 1. ADX (Extended Periods)
    def calculate_adx(df, period=14):
        high = df['High']
        low = df['Low']
        close = df['Close']
        
        # True Range
        tr1 = high - low
        tr2 = abs(high - close.shift())
        tr3 = abs(low - close.shift())
        tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
        atr = tr.rolling(window=period, min_periods=1).mean()
        
        # Directional movements
        up_move = high - high.shift()
        down_move = low.shift() - low
        
        pos_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0)
        neg_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0)
        
        pos_di = 100 * pd.Series(pos_dm).rolling(window=period, min_periods=1).mean() / atr.replace(0, np.nan)
        neg_di = 100 * pd.Series(neg_dm).rolling(window=period, min_periods=1).mean() / atr.replace(0, np.nan)
        
        di_sum = pos_di + neg_di
        dx = 100 * abs(pos_di - neg_di) / di_sum.replace(0, np.nan)
        adx = dx.rolling(window=period, min_periods=1).mean()
        
        return adx.fillna(0), pos_di.fillna(0), neg_di.fillna(0)
    
    for period in [7, 10, 14, 20, 21, 28]:
        adx, plus_di, minus_di = calculate_adx(df, period)
        df[f'ADX_{period}'] = adx
        df[f'plus_DI_{period}'] = plus_di
        df[f'minus_DI_{period}'] = minus_di
        df[f'DI_diff_{period}'] = plus_di - minus_di
        df[f'ADX_slope_{period}'] = adx.diff()
    
    # 2. Aroon (Extended Periods)
    def calculate_aroon(df, period=25):
        high = df['High']
        low = df['Low']
        
        aroon_up = high.rolling(window=period + 1, min_periods=1).apply(
            lambda x: (period - (period - x.argmax())) / period * 100 if len(x) > 0 else 50
        )
        aroon_down = low.rolling(window=period + 1, min_periods=1).apply(
            lambda x: (period - (period - x.argmin())) / period * 100 if len(x) > 0 else 50
        )
        
        return aroon_up, aroon_down
    
    for period in [10, 14, 20, 25, 50]:
        aroon_up, aroon_down = calculate_aroon(df, period)
        df[f'aroon_up_{period}'] = aroon_up
        df[f'aroon_down_{period}'] = aroon_down
        df[f'aroon_oscillator_{period}'] = aroon_up - aroon_down
    
    # 3. Vortex Indicator (Extended)
    def calculate_vortex(df, period=14):
        high = df['High']
        low = df['Low']
        close = df['Close']
        
        vm_plus = abs(high - low.shift())
        vm_minus = abs(low - high.shift())
        
        tr = pd.concat([
            high - low,
            abs(high - close.shift()),
            abs(low - close.shift())
        ], axis=1).max(axis=1)
        
        sum_vm_plus = vm_plus.rolling(window=period, min_periods=1).sum()
        sum_vm_minus = vm_minus.rolling(window=period, min_periods=1).sum()
        sum_tr = tr.rolling(window=period, min_periods=1).sum()
        
        vi_plus = sum_vm_plus / sum_tr.replace(0, np.nan)
        vi_minus = sum_vm_minus / sum_tr.replace(0, np.nan)
        
        return vi_plus.fillna(1), vi_minus.fillna(1)
    
    for period in [10, 14, 21, 28]:
        vi_plus, vi_minus = calculate_vortex(df, period)
        df[f'VI_plus_{period}'] = vi_plus
        df[f'VI_minus_{period}'] = vi_minus
        df[f'VI_diff_{period}'] = vi_plus - vi_minus
    
    # 4. Mass Index
    def calculate_mass_index(df, period=25, ema_period=9):
        high_low = df['High'] - df['Low']
        ema1 = high_low.ewm(span=ema_period, adjust=False).mean()
        ema2 = ema1.ewm(span=ema_period, adjust=False).mean()
        
        mass = (ema1 / ema2).rolling(window=period).sum()
        return mass
    
    df['mass_index'] = calculate_mass_index(df)
    df['mass_index_20'] = calculate_mass_index(df, 20)
    
    # 5. Parabolic SAR (Multiple AF values)
    def calculate_psar(df, af_start=0.02, af_increment=0.02, af_max=0.2):
        high = df['High'].values
        low = df['Low'].values
        close = df['Close'].values
        
        psar = np.zeros_like(close)
        psar[0] = close[0]
        bull = True
        af = af_start
        ep = high[0]
        hp = high[0]
        lp = low[0]
        
        for i in range(1, len(close)):
            if bull:
                psar[i] = psar[i-1] + af * (ep - psar[i-1])
                
                if low[i] < psar[i]:
                    bull = False
                    psar[i] = hp
                    ep = low[i]
                    af = af_start
                    lp = low[i]
                    hp = high[i]
                else:
                    if high[i] > ep:
                        ep = high[i]
                        af = min(af + af_increment, af_max)
                    if high[i] > hp:
                        hp = high[i]
            else:
                psar[i] = psar[i-1] + af * (ep - psar[i-1])
                
                if high[i] > psar[i]:
                    bull = True
                    psar[i] = lp
                    ep = high[i]
                    af = af_start
                    hp = high[i]
                    lp = low[i]
                else:
                    if low[i] < ep:
                        ep = low[i]
                        af = min(af + af_increment, af_max)
                    if low[i] < lp:
                        lp = low[i]
        
        return psar
    
    # Standard PSAR
    df['PSAR'] = calculate_psar(df)
    df['PSAR_distance'] = df['Close'] - df['PSAR']
    df['PSAR_signal'] = np.where(df['Close'] > df['PSAR'], 1, -1)
    
    # Alternative PSAR settings
    df['PSAR_aggressive'] = calculate_psar(df, 0.03, 0.03, 0.3)
    df['PSAR_conservative'] = calculate_psar(df, 0.01, 0.01, 0.1)
    
    # 6. Supertrend (Multiple Settings)
    def calculate_supertrend(df, period=10, multiplier=3):
        hl_avg = (df['High'] + df['Low']) / 2
        atr = calculate_atr(df, period)
        
        # Basic bands
        basic_upper = hl_avg + multiplier * atr
        basic_lower = hl_avg - multiplier * atr
        
        # Initialize
        supertrend = np.zeros(len(df))
        direction = np.zeros(len(df))
        
        for i in range(period, len(df)):
            # Upper band
            if basic_upper.iloc[i] < supertrend[i-1] or df['Close'].iloc[i-1] > supertrend[i-1]:
                upper_band = basic_upper.iloc[i]
            else:
                upper_band = supertrend[i-1]
            
            # Lower band
            if basic_lower.iloc[i] > supertrend[i-1] or df['Close'].iloc[i-1] < supertrend[i-1]:
                lower_band = basic_lower.iloc[i]
            else:
                lower_band = supertrend[i-1]
            
            # Direction
            if i == period:
                if df['Close'].iloc[i] <= upper_band:
                    supertrend[i] = upper_band
                    direction[i] = -1
                else:
                    supertrend[i] = lower_band
                    direction[i] = 1
            else:
                if direction[i-1] == -1:
                    if df['Close'].iloc[i] <= upper_band:
                        supertrend[i] = upper_band
                        direction[i] = -1
                    else:
                        supertrend[i] = lower_band
                        direction[i] = 1
                else:
                    if df['Close'].iloc[i] >= lower_band:
                        supertrend[i] = lower_band
                        direction[i] = 1
                    else:
                        supertrend[i] = upper_band
                        direction[i] = -1
        
        return supertrend, direction
    
    for period, multiplier in [(7, 2), (7, 3), (10, 2), (10, 3), (14, 2), (14, 3), (20, 2), (20, 3)]:
        st, st_dir = calculate_supertrend(df, period, multiplier)
        df[f'supertrend_{period}_{multiplier}'] = st
        df[f'supertrend_dir_{period}_{multiplier}'] = st_dir
        df[f'supertrend_distance_{period}_{multiplier}'] = df['Close'] - st
    
    # 7. Volatility Stop (VSTOP) - Multiple Settings
    def calculate_vstop(df, period=20, multiplier=2):
        atr = calculate_atr(df, period)
        vstop_long = df['Close'] - multiplier * atr
        vstop_short = df['Close'] + multiplier * atr
        
        vstop = np.zeros(len(df))
        direction = np.zeros(len(df))
        vstop[0] = vstop_long.iloc[0]
        direction[0] = 1
        
        for i in range(1, len(df)):
            if df['Close'].iloc[i] > vstop[i-1]:
                vstop[i] = max(vstop_long.iloc[i], vstop[i-1])
                direction[i] = 1
            else:
                vstop[i] = min(vstop_short.iloc[i], vstop[i-1])
                direction[i] = -1
                
        return vstop, direction
    
    for period in [14, 20, 30]:
        for mult in [1.5, 2, 2.5, 3]:
            vstop, vstop_dir = calculate_vstop(df, period, mult)
            df[f'VSTOP_{period}_{mult}'] = vstop
            df[f'VSTOP_direction_{period}_{mult}'] = vstop_dir
            df[f'VSTOP_distance_{period}_{mult}'] = df['Close'] - vstop
    
    # ==================== ICHIMOKU CLOUD (Complete) ====================
    def calculate_ichimoku(df):
        # Tenkan-sen
        high_9 = df['High'].rolling(window=9, min_periods=1).max()
        low_9 = df['Low'].rolling(window=9, min_periods=1).min()
        tenkan_sen = (high_9 + low_9) / 2
        
        # Kijun-sen
        high_26 = df['High'].rolling(window=26, min_periods=1).max()
        low_26 = df['Low'].rolling(window=26, min_periods=1).min()
        kijun_sen = (high_26 + low_26) / 2
        
        # Senkou Spans
        senkou_span_a = ((tenkan_sen + kijun_sen) / 2).shift(26)
        
        high_52 = df['High'].rolling(window=52, min_periods=1).max()
        low_52 = df['Low'].rolling(window=52, min_periods=1).min()
        senkou_span_b = ((high_52 + low_52) / 2).shift(26)
        
        # Chikou Span
        chikou_span = df['Close'].shift(-26)
        
        return tenkan_sen, kijun_sen, senkou_span_a, senkou_span_b, chikou_span
    
    tenkan, kijun, span_a, span_b, chikou = calculate_ichimoku(df)
    df['ichimoku_tenkan'] = tenkan
    df['ichimoku_kijun'] = kijun
    df['ichimoku_span_a'] = span_a
    df['ichimoku_span_b'] = span_b
    df['ichimoku_chikou'] = chikou
    df['ichimoku_cloud_thickness'] = abs(span_a - span_b)
    df['ichimoku_above_cloud'] = np.where(
        df['Close'] > df[['ichimoku_span_a', 'ichimoku_span_b']].max(axis=1), 1, 0
    )
    df['ichimoku_below_cloud'] = np.where(
        df['Close'] < df[['ichimoku_span_a', 'ichimoku_span_b']].min(axis=1), 1, 0
    )
    df['ichimoku_in_cloud'] = 1 - df['ichimoku_above_cloud'] - df['ichimoku_below_cloud']
    df['ichimoku_tenkan_kijun_cross'] = np.where(tenkan > kijun, 1, -1)
    df['ichimoku_price_tenkan_cross'] = np.where(df['Close'] > tenkan, 1, -1)
    
    # ==================== OSCILLATORS & WAVES ====================
    
    # 1. Awesome Oscillator
    df['AO'] = df['Close'].rolling(window=5, min_periods=1).mean() - df['Close'].rolling(window=34, min_periods=1).mean()
    df['AO_diff'] = df['AO'].diff()
    df['AO_ma'] = df['AO'].rolling(window=5).mean()
    
    # 2. Chande Momentum Oscillator
    def calculate_cmo(prices, period=14):
        price_diff = prices.diff()
        up_sum = price_diff.where(price_diff > 0, 0).rolling(period).sum()
        down_sum = abs(price_diff.where(price_diff < 0, 0)).rolling(period).sum()
        
        cmo = 100 * (up_sum - down_sum) / (up_sum + down_sum).replace(0, np.nan)
        return cmo
    
    for period in [9, 14, 20]:
        df[f'CMO_{period}'] = calculate_cmo(df['Close'], period)
    
    # 3. Detrended Price Oscillator
    def calculate_dpo(prices, period=20):
        shift = int(period / 2) + 1
        ma = prices.rolling(window=period).mean()
        dpo = prices.shift(shift) - ma
        return dpo
    
    for period in [10, 20, 30]:
        df[f'DPO_{period}'] = calculate_dpo(df['Close'], period)
    
    # 4. Percentage Price Oscillator
    def calculate_ppo(prices, fast=12, slow=26, signal=9):
        ema_fast = prices.ewm(span=fast, adjust=False).mean()
        ema_slow = prices.ewm(span=slow, adjust=False).mean()
        
        ppo = ((ema_fast - ema_slow) / ema_slow) * 100
        ppo_signal = ppo.ewm(span=signal, adjust=False).mean()
        ppo_hist = ppo - ppo_signal
        
        return ppo, ppo_signal, ppo_hist
    
    for fast, slow, signal in [(12, 26, 9), (5, 35, 5)]:
        ppo, ppo_signal, ppo_hist = calculate_ppo(df['Close'], fast, slow, signal)
        df[f'PPO_{fast}_{slow}'] = ppo
        df[f'PPO_{fast}_{slow}_signal'] = ppo_signal
        df[f'PPO_{fast}_{slow}_hist'] = ppo_hist
    
    # 5. Price Oscillator
    def calculate_price_oscillator(prices, short_period=10, long_period=20):
        short_ma = prices.rolling(window=short_period).mean()
        long_ma = prices.rolling(window=long_period).mean()
        
        po = short_ma - long_ma
        return po
    
    for short, long in [(5, 10), (10, 20), (20, 50)]:
        df[f'PO_{short}_{long}'] = calculate_price_oscillator(df['Close'], short, long)
    
    # 6. QStick Indicator
    def calculate_qstick(df, period=14):
        diff = df['Close'] - df['Open']
        qstick = diff.rolling(window=period).mean()
        return qstick
    
    for period in [8, 14, 20]:
        df[f'QStick_{period}'] = calculate_qstick(df, period)
    
    # 7. Random Walk Index
    def calculate_rwi(df, period=14):
        high = df['High']
        low = df['Low']
        close = df['Close']
        
        atr = calculate_atr(df, period)
        
        rwi_high = (high - low.shift(period)) / (atr * np.sqrt(period))
        rwi_low = (high.shift(period) - low) / (atr * np.sqrt(period))
        
        return rwi_high, rwi_low
    
    for period in [10, 14, 20]:
        rwi_h, rwi_l = calculate_rwi(df, period)
        df[f'RWI_high_{period}'] = rwi_h
        df[f'RWI_low_{period}'] = rwi_l
    
    # 8. Schaff Trend Cycle
    def calculate_stc(df, fast=23, slow=50, cycle=10):
        # MACD
        ema_fast = df['Close'].ewm(span=fast, adjust=False).mean()
        ema_slow = df['Close'].ewm(span=slow, adjust=False).mean()
        macd = ema_fast - ema_slow
        
        # Stochastic of MACD
        lowest_macd = macd.rolling(cycle).min()
        highest_macd = macd.rolling(cycle).max()
        
        stoch = 100 * (macd - lowest_macd) / (highest_macd - lowest_macd).replace(0, np.nan)
        stc = stoch.rolling(cycle).mean()
        
        return stc
    
    df['STC'] = calculate_stc(df)
    df['STC_20_40_10'] = calculate_stc(df, 20, 40, 10)
    
    # ==================== ADVANCED WAVE ANALYSIS ====================
    
    # 1. Sine Wave Indicator
    def calculate_sine_wave(prices, period=20):
        # Hilbert Transform approach (simplified)
        if scipy_available:
            analytic_signal = hilbert(prices.fillna(method='ffill'))
            phase = np.unwrap(np.angle(analytic_signal))
            
            sine = np.sin(phase)
            lead_sine = np.sin(phase + np.pi/4)
            
            return pd.Series(sine, index=prices.index), pd.Series(lead_sine, index=prices.index)
        else:
            # Simplified version
            x = np.arange(len(prices))
            sine = np.sin(2 * np.pi * x / period)
            lead_sine = np.sin(2 * np.pi * x / period + np.pi/4)
            
            return pd.Series(sine, index=prices.index), pd.Series(lead_sine, index=prices.index)
    
    sine, lead_sine = calculate_sine_wave(df['Close'])
    df['sine_wave'] = sine
    df['lead_sine'] = lead_sine
    
    # 2. Cyber Cycle
    def calculate_cyber_cycle(prices, alpha=0.07):
        smooth = (prices + 2 * prices.shift(1) + 2 * prices.shift(2) + prices.shift(3)) / 6
        
        cycle = smooth.copy()
        for i in range(6, len(smooth)):
            cycle.iloc[i] = (1 - 0.5 * alpha) ** 2 * (smooth.iloc[i] - 2 * smooth.iloc[i-1] + smooth.iloc[i-2]) + \
                            2 * (1 - alpha) * cycle.iloc[i-1] - (1 - alpha) ** 2 * cycle.iloc[i-2]
        
        return cycle
    
    df['cyber_cycle'] = calculate_cyber_cycle(df['Close'])
    
    # 3. Mesa Adaptive Moving Average (MAMA)
    def calculate_mama_fama(prices, fast_limit=0.5, slow_limit=0.05):
        # Simplified MAMA/FAMA calculation
        mama = prices.ewm(span=20, adjust=False).mean()
        fama = mama.ewm(span=10, adjust=False).mean()
        
        return mama, fama
    
    df['MAMA'], df['FAMA'] = calculate_mama_fama(df['Close'])
    df['MAMA_FAMA_diff'] = df['MAMA'] - df['FAMA']
    
    # 4. Instantaneous Trendline
    def calculate_instantaneous_trendline(prices, alpha=0.07):
        # Simplified version of Ehlers' Instantaneous Trendline
        it = prices.copy()
        
        for i in range(6, len(prices)):
            it.iloc[i] = (alpha - alpha**2/4) * prices.iloc[i] + \
                        0.5 * alpha**2 * prices.iloc[i-1] - \
                        (alpha - 0.75 * alpha**2) * prices.iloc[i-2] + \
                        2 * (1 - alpha) * it.iloc[i-1] - \
                        (1 - alpha)**2 * it.iloc[i-2]
        
        return it
    
    df['instantaneous_trendline'] = calculate_instantaneous_trendline(df['Close'])
    
    # 5. Dominant Cycle Period
    def calculate_dominant_cycle(prices, min_period=8, max_period=50):
        # Simplified dominant cycle calculation
        cycles = []
        for period in range(min_period, max_period + 1):
            cycle_component = np.sin(2 * np.pi * np.arange(len(prices)) / period)
            correlation = np.corrcoef(prices.fillna(method='ffill'), cycle_component)[0, 1]
            cycles.append(abs(correlation))
        
        dominant_period = min_period + np.argmax(cycles)
        return dominant_period
    
    # Calculate dominant cycle for windows
    dom_cycles = []
    for i in range(100, len(df)):
        window = df['Close'].iloc[i-100:i]
        dom_cycle = calculate_dominant_cycle(window)
        dom_cycles.append(dom_cycle)
    
    # Pad the beginning
    dom_cycles = [20] * 100 + dom_cycles
    df['dominant_cycle_period'] = dom_cycles
    
    # ==================== PATTERN RECOGNITION ====================
    
    # 1. Candlestick Patterns
    # Basic patterns
    df['doji'] = np.where(abs(df['Close'] - df['Open']) / (df['High'] - df['Low']).replace(0, np.nan) < 0.1, 1, 0)
    df['hammer'] = np.where(
        ((df['Low'] - np.minimum(df['Close'], df['Open'])) > 2 * abs(df['Close'] - df['Open'])) &
        ((df['High'] - np.maximum(df['Close'], df['Open'])) < 0.3 * abs(df['Close'] - df['Open'])),
        1, 0
    )
    df['shooting_star'] = np.where(
        ((df['High'] - np.maximum(df['Close'], df['Open'])) > 2 * abs(df['Close'] - df['Open'])) &
        ((np.minimum(df['Close'], df['Open']) - df['Low']) < 0.3 * abs(df['Close'] - df['Open'])),
        1, 0
    )
    
    # 2. Three Line Strike
    def detect_three_line_strike(df):
        pattern = np.zeros(len(df))
        
        for i in range(3, len(df)):
            # Bullish pattern
            if (df['Close'].iloc[i-3] < df['Open'].iloc[i-3] and
                df['Close'].iloc[i-2] < df['Open'].iloc[i-2] and
                df['Close'].iloc[i-1] < df['Open'].iloc[i-1] and
                df['Close'].iloc[i] > df['Open'].iloc[i] and
                df['Close'].iloc[i] > df['Open'].iloc[i-3]):
                pattern[i] = 1
            
            # Bearish pattern
            elif (df['Close'].iloc[i-3] > df['Open'].iloc[i-3] and
                  df['Close'].iloc[i-2] > df['Open'].iloc[i-2] and
                  df['Close'].iloc[i-1] > df['Open'].iloc[i-1] and
                  df['Close'].iloc[i] < df['Open'].iloc[i] and
                  df['Close'].iloc[i] < df['Open'].iloc[i-3]):
                pattern[i] = -1
        
        return pattern
    
    df['three_line_strike'] = detect_three_line_strike(df)
    
    # ==================== FIBONACCI FEATURES ====================
    
    # Fibonacci Retracement Levels
    def calculate_fib_levels(df, period=50):
        high = df['High'].rolling(window=period, min_periods=1).max()
        low = df['Low'].rolling(window=period, min_periods=1).min()
        diff = high - low
        
        fib_levels = {
            '0': high,
            '236': high - 0.236 * diff,
            '382': high - 0.382 * diff,
            '500': high - 0.500 * diff,
            '618': high - 0.618 * diff,
            '786': high - 0.786 * diff,
            '100': low,
            '1272': high + 0.272 * diff,
            '1618': high + 0.618 * diff
        }
        
        return fib_levels
    
    for period in [21, 34, 50, 89, 144, 233]:
        fib_levels = calculate_fib_levels(df, period)
        for level, values in fib_levels.items():
            df[f'fib_{level}_{period}'] = values
            df[f'fib_{level}_{period}_distance'] = df['Close'] - values
    
    # ==================== PIVOT POINTS (All Variations) ====================
    
    # 1. Standard Pivot Points
    def calculate_pivot_points(df):
        pivot = (df['High'].shift(1) + df['Low'].shift(1) + df['Close'].shift(1)) / 3
        
        r1 = 2 * pivot - df['Low'].shift(1)
        s1 = 2 * pivot - df['High'].shift(1)
        r2 = pivot + (df['High'].shift(1) - df['Low'].shift(1))
        s2 = pivot - (df['High'].shift(1) - df['Low'].shift(1))
        r3 = df['High'].shift(1) + 2 * (pivot - df['Low'].shift(1))
        s3 = df['Low'].shift(1) - 2 * (df['High'].shift(1) - pivot)
        r4 = r3 + (df['High'].shift(1) - df['Low'].shift(1))
        s4 = s3 - (df['High'].shift(1) - df['Low'].shift(1))
        
        return pivot, r1, s1, r2, s2, r3, s3, r4, s4
    
    pivot, r1, s1, r2, s2, r3, s3, r4, s4 = calculate_pivot_points(df)
    df['pivot_point'] = pivot
    df['pivot_r1'] = r1
    df['pivot_s1'] = s1
    df['pivot_r2'] = r2
    df['pivot_s2'] = s2
    df['pivot_r3'] = r3
    df['pivot_s3'] = s3
    df['pivot_r4'] = r4
    df['pivot_s4'] = s4
    
    # 2. Camarilla Pivot Points
    def calculate_camarilla_pivots(df):
        h = df['High'].shift(1)
        l = df['Low'].shift(1)
        c = df['Close'].shift(1)
        
        range_hl = h - l
        
        r4 = c + range_hl * 1.1 / 2
        r3 = c + range_hl * 1.1 / 4
        r2 = c + range_hl * 1.1 / 6
        r1 = c + range_hl * 1.1 / 12
        
        s1 = c - range_hl * 1.1 / 12
        s2 = c - range_hl * 1.1 / 6
        s3 = c - range_hl * 1.1 / 4
        s4 = c - range_hl * 1.1 / 2
        
        r5 = r4 + 1.168 * (r4 - r3)
        s5 = s4 - 1.168 * (s3 - s4)
        
        return r5, r4, r3, r2, r1, s1, s2, s3, s4, s5
    
    cam_r5, cam_r4, cam_r3, cam_r2, cam_r1, cam_s1, cam_s2, cam_s3, cam_s4, cam_s5 = calculate_camarilla_pivots(df)
    df['camarilla_r5'] = cam_r5
    df['camarilla_r4'] = cam_r4
    df['camarilla_r3'] = cam_r3
    df['camarilla_r2'] = cam_r2
    df['camarilla_r1'] = cam_r1
    df['camarilla_s1'] = cam_s1
    df['camarilla_s2'] = cam_s2
    df['camarilla_s3'] = cam_s3
    df['camarilla_s4'] = cam_s4
    df['camarilla_s5'] = cam_s5
    
    # 3. Woodie's Pivot Points
    def calculate_woodie_pivots(df):
        h = df['High'].shift(1)
        l = df['Low'].shift(1)
        c = df['Close'].shift(1)
        
        pivot = (h + l + 2 * c) / 4
        r1 = 2 * pivot - l
        s1 = 2 * pivot - h
        r2 = pivot + (h - l)
        s2 = pivot - (h - l)
        r3 = h + 2 * (pivot - l)
        s3 = l - 2 * (h - pivot)
        
        return pivot, r1, s1, r2, s2, r3, s3
    
    woodie_pivot, woodie_r1, woodie_s1, woodie_r2, woodie_s2, woodie_r3, woodie_s3 = calculate_woodie_pivots(df)
    df['woodie_pivot'] = woodie_pivot
    df['woodie_r1'] = woodie_r1
    df['woodie_s1'] = woodie_s1
    df['woodie_r2'] = woodie_r2
    df['woodie_s2'] = woodie_s2
    df['woodie_r3'] = woodie_r3
    df['woodie_s3'] = woodie_s3
    
    # 4. Fibonacci Pivot Points
    def calculate_fibonacci_pivots(df):
        h = df['High'].shift(1)
        l = df['Low'].shift(1)
        c = df['Close'].shift(1)
        
        pivot = (h + l + c) / 3
        range_hl = h - l
        
        r1 = pivot + 0.382 * range_hl
        r2 = pivot + 0.618 * range_hl
        r3 = pivot + range_hl
        
        s1 = pivot - 0.382 * range_hl
        s2 = pivot - 0.618 * range_hl
        s3 = pivot - range_hl
        
        return pivot, r1, r2, r3, s1, s2, s3
    
    fib_pivot, fib_r1, fib_r2, fib_r3, fib_s1, fib_s2, fib_s3 = calculate_fibonacci_pivots(df)
    df['fib_pivot'] = fib_pivot
    df['fib_pivot_r1'] = fib_r1
    df['fib_pivot_r2'] = fib_r2
    df['fib_pivot_r3'] = fib_r3
    df['fib_pivot_s1'] = fib_s1
    df['fib_pivot_s2'] = fib_s2
    df['fib_pivot_s3'] = fib_s3
    
    # 5. DeMark Pivot Points
    def calculate_demark_pivots(df):
        h = df['High'].shift(1)
        l = df['Low'].shift(1)
        c = df['Close'].shift(1)
        o = df['Open'].shift(1)
        
        # X value calculation
        x = np.where(c < o, h + 2 * l + c,
                    np.where(c > o, 2 * h + l + c,
                            h + l + 2 * c))
        
        pivot = x / 4
        r1 = x / 2 - l
        s1 = x / 2 - h
        
        return pivot, r1, s1
    
    demark_pivot, demark_r1, demark_s1 = calculate_demark_pivots(df)
    df['demark_pivot'] = demark_pivot
    df['demark_r1'] = demark_r1
    df['demark_s1'] = demark_s1
    
    # ==================== MARKET INTERNALS ====================
    
    # 1. Advance/Decline Oscillator (simulated)
    df['advance_decline'] = np.where(df['Close'] > df['Close'].shift(), 1, -1)
    df['ad_oscillator'] = df['advance_decline'].rolling(window=10).sum()
    df['ad_oscillator_ma'] = df['ad_oscillator'].rolling(window=20).mean()
    
    # 2. McClellan Oscillator (simulated)
    df['mcclellan_short'] = df['advance_decline'].ewm(span=19, adjust=False).mean()
    df['mcclellan_long'] = df['advance_decline'].ewm(span=39, adjust=False).mean()
    df['mcclellan_oscillator'] = df['mcclellan_short'] - df['mcclellan_long']
    
    # 3. TRIN (simulated)
    df['trin'] = (df['advance_decline'].rolling(20).sum() / df['Volume'].rolling(20).mean()).replace(0, np.nan)
    
    # ==================== GANN INDICATORS ====================
    
    # 1. Gann Angles (simplified)
    def calculate_gann_angles(df, period=50):
        # Calculate price range
        price_range = df['High'].rolling(period).max() - df['Low'].rolling(period).min()
        time_units = period
        
        # Basic Gann angles (price/time ratios)
        gann_1x1 = df['Low'].rolling(period).min() + (price_range / time_units) * np.arange(len(df))
        gann_2x1 = df['Low'].rolling(period).min() + (2 * price_range / time_units) * np.arange(len(df))
        gann_1x2 = df['Low'].rolling(period).min() + (price_range / (2 * time_units)) * np.arange(len(df))
        
        return gann_1x1, gann_2x1, gann_1x2
    
    # Apply Gann calculations safely
    gann_1x1, gann_2x1, gann_1x2 = calculate_gann_angles(df)
    df['gann_1x1'] = pd.Series(gann_1x1[-len(df):], index=df.index)
    df['gann_2x1'] = pd.Series(gann_2x1[-len(df):], index=df.index)
    df['gann_1x2'] = pd.Series(gann_1x2[-len(df):], index=df.index)
    
    # 2. Gann HiLo Activator
    def calculate_gann_hilo(df, period=3):
        hilo_support = df['Low'].rolling(period).mean()
        hilo_resistance = df['High'].rolling(period).mean()
        
        hilo = np.where(df['Close'] > hilo_resistance.shift(), hilo_support, hilo_resistance)
        
        return hilo
    
    df['gann_hilo'] = calculate_gann_hilo(df)
    df['gann_hilo_signal'] = np.where(df['Close'] > df['gann_hilo'], 1, -1)
    
    # ==================== ELDER RAY ====================
    
    def calculate_elder_ray(df, period=13):
        ema = df['Close'].ewm(span=period, adjust=False).mean()
        bull_power = df['High'] - ema
        bear_power = df['Low'] - ema
        
        return bull_power, bear_power
    
    for period in [13, 21]:
        bull, bear = calculate_elder_ray(df, period)
        df[f'elder_bull_{period}'] = bull
        df[f'elder_bear_{period}'] = bear
        df[f'elder_ray_{period}'] = bull + bear
    
    # ==================== MARKET PROFILE ====================
    
    def calculate_market_profile(df, period=24):
        # Value Area calculation (simplified)
        for i in range(period, len(df)):
            period_data = df.iloc[i-period:i]
            
            # Create price histogram
            prices = period_data['Close'].values
            volumes = period_data['Volume'].values
            
            # Find POC (Point of Control)
            price_levels = np.linspace(prices.min(), prices.max(), 20)
            volume_profile = np.zeros(len(price_levels)-1)
            
            for j in range(len(prices)):
                idx = np.digitize(prices[j], price_levels) - 1
                if 0 <= idx < len(volume_profile):
                    volume_profile[idx] += volumes[j]
            
            # POC is the price level with highest volume
            poc_idx = np.argmax(volume_profile)
            poc_price = (price_levels[poc_idx] + price_levels[poc_idx+1]) / 2
            
            # Value area (70% of volume)
            total_volume = volume_profile.sum()
            target_volume = total_volume * 0.7
            
            # Expand from POC until we reach 70% volume
            current_volume = volume_profile[poc_idx]
            low_idx, high_idx = poc_idx, poc_idx
            
            while current_volume < target_volume and (low_idx > 0 or high_idx < len(volume_profile)-1):
                if low_idx > 0 and high_idx < len(volume_profile)-1:
                    if volume_profile[low_idx-1] > volume_profile[high_idx+1]:
                        low_idx -= 1
                        current_volume += volume_profile[low_idx]
                    else:
                        high_idx += 1
                        current_volume += volume_profile[high_idx]
                elif low_idx > 0:
                    low_idx -= 1
                    current_volume += volume_profile[low_idx]
                elif high_idx < len(volume_profile)-1:
                    high_idx += 1
                    current_volume += volume_profile[high_idx]
            
            vah = price_levels[high_idx+1] if high_idx < len(price_levels)-1 else price_levels[-1]
            val = price_levels[low_idx]
            
            df.loc[df.index[i], 'market_profile_poc'] = poc_price
            df.loc[df.index[i], 'market_profile_vah'] = vah
            df.loc[df.index[i], 'market_profile_val'] = val
            df.loc[df.index[i], 'market_profile_value_area_width'] = vah - val
    
    # Initialize columns
    df['market_profile_poc'] = np.nan
    df['market_profile_vah'] = np.nan
    df['market_profile_val'] = np.nan
    df['market_profile_value_area_width'] = np.nan
    
    calculate_market_profile(df)
    
    # Forward fill market profile values
    df['market_profile_poc'] = df['market_profile_poc'].ffill()
    df['market_profile_vah'] = df['market_profile_vah'].ffill()
    df['market_profile_val'] = df['market_profile_val'].ffill()
    df['market_profile_value_area_width'] = df['market_profile_value_area_width'].ffill()
    
    # ==================== VOLUME PROFILE (Enhanced) ====================
    # Enhanced Volume Profile with TPO concept
    def calculate_enhanced_volume_profile(df, period=24, n_bins=30):
        """Enhanced Volume Profile with Time Price Opportunity (TPO)"""
        volume_profile_features = {}
        
        for i in range(period, len(df)):
            window_data = df.iloc[i-period:i]
            
            # Create price bins
            price_range = window_data['High'].max() - window_data['Low'].min()
            if price_range > 0:
                # Volume at Price (VAP)
                price_bins = np.linspace(window_data['Low'].min(), window_data['High'].max(), n_bins)
                volume_distribution = np.zeros(len(price_bins) - 1)
                
                for j in range(len(window_data)):
                    # Distribute volume across the range of the candle
                    candle_high = window_data['High'].iloc[j]
                    candle_low = window_data['Low'].iloc[j]
                    candle_volume = window_data['Volume'].iloc[j]
                    
                    # Find bins that this candle touches
                    touched_bins = np.where((price_bins[:-1] <= candle_high) & (price_bins[1:] >= candle_low))[0]
                    
                    if len(touched_bins) > 0:
                        # Distribute volume equally among touched bins
                        volume_per_bin = candle_volume / len(touched_bins)
                        volume_distribution[touched_bins] += volume_per_bin
                
                # Calculate key metrics
                total_volume = volume_distribution.sum()
                if total_volume > 0:
                    # VPOC (Volume Point of Control)
                    vpoc_idx = np.argmax(volume_distribution)
                    vpoc_price = (price_bins[vpoc_idx] + price_bins[vpoc_idx + 1]) / 2
                    
                    # High Volume Nodes (HVN) and Low Volume Nodes (LVN)
                    volume_threshold_high = np.percentile(volume_distribution, 70)
                    volume_threshold_low = np.percentile(volume_distribution, 30)
                    
                    hvn_count = np.sum(volume_distribution > volume_threshold_high)
                    lvn_count = np.sum(volume_distribution < volume_threshold_low)
                    
                    # Volume Weighted Average Price for the period
                    prices_mid = (price_bins[:-1] + price_bins[1:]) / 2
                    period_vwap = np.sum(prices_mid * volume_distribution) / total_volume
                    
                    # Store features
                    df.loc[df.index[i], 'volume_profile_vpoc'] = vpoc_price
                    df.loc[df.index[i], 'volume_profile_hvn_count'] = hvn_count
                    df.loc[df.index[i], 'volume_profile_lvn_count'] = lvn_count
                    df.loc[df.index[i], 'volume_profile_period_vwap'] = period_vwap
                    df.loc[df.index[i], 'volume_profile_vpoc_distance'] = df['Close'].iloc[i] - vpoc_price
    
    # Initialize columns
    df['volume_profile_vpoc'] = np.nan
    df['volume_profile_hvn_count'] = np.nan
    df['volume_profile_lvn_count'] = np.nan
    df['volume_profile_period_vwap'] = np.nan
    df['volume_profile_vpoc_distance'] = np.nan
    
    calculate_enhanced_volume_profile(df)
    
    # Fill NaN values
    for col in ['volume_profile_vpoc', 'volume_profile_hvn_count', 'volume_profile_lvn_count', 
                'volume_profile_period_vwap', 'volume_profile_vpoc_distance']:
        df[col] = df[col].ffill().fillna(0)
    
    # Original simple volume profile
    price_bins = pd.cut(df['Close'], bins=20, labels=False)
    df['volume_profile_level'] = price_bins
    
    # ==================== HEIKIN ASHI ====================
    def calculate_heikin_ashi(df):
        ha_close = (df['Open'] + df['High'] + df['Low'] + df['Close']) / 4
        ha_open = ha_close.copy()
        ha_open.iloc[0] = (df['Open'].iloc[0] + df['Close'].iloc[0]) / 2
        
        for i in range(1, len(ha_open)):
            ha_open.iloc[i] = (ha_open.iloc[i-1] + ha_close.iloc[i-1]) / 2
        
        ha_high = df[['High', 'Open', 'Close']].max(axis=1)
        ha_low = df[['Low', 'Open', 'Close']].min(axis=1)
        
        return ha_open, ha_high, ha_low, ha_close
    
    ha_open, ha_high, ha_low, ha_close = calculate_heikin_ashi(df)
    df['HA_open'] = ha_open
    df['HA_high'] = ha_high
    df['HA_low'] = ha_low
    df['HA_close'] = ha_close
    df['HA_body'] = df['HA_close'] - df['HA_open']
    df['HA_wickup'] = df['HA_high'] - df[['HA_open', 'HA_close']].max(axis=1)
    df['HA_wickdown'] = df[['HA_open', 'HA_close']].min(axis=1) - df['HA_low']
    df['HA_trend'] = np.where(df['HA_close'] > df['HA_open'], 1, -1)
    df['HA_body_ratio'] = abs(df['HA_body']) / (df['HA_high'] - df['HA_low']).replace(0, np.nan)
    
    # ==================== RENKO ====================
    def calculate_renko_features(df, brick_size=None):
        if brick_size is None:
            # Dynamic brick size based on ATR
            brick_size = calculate_atr(df, 14).mean()
        
        renko_prices = []
        renko_directions = []
        current_price = df['Close'].iloc[0]
        
        for i in range(len(df)):
            price = df['Close'].iloc[i]
            
            if len(renko_prices) == 0:
                renko_prices.append(current_price)
                renko_directions.append(0)
                continue
            
            price_diff = price - current_price
            
            if abs(price_diff) >= brick_size:
                num_bricks = int(abs(price_diff) / brick_size)
                direction = 1 if price_diff > 0 else -1
                
                for _ in range(num_bricks):
                    current_price += direction * brick_size
                    renko_prices.append(current_price)
                    renko_directions.append(direction)
            
            # Pad to match original length
            while len(renko_prices) < i + 1:
                renko_prices.append(renko_prices[-1])
                renko_directions.append(0)
        
        # Ensure same length
        renko_prices = renko_prices[:len(df)]
        renko_directions = renko_directions[:len(df)]
        
        return pd.Series(renko_prices, index=df.index), pd.Series(renko_directions, index=df.index)
    
    df['renko_price'], df['renko_direction'] = calculate_renko_features(df)
    df['renko_diff'] = df['Close'] - df['renko_price']
    
    # ==================== KAGI ====================
    def calculate_kagi_features(df, reversal_pct=0.04):
        kagi_prices = []
        kagi_directions = []
        
        current_price = df['Close'].iloc[0]
        current_direction = 0
        reversal_price = current_price
        
        for i in range(len(df)):
            price = df['Close'].iloc[i]
            
            if current_direction == 0:  # Initial state
                if price > current_price * (1 + reversal_pct):
                    current_direction = 1
                elif price < current_price * (1 - reversal_pct):
                    current_direction = -1
                current_price = price
            
            elif current_direction == 1:  # Uptrend
                if price > current_price:
                    current_price = price
                elif price < current_price * (1 - reversal_pct):
                    current_direction = -1
                    reversal_price = current_price
                    current_price = price
            
            else:  # Downtrend
                if price < current_price:
                    current_price = price
                elif price > current_price * (1 + reversal_pct):
                    current_direction = 1
                    reversal_price = current_price
                    current_price = price
            
            kagi_prices.append(current_price)
            kagi_directions.append(current_direction)
        
        return pd.Series(kagi_prices, index=df.index), pd.Series(kagi_directions, index=df.index)
    
    df['kagi_price'], df['kagi_direction'] = calculate_kagi_features(df)
    df['kagi_diff'] = df['Close'] - df['kagi_price']
    
    # ==================== POINT AND FIGURE ====================
    def calculate_pnf_features(df, box_size=None, reversal_boxes=3):
        if box_size is None:
            box_size = calculate_atr(df, 14).mean()
        
        pnf_column_type = []  # 1 for X (up), -1 for O (down)
        pnf_column_height = []
        
        current_column_type = 0
        current_column_height = 0
        current_price = df['Close'].iloc[0]
        
        for i in range(len(df)):
            price = df['Close'].iloc[i]
            
            if current_column_type == 0:  # Initialize
                if price >= current_price + box_size:
                    current_column_type = 1
                    current_column_height = int((price - current_price) / box_size)
                elif price <= current_price - box_size:
                    current_column_type = -1
                    current_column_height = int((current_price - price) / box_size)
            
            elif current_column_type == 1:  # X column (uptrend)
                if price >= current_price + box_size:
                    boxes_to_add = int((price - current_price) / box_size)
                    current_column_height += boxes_to_add
                    current_price += boxes_to_add * box_size
                elif price <= current_price - (reversal_boxes * box_size):
                    # Reversal to O column
                    current_column_type = -1
                    current_column_height = reversal_boxes
                    current_price -= reversal_boxes * box_size
            
            else:  # O column (downtrend)
                if price <= current_price - box_size:
                    boxes_to_add = int((current_price - price) / box_size)
                    current_column_height += boxes_to_add
                    current_price -= boxes_to_add * box_size
                elif price >= current_price + (reversal_boxes * box_size):
                    # Reversal to X column
                    current_column_type = 1
                    current_column_height = reversal_boxes
                    current_price += reversal_boxes * box_size
            
            pnf_column_type.append(current_column_type)
            pnf_column_height.append(current_column_height)
        
        return pd.Series(pnf_column_type, index=df.index), pd.Series(pnf_column_height, index=df.index)
    
    df['pnf_column_type'], df['pnf_column_height'] = calculate_pnf_features(df)
    
    # ==================== LINE BREAK ====================
    def calculate_line_break_features(df, num_lines=3):
        line_break_prices = []
        line_break_colors = []  # 1 for white (up), -1 for black (down)
        
        price_lines = []
        
        for i in range(len(df)):
            price = df['Close'].iloc[i]
            
            if len(price_lines) < num_lines:
                price_lines.append(price)
                if len(price_lines) == 1:
                    line_break_colors.append(0)
                else:
                    line_break_colors.append(1 if price > price_lines[-2] else -1)
            else:
                if price > max(price_lines):
                    price_lines.append(price)
                    price_lines.pop(0)
                    line_break_colors.append(1)
                elif price < min(price_lines):
                    price_lines.append(price)
                    price_lines.pop(0)
                    line_break_colors.append(-1)
                else:
                    line_break_colors.append(line_break_colors[-1])
            
            line_break_prices.append(price_lines[-1] if price_lines else price)
        
        return pd.Series(line_break_prices, index=df.index), pd.Series(line_break_colors, index=df.index)
    
    df['line_break_price'], df['line_break_color'] = calculate_line_break_features(df)
    
    # ==================== ELLIOTT WAVE (Enhanced) ====================
    def detect_elliott_waves(prices, window=10):
        # Simplified peak/trough detection
        wave_count = np.zeros(len(prices))
        
        # Use percentile-based thresholds
        high_thresh = np.percentile(prices, 80)
        low_thresh = np.percentile(prices, 20)
        
        wave_count[prices > high_thresh] = 1
        wave_count[prices < low_thresh] = -1
        
        return pd.Series(wave_count).rolling(window=window*5, min_periods=1).sum()
    
    df['elliott_wave_count'] = detect_elliott_waves(df['Close'].values)
    
    def advanced_elliott_wave_detection(df, lookback=100):
        """Enhanced Elliott Wave detection with wave degree identification"""
        prices = df['Close'].values
        highs = df['High'].values
        lows = df['Low'].values
        
        # Find swing points using local extrema
        # Different window sizes for different wave degrees
        wave_degrees = {
            'minor': 5,
            'intermediate': 21,
            'primary': 55,
            'grand': 89,
            'super': 144
        }
        
        elliott_features = {}
        
        if scipy_available:
            for degree, window in wave_degrees.items():
                # Find local maxima and minima
                local_max = argrelextrema(highs, np.greater, order=window)[0]
                local_min = argrelextrema(lows, np.less, order=window)[0]
                
                # Wave counting
                wave_pattern = np.zeros(len(df))
                
                # Mark highs as positive, lows as negative
                wave_pattern[local_max] = 1
                wave_pattern[local_min] = -1
                
                # Smooth to get wave structure
                wave_smooth = pd.Series(wave_pattern).rolling(window=window*2, min_periods=1).sum()
                
                elliott_features[f'elliott_{degree}_wave'] = wave_smooth
                elliott_features[f'elliott_{degree}_strength'] = abs(wave_smooth)
                
                # Detect potential wave 3 (strongest trend)
                momentum = pd.Series(prices).pct_change(window).rolling(window).std()
                elliott_features[f'elliott_{degree}_wave3_prob'] = momentum / momentum.rolling(50).mean()
        else:
            # Simplified version without scipy
            for degree, window in wave_degrees.items():
                # Use rolling max/min to find extrema
                roll_max = pd.Series(highs).rolling(window*2+1, center=True).max()
                roll_min = pd.Series(lows).rolling(window*2+1, center=True).min()
                
                wave_pattern = np.zeros(len(df))
                wave_pattern[highs == roll_max] = 1
                wave_pattern[lows == roll_min] = -1
                
                wave_smooth = pd.Series(wave_pattern).rolling(window=window*2, min_periods=1).sum()
                
                elliott_features[f'elliott_{degree}_wave'] = wave_smooth
                elliott_features[f'elliott_{degree}_strength'] = abs(wave_smooth)
                
                momentum = pd.Series(prices).pct_change(window).rolling(window).std()
                elliott_features[f'elliott_{degree}_wave3_prob'] = momentum / momentum.rolling(50).mean()
        
        return elliott_features
    
    elliott_advanced = advanced_elliott_wave_detection(df)
    for name, values in elliott_advanced.items():
        df[name] = values
    
    # ==================== HARMONIC PATTERNS (Complete) ====================
    def detect_harmonic_patterns(df, lookback=50):
        high = df['High'].rolling(window=lookback, min_periods=1).max()
        low = df['Low'].rolling(window=lookback, min_periods=1).min()
        current = df['Close']
        
        range_hl = high - low
        range_hl = range_hl.replace(0, np.nan)
        retracement = (high - current) / range_hl
        
        # All Fibonacci ratios for harmonic patterns
        harmonic_ratios = {
            '382': 0.382,
            '500': 0.500,
            '618': 0.618,
            '707': 0.707,
            '786': 0.786,
            '886': 0.886,
            '1000': 1.000,
            '1130': 1.130,
            '1272': 1.272,
            '1414': 1.414,
            '1618': 1.618,
            '2000': 2.000,
            '2236': 2.236,
            '2618': 2.618,
            '3141': 3.141,
            '3618': 3.618
        }
        
        harmonic_scores = {}
        for name, ratio in harmonic_ratios.items():
            harmonic_scores[f'harmonic_{name}'] = (abs(retracement - ratio) < 0.05).astype(int)
        
        return harmonic_scores
    
    harmonic_patterns = detect_harmonic_patterns(df)
    for name, values in harmonic_patterns.items():
        df[name] = values
    
    def advanced_harmonic_patterns(df, lookback=100):
        """Detect Gartley, Butterfly, Bat, Crab, Shark, and Cypher patterns"""
        
        # Key Fibonacci ratios for different patterns
        pattern_ratios = {
            'gartley': {'XA_BC': 0.618, 'AB_CD': 1.272, 'XA_AD': 0.786},
            'butterfly': {'XA_BC': 0.786, 'AB_CD': 1.618, 'XA_AD': 1.272},
            'bat': {'XA_BC': 0.886, 'AB_CD': 2.618, 'XA_AD': 0.886},
            'crab': {'XA_BC': 0.886, 'AB_CD': 3.618, 'XA_AD': 1.618},
            'shark': {'XA_BC': 1.130, 'AB_CD': 1.618, 'XA_AD': 0.886},
            'cypher': {'XA_BC': 0.382, 'AB_CD': 2.000, 'XA_AD': 0.786},
            'nen_star': {'XA_BC': 0.382, 'AB_CD': 1.272, 'XA_AD': 0.618},
            'black_swan': {'XA_BC': 1.382, 'AB_CD': 2.618, 'XA_AD': 1.272},
            'white_swan': {'XA_BC': 0.236, 'AB_CD': 3.618, 'XA_AD': 0.500}
        }
        
        harmonic_features = {}
        
        # Find swing points
        highs = df['High'].values
        lows = df['Low'].values
        
        if scipy_available:
            # Find local extrema
            high_points = argrelextrema(highs, np.greater, order=10)[0]
            low_points = argrelextrema(lows, np.less, order=10)[0]
        else:
            # Simplified without scipy
            high_points = []
            low_points = []
        
        # Pattern detection scores
        for pattern_name, ratios in pattern_ratios.items():
            pattern_score = np.zeros(len(df))
            
            # Simplified pattern detection based on retracement levels
            for i in range(lookback, len(df)):
                window_high = df['High'].iloc[i-lookback:i].max()
                window_low = df['Low'].iloc[i-lookback:i].min()
                window_range = window_high - window_low
                
                if window_range > 0:
                    current_retracement = (df['Close'].iloc[i] - window_low) / window_range
                    
                    # Check if current retracement matches pattern ratios
                    score = 0
                    for ratio_name, ratio_value in ratios.items():
                        if abs(current_retracement - ratio_value) < 0.05:
                            score += 1
                    
                    pattern_score[i] = score / len(ratios)
            
            harmonic_features[f'harmonic_{pattern_name}_score'] = pattern_score
        
        # AB=CD pattern detection
        abcd_pattern = np.zeros(len(df))
        for i in range(20, len(df)):
            # Look for equal moves
            move1 = df['Close'].iloc[i-20] - df['Close'].iloc[i-15]
            move2 = df['Close'].iloc[i-10] - df['Close'].iloc[i-5]
            move3 = df['Close'].iloc[i-5] - df['Close'].iloc[i]
            
            if abs(move1) > 0 and abs(move3) > 0:
                if abs(move1 / move3 - 1) < 0.1:  # Similar magnitude moves
                    abcd_pattern[i] = 1
        
        harmonic_features['harmonic_abcd_pattern'] = abcd_pattern
        
        return harmonic_features
    
    harmonic_advanced = advanced_harmonic_patterns(df)
    for name, values in harmonic_advanced.items():
        df[name] = values
    
    # ==================== SMART MONEY CONCEPTS (Ultimate) ====================
    def enhanced_smc_analysis(df):
        """Advanced Smart Money Concepts including more sophisticated patterns"""
        
        # 1. Market Structure Analysis
        # Higher Highs/Higher Lows (Bullish) vs Lower Highs/Lower Lows (Bearish)
        swing_lookback = 10
        
        # Find swing points
        swing_highs = df['High'].rolling(swing_lookback*2+1, center=True).max() == df['High']
        swing_lows = df['Low'].rolling(swing_lookback*2+1, center=True).min() == df['Low']
        
        # Track market structure
        hh = np.zeros(len(df))  # Higher High
        hl = np.zeros(len(df))  # Higher Low
        lh = np.zeros(len(df))  # Lower High
        ll = np.zeros(len(df))  # Lower Low
        
        prev_high = 0
        prev_low = float('inf')
        
        for i in range(len(df)):
            if swing_highs.iloc[i]:
                if df['High'].iloc[i] > prev_high:
                    hh[i] = 1
                else:
                    lh[i] = 1
                prev_high = df['High'].iloc[i]
            
            if swing_lows.iloc[i]:
                if df['Low'].iloc[i] > prev_low:
                    hl[i] = 1
                else:
                    ll[i] = 1
                prev_low = df['Low'].iloc[i]
        
        # Market structure bias
        bullish_structure = pd.Series(hh + hl).rolling(50).sum()
        bearish_structure = pd.Series(lh + ll).rolling(50).sum()
        structure_bias = bullish_structure - bearish_structure
        
        # 2. Order Blocks (Enhanced)
        # Identify last up/down candle before strong move
        strong_move_threshold = df['Close'].pct_change().abs().rolling(20).std() * 2
        strong_bullish_move = df['Close'].pct_change() > strong_move_threshold
        strong_bearish_move = df['Close'].pct_change() < -strong_move_threshold
        
        # Bullish order block: Last bearish candle before strong up move
        bearish_candle = df['Close'] < df['Open']
        bullish_ob = bearish_candle.shift(1) & strong_bullish_move
        
        # Bearish order block: Last bullish candle before strong down move
        bullish_candle = df['Close'] > df['Open']
        bearish_ob = bullish_candle.shift(1) & strong_bearish_move
        
        # 3. Liquidity Zones (Enhanced)
        # Areas where stops are likely to be placed
        lookback = 50
        
        # Swing high/low liquidity
        swing_high_liquidity = df['High'].rolling(lookback).max()
        swing_low_liquidity = df['Low'].rolling(lookback).min()
        
        # Equal highs/lows (liquidity pools)
        high_counts = df['High'].round(0).rolling(20).apply(lambda x: x.value_counts().max())
        low_counts = df['Low'].round(0).rolling(20).apply(lambda x: x.value_counts().max())
        
        equal_highs = high_counts > 2
        equal_lows = low_counts > 2
        
        # 4. Imbalance / Fair Value Gap (Enhanced)
        # Gap between previous high and next low (or vice versa)
        fvg_bullish = (df['Low'].shift(-1) > df['High'].shift(1)) & (df['Close'] > df['Open'])
        fvg_bearish = (df['High'].shift(-1) < df['Low'].shift(1)) & (df['Close'] < df['Open'])
        
        # Size of the gap
        fvg_size_bullish = np.where(fvg_bullish, df['Low'].shift(-1) - df['High'].shift(1), 0)
        fvg_size_bearish = np.where(fvg_bearish, df['Low'].shift(1) - df['High'].shift(-1), 0)
        
        # 5. Premium/Discount Zones
        # Price relative to recent range
        range_high = df['High'].rolling(50).max()
        range_low = df['Low'].rolling(50).min()
        range_mid = (range_high + range_low) / 2
        
        premium_zone = df['Close'] > range_mid + (range_high - range_mid) * 0.5
        discount_zone = df['Close'] < range_mid - (range_mid - range_low) * 0.5
        equilibrium = ~premium_zone & ~discount_zone
        
        # 6. Institutional Candles
        # Large body candles with specific characteristics
        body_size = abs(df['Close'] - df['Open'])
        avg_body = body_size.rolling(20).mean()
        
        # Institutional buying: Large bullish candle with small wicks
        inst_buying = (
            (df['Close'] > df['Open']) & 
            (body_size > avg_body * 1.5) &
            (df['High'] - df['Close'] < body_size * 0.25) &
            (df['Open'] - df['Low'] < body_size * 0.25)
        )
        
        # Institutional selling: Large bearish candle with small wicks
        inst_selling = (
            (df['Close'] < df['Open']) & 
            (body_size > avg_body * 1.5) &
            (df['High'] - df['Open'] < body_size * 0.25) &
            (df['Close'] - df['Low'] < body_size * 0.25)
        )
        
        # 7. Change of Character (ChoCH)
        # When market structure changes
        choch = ((hh + hl) > 0) & (bearish_structure.shift(1) > bullish_structure.shift(1))
        choch |= ((lh + ll) > 0) & (bullish_structure.shift(1) > bearish_structure.shift(1))
        
        # 8. Breaker Blocks
        # Failed order blocks that become support/resistance
        breaker_bull = bearish_ob.shift(10) & (df['Close'] > df['High'].shift(10))
        breaker_bear = bullish_ob.shift(10) & (df['Close'] < df['Low'].shift(10))
        
        # 9. Mitigation Blocks
        # Areas where previous imbalances are filled
        mitigation_bull = fvg_bullish.shift(20) & (df['Low'] <= df['Low'].shift(-1).shift(20))
        mitigation_bear = fvg_bearish.shift(20) & (df['High'] >= df['High'].shift(-1).shift(20))
        
        # 10. Inducement
        # False breakouts to trap retail traders
        recent_high = df['High'].rolling(20).max()
        recent_low = df['Low'].rolling(20).min()
        
        inducement_high = (df['High'] > recent_high.shift(1)) & (df['Close'] < recent_high)
        inducement_low = (df['Low'] < recent_low.shift(1)) & (df['Close'] > recent_low)
        
        # Compile all SMC features
        smc_features = {
            'smc_structure_bias': structure_bias,
            'smc_bullish_structure': (structure_bias > 0).astype(int),
            'smc_bearish_structure': (structure_bias < 0).astype(int),
            'smc_bullish_ob': bullish_ob.astype(int),
            'smc_bearish_ob': bearish_ob.astype(int),
            'smc_swing_high_liquidity': swing_high_liquidity,
            'smc_swing_low_liquidity': swing_low_liquidity,
            'smc_equal_highs': equal_highs.astype(int),
            'smc_equal_lows': equal_lows.astype(int),
            'smc_fvg_bullish': fvg_bullish.astype(int),
            'smc_fvg_bearish': fvg_bearish.astype(int),
            'smc_fvg_size_bull': fvg_size_bullish,
            'smc_fvg_size_bear': fvg_size_bearish,
            'smc_premium_zone': premium_zone.astype(int),
            'smc_discount_zone': discount_zone.astype(int),
            'smc_equilibrium': equilibrium.astype(int),
            'smc_inst_buying': inst_buying.astype(int),
            'smc_inst_selling': inst_selling.astype(int),
            'smc_choch': choch.astype(int),
            'smc_breaker_bull': breaker_bull.astype(int),
            'smc_breaker_bear': breaker_bear.astype(int),
            'smc_mitigation_bull': mitigation_bull.astype(int),
            'smc_mitigation_bear': mitigation_bear.astype(int),
            'smc_inducement_high': inducement_high.astype(int),
            'smc_inducement_low': inducement_low.astype(int),
            'smc_liquidity_grab': (
                ((df['High'] > swing_high_liquidity.shift(1)) & (df['Close'] < df['Open'])) |
                ((df['Low'] < swing_low_liquidity.shift(1)) & (df['Close'] > df['Open']))
            ).astype(int),
            'smc_market_structure_shift': (abs(structure_bias.diff()) > 5).astype(int)
        }
        
        return smc_features
    
    # Apply enhanced SMC analysis
    enhanced_smc = enhanced_smc_analysis(df)
    for name, values in enhanced_smc.items():
        df[name] = values
    
    # Keep original SMC features for compatibility
    def calculate_smc_features(df):
        # Order blocks (simplified version)
        # Look for strong moves after consolidation
        volatility = df['Close'].rolling(window=10).std()
        low_vol = volatility < volatility.rolling(window=50).mean()
        
        # Fair Value Gap (FVG)
        fvg_up = (df['Low'].shift(-1) > df['High'].shift(1)).astype(int)
        fvg_down = (df['High'].shift(-1) < df['Low'].shift(1)).astype(int)
        
        # Break of Structure (BOS)
        swing_high = df['High'].rolling(window=20).max()
        swing_low = df['Low'].rolling(window=20).min()
        bos_bullish = (df['Close'] > swing_high.shift(1)).astype(int)
        bos_bearish = (df['Close'] < swing_low.shift(1)).astype(int)
        
        # Liquidity zones (simplified)
        liquidity_high = df['High'].rolling(window=50).max()
        liquidity_low = df['Low'].rolling(window=50).min()
        
        return {
            'smc_low_volatility': low_vol.astype(int),
            'smc_fvg_up': fvg_up,
            'smc_fvg_down': fvg_down,
            'smc_bos_bullish': bos_bullish,
            'smc_bos_bearish': bos_bearish,
            'smc_liquidity_distance_high': liquidity_high - df['Close'],
            'smc_liquidity_distance_low': df['Close'] - liquidity_low
        }
    
    smc_features = calculate_smc_features(df)
    for name, values in smc_features.items():
        df[name] = values
    
    # ==================== ORDER FLOW ANALYSIS (Ultimate) ====================
    def enhanced_order_flow_analysis(df):
        """Advanced Order Flow metrics using price action and volume"""
        
        # Delta calculation (approximated from price and volume)
        price_move = df['Close'] - df['Open']
        typical_range = (df['High'] - df['Low']).rolling(20).mean()
        
        # Volume Delta approximation
        # Positive price move = more buying, negative = more selling
        delta_ratio = price_move / typical_range.replace(0, np.nan)
        delta_ratio = delta_ratio.fillna(0).clip(-1, 1)
        
        # Split volume into buy/sell
        buy_volume = df['Volume'] * (0.5 + 0.5 * delta_ratio)
        sell_volume = df['Volume'] * (0.5 - 0.5 * delta_ratio)
        delta = buy_volume - sell_volume
        
        # Cumulative Delta
        cumulative_delta = delta.cumsum()
        
        # Delta Divergence
        price_change = df['Close'].pct_change(20)
        delta_change = cumulative_delta.pct_change(20)
        delta_divergence = np.sign(price_change) != np.sign(delta_change)
        
        # Absorption Detection
        # Large volume with small price move = absorption
        volume_zscore = (df['Volume'] - df['Volume'].rolling(50).mean()) / df['Volume'].rolling(50).std()
        price_move_pct = abs(df['Close'].pct_change())
        absorption = (volume_zscore > 2) & (price_move_pct < price_move_pct.rolling(50).mean())
        
        # Exhaustion Detection
        # Large move with decreasing volume = exhaustion
        volume_ma_ratio = df['Volume'] / df['Volume'].rolling(20).mean()
        price_momentum = df['Close'].pct_change(5).abs()
        exhaustion = (price_momentum > price_momentum.rolling(50).quantile(0.8)) & (volume_ma_ratio < 0.7)
        
        # Initiative vs Responsive Activity
        # Initiative: Breaking out of value area with volume
        value_high = df['High'].rolling(20).quantile(0.7)
        value_low = df['Low'].rolling(20).quantile(0.3)
        
        initiative_buying = (df['Close'] > value_high) & (volume_zscore > 1)
        initiative_selling = (df['Close'] < value_low) & (volume_zscore > 1)
        responsive_activity = (df['Close'] <= value_high) & (df['Close'] >= value_low) & (volume_zscore > 0.5)
        
        # POC (Point of Control) Migration
        poc = df.groupby(pd.cut(df['Close'], bins=50))['Volume'].sum().idxmax()
        poc_price = poc.mid if hasattr(poc, 'mid') else df['Close'].median()
        poc_migration = df['Close'] - poc_price
        
        # Store all features
        order_flow_features = {
            'of_delta': delta,
            'of_cumulative_delta': cumulative_delta,
            'of_delta_divergence': delta_divergence.astype(int),
            'of_absorption': absorption.astype(int),
            'of_exhaustion': exhaustion.astype(int),
            'of_initiative_buying': initiative_buying.astype(int),
            'of_initiative_selling': initiative_selling.astype(int),
            'of_responsive': responsive_activity.astype(int),
            'of_buy_volume': buy_volume,
            'of_sell_volume': sell_volume,
            'of_volume_delta_ratio': (buy_volume / sell_volume.replace(0, np.nan)).fillna(1),
            'of_poc_migration': poc_migration,
            'of_delta_acceleration': delta.diff(),
            'of_volume_imbalance': abs(buy_volume - sell_volume) / df['Volume']
        }
        
        return order_flow_features
    
    order_flow_enhanced = enhanced_order_flow_analysis(df)
    for name, values in order_flow_enhanced.items():
        df[name] = values
    
    # Original Order Flow Imbalance (keep for compatibility)
    def calculate_order_flow_imbalance(df):
        # Simulate order flow using price and volume
        price_change = df['Close'].pct_change()
        
        # Aggressive buying/selling detection
        aggressive_buying = (price_change > 0) & (df['Volume'] > df['Volume'].rolling(20).mean())
        aggressive_selling = (price_change < 0) & (df['Volume'] > df['Volume'].rolling(20).mean())
        
        # Order flow imbalance
        buy_pressure = np.where(aggressive_buying, df['Volume'] * abs(price_change), 0)
        sell_pressure = np.where(aggressive_selling, df['Volume'] * abs(price_change), 0)
        
        imbalance = pd.Series(buy_pressure).rolling(10).sum() - pd.Series(sell_pressure).rolling(10).sum()
        
        return imbalance, buy_pressure, sell_pressure
    
    imbalance, buy_pressure, sell_pressure = calculate_order_flow_imbalance(df)
    df['order_flow_imbalance'] = imbalance
    df['buy_pressure'] = buy_pressure
    df['sell_pressure'] = sell_pressure
    # Convert to pandas Series for safe division
    sell_pressure_sum = pd.Series(sell_pressure).rolling(20).sum()
    sell_pressure_sum = sell_pressure_sum.replace(0, np.nan)
    df['order_flow_ratio'] = pd.Series(buy_pressure).rolling(20).sum() / sell_pressure_sum
    
    # ==================== STATISTICAL FEATURES ====================
    
    # 1. Z-Score and Normal Distribution
    def calculate_bell_curve_features(df, period=50):
        # Calculate z-scores for price position
        price_mean = df['Close'].rolling(window=period, min_periods=1).mean()
        price_std = df['Close'].rolling(window=period, min_periods=1).std()
        
        z_score = (df['Close'] - price_mean) / price_std.replace(0, np.nan)
        
        # Probability within normal distribution
        if scipy_available:
            probability = stats.norm.cdf(z_score)
        else:
            # Approximate normal CDF without scipy
            # Using approximation: Φ(x) ≈ 0.5 * (1 + tanh(0.797884560803 * x))
            probability = 0.5 * (1 + np.tanh(0.797884560803 * z_score))
        
        return z_score.fillna(0), probability
    
    for period in [20, 50, 100]:
        z_score, norm_probability = calculate_bell_curve_features(df, period)
        df[f'price_z_score_{period}'] = z_score
        df[f'price_norm_probability_{period}'] = norm_probability
        df[f'price_extreme_{period}'] = (abs(z_score) > 2).astype(int)
    
    # 2. Percentile Rank
    for period in [20, 50, 100]:
        df[f'price_percentile_{period}'] = df['Close'].rolling(period).apply(
            lambda x: stats.percentileofscore(x, x.iloc[-1]) / 100 if len(x) == period else 0.5
        )
    
    # 3. Linear Regression Features
    def calculate_linear_regression_features(prices, period):
        slopes = []
        intercepts = []
        r_squared = []
        std_errors = []
        
        for i in range(period, len(prices)):
            y = prices.iloc[i-period:i].values
            x = np.arange(period)
            
            # Linear regression
            slope, intercept = np.polyfit(x, y, 1)
            y_pred = slope * x + intercept
            
            # R-squared
            ss_res = np.sum((y - y_pred) ** 2)
            ss_tot = np.sum((y - np.mean(y)) ** 2)
            r2 = 1 - (ss_res / ss_tot) if ss_tot != 0 else 0
            
            # Standard error
            std_err = np.sqrt(ss_res / (period - 2)) if period > 2 else 0
            
            slopes.append(slope)
            intercepts.append(intercept)
            r_squared.append(r2)
            std_errors.append(std_err)
        
        # Pad the beginning
        slopes = [0] * period + slopes
        intercepts = [prices.iloc[0]] * period + intercepts
        r_squared = [0] * period + r_squared
        std_errors = [0] * period + std_errors
        
        return slopes, intercepts, r_squared, std_errors
    
    for period in [10, 20, 50]:
        slopes, intercepts, r2, std_err = calculate_linear_regression_features(df['Close'], period)
        df[f'linreg_slope_{period}'] = slopes
        df[f'linreg_intercept_{period}'] = intercepts
        df[f'linreg_r2_{period}'] = r2
        df[f'linreg_stderr_{period}'] = std_err
        df[f'linreg_angle_{period}'] = np.arctan(slopes) * 180 / np.pi
    
    # 4. Polynomial Regression Features
    def calculate_poly_regression(prices, degree=2, period=100):
        results = np.zeros(len(prices))
        upper = np.zeros(len(prices))
        lower = np.zeros(len(prices))
        
        for i in range(period, len(prices)):
            y = prices.iloc[i-period:i].values
            x = np.arange(period)
            
            # Fit polynomial
            coeffs = np.polyfit(x, y, degree)
            poly = np.poly1d(coeffs)
            
            # Current value
            results[i] = poly(period-1)
            
            # Calculate channel based on residuals
            fitted = poly(x)
            residuals = y - fitted
            std_dev = np.std(residuals)
            
            upper[i] = results[i] + 2 * std_dev
            lower[i] = results[i] - 2 * std_dev
        
        return results, upper, lower
    
    for degree in [2, 3]:
        for period in [50, 100]:
            poly_reg, poly_upper, poly_lower = calculate_poly_regression(df['Close'], degree, period)
            df[f'poly{degree}_regression_{period}'] = poly_reg
            df[f'poly{degree}_upper_{period}'] = poly_upper
            df[f'poly{degree}_lower_{period}'] = poly_lower
            # Convert to pandas Series for safe division
            poly_diff = pd.Series(poly_upper - poly_lower)
            poly_diff = poly_diff.replace(0, np.nan)
            df[f'poly{degree}_position_{period}'] = (df['Close'] - poly_lower) / poly_diff
    
    # ==================== TRAILING STOPS ====================
    
    # 1. ATR Trailing Stop (Multiple Settings)
    def calculate_trailing_stop_atr(df, period=14, multiplier=3):
        atr = calculate_atr(df, period)
        
        trailing_stop_long = np.zeros(len(df))
        trailing_stop_short = np.zeros(len(df))
        
        trailing_stop_long[0] = df['Close'].iloc[0] - multiplier * atr.iloc[0]
        trailing_stop_short[0] = df['Close'].iloc[0] + multiplier * atr.iloc[0]
        
        for i in range(1, len(df)):
            # Long trailing stop
            trailing_stop_long[i] = max(
                df['Close'].iloc[i] - multiplier * atr.iloc[i],
                trailing_stop_long[i-1]
            )
            
            # Short trailing stop
            trailing_stop_short[i] = min(
                df['Close'].iloc[i] + multiplier * atr.iloc[i],
                trailing_stop_short[i-1]
            )
        
        return trailing_stop_long, trailing_stop_short
    
    for period in [10, 14, 20]:
        for mult in [1.5, 2, 2.5, 3, 4]:
            tsl_long, tsl_short = calculate_trailing_stop_atr(df, period, mult)
            df[f'trailing_stop_long_{period}_{mult}'] = tsl_long
            df[f'trailing_stop_short_{period}_{mult}'] = tsl_short
            df[f'trailing_stop_distance_{period}_{mult}'] = np.minimum(
                df['Close'] - tsl_long,
                tsl_short - df['Close']
            )
    
    # 2. Percentage Trailing Stop
    def calculate_trailing_stop_pct(df, pct=0.05):
        trailing_stop_long = np.zeros(len(df))
        trailing_stop_short = np.zeros(len(df))
        
        trailing_stop_long[0] = df['Close'].iloc[0] * (1 - pct)
        trailing_stop_short[0] = df['Close'].iloc[0] * (1 + pct)
        
        for i in range(1, len(df)):
            trailing_stop_long[i] = max(
                df['Close'].iloc[i] * (1 - pct),
                trailing_stop_long[i-1]
            )
            
            trailing_stop_short[i] = min(
                df['Close'].iloc[i] * (1 + pct),
                trailing_stop_short[i-1]
            )
        
        return trailing_stop_long, trailing_stop_short
    
    for pct in [0.02, 0.03, 0.05, 0.07, 0.10]:
        tsl_long, tsl_short = calculate_trailing_stop_pct(df, pct)
        df[f'trailing_stop_pct_long_{int(pct*100)}'] = tsl_long
        df[f'trailing_stop_pct_short_{int(pct*100)}'] = tsl_short
    
    # ==================== MISCELLANEOUS INDICATORS ====================
    
    # 1. Balance of Power
    def calculate_balance_of_power(df):
        bop = (df['Close'] - df['Open']) / (df['High'] - df['Low']).replace(0, np.nan)
        return bop.fillna(0)
    
    df['balance_of_power'] = calculate_balance_of_power(df)
    df['bop_sma'] = df['balance_of_power'].rolling(14).mean()
    
    # 2. Elder's Force Index (Extended)
    def calculate_force_index(df, period=13):
        fi = df['Close'].diff() * df['Volume']
        return fi.ewm(span=period, adjust=False).mean()
    
    for period in [2, 13, 20, 50]:
        df[f'force_index_{period}'] = calculate_force_index(df, period)
    
    # 3. Accumulation Swing Index
    def calculate_asi(df):
        # Simplified ASI calculation
        limit_move = 0.5  # Simplified
        
        r = np.zeros(len(df))
        for i in range(1, len(df)):
            high_change = abs(df['High'].iloc[i] - df['Close'].iloc[i-1])
            low_change = abs(df['Low'].iloc[i] - df['Close'].iloc[i-1])
            high_low = abs(df['High'].iloc[i] - df['Low'].iloc[i])
            prev_close_open = abs(df['Close'].iloc[i-1] - df['Open'].iloc[i-1])
            
            r[i] = max(high_change, low_change, high_low)
        
        k = np.maximum(abs(df['High'] - df['Close'].shift()), abs(df['Low'] - df['Close'].shift()))
        
        swing_index = 50 * ((df['Close'] - df['Close'].shift() + 0.5 * (df['Close'] - df['Open']) + 
                            0.25 * (df['Close'].shift() - df['Open'].shift())) / r) * (k / limit_move)
        
        asi = swing_index.cumsum()
        return asi
    
    df['asi'] = calculate_asi(df)
    
    # 4. Relative Vigor Index
    def calculate_rvi(df, period=10):
        co = df['Close'] - df['Open']
        hl = df['High'] - df['Low']
        
        # Smooth with SMA
        numerator = co.rolling(4).mean()
        denominator = hl.rolling(4).mean()
        
        rvi = numerator.rolling(period).sum() / denominator.rolling(period).sum().replace(0, np.nan)
        signal = rvi.rolling(4).mean()
        
        return rvi.fillna(0), signal.fillna(0)
    
    for period in [10, 14]:
        rvi, rvi_signal = calculate_rvi(df, period)
        df[f'rvi_{period}'] = rvi
        df[f'rvi_signal_{period}'] = rvi_signal
    
    # 5. Psychological Line
    def calculate_psychological_line(df, period=12):
        up_days = (df['Close'] > df['Close'].shift()).astype(int)
        psy_line = up_days.rolling(period).sum() / period * 100
        return psy_line
    
    df['psychological_line'] = calculate_psychological_line(df)
    df['psychological_line_20'] = calculate_psychological_line(df, 20)
    
    # 6. Hurst Exponent
    def calculate_hurst_exponent(prices, period=100):
        hurst_values = []
        
        for i in range(period, len(prices)):
            ts = prices.iloc[i-period:i].values
            
            # Calculate R/S
            lags = range(2, min(period//2, 20))
            tau = []
            
            for lag in lags:
                # Divide into chunks
                chunks = [ts[j:j+lag] for j in range(0, len(ts), lag)]
                chunks = [chunk for chunk in chunks if len(chunk) == lag]
                
                if not chunks:
                    continue
                
                # Calculate R/S for each chunk
                rs_values = []
                for chunk in chunks:
                    mean_chunk = np.mean(chunk)
                    deviations = chunk - mean_chunk
                    Z = np.cumsum(deviations)
                    R = np.max(Z) - np.min(Z)
                    S = np.std(chunk, ddof=1)
                    
                    if S != 0:
                        rs_values.append(R / S)
                
                if rs_values:
                    tau.append(np.mean(rs_values))
            
            # Calculate Hurst exponent
            if len(tau) > 1:
                log_lags = np.log(list(lags)[:len(tau)])
                log_tau = np.log(tau)
                hurst = np.polyfit(log_lags, log_tau, 1)[0]
                hurst_values.append(hurst)
            else:
                hurst_values.append(0.5)
        
        # Pad the beginning
        hurst_values = [0.5] * period + hurst_values
        
        return hurst_values
    
    df['hurst_exponent'] = calculate_hurst_exponent(df['Close'])
    
    # 7. Fractal Dimension
    def calculate_fractal_dimension(prices, period=30):
        fd_values = []
        
        for i in range(period, len(prices)):
            ts = prices.iloc[i-period:i].values
            
            # Higuchi method
            k_max = 5
            L = []
            
            for k in range(1, k_max + 1):
                Lk = []
                
                for m in range(k):
                    Lm = 0
                    for j in range(1, int((period - m) / k)):
                        Lm += abs(ts[m + j * k] - ts[m + (j - 1) * k])
                    
                    Lm = Lm * (period - 1) / (k * int((period - m) / k))
                    Lk.append(Lm)
                
                L.append(np.mean(Lk))
            
            # Calculate fractal dimension
            if len(L) > 1:
                log_k = np.log(range(1, len(L) + 1))
                log_L = np.log(L)
                fd = -np.polyfit(log_k, log_L, 1)[0]
                fd_values.append(fd)
            else:
                fd_values.append(1.5)
        
        # Pad the beginning
        fd_values = [1.5] * period + fd_values
        
        return fd_values
    
    df['fractal_dimension'] = calculate_fractal_dimension(df['Close'])
    
    # ==================== VOLATILITY CONES ====================
    
    def calculate_volatility_cones(df):
        periods = [5, 10, 20, 30, 50, 100]
        percentiles = [10, 25, 50, 75, 90]
        
        for period in periods:
            # Calculate rolling volatility
            returns = df['Close'].pct_change()
            rolling_vol = returns.rolling(period).std() * np.sqrt(252)
            
            # Calculate percentiles
            for pct in percentiles:
                df[f'vol_cone_{period}_{pct}'] = rolling_vol.rolling(252).quantile(pct/100)
            
            # Current volatility position
            df[f'vol_cone_{period}_position'] = rolling_vol
    
    calculate_volatility_cones(df)
    
    # ==================== MICROSTRUCTURE FEATURES ====================
    
    # 1. Tick Rule
    df['tick_rule'] = np.sign(df['Close'].diff())
    df['tick_rule'] = df['tick_rule'].fillna(0)
    
    # 2. Roll's Implicit Spread Estimator
    def calculate_roll_spread(prices, period=20):
        price_changes = prices.diff()
        
        spread_estimates = []
        for i in range(period, len(prices)):
            changes = price_changes.iloc[i-period:i]
            cov = changes.cov(changes.shift(1))
            
            if cov < 0:
                spread = 2 * np.sqrt(-cov)
            else:
                spread = 0
            
            spread_estimates.append(spread)
        
        # Pad the beginning
        spread_estimates = [0] * period + spread_estimates
        
        return spread_estimates
    
    df['roll_spread'] = calculate_roll_spread(df['Close'])
    
    # 3. Kyle's Lambda (simplified)
    def calculate_kyle_lambda(df, period=20):
        # Simplified: price impact of volume
        price_changes = df['Close'].pct_change()
        
        lambdas = []
        for i in range(period, len(df)):
            price_change = price_changes.iloc[i-period:i]
            volume = df['Volume'].iloc[i-period:i]
            
            # Simple regression of |price change| on volume
            if volume.std() > 0:
                coef = np.cov(abs(price_change), volume)[0, 1] / volume.var()
                lambdas.append(coef)
            else:
                lambdas.append(0)
        
        # Pad the beginning
        lambdas = [0] * period + lambdas
        
        return lambdas
    
    df['kyle_lambda'] = calculate_kyle_lambda(df)
    
    # 4. Amihud Illiquidity
    def calculate_amihud_illiquidity(df, period=20):
        illiquidity = abs(df['Close'].pct_change()) / (df['Volume'] * df['Close'])
        illiquidity = illiquidity.replace([np.inf, -np.inf], np.nan)
        
        rolling_illiquidity = illiquidity.rolling(period).mean()
        return rolling_illiquidity.fillna(0)
    
    df['amihud_illiquidity'] = calculate_amihud_illiquidity(df)
    
    # ==================== SEASONALITY FEATURES ====================
    
    # Time-based features
    if 'datetime' in df.columns:
        dt = pd.to_datetime(df['datetime'])
        df['hour'] = dt.dt.hour
        df['day_of_week'] = dt.dt.dayofweek
        df['day_of_month'] = dt.dt.day
        df['week_of_year'] = dt.dt.isocalendar().week
        df['month'] = dt.dt.month
        df['quarter'] = dt.dt.quarter
        
        # Cyclical encoding
        df['hour_sin'] = np.sin(2 * np.pi * df['hour'] / 24)
        df['hour_cos'] = np.cos(2 * np.pi * df['hour'] / 24)
        df['day_sin'] = np.sin(2 * np.pi * df['day_of_week'] / 7)
        df['day_cos'] = np.cos(2 * np.pi * df['day_of_week'] / 7)
        df['month_sin'] = np.sin(2 * np.pi * df['month'] / 12)
        df['month_cos'] = np.cos(2 * np.pi * df['month'] / 12)
    
    # ==================== LAG FEATURES (Extended) ====================
    
    # Price and return lags
    for lag in [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 12, 24, 48, 72, 96, 120, 144, 168]:
        df[f'close_lag_{lag}'] = df['Close'].shift(lag)
        df[f'return_lag_{lag}'] = df['price_change'].shift(lag)
        df[f'volume_lag_{lag}'] = df['Volume'].shift(lag)
        df[f'high_lag_{lag}'] = df['High'].shift(lag)
        df[f'low_lag_{lag}'] = df['Low'].shift(lag)
    
    # Moving average lags
    for ma_period in [5, 10, 20, 50]:
        for lag in [1, 2, 3, 5, 10]:
            df[f'sma_{ma_period}_lag_{lag}'] = df[f'SMA_{ma_period}'].shift(lag)
    
    # ==================== INTERACTION FEATURES ====================
    
    # Price-Volume interactions
    df['price_volume_interaction'] = df['price_change'] * (df['Volume'] / df['Volume'].rolling(20).mean().replace(0, np.nan))
    df['price_volume_correlation'] = df['Close'].rolling(20).corr(df['Volume'])
    
    # Trend-Momentum interactions
    df['trend_strength'] = df['ADX_14'] * np.sign(df['DI_diff_14'])
    df['trend_momentum'] = df['SMA_50_slope'] * df['RSI_14']
    
    # Volatility-Volume interactions
    df['volatility_volume'] = df['ATR_14'] *  (df['Volume'] / df['Volume'].rolling(20).mean().replace(0, np.nan))
    df['volatility_surge'] = df['ATR_14'] / df['ATR_14'].rolling(50).mean()
    
    # Multi-indicator confirmations
    df['momentum_confirmation'] = (
        (df['RSI_14'] > 50).astype(int) + 
        (df['MACD_12_26_9_histogram'] > 0).astype(int) + 
        (df['stoch_k_14_3'] > 50).astype(int) +
        (df['MFI_14'] > 50).astype(int) +
        (df['CMO_14'] > 0).astype(int)
    ) / 5
    
    df['trend_confirmation'] = (
        (df['Close'] > df['SMA_50']).astype(int) +
        (df['SMA_20'] > df['SMA_50']).astype(int) +
        (df['ADX_14'] > 25).astype(int) +
        (df['aroon_up_25'] > df['aroon_down_25']).astype(int) +
        (df['supertrend_dir_10_3'] > 0).astype(int)
    ) / 5
    
    df['reversal_signals'] = (
        (df['RSI_14'] < 30).astype(int) + (df['RSI_14'] > 70).astype(int) +
        (df['williams_r_14'] < -80).astype(int) + (df['williams_r_14'] > -20).astype(int) +
        (df['CCI_20'] < -100).astype(int) + (df['CCI_20'] > 100).astype(int) +
        (df['stoch_k_14_3'] < 20).astype(int) + (df['stoch_k_14_3'] > 80).astype(int)
    )
    
    # ==================== RATIO FEATURES ====================
    
    # Price ratios
    df['high_low_spread'] = (df['High'] - df['Low']) / df['Close']
    df['close_to_high'] = (df['High'] - df['Close']) / (df['High'] - df['Low']).replace(0, np.nan)
    df['close_to_low'] = (df['Close'] - df['Low']) / (df['High'] - df['Low']).replace(0, np.nan)
    
    # Volume ratios
    df['volume_to_market_cap'] = df['Volume'] / df['Volume'].rolling(252).mean()
    df['relative_volume'] = df['Volume'] / df['Volume'].rolling(20).mean()
    
    # Volatility ratios
    df['atr_to_close'] = df['ATR_14'] / df['Close']
    df['volatility_ratio'] = df['hist_volatility_20'] / df['hist_volatility_100']
    
    # ==================== MARKET SENTIMENT FEATURES ====================
    
    df['volume_surge'] = df['Volume'] / df['Volume'].rolling(window=20, min_periods=1).mean().replace(0, np.nan)
    df['large_volume_flag'] = (df['volume_surge'] > 2).astype(int)
    df['extreme_volume_flag'] = (df['volume_surge'] > 3).astype(int)
    df['price_acceleration'] = df['price_change'].diff()
    df['price_jerk'] = df['price_acceleration'].diff()  # Third derivative
    df['momentum_strength'] = df['price_change'].rolling(window=10, min_periods=1).mean()
    df['momentum_acceleration'] = df['momentum_strength'].diff()
    
    # Volatility features
    for window in [5, 10, 20, 30, 50, 100]:
        df[f'volatility_{window}'] = df['price_change'].rolling(window=window, min_periods=1).std()
        df[f'volatility_ratio_{window}'] = df[f'volatility_{window}'] / df[f'volatility_{window}'].rolling(window=50, min_periods=1).mean().replace(0, np.nan)
    
    # ==================== TARGET ====================
    df['target'] = df['Close'].shift(-1)
    
    # ==================== FINAL PROCESSING ====================
    
    print(f"\nCreated {len(df.columns)} features!")
    print("\nFeature categories created:")
    print("- Basic Features")
    print("- Price Statistics (multiple windows)")
    print("- Moving Averages (15 types, multiple periods each)")
    print("- Momentum Indicators (RSI, MACD, Stochastic, Ultimate, Williams, CCI, MFI, ROC, etc.)")
    print("- Volatility Indicators (ATR, Bollinger Bands, Keltner, Donchian, Historical Vol, etc.)")
    print("- Volume Indicators (OBV, VWAP, A/D, CMF, PVT, Force Index, etc.)")
    print("- Trend Indicators (ADX, Aroon, Vortex, Mass Index, PSAR, Supertrend, etc.)")
    print("- Oscillators (AO, CMO, DPO, PPO, etc.)")
    print("- Advanced Wave Analysis (Elliott, Sine, Cyber Cycle, MESA, etc.)")
    print("- Pattern Recognition (Candlesticks, Harmonics, etc.)")
    print("- Fibonacci Features (Retracements, Extensions)")
    print("- Pivot Points (5 types)")
    print("- Market Profile & Volume Profile")
    print("- Smart Money Concepts (Enhanced)")
    print("- Order Flow Analysis (Enhanced)")
    print("- Statistical Features")
    print("- Chart Types (Heikin Ashi, Renko, Kagi, P&F, Line Break)")
    print("- Market Microstructure")
    print("- Seasonality Features")
    print("- Interaction Features")
    print("- And many more...")
    
    # Fill NaN values
    print("\nCleaning data...")
    # First, forward fill
    df = df.ffill()
    # Then, backward fill for any remaining NaN at the beginning
    df = df.bfill()
    # Finally, fill any remaining NaN with 0 (shouldn't be many)
    df = df.fillna(0)
    
    # Replace infinite values
    df = df.replace([np.inf, -np.inf], 0)
    
    # Drop rows where target is NaN (last row)
    df = df[df['target'].notna()]
    
    print(f"Data after cleaning: {len(df)} rows (removed {original_len - len(df)} rows)")
    print(f"Total features created: {len(df.columns) - 7}")  # Subtract original columns
    
    return df