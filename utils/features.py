import numpy as np
import pandas as pd


def create_robust_features(data):
        """Create ALL technical features with improved NaN handling."""
        df = data.copy()
        
        print("Creating comprehensive technical features...")
        
        # Import scipy for advanced analysis
        try:
            from scipy.signal import argrelextrema
            from scipy import stats
            scipy_available = True
        except ImportError:
            print("Warning: scipy not available. Some advanced features will be simplified.")
            scipy_available = False
        
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
        
        # Safe price position calculation
        hl_range = df['High'] - df['Low']
        df['price_position'] = np.where(hl_range > 0, 
                                    (df['Close'] - df['Low']) / hl_range, 
                                    0.5)
        
        # ==================== PRICE STATISTICS ====================
        for window in [3, 5, 7, 10, 14, 20, 30, 50]:
            df[f'price_mean_{window}'] = df['Close'].rolling(window=window, min_periods=1).mean()
            df[f'price_std_{window}'] = df['Close'].rolling(window=window, min_periods=1).std()
            df[f'price_min_{window}'] = df['Close'].rolling(window=window, min_periods=1).min()
            df[f'price_max_{window}'] = df['Close'].rolling(window=window, min_periods=1).max()
        
        # ==================== MOVING AVERAGES ====================
        for window in [3, 5, 10, 20, 50, 100, 200]:
            df[f'SMA_{window}'] = df['Close'].rolling(window=window, min_periods=1).mean()
            df[f'SMA_{window}_ratio'] = df['Close'] / df[f'SMA_{window}'].replace(0, np.nan)
        
        for span in [3, 5, 8, 12, 20, 26, 50]:
            df[f'EMA_{span}'] = df['Close'].ewm(span=span, adjust=False).mean()
            df[f'EMA_{span}_ratio'] = df['Close'] / df[f'EMA_{span}'].replace(0, np.nan)
        
        # ==================== NEW: WEIGHTED MOVING AVERAGE (WMA) ====================
        def calculate_wma(prices, window):
            weights = np.arange(1, window + 1)
            wma = prices.rolling(window=window, min_periods=1).apply(
                lambda x: np.dot(x, weights[-len(x):]) / weights[-len(x):].sum() if len(x) > 0 else x.mean()
            )
            return wma
        
        for window in [5, 10, 20, 50]:
            df[f'WMA_{window}'] = calculate_wma(df['Close'], window)
            df[f'WMA_{window}_ratio'] = df['Close'] / df[f'WMA_{window}'].replace(0, np.nan)
        
        # ==================== MOMENTUM INDICATORS ====================
        
        # 1. RSI
        def calculate_rsi(prices, window=14):
            delta = prices.diff()
            gain = delta.where(delta > 0, 0)
            loss = -delta.where(delta < 0, 0)
            avg_gain = gain.rolling(window=window, min_periods=1).mean()
            avg_loss = loss.rolling(window=window, min_periods=1).mean()
            rs = avg_gain / avg_loss.replace(0, np.nan)
            rsi = 100 - (100 / (1 + rs))
            return rsi.fillna(50)  # Neutral RSI when undefined
        
        for period in [7, 14, 21, 28]:
            df[f'RSI_{period}'] = calculate_rsi(df['Close'], window=period)
        
        # 2. MACD
        df['MACD_12_26'] = df['EMA_12'] - df['EMA_26']
        df['MACD_signal'] = df['MACD_12_26'].ewm(span=9, adjust=False).mean()
        df['MACD_histogram'] = df['MACD_12_26'] - df['MACD_signal']
        df['MACD_cross'] = np.where(df['MACD_12_26'] > df['MACD_signal'], 1, -1)
        
        # 3. Stochastic Oscillator
        for period in [5, 14, 21]:
            low_min = df['Low'].rolling(window=period, min_periods=1).min()
            high_max = df['High'].rolling(window=period, min_periods=1).max()
            hl_range = high_max - low_min
            df[f'stoch_k_{period}'] = np.where(hl_range > 0,
                                            100 * ((df['Close'] - low_min) / hl_range),
                                            50)
            df[f'stoch_d_{period}'] = pd.Series(df[f'stoch_k_{period}']).rolling(window=3, min_periods=1).mean()
            df[f'stoch_cross_{period}'] = np.where(df[f'stoch_k_{period}'] > df[f'stoch_d_{period}'], 1, -1)
        
        # 4. Stochastic RSI
        for period in [14, 21]:
            rsi = df[f'RSI_{period}']
            rsi_min = rsi.rolling(window=period, min_periods=1).min()
            rsi_max = rsi.rolling(window=period, min_periods=1).max()
            rsi_range = rsi_max - rsi_min
            df[f'stoch_rsi_{period}'] = np.where(rsi_range > 0,
                                                (rsi - rsi_min) / rsi_range,
                                                0.5)
        
        # 5. ADX
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
        
        for period in [14, 21]:
            adx, plus_di, minus_di = calculate_adx(df, period)
            df[f'ADX_{period}'] = adx
            df[f'plus_DI_{period}'] = plus_di
            df[f'minus_DI_{period}'] = minus_di
            df[f'DI_diff_{period}'] = plus_di - minus_di
        
        # 6. Aroon
        def calculate_aroon(df, period=25):
            high = df['High']
            low = df['Low']
            
            # Using a more efficient approach
            aroon_up = high.rolling(window=period + 1, min_periods=1).apply(
                lambda x: (period - (period - x.argmax())) / period * 100 if len(x) > 0 else 50
            )
            aroon_down = low.rolling(window=period + 1, min_periods=1).apply(
                lambda x: (period - (period - x.argmin())) / period * 100 if len(x) > 0 else 50
            )
            
            return aroon_up, aroon_down
        
        for period in [14, 25]:
            aroon_up, aroon_down = calculate_aroon(df, period)
            df[f'aroon_up_{period}'] = aroon_up
            df[f'aroon_down_{period}'] = aroon_down
            df[f'aroon_oscillator_{period}'] = aroon_up - aroon_down
        
        # 7. CCI
        def calculate_cci(df, period=20):
            typical_price = (df['High'] + df['Low'] + df['Close']) / 3
            sma_tp = typical_price.rolling(window=period, min_periods=1).mean()
            mad = typical_price.rolling(window=period, min_periods=1).apply(
                lambda x: np.mean(np.abs(x - x.mean())) if len(x) > 0 else 1
            )
            cci = (typical_price - sma_tp) / (0.015 * mad.replace(0, np.nan))
            return cci.fillna(0)
        
        for period in [14, 20, 28]:
            df[f'CCI_{period}'] = calculate_cci(df, period)
        
        # 8. ATR
        def calculate_atr(df, period):
            high_low = df['High'] - df['Low']
            high_close = np.abs(df['High'] - df['Close'].shift())
            low_close = np.abs(df['Low'] - df['Close'].shift())
            ranges = pd.concat([high_low, high_close, low_close], axis=1)
            true_range = ranges.max(axis=1)
            return true_range.rolling(window=period, min_periods=1).mean()
        
        for period in [7, 14, 21]:
            df[f'ATR_{period}'] = calculate_atr(df, period)
            df[f'ATR_{period}_ratio'] = df[f'ATR_{period}'] / df['Close'].replace(0, np.nan)
        
        # 9. Bollinger Bands
        for window in [20, 30]:
            bb_mean = df['Close'].rolling(window=window, min_periods=1).mean()
            bb_std = df['Close'].rolling(window=window, min_periods=1).std()
            df[f'BB_upper_{window}'] = bb_mean + (bb_std * 2)
            df[f'BB_lower_{window}'] = bb_mean - (bb_std * 2)
            df[f'BB_width_{window}'] = df[f'BB_upper_{window}'] - df[f'BB_lower_{window}']
            bb_width = df[f'BB_width_{window}'].replace(0, np.nan)
            df[f'BB_position_{window}'] = (df['Close'] - df[f'BB_lower_{window}']) / bb_width
            df[f'BB_bandwidth_{window}'] = bb_width / bb_mean.replace(0, np.nan)
        
        # 10. VWAP
        cumulative_pv = (df['Volume'] * df['typical_price']).cumsum()
        cumulative_volume = df['Volume'].cumsum()
        df['VWAP'] = cumulative_pv / cumulative_volume.replace(0, np.nan)
        df['VWAP_ratio'] = df['Close'] / df['VWAP'].replace(0, np.nan)
        
        # Rolling VWAP
        for period in [20, 50]:
            rolling_pv = (df['Volume'] * df['typical_price']).rolling(window=period, min_periods=1).sum()
            rolling_volume = df['Volume'].rolling(window=period, min_periods=1).sum()
            df[f'VWAP_{period}'] = rolling_pv / rolling_volume.replace(0, np.nan)
            df[f'VWAP_{period}_ratio'] = df['Close'] / df[f'VWAP_{period}'].replace(0, np.nan)
        
        # 11. Parabolic SAR
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
        
        df['PSAR'] = calculate_psar(df)
        df['PSAR_distance'] = df['Close'] - df['PSAR']
        df['PSAR_signal'] = np.where(df['Close'] > df['PSAR'], 1, -1)
        
        # 12. Ichimoku Cloud
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
        
        # 13. Fibonacci Retracement
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
                '100': low
            }
            
            return fib_levels
        
        for period in [50, 100]:
            fib_levels = calculate_fib_levels(df, period)
            for level, values in fib_levels.items():
                df[f'fib_{level}_{period}'] = values
                df[f'fib_{level}_{period}_distance'] = df['Close'] - values
        
        # 14. Donchian Channels
        for period in [20, 50]:
            df[f'donchian_high_{period}'] = df['High'].rolling(window=period, min_periods=1).max()
            df[f'donchian_low_{period}'] = df['Low'].rolling(window=period, min_periods=1).min()
            df[f'donchian_mid_{period}'] = (df[f'donchian_high_{period}'] + df[f'donchian_low_{period}']) / 2
            df[f'donchian_width_{period}'] = df[f'donchian_high_{period}'] - df[f'donchian_low_{period}']
            width = df[f'donchian_width_{period}'].replace(0, np.nan)
            df[f'donchian_position_{period}'] = (df['Close'] - df[f'donchian_low_{period}']) / width
        
        # ==================== NEW: PIVOT POINTS ====================
        def calculate_pivot_points(df):
            # Standard Pivot Points
            pivot = (df['High'].shift(1) + df['Low'].shift(1) + df['Close'].shift(1)) / 3
            
            # Support and Resistance levels
            r1 = 2 * pivot - df['Low'].shift(1)
            s1 = 2 * pivot - df['High'].shift(1)
            r2 = pivot + (df['High'].shift(1) - df['Low'].shift(1))
            s2 = pivot - (df['High'].shift(1) - df['Low'].shift(1))
            r3 = df['High'].shift(1) + 2 * (pivot - df['Low'].shift(1))
            s3 = df['Low'].shift(1) - 2 * (df['High'].shift(1) - pivot)
            
            return pivot, r1, s1, r2, s2, r3, s3
        
        pivot, r1, s1, r2, s2, r3, s3 = calculate_pivot_points(df)
        df['pivot_point'] = pivot
        df['pivot_r1'] = r1
        df['pivot_s1'] = s1
        df['pivot_r2'] = r2
        df['pivot_s2'] = s2
        df['pivot_r3'] = r3
        df['pivot_s3'] = s3
        
        # Distance from pivot levels
        df['distance_from_pivot'] = df['Close'] - df['pivot_point']
        df['pivot_position'] = np.where(df['Close'] > df['pivot_point'], 1, -1)
        
        # ==================== NEW: ACCUMULATION/DISTRIBUTION LINE ====================
        def calculate_ad_line(df):
            clv = ((df['Close'] - df['Low']) - (df['High'] - df['Close'])) / (df['High'] - df['Low']).replace(0, np.nan)
            clv = clv.fillna(0)
            ad = (clv * df['Volume']).cumsum()
            return ad
        
        df['AD_line'] = calculate_ad_line(df)
        df['AD_line_ema'] = df['AD_line'].ewm(span=20, adjust=False).mean()
        df['AD_divergence'] = df['AD_line'] - df['AD_line_ema']
        
        # ==================== NEW: PRICE VOLUME TREND (PVT) ====================
        def calculate_pvt(df):
            price_change = df['Close'].pct_change()
            pvt = (price_change * df['Volume']).fillna(0).cumsum()
            return pvt
        
        df['PVT'] = calculate_pvt(df)
        df['PVT_signal'] = df['PVT'].ewm(span=10, adjust=False).mean()
        df['PVT_divergence'] = df['PVT'] - df['PVT_signal']
        
        # ==================== NEW: SUPERTREND ====================
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
        
        for period, multiplier in [(10, 3), (7, 2)]:
            st, st_dir = calculate_supertrend(df, period, multiplier)
            df[f'supertrend_{period}_{multiplier}'] = st
            df[f'supertrend_dir_{period}_{multiplier}'] = st_dir
            df[f'supertrend_distance_{period}_{multiplier}'] = df['Close'] - st
        
        # 15. Awesome Oscillator
        df['AO'] = df['Close'].rolling(window=5, min_periods=1).mean() - df['Close'].rolling(window=34, min_periods=1).mean()
        df['AO_diff'] = df['AO'].diff()
        
        # ==================== ENHANCED VOLUME PROFILE ====================
        # Volume Profile (Enhanced version with TPO concept)
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
        
        # 17. OBV
        df['OBV'] = (np.sign(df['Close'].diff()) * df['Volume']).fillna(0).cumsum()
        df['OBV_EMA'] = df['OBV'].ewm(span=20, adjust=False).mean()
        df['OBV_signal'] = df['OBV'] - df['OBV_EMA']
        
        # 18. Vortex Indicator
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
        
        for period in [14, 21]:
            vi_plus, vi_minus = calculate_vortex(df, period)
            df[f'VI_plus_{period}'] = vi_plus
            df[f'VI_minus_{period}'] = vi_minus
            df[f'VI_diff_{period}'] = vi_plus - vi_minus
        
        # 19. Elliott Wave (Simplified)
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
        
        # ==================== ENHANCED ELLIOTT WAVE ANALYSIS ====================
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
                'primary': 55
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
        
        # 20. Harmonic Patterns (Simplified)
        def detect_harmonic_patterns(df, lookback=50):
            high = df['High'].rolling(window=lookback, min_periods=1).max()
            low = df['Low'].rolling(window=lookback, min_periods=1).min()
            current = df['Close']
            
            range_hl = high - low
            range_hl = range_hl.replace(0, np.nan)
            retracement = (high - current) / range_hl
            
            # Harmonic levels
            harmonic_618 = abs(retracement - 0.618) < 0.05
            harmonic_786 = abs(retracement - 0.786) < 0.05
            harmonic_886 = abs(retracement - 0.886) < 0.05
            
            return harmonic_618.astype(int) + harmonic_786.astype(int) + harmonic_886.astype(int)
        
        df['harmonic_pattern_strength'] = detect_harmonic_patterns(df)
        
        # ==================== ADVANCED HARMONIC PATTERNS ====================
        def advanced_harmonic_patterns(df, lookback=100):
            """Detect Gartley, Butterfly, Bat, and Crab patterns"""
            
            # Key Fibonacci ratios for different patterns
            pattern_ratios = {
                'gartley': {'XA_BC': 0.618, 'AB_CD': 1.272, 'XA_AD': 0.786},
                'butterfly': {'XA_BC': 0.786, 'AB_CD': 1.618, 'XA_AD': 1.272},
                'bat': {'XA_BC': 0.886, 'AB_CD': 2.618, 'XA_AD': 0.886},
                'crab': {'XA_BC': 0.886, 'AB_CD': 3.618, 'XA_AD': 1.618}
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
        
        # 21. DEMA
        for period in [10, 20, 50]:
            ema = df['Close'].ewm(span=period, adjust=False).mean()
            ema_of_ema = ema.ewm(span=period, adjust=False).mean()
            df[f'DEMA_{period}'] = 2 * ema - ema_of_ema
            df[f'DEMA_{period}_ratio'] = df['Close'] / df[f'DEMA_{period}'].replace(0, np.nan)
        
        # 22. Market Sentiment Features
        df['volume_surge'] = df['Volume'] / df['Volume'].rolling(window=20, min_periods=1).mean().replace(0, np.nan)
        df['large_volume_flag'] = (df['volume_surge'] > 2).astype(int)
        df['price_acceleration'] = df['price_change'].diff()
        df['momentum_strength'] = df['price_change'].rolling(window=10, min_periods=1).mean()
        df['volatility_20'] = df['price_change'].rolling(window=20, min_periods=1).std()
        df['volatility_ratio'] = df['volatility_20'] / df['volatility_20'].rolling(window=50, min_periods=1).mean().replace(0, np.nan)
        
        # Additional indicators
        # Williams %R
        for period in [14, 28]:
            highest_high = df['High'].rolling(window=period, min_periods=1).max()
            lowest_low = df['Low'].rolling(window=period, min_periods=1).min()
            hl_range = highest_high - lowest_low
            df[f'williams_r_{period}'] = np.where(hl_range > 0,
                                                -100 * ((highest_high - df['Close']) / hl_range),
                                                -50)
        
        # MFI
        def calculate_mfi(df, period=14):
            typical_price = (df['High'] + df['Low'] + df['Close']) / 3
            money_flow = typical_price * df['Volume']
            
            positive_flow = np.where(typical_price > typical_price.shift(), money_flow, 0)
            negative_flow = np.where(typical_price < typical_price.shift(), money_flow, 0)
            
            positive_mf = pd.Series(positive_flow).rolling(window=period, min_periods=1).sum()
            negative_mf = pd.Series(negative_flow).rolling(window=period, min_periods=1).sum()
            
            mfi = 100 - (100 / (1 + positive_mf / negative_mf.replace(0, np.nan)))
            return mfi.fillna(50)
        
        df['MFI_14'] = calculate_mfi(df)
        
        # CMF
        def calculate_cmf(df, period=20):
            clv = ((df['Close'] - df['Low']) - (df['High'] - df['Close'])) / (df['High'] - df['Low']).replace(0, np.nan)
            mfv = clv * df['Volume']
            cmf = mfv.rolling(window=period, min_periods=1).sum() / df['Volume'].rolling(window=period, min_periods=1).sum().replace(0, np.nan)
            return cmf.fillna(0)
        
        df['CMF_20'] = calculate_cmf(df)
        
        # ROC
        for period in [10, 20, 30]:
            shifted_close = df['Close'].shift(period)
            df[f'ROC_{period}'] = ((df['Close'] - shifted_close) / shifted_close.replace(0, np.nan)) * 100
        
        # Volume indicators
        for window in [5, 10, 20, 50]:
            df[f'volume_mean_{window}'] = df['Volume'].rolling(window=window, min_periods=1).mean()
            df[f'volume_ratio_{window}'] = df['Volume'] / df[f'volume_mean_{window}'].replace(0, np.nan)
        
       
        # Lag features
        for lag in [1, 2, 3, 6, 12, 24, 48]:
            df[f'close_lag_{lag}'] = df['Close'].shift(lag)
            df[f'volume_lag_{lag}'] = df['Volume'].shift(lag)
            df[f'return_lag_{lag}'] = df['price_change'].shift(lag)
        
        # Interaction features
        df['price_volume_interaction'] = df['price_change'] * df['volume_ratio_20']
        df['trend_strength'] = df['ADX_14'] * np.sign(df['DI_diff_14'])
        df['momentum_confirmation'] = ((df['RSI_14'] > 50).astype(int) + 
                                    (df['MACD_histogram'] > 0).astype(int) + 
                                    (df['stoch_k_14'] > 50).astype(int)) / 3
        
        # ==================== NEW: HEIKIN ASHI ====================
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
        
        # ==================== NEW: VOLATILITY STOP (VSTOP) ====================
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
        
        vstop, vstop_dir = calculate_vstop(df)
        df['VSTOP'] = vstop
        df['VSTOP_direction'] = vstop_dir
        df['VSTOP_distance'] = df['Close'] - vstop
        
        # ==================== NEW: KELTNER CHANNELS ====================
        def calculate_keltner_channels(df, ema_period=20, atr_period=10, multiplier=2):
            ema = df['Close'].ewm(span=ema_period, adjust=False).mean()
            atr = calculate_atr(df, atr_period)
            
            upper = ema + multiplier * atr
            lower = ema - multiplier * atr
            
            return ema, upper, lower
        
        for period in [20, 50]:
            kc_ema, kc_upper, kc_lower = calculate_keltner_channels(df, ema_period=period)
            df[f'KC_middle_{period}'] = kc_ema
            df[f'KC_upper_{period}'] = kc_upper
            df[f'KC_lower_{period}'] = kc_lower
            df[f'KC_width_{period}'] = kc_upper - kc_lower
            df[f'KC_position_{period}'] = (df['Close'] - kc_lower) / (kc_upper - kc_lower).replace(0, np.nan)
        
        # ==================== NEW: ADAPTIVE MOVING AVERAGE (AMA) ====================
        def calculate_ama(prices, er_period=10, fast_period=2, slow_period=30):
            # Efficiency Ratio
            change = abs(prices.diff(er_period))
            volatility = prices.diff().abs().rolling(window=er_period).sum()
            er = change / volatility.replace(0, np.nan)
            er = er.fillna(0.5)
            
            # Smoothing constants
            fast_sc = 2 / (fast_period + 1)
            slow_sc = 2 / (slow_period + 1)
            
            # Variable smoothing
            sc = er * (fast_sc - slow_sc) + slow_sc
            sc = sc ** 2
            
            # AMA calculation
            ama = prices.copy()
            for i in range(er_period, len(prices)):
                ama.iloc[i] = ama.iloc[i-1] + sc.iloc[i] * (prices.iloc[i] - ama.iloc[i-1])
                
            return ama, er
        
        ama, efficiency_ratio = calculate_ama(df['Close'])
        df['AMA'] = ama
        df['AMA_ER'] = efficiency_ratio
        df['AMA_distance'] = df['Close'] - df['AMA']
        
        # ==================== NEW: CUMULATIVE VOLUME DELTA ====================
        def calculate_cvd(df):
            # Approximate bid/ask volume
            price_change = df['Close'] - df['Close'].shift(1)
            
            # If price goes up, assume more buying volume
            buy_volume = np.where(price_change > 0, df['Volume'], 
                                np.where(price_change < 0, df['Volume'] * 0.3, df['Volume'] * 0.5))
            sell_volume = df['Volume'] - buy_volume
            
            volume_delta = buy_volume - sell_volume
            cvd = volume_delta.cumsum()
            
            return cvd, volume_delta
        
        cvd, volume_delta = calculate_cvd(df)
        df['CVD'] = cvd
        df['volume_delta'] = volume_delta
        df['CVD_sma'] = df['CVD'].rolling(window=20, min_periods=1).mean()
        df['CVD_divergence'] = df['CVD'] - df['CVD_sma']
        
        # ==================== NEW: POLYNOMIAL REGRESSION CHANNEL ====================
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
        
        poly_reg, poly_upper, poly_lower = calculate_poly_regression(df['Close'], degree=2, period=50)
        df['poly_regression'] = poly_reg
        df['poly_upper'] = poly_upper
        df['poly_lower'] = poly_lower
        # Convert to pandas Series for safe division
        poly_diff = pd.Series(poly_upper - poly_lower)
        poly_diff = poly_diff.replace(0, np.nan)
        df['poly_position'] = (df['Close'] - poly_lower) / poly_diff
        
        # ==================== NEW: R-SQUARED ====================
        def calculate_r_squared(df, period=20):
            r_squared = np.zeros(len(df))
            
            for i in range(period, len(df)):
                y = df['Close'].iloc[i-period:i].values
                x = np.arange(period)
                
                # Linear regression
                slope, intercept = np.polyfit(x, y, 1)
                y_pred = slope * x + intercept
                
                # R-squared
                ss_res = np.sum((y - y_pred) ** 2)
                ss_tot = np.sum((y - np.mean(y)) ** 2)
                
                r_squared[i] = 1 - (ss_res / ss_tot) if ss_tot != 0 else 0
                
            return r_squared
        
        df['r_squared'] = calculate_r_squared(df, period=20)
        df['r_squared_smooth'] = pd.Series(df['r_squared']).rolling(window=5, min_periods=1).mean()
        
        # ==================== ENHANCED SMART MONEY CONCEPTS (SMC) ====================
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
                'smc_liquidity_grab': (
                    ((df['High'] > swing_high_liquidity.shift(1)) & (df['Close'] < df['Open'])) |
                    ((df['Low'] < swing_low_liquidity.shift(1)) & (df['Close'] > df['Open']))
                ).astype(int)
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
        
        # ==================== NEW: BELL CURVE ANALYSIS ====================
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
        
        z_score, norm_probability = calculate_bell_curve_features(df)
        df['price_z_score'] = z_score
        df['price_norm_probability'] = norm_probability
        df['price_extreme'] = (abs(z_score) > 2).astype(int)
        
        # ==================== NEW: ADVANCED PIVOT POINTS ====================
        # Camarilla Pivot Points
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
            
            return r4, r3, r2, r1, s1, s2, s3, s4
        
        cam_r4, cam_r3, cam_r2, cam_r1, cam_s1, cam_s2, cam_s3, cam_s4 = calculate_camarilla_pivots(df)
        df['camarilla_r4'] = cam_r4
        df['camarilla_r3'] = cam_r3
        df['camarilla_s3'] = cam_s3
        df['camarilla_s4'] = cam_s4
        
        # Woodie's Pivot Points
        def calculate_woodie_pivots(df):
            h = df['High'].shift(1)
            l = df['Low'].shift(1)
            c = df['Close'].shift(1)
            
            pivot = (h + l + 2 * c) / 4
            r1 = 2 * pivot - l
            s1 = 2 * pivot - h
            r2 = pivot + (h - l)
            s2 = pivot - (h - l)
            
            return pivot, r1, s1, r2, s2
        
        woodie_pivot, woodie_r1, woodie_s1, woodie_r2, woodie_s2 = calculate_woodie_pivots(df)
        df['woodie_pivot'] = woodie_pivot
        df['woodie_r1'] = woodie_r1
        df['woodie_s1'] = woodie_s1
        
        # ==================== NEW: TRAILING STOP LOSS ATR ====================
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
        
        tsl_long, tsl_short = calculate_trailing_stop_atr(df)
        df['trailing_stop_long'] = tsl_long
        df['trailing_stop_short'] = tsl_short
        df['trailing_stop_distance'] = np.minimum(
            df['Close'] - tsl_long,
            tsl_short - df['Close']
        )
        
        # ==================== NEW: ADVANCED MARKET PROFILE ====================
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
        
        # ==================== ENHANCED ORDER FLOW ANALYSIS ====================
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
                'of_volume_delta_ratio': (buy_volume / sell_volume.replace(0, np.nan)).fillna(1)
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
        
        # ==================== TARGET ====================
        df['target'] = df['Close'].shift(-1)
        
        print(f"Created {len(df.columns)} features!")
        print("Including enhanced implementations of:")
        print("- Elliott Wave Analysis (3 wave degrees)")
        print("- Advanced Harmonic Patterns (Gartley, Butterfly, Bat, Crab)")
        print("- Enhanced Volume Profile with TPO concepts")
        print("- Advanced Order Flow Analysis")
        print("- Comprehensive Smart Money Concepts")
        print("- Plus all standard technical indicators")
        
        print(f"Created {len(df.columns)} features!")
        
        # Fill NaN values
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
        
        return df