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
    
    # Try to import pandas-ta
    try:
        import pandas_ta as pta
        pandas_ta_available = True
        print("pandas-ta available - using comprehensive indicators")
    except ImportError:
        print("Warning: pandas-ta not available.")
        pandas_ta_available = False
    
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
    
    # ==================== ALL TA-LIB INDICATORS ====================
    if talib_available:
        print("\nAdding ALL TA-Lib indicators...")
        
        # Prepare OHLCV data
        high = df['High'].values
        low = df['Low'].values
        close = df['Close'].values
        open_ = df['Open'].values
        volume = df['Volume'].values
        
        # 1. Cycle Indicators
        try:
            try:
                df['talib_HT_DCPERIOD'] = talib.HT_DCPERIOD(close)
            except Exception as e:
                print(f"talib.HT_DCPERIOD failed: {e}")
            
            try:
                df['talib_HT_DCPHASE'] = talib.HT_DCPHASE(close)
            except Exception as e:
                print(f"talib.HT_DCPHASE failed: {e}")
            
            try:
                ht_phasor_inphase, ht_phasor_quad = talib.HT_PHASOR(close)
                df['talib_HT_PHASOR_inphase'] = ht_phasor_inphase
                df['talib_HT_PHASOR_quadrature'] = ht_phasor_quad
            except Exception as e:
                print(f"talib.HT_PHASOR failed: {e}")
            
            try:
                ht_sine, ht_leadsine = talib.HT_SINE(close)
                df['talib_HT_SINE'] = ht_sine
                df['talib_HT_LEADSINE'] = ht_leadsine
            except Exception as e:
                print(f"talib.HT_SINE failed: {e}")
            
            try:
                df['talib_HT_TRENDMODE'] = talib.HT_TRENDMODE(close)
            except Exception as e:
                print(f"talib.HT_TRENDMODE failed: {e}")
        except Exception as e:
            print(f"Hilbert Transform indicators section failed: {e}")
        
        # 2. Math Operators
        try:
            # These need two arrays, so we'll use different price combinations
            try:
                df['talib_ADD_hl'] = talib.ADD(high, low)
            except Exception as e:
                print(f"talib.ADD(high, low) failed: {e}")
            
            try:
                df['talib_ADD_oc'] = talib.ADD(open_, close)
            except Exception as e:
                print(f"talib.ADD(open_, close) failed: {e}")
            
            try:
                df['talib_DIV_ch'] = talib.DIV(close, high)
            except Exception as e:
                print(f"talib.DIV(close, high) failed: {e}")
            
            try:
                df['talib_DIV_cl'] = talib.DIV(close, low)
            except Exception as e:
                print(f"talib.DIV(close, low) failed: {e}")
            
            try:
                df['talib_MAX_high'] = talib.MAX(high, timeperiod=30)
            except Exception as e:
                print(f"talib.MAX(high, low) failed: {e}")
            
            try:
                df['talib_MIN_low'] = talib.MIN(low, timeperiod=30)
            except Exception as e:
                print(f"talib.MIN(high, low) failed: {e}")
            
            try:
                df['talib_MULT_cv'] = talib.MULT(close, volume)
            except Exception as e:
                print(f"talib.MULT(close, volume) failed: {e}")
            
            try:
                df['talib_SUB_hl'] = talib.SUB(high, low)
            except Exception as e:
                print(f"talib.SUB(high, low) failed: {e}")
            
            try:
                df['talib_SUB_co'] = talib.SUB(close, open_)
            except Exception as e:
                print(f"talib.SUB(close, open_) failed: {e}")
            
            # Single array functions with different periods
            for period in [5, 10, 20, 30]:
                try:
                    df[f'talib_MAXINDEX_{period}'] = talib.MAXINDEX(close, timeperiod=period)
                except Exception as e:
                    print(f"talib.MAXINDEX({period}) failed: {e}")
                
                try:
                    df[f'talib_MININDEX_{period}'] = talib.MININDEX(close, timeperiod=period)
                except Exception as e:
                    print(f"talib.MININDEX({period}) failed: {e}")
                
                try:
                    min_val, max_val = talib.MINMAX(close, timeperiod=period)
                    df[f'talib_MINMAX_min_{period}'] = min_val
                    df[f'talib_MINMAX_max_{period}'] = max_val
                except Exception as e:
                    print(f"talib.MINMAX({period}) failed: {e}")
                
                try:
                    minidx, maxidx = talib.MINMAXINDEX(close, timeperiod=period)
                    df[f'talib_MINMAXINDEX_minidx_{period}'] = minidx
                    df[f'talib_MINMAXINDEX_maxidx_{period}'] = maxidx
                except Exception as e:
                    print(f"talib.MINMAXINDEX({period}) failed: {e}")
                
                try:
                    df[f'talib_SUM_{period}'] = talib.SUM(close, timeperiod=period)
                except Exception as e:
                    print(f"talib.SUM({period}) failed: {e}")
        except Exception as e:
            print(f"Math operators section failed: {e}")
        
        # 3. Math Transform
        try:
            # Normalize close prices to [-1, 1] range for trigonometric functions
            normalized_close = 2 * (close - np.min(close)) / (np.max(close) - np.min(close)) - 1
            normalized_close = np.clip(normalized_close, -1, 1)
            
            try:
                df['talib_ACOS'] = talib.ACOS(normalized_close)
            except Exception as e:
                print(f"talib.ACOS failed: {e}")
            
            try:
                df['talib_ASIN'] = talib.ASIN(normalized_close)
            except Exception as e:
                print(f"talib.ASIN failed: {e}")
            
            try:
                df['talib_ATAN'] = talib.ATAN(close)
            except Exception as e:
                print(f"talib.ATAN failed: {e}")
            
            try:
                df['talib_CEIL'] = talib.CEIL(close)
            except Exception as e:
                print(f"talib.CEIL failed: {e}")
            
            try:
                df['talib_COS'] = talib.COS(close)
            except Exception as e:
                print(f"talib.COS failed: {e}")
            
            try:
                df['talib_COSH'] = talib.COSH(close / 1000)  # Scale down to avoid overflow
            except Exception as e:
                print(f"talib.COSH failed: {e}")
            
            try:
                df['talib_EXP'] = talib.EXP(close / 1000)  # Scale down to avoid overflow
            except Exception as e:
                print(f"talib.EXP failed: {e}")
            
            try:
                df['talib_FLOOR'] = talib.FLOOR(close)
            except Exception as e:
                print(f"talib.FLOOR failed: {e}")
            
            try:
                df['talib_LN'] = talib.LN(close)
            except Exception as e:
                print(f"talib.LN failed: {e}")
            
            try:
                df['talib_LOG10'] = talib.LOG10(close)
            except Exception as e:
                print(f"talib.LOG10 failed: {e}")
            
            try:
                df['talib_SIN'] = talib.SIN(close)
            except Exception as e:
                print(f"talib.SIN failed: {e}")
            
            try:
                df['talib_SINH'] = talib.SINH(close / 1000)  # Scale down
            except Exception as e:
                print(f"talib.SINH failed: {e}")
            
            try:
                df['talib_SQRT'] = talib.SQRT(close)
            except Exception as e:
                print(f"talib.SQRT failed: {e}")
            
            try:
                df['talib_TAN'] = talib.TAN(close)
            except Exception as e:
                print(f"talib.TAN failed: {e}")
            
            try:
                df['talib_TANH'] = talib.TANH(close / 1000)  # Scale down
            except Exception as e:
                print(f"talib.TANH failed: {e}")
        except Exception as e:
            print(f"Math transform section failed: {e}")
        
        # 4. Momentum Indicators
        try:
            # ADX family
            for period in [7, 14, 21, 28]:
                df[f'talib_ADX_{period}'] = talib.ADX(high, low, close, timeperiod=period)
                df[f'talib_ADXR_{period}'] = talib.ADXR(high, low, close, timeperiod=period)
                df[f'talib_DX_{period}'] = talib.DX(high, low, close, timeperiod=period)
                df[f'talib_MINUS_DI_{period}'] = talib.MINUS_DI(high, low, close, timeperiod=period)
                df[f'talib_PLUS_DI_{period}'] = talib.PLUS_DI(high, low, close, timeperiod=period)
                df[f'talib_MINUS_DM_{period}'] = talib.MINUS_DM(high, low, timeperiod=period)
                df[f'talib_PLUS_DM_{period}'] = talib.PLUS_DM(high, low, timeperiod=period)
            
            # APO
            for fast, slow in [(12, 26), (5, 35), (10, 20)]:
                df[f'talib_APO_{fast}_{slow}'] = talib.APO(close, fastperiod=fast, slowperiod=slow)
            
            # AROON
            for period in [14, 25, 50]:
                aroon_down, aroon_up = talib.AROON(high, low, timeperiod=period)
                df[f'talib_AROON_down_{period}'] = aroon_down
                df[f'talib_AROON_up_{period}'] = aroon_up
                df[f'talib_AROONOSC_{period}'] = talib.AROONOSC(high, low, timeperiod=period)
            
            # BOP
            df['talib_BOP'] = talib.BOP(open_, high, low, close)
            
            # CCI
            for period in [14, 20, 30]:
                df[f'talib_CCI_{period}'] = talib.CCI(high, low, close, timeperiod=period)
            
            # CMO
            for period in [14, 20, 30]:
                df[f'talib_CMO_{period}'] = talib.CMO(close, timeperiod=period)
            
            # MACD variations
            for fast, slow, signal in [(12, 26, 9), (5, 35, 5), (3, 10, 16)]:
                macd, macdsignal, macdhist = talib.MACD(close, fastperiod=fast, slowperiod=slow, signalperiod=signal)
                df[f'talib_MACD_{fast}_{slow}_{signal}'] = macd
                df[f'talib_MACD_{fast}_{slow}_{signal}_signal'] = macdsignal
                df[f'talib_MACD_{fast}_{slow}_{signal}_hist'] = macdhist
            
            # MACDEXT
            for fast, slow, signal in [(12, 26, 9), (5, 35, 5), (3, 10, 16)]:
                macd_ext, macd_ext_signal, macd_ext_hist = talib.MACDEXT(
                    close, 
                    fastperiod=fast, 
                    fastmatype=0,
                    slowperiod=slow, 
                    slowmatype=0,
                    signalperiod=signal,
                    signalmatype=0
                )
                df[f'talib_MACDEXT_{fast}_{slow}_{signal}'] = macd_ext
                df[f'talib_MACDEXT_{fast}_{slow}_{signal}_signal'] = macd_ext_signal
                df[f'talib_MACDEXT_{fast}_{slow}_{signal}_hist'] = macd_ext_hist
            
            # MACDFIX
            macdfix, macdfix_signal, macdfix_hist = talib.MACDFIX(close, signalperiod=9)
            df['talib_MACDFIX'] = macdfix
            df['talib_MACDFIX_signal'] = macdfix_signal
            df['talib_MACDFIX_hist'] = macdfix_hist
            
            # MFI
            for period in [14, 20, 30]:
                df[f'talib_MFI_{period}'] = talib.MFI(high, low, close, volume, timeperiod=period)
            
            # MOM
            for period in [10, 20, 30]:
                df[f'talib_MOM_{period}'] = talib.MOM(close, timeperiod=period)
            
            # PPO
            for fast, slow in [(12, 26), (5, 35)]:
                df[f'talib_PPO_{fast}_{slow}'] = talib.PPO(close, fastperiod=fast, slowperiod=slow)
            
            # ROC family
            for period in [10, 20, 30]:
                df[f'talib_ROC_{period}'] = talib.ROC(close, timeperiod=period)
                df[f'talib_ROCP_{period}'] = talib.ROCP(close, timeperiod=period)
                df[f'talib_ROCR_{period}'] = talib.ROCR(close, timeperiod=period)
                df[f'talib_ROCR100_{period}'] = talib.ROCR100(close, timeperiod=period)
            
            # RSI
            for period in [9, 14, 21, 28]:
                df[f'talib_RSI_{period}'] = talib.RSI(close, timeperiod=period)
            
            # STOCH
            for k_period, d_period in [(14, 3), (21, 3), (9, 3)]:
                slowk, slowd = talib.STOCH(high, low, close, fastk_period=k_period, slowk_period=3, slowd_period=d_period)
                df[f'talib_STOCH_k_{k_period}_{d_period}'] = slowk
                df[f'talib_STOCH_d_{k_period}_{d_period}'] = slowd
            
            # STOCHF
            for k_period, d_period in [(14, 3), (21, 3)]:
                fastk, fastd = talib.STOCHF(high, low, close, fastk_period=k_period, fastd_period=d_period)
                df[f'talib_STOCHF_k_{k_period}_{d_period}'] = fastk
                df[f'talib_STOCHF_d_{k_period}_{d_period}'] = fastd
            
            # STOCHRSI
            for period in [14, 21]:
                fastk, fastd = talib.STOCHRSI(close, timeperiod=period, fastk_period=5, fastd_period=3)
                df[f'talib_STOCHRSI_k_{period}'] = fastk
                df[f'talib_STOCHRSI_d_{period}'] = fastd
            
            # TRIX
            for period in [14, 30]:
                df[f'talib_TRIX_{period}'] = talib.TRIX(close, timeperiod=period)
            
            # ULTOSC
            df['talib_ULTOSC'] = talib.ULTOSC(high, low, close, timeperiod1=7, timeperiod2=14, timeperiod3=28)
            df['talib_ULTOSC_custom'] = talib.ULTOSC(high, low, close, timeperiod1=5, timeperiod2=10, timeperiod3=20)
            
            # WILLR
            for period in [14, 20, 50]:
                df[f'talib_WILLR_{period}'] = talib.WILLR(high, low, close, timeperiod=period)
        except Exception as e:
            print(f"Some momentum indicators failed: {e}")
        
        # 5. Overlap Studies
        try:
            # BBANDS
            for period in [20, 50]:
                for nbdev in [1, 2, 2.5, 3]:
                    upper, middle, lower = talib.BBANDS(close, timeperiod=period, nbdevup=nbdev, nbdevdn=nbdev)
                    df[f'talib_BBANDS_upper_{period}_{nbdev}'] = upper
                    df[f'talib_BBANDS_middle_{period}_{nbdev}'] = middle
                    df[f'talib_BBANDS_lower_{period}_{nbdev}'] = lower
            
            # DEMA
            for period in [10, 20, 30, 50]:
                df[f'talib_DEMA_{period}'] = talib.DEMA(close, timeperiod=period)
            
            # EMA  
            for period in [5, 10, 12, 20, 26, 50, 100, 200]:
                df[f'talib_EMA_{period}'] = talib.EMA(close, timeperiod=period)
            
            # HT_TRENDLINE
            df['talib_HT_TRENDLINE'] = talib.HT_TRENDLINE(close)
            
            # KAMA
            for period in [10, 20, 30]:
                df[f'talib_KAMA_{period}'] = talib.KAMA(close, timeperiod=period)
            
            # MA
            for period in [5, 10, 20, 50, 100, 200]:
                df[f'talib_MA_{period}'] = talib.MA(close, timeperiod=period)
            
            # MAMA
            mama, fama = talib.MAMA(close, fastlimit=0.5, slowlimit=0.05)
            df['talib_MAMA'] = mama
            df['talib_FAMA'] = fama
            
            # MIDPOINT
            for period in [7, 14, 28]:
                df[f'talib_MIDPOINT_{period}'] = talib.MIDPOINT(close, timeperiod=period)
            
            # MIDPRICE
            for period in [7, 14, 28]:
                df[f'talib_MIDPRICE_{period}'] = talib.MIDPRICE(high, low, timeperiod=period)
            
            # SAR
            df['talib_SAR'] = talib.SAR(high, low, acceleration=0.02, maximum=0.2)
            df['talib_SAR_aggressive'] = talib.SAR(high, low, acceleration=0.03, maximum=0.3)
            
            # SAREXT
            df['talib_SAREXT'] = talib.SAREXT(high, low)
            
            # SMA
            for period in [5, 10, 20, 50, 100, 200]:
                df[f'talib_SMA_{period}'] = talib.SMA(close, timeperiod=period)
            
            # T3
            for period in [5, 10, 20]:
                df[f'talib_T3_{period}'] = talib.T3(close, timeperiod=period, vfactor=0.7)
            
            # TEMA
            for period in [10, 20, 30]:
                df[f'talib_TEMA_{period}'] = talib.TEMA(close, timeperiod=period)
            
            # TRIMA
            for period in [10, 20, 30]:
                df[f'talib_TRIMA_{period}'] = talib.TRIMA(close, timeperiod=period)
            
            # WMA
            for period in [10, 20, 30]:
                df[f'talib_WMA_{period}'] = talib.WMA(close, timeperiod=period)
        except Exception as e:
            print(f"Some overlap studies failed: {e}")
        
        # 6. Pattern Recognition - ALL 61 patterns
        try:
            pattern_functions = [
                ('CDL2CROWS', talib.CDL2CROWS),
                ('CDL3BLACKCROWS', talib.CDL3BLACKCROWS),
                ('CDL3INSIDE', talib.CDL3INSIDE),
                ('CDL3LINESTRIKE', talib.CDL3LINESTRIKE),
                ('CDL3OUTSIDE', talib.CDL3OUTSIDE),
                ('CDL3STARSINSOUTH', talib.CDL3STARSINSOUTH),
                ('CDL3WHITESOLDIERS', talib.CDL3WHITESOLDIERS),
                ('CDLABANDONEDBABY', talib.CDLABANDONEDBABY),
                ('CDLADVANCEBLOCK', talib.CDLADVANCEBLOCK),
                ('CDLBELTHOLD', talib.CDLBELTHOLD),
                ('CDLBREAKAWAY', talib.CDLBREAKAWAY),
                ('CDLCLOSINGMARUBOZU', talib.CDLCLOSINGMARUBOZU),
                ('CDLCONCEALBABYSWALL', talib.CDLCONCEALBABYSWALL),
                ('CDLCOUNTERATTACK', talib.CDLCOUNTERATTACK),
                ('CDLDARKCLOUDCOVER', talib.CDLDARKCLOUDCOVER),
                ('CDLDOJI', talib.CDLDOJI),
                ('CDLDOJISTAR', talib.CDLDOJISTAR),
                ('CDLDRAGONFLYDOJI', talib.CDLDRAGONFLYDOJI),
                ('CDLENGULFING', talib.CDLENGULFING),
                ('CDLEVENINGDOJISTAR', talib.CDLEVENINGDOJISTAR),
                ('CDLEVENINGSTAR', talib.CDLEVENINGSTAR),
                ('CDLGAPSIDESIDEWHITE', talib.CDLGAPSIDESIDEWHITE),
                ('CDLGRAVESTONEDOJI', talib.CDLGRAVESTONEDOJI),
                ('CDLHAMMER', talib.CDLHAMMER),
                ('CDLHANGINGMAN', talib.CDLHANGINGMAN),
                ('CDLHARAMI', talib.CDLHARAMI),
                ('CDLHARAMICROSS', talib.CDLHARAMICROSS),
                ('CDLHIGHWAVE', talib.CDLHIGHWAVE),
                ('CDLHIKKAKE', talib.CDLHIKKAKE),
                ('CDLHIKKAKEMOD', talib.CDLHIKKAKEMOD),
                ('CDLHOMINGPIGEON', talib.CDLHOMINGPIGEON),
                ('CDLIDENTICAL3CROWS', talib.CDLIDENTICAL3CROWS),
                ('CDLINNECK', talib.CDLINNECK),
                ('CDLINVERTEDHAMMER', talib.CDLINVERTEDHAMMER),
                ('CDLKICKING', talib.CDLKICKING),
                ('CDLKICKINGBYLENGTH', talib.CDLKICKINGBYLENGTH),
                ('CDLLADDERBOTTOM', talib.CDLLADDERBOTTOM),
                ('CDLLONGLEGGEDDOJI', talib.CDLLONGLEGGEDDOJI),
                ('CDLLONGLINE', talib.CDLLONGLINE),
                ('CDLMARUBOZU', talib.CDLMARUBOZU),
                ('CDLMATCHINGLOW', talib.CDLMATCHINGLOW),
                ('CDLMATHOLD', talib.CDLMATHOLD),
                ('CDLMORNINGDOJISTAR', talib.CDLMORNINGDOJISTAR),
                ('CDLMORNINGSTAR', talib.CDLMORNINGSTAR),
                ('CDLONNECK', talib.CDLONNECK),
                ('CDLPIERCING', talib.CDLPIERCING),
                ('CDLRICKSHAWMAN', talib.CDLRICKSHAWMAN),
                ('CDLRISEFALL3METHODS', talib.CDLRISEFALL3METHODS),
                ('CDLSEPARATINGLINES', talib.CDLSEPARATINGLINES),
                ('CDLSHOOTINGSTAR', talib.CDLSHOOTINGSTAR),
                ('CDLSHORTLINE', talib.CDLSHORTLINE),
                ('CDLSPINNINGTOP', talib.CDLSPINNINGTOP),
                ('CDLSTALLEDPATTERN', talib.CDLSTALLEDPATTERN),
                ('CDLSTICKSANDWICH', talib.CDLSTICKSANDWICH),
                ('CDLTAKURI', talib.CDLTAKURI),
                ('CDLTASUKIGAP', talib.CDLTASUKIGAP),
                ('CDLTHRUSTING', talib.CDLTHRUSTING),
                ('CDLTRISTAR', talib.CDLTRISTAR),
                ('CDLUNIQUE3RIVER', talib.CDLUNIQUE3RIVER),
                ('CDLUPSIDEGAP2CROWS', talib.CDLUPSIDEGAP2CROWS),
                ('CDLXSIDEGAP3METHODS', talib.CDLXSIDEGAP3METHODS)
            ]
            
            for name, func in pattern_functions:
                try:
                    df[f'talib_{name}'] = func(open_, high, low, close)
                except Exception as e:
                    print(f"talib.{name} failed: {e}")
                    
        except Exception as e:
            print(f"Pattern recognition section failed: {e}")
        
        # 7. Price Transform
        try:
            try:
                df['talib_AVGPRICE'] = talib.AVGPRICE(open_, high, low, close)
            except Exception as e:
                print(f"talib.AVGPRICE failed: {e}")
            
            try:
                df['talib_MEDPRICE'] = talib.MEDPRICE(high, low)
            except Exception as e:
                print(f"talib.MEDPRICE failed: {e}")
            
            try:
                df['talib_TYPPRICE'] = talib.TYPPRICE(high, low, close)
            except Exception as e:
                print(f"talib.TYPPRICE failed: {e}")
            
            try:
                df['talib_WCLPRICE'] = talib.WCLPRICE(high, low, close)
            except Exception as e:
                print(f"talib.WCLPRICE failed: {e}")
        except Exception as e:
            print(f"Price transform section failed: {e}")
        
        # 8. Statistic Functions
        try:
            # BETA
            for period in [5, 20, 60]:
                try:
                    df[f'talib_BETA_{period}'] = talib.BETA(high, low, timeperiod=period)
                except Exception as e:
                    print(f"talib.BETA({period}) failed: {e}")
            
            # CORREL
            for period in [10, 30, 60]:
                try:
                    df[f'talib_CORREL_{period}'] = talib.CORREL(high, low, timeperiod=period)
                except Exception as e:
                    print(f"talib.CORREL({period}) failed: {e}")
            
            # Linear Regression family
            for period in [14, 20, 50]:
                try:
                    df[f'talib_LINEARREG_{period}'] = talib.LINEARREG(close, timeperiod=period)
                except Exception as e:
                    print(f"talib.LINEARREG({period}) failed: {e}")
                
                try:
                    df[f'talib_LINEARREG_ANGLE_{period}'] = talib.LINEARREG_ANGLE(close, timeperiod=period)
                except Exception as e:
                    print(f"talib.LINEARREG_ANGLE({period}) failed: {e}")
                
                try:
                    df[f'talib_LINEARREG_INTERCEPT_{period}'] = talib.LINEARREG_INTERCEPT(close, timeperiod=period)
                except Exception as e:
                    print(f"talib.LINEARREG_INTERCEPT({period}) failed: {e}")
                
                try:
                    df[f'talib_LINEARREG_SLOPE_{period}'] = talib.LINEARREG_SLOPE(close, timeperiod=period)
                except Exception as e:
                    print(f"talib.LINEARREG_SLOPE({period}) failed: {e}")
                
                try:
                    df[f'talib_TSF_{period}'] = talib.TSF(close, timeperiod=period)
                except Exception as e:
                    print(f"talib.TSF({period}) failed: {e}")
            
            # STDDEV
            for period in [5, 10, 20, 50]:
                try:
                    df[f'talib_STDDEV_{period}'] = talib.STDDEV(close, timeperiod=period, nbdev=1)
                except Exception as e:
                    print(f"talib.STDDEV({period}) failed: {e}")
            
            # VAR
            for period in [5, 10, 20]:
                try:
                    df[f'talib_VAR_{period}'] = talib.VAR(close, timeperiod=period)
                except Exception as e:
                    print(f"talib.VAR({period}) failed: {e}")
        except Exception as e:
            print(f"Statistic functions section failed: {e}")
        
        # 9. Volatility Indicators
        try:
            # ATR
            for period in [7, 14, 21]:
                try:
                    df[f'talib_ATR_{period}'] = talib.ATR(high, low, close, timeperiod=period)
                except Exception as e:
                    print(f"talib.ATR({period}) failed: {e}")
            
            # NATR
            for period in [7, 14, 21]:
                try:
                    df[f'talib_NATR_{period}'] = talib.NATR(high, low, close, timeperiod=period)
                except Exception as e:
                    print(f"talib.NATR({period}) failed: {e}")
            
            # TRANGE
            try:
                df['talib_TRANGE'] = talib.TRANGE(high, low, close)
            except Exception as e:
                print(f"talib.TRANGE failed: {e}")
        except Exception as e:
            print(f"Volatility indicators section failed: {e}")
        
        # 10. Volume Indicators
        try:
            try:
                df['talib_AD'] = talib.AD(high, low, close, volume)
            except Exception as e:
                print(f"talib.AD failed: {e}")
            
            try:
                df['talib_ADOSC'] = talib.ADOSC(high, low, close, volume, fastperiod=3, slowperiod=10)
            except Exception as e:
                print(f"talib.ADOSC failed: {e}")
            
            try:
                df['talib_OBV'] = talib.OBV(close, volume)
            except Exception as e:
                print(f"talib.OBV failed: {e}")
        except Exception as e:
            print(f"Volume indicators section failed: {e}")
    
    # ==================== PANDAS-TA INDICATORS (NON-DUPLICATES) ====================
    if pandas_ta_available:
        print("\nAdding pandas-ta indicators (non-duplicates)...")
        
        # Set pandas-ta to use all cores
        df.ta.cores = 0
        
        # Only add indicators not already covered by TA-Lib
        # 1. Candle patterns (additional ones)
        try:
            try:
                df.ta.cdl_z(append=True)
            except Exception as e:
                print(f"pandas-ta CDL_Z failed: {e}")
        except Exception as e:
            print(f"Candle patterns section failed: {e}")
        
        # 2. Momentum indicators (non-duplicates)
        try:
            try:
                df.ta.ao(append=True)  # Awesome Oscillator
            except Exception as e:
                print(f"pandas-ta AO failed: {e}")
            
            try:
                df.ta.bias(length=26, append=True)  # Bias
            except Exception as e:
                print(f"pandas-ta BIAS failed: {e}")
            
            try:
                df.ta.brar(append=True)  # BRAR
            except Exception as e:
                print(f"pandas-ta BRAR failed: {e}")
            
            try:
                df.ta.cfo(append=True)  # Chande Forecast Oscillator
            except Exception as e:
                print(f"pandas-ta CFO failed: {e}")
            
            try:
                df.ta.cg(length=10, append=True)  # Center of Gravity
            except Exception as e:
                print(f"pandas-ta CG failed: {e}")
            
            try:
                df.ta.coppock(append=True)  # Coppock Curve
            except Exception as e:
                print(f"pandas-ta COPPOCK failed: {e}")
            
            try:
                df.ta.cti(append=True)  # Correlation Trend Indicator
            except Exception as e:
                print(f"pandas-ta CTI failed: {e}")
            
            try:
                df.ta.er(length=10, append=True)  # Efficiency Ratio
            except Exception as e:
                print(f"pandas-ta ER failed: {e}")
            
            try:
                df.ta.eri(append=True)  # Elder Ray Index
            except Exception as e:
                print(f"pandas-ta ERI failed: {e}")
            
            try:
                df.ta.fisher(length=9, append=True)  # Fisher Transform
            except Exception as e:
                print(f"pandas-ta FISHER failed: {e}")
            
            try:
                df.ta.inertia(length=14, append=True)  # Inertia
            except Exception as e:
                print(f"pandas-ta INERTIA failed: {e}")
            
            try:
                df.ta.kdj(append=True)  # KDJ
            except Exception as e:
                print(f"pandas-ta KDJ failed: {e}")
            
            try:
                df.ta.kst(append=True)  # Know Sure Thing
            except Exception as e:
                print(f"pandas-ta KST failed: {e}")
            
            try:
                df.ta.pgo(append=True)  # Pretty Good Oscillator
            except Exception as e:
                print(f"pandas-ta PGO failed: {e}")
            
            try:
                df.ta.psl(append=True)  # Psychological Line
            except Exception as e:
                print(f"pandas-ta PSL failed: {e}")
            
            try:
                df.ta.pvo(append=True)  # Price Volume Oscillator
            except Exception as e:
                print(f"pandas-ta PVO failed: {e}")
            
            try:
                df.ta.qqe(append=True)  # Quantitative Qualitative Estimation
            except Exception as e:
                print(f"pandas-ta QQE failed: {e}")
            
            try:
                df.ta.rsx(append=True)  # Relative Strength Xtra
            except Exception as e:
                print(f"pandas-ta RSX failed: {e}")
            
            try:
                df.ta.rvgi(append=True)  # Relative Vigor Index
            except Exception as e:
                print(f"pandas-ta RVGI failed: {e}")
            
            try:
                df.ta.rvi(append=True)  # Relative Volatility Index
            except Exception as e:
                print(f"pandas-ta RVI failed: {e}")
            
            try:
                df.ta.slope(length=10, append=True)  # Slope
            except Exception as e:
                print(f"pandas-ta SLOPE failed: {e}")
            
            try:
                df.ta.smi(append=True)  # Stochastic Momentum Index
            except Exception as e:
                print(f"pandas-ta SMI failed: {e}")
            
            try:
                df.ta.squeeze(append=True)  # Squeeze
            except Exception as e:
                print(f"pandas-ta SQUEEZE failed: {e}")
            
            try:
                df.ta.squeeze_pro(append=True)  # Squeeze Pro
            except Exception as e:
                print(f"pandas-ta SQUEEZE_PRO failed: {e}")
            
            try:
                df.ta.stc(append=True)  # Schaff Trend Cycle
            except Exception as e:
                print(f"pandas-ta STC failed: {e}")
            
            try:
                df.ta.td_seq(append=True)  # TD Sequential
            except Exception as e:
                print(f"pandas-ta TD_SEQ failed: {e}")
            
            try:
                df.ta.tsi(append=True)  # True Strength Index
            except Exception as e:
                print(f"pandas-ta TSI failed: {e}")
            
            try:
                df.ta.uo(append=True)  # Ultimate Oscillator
            except Exception as e:
                print(f"pandas-ta UO failed: {e}")
                
        except Exception as e:
            print(f"Momentum indicators section failed: {e}")
        
        # 3. Overlap indicators (non-duplicates)
        try:
            try:
                df.ta.alma(append=True)  # Arnaud Legoux Moving Average
            except Exception as e:
                print(f"pandas-ta ALMA failed: {e}")
            
            try:
                df.ta.fwma(append=True)  # Fibonacci's Weighted Moving Average
            except Exception as e:
                print(f"pandas-ta FWMA failed: {e}")
            
            try:
                df.ta.hma(append=True)  # Hull Moving Average
            except Exception as e:
                print(f"pandas-ta HMA failed: {e}")
            
            try:
                df.ta.hwma(append=True)  # Holt-Winter Moving Average
            except Exception as e:
                print(f"pandas-ta HWMA failed: {e}")
            
            # Fixed Ichimoku calculation
            try:
                ichimoku_result = df.ta.ichimoku(append=False)
                if ichimoku_result is not None and isinstance(ichimoku_result, pd.DataFrame):
                    for col in ichimoku_result.columns:
                        df[f'pta_ichimoku_{col}'] = ichimoku_result[col]
            except Exception as e:
                print(f"pandas-ta ICHIMOKU failed: {e}")
            
            try:
                df.ta.jma(append=True)  # Jurik Moving Average
            except Exception as e:
                print(f"pandas-ta JMA failed: {e}")
            
            try:
                df.ta.pwma(append=True)  # Pascal's Weighted Moving Average
            except Exception as e:
                print(f"pandas-ta PWMA failed: {e}")
            
            try:
                df.ta.rma(append=True)  # Wilders' Moving Average
            except Exception as e:
                print(f"pandas-ta RMA failed: {e}")
            
            try:
                df.ta.sinwma(append=True)  # Sine Weighted Moving Average
            except Exception as e:
                print(f"pandas-ta SINWMA failed: {e}")
            
            try:
                df.ta.ssf(append=True)  # Ehlers Super Smoother Filter
            except Exception as e:
                print(f"pandas-ta SSF failed: {e}")
            
            try:
                df.ta.supertrend(append=True)  # SuperTrend
            except Exception as e:
                print(f"pandas-ta SUPERTREND failed: {e}")
            
            try:
                df.ta.swma(append=True)  # Symmetric Weighted Moving Average
            except Exception as e:
                print(f"pandas-ta SWMA failed: {e}")
            
            try:
                df.ta.vidya(append=True)  # Variable Index Dynamic Average
            except Exception as e:
                print(f"pandas-ta VIDYA failed: {e}")
            
            try:
                df.ta.vwma(append=True)  # Volume Weighted Moving Average
            except Exception as e:
                print(f"pandas-ta VWMA failed: {e}")
            
            try:
                df.ta.zlma(append=True)  # Zero Lag Moving Average
            except Exception as e:
                print(f"pandas-ta ZLMA failed: {e}")
                
        except Exception as e:
            print(f"Overlap indicators section failed: {e}")
        
        # 4. Performance indicators
        try:
            try:
                df.ta.log_return(append=True)
            except Exception as e:
                print(f"pandas-ta LOG_RETURN failed: {e}")
            
            try:
                df.ta.percent_return(append=True)
            except Exception as e:
                print(f"pandas-ta PERCENT_RETURN failed: {e}")
        except Exception as e:
            print(f"Performance indicators section failed: {e}")
        
        # 5. Statistics (non-duplicates)
        try:
            try:
                df.ta.entropy(append=True)
            except Exception as e:
                print(f"pandas-ta ENTROPY failed: {e}")
            
            try:
                df.ta.kurtosis(length=14, append=True)
            except Exception as e:
                print(f"pandas-ta KURTOSIS failed: {e}")
            
            try:
                df.ta.mad(append=True)  # Mean Absolute Deviation
            except Exception as e:
                print(f"pandas-ta MAD failed: {e}")
            
            try:
                df.ta.median(append=True)
            except Exception as e:
                print(f"pandas-ta MEDIAN failed: {e}")
            
            try:
                df.ta.quantile(append=True)
            except Exception as e:
                print(f"pandas-ta QUANTILE failed: {e}")
            
            try:
                df.ta.skew(length=14, append=True)
            except Exception as e:
                print(f"pandas-ta SKEW failed: {e}")
            
            try:
                df.ta.tos_stdevall(append=True)  # ThinkOrSwim Standard Deviation All
            except Exception as e:
                print(f"pandas-ta TOS_STDEVALL failed: {e}")
            
            try:
                df.ta.variance(append=True)
            except Exception as e:
                print(f"pandas-ta VARIANCE failed: {e}")
            
            try:
                df.ta.zscore(append=True)
            except Exception as e:
                print(f"pandas-ta ZSCORE failed: {e}")
        except Exception as e:
            print(f"Statistics indicators section failed: {e}")
        
        # 6. Trend indicators (non-duplicates)
        try:
            try:
                df.ta.amat(append=True)  # Archer Moving Averages Trends
            except Exception as e:
                print(f"pandas-ta AMAT failed: {e}")
            
            try:
                df.ta.chop(append=True)  # Choppiness Index
            except Exception as e:
                print(f"pandas-ta CHOP failed: {e}")
            
            try:
                df.ta.cksp(append=True)  # Chande Kroll Stop
            except Exception as e:
                print(f"pandas-ta CKSP failed: {e}")
            
            try:
                df.ta.decay(append=True)  # Linear Decay
            except Exception as e:
                print(f"pandas-ta DECAY failed: {e}")
            
            try:
                df.ta.dpo(append=True)  # Detrended Price Oscillator
            except Exception as e:
                print(f"pandas-ta DPO failed: {e}")
            
            try:
                df.ta.long_run(append=True)  # Long Run
            except Exception as e:
                print(f"pandas-ta LONG_RUN failed: {e}")
            
            try:
                df.ta.short_run(append=True)  # Short Run
            except Exception as e:
                print(f"pandas-ta SHORT_RUN failed: {e}")
            
            try:
                df.ta.qstick(append=True)  # QStick
            except Exception as e:
                print(f"pandas-ta QSTICK failed: {e}")
            
            try:
                df.ta.ttm_trend(append=True)  # TTM Trend
            except Exception as e:
                print(f"pandas-ta TTM_TREND failed: {e}")
            
            try:
                df.ta.vhf(append=True)  # Vertical Horizontal Filter
            except Exception as e:
                print(f"pandas-ta VHF failed: {e}")
            
            try:
                df.ta.vortex(append=True)  # Vortex Indicator
            except Exception as e:
                print(f"pandas-ta VORTEX failed: {e}")
        except Exception as e:
            print(f"Trend indicators section failed: {e}")
        
        # 7. Utility indicators
        try:
            try:
                df.ta.above(append=True)
            except Exception as e:
                print(f"pandas-ta ABOVE failed: {e}")
            
            try:
                df.ta.above_value(close=df['Close'], value=df['Close'].mean(), append=True)
            except Exception as e:
                print(f"pandas-ta ABOVE_VALUE failed: {e}")
            
            try:
                df.ta.below(append=True)
            except Exception as e:
                print(f"pandas-ta BELOW failed: {e}")
            
            try:
                df.ta.below_value(close=df['Close'], value=df['Close'].mean(), append=True)
            except Exception as e:
                print(f"pandas-ta BELOW_VALUE failed: {e}")
            
            try:
                df.ta.cross(append=True)
            except Exception as e:
                print(f"pandas-ta CROSS failed: {e}")
            
            try:
                df.ta.decreasing(append=True)
            except Exception as e:
                print(f"pandas-ta DECREASING failed: {e}")
            
            try:
                df.ta.increasing(append=True)
            except Exception as e:
                print(f"pandas-ta INCREASING failed: {e}")
        except Exception as e:
            print(f"Utility indicators section failed: {e}")
        
        # 8. Volatility indicators (non-duplicates)
        try:
            try:
                df.ta.aberration(append=True)  # Aberration
            except Exception as e:
                print(f"pandas-ta ABERRATION failed: {e}")
            
            try:
                df.ta.accbands(append=True)  # Acceleration Bands
            except Exception as e:
                print(f"pandas-ta ACCBANDS failed: {e}")
            
            try:
                df.ta.donchian(append=True)  # Donchian Channel
            except Exception as e:
                print(f"pandas-ta DONCHIAN failed: {e}")
            
            try:
                df.ta.ebsw(append=True)  # Even Better Sinewave
            except Exception as e:
                print(f"pandas-ta EBSW failed: {e}")
            
            try:
                df.ta.hwc(append=True)  # Holt-Winter Channel
            except Exception as e:
                print(f"pandas-ta HWC failed: {e}")
            
            try:
                df.ta.kc(append=True)  # Keltner Channel
            except Exception as e:
                print(f"pandas-ta KC failed: {e}")
            
            try:
                df.ta.massi(append=True)  # Mass Index
            except Exception as e:
                print(f"pandas-ta MASSI failed: {e}")
            
            try:
                df.ta.pdist(append=True)  # Price Distance
            except Exception as e:
                print(f"pandas-ta PDIST failed: {e}")
            
            try:
                df.ta.thermo(append=True)  # Ehlers Thermometer
            except Exception as e:
                print(f"pandas-ta THERMO failed: {e}")
            
            try:
                df.ta.true_range(append=True)  # True Range
            except Exception as e:
                print(f"pandas-ta TRUE_RANGE failed: {e}")
            
            try:
                df.ta.ui(append=True)  # Ulcer Index
            except Exception as e:
                print(f"pandas-ta UI failed: {e}")
        except Exception as e:
            print(f"Volatility indicators section failed: {e}")
        
        # 9. Volume indicators (non-duplicates)
        try:
            try:
                df.ta.aobv(append=True)  # Archer On Balance Volume
            except Exception as e:
                print(f"pandas-ta AOBV failed: {e}")
            
            try:
                df.ta.cmf(append=True)  # Chaikin Money Flow
            except Exception as e:
                print(f"pandas-ta CMF failed: {e}")
            
            try:
                df.ta.efi(append=True)  # Elder's Force Index
            except Exception as e:
                print(f"pandas-ta EFI failed: {e}")
            
            try:
                df.ta.eom(append=True)  # Ease of Movement
            except Exception as e:
                print(f"pandas-ta EOM failed: {e}")
            
            try:
                df.ta.kvo(append=True)  # Klinger Volume Oscillator
            except Exception as e:
                print(f"pandas-ta KVO failed: {e}")
            
            try:
                df.ta.nvi(append=True)  # Negative Volume Index
            except Exception as e:
                print(f"pandas-ta NVI failed: {e}")
            
            try:
                df.ta.pvi(append=True)  # Positive Volume Index
            except Exception as e:
                print(f"pandas-ta PVI failed: {e}")
            
            try:
                df.ta.pvol(append=True)  # Price-Volume
            except Exception as e:
                print(f"pandas-ta PVOL failed: {e}")
            
            try:
                df.ta.pvr(append=True)  # Price Volume Rank
            except Exception as e:
                print(f"pandas-ta PVR failed: {e}")
            
            try:
                df.ta.pvt(append=True)  # Price Volume Trend
            except Exception as e:
                print(f"pandas-ta PVT failed: {e}")
            
            try:
                df.ta.vp(append=True)  # Volume Profile
            except Exception as e:
                print(f"pandas-ta VP failed: {e}")
        except Exception as e:
            print(f"Volume indicators section failed: {e}")
        
        # 10. Custom indicators
        try:
            # HA (Heikin-Ashi)
            try:
                ha_df = df.ta.ha(append=False)
                if ha_df is not None:
                    for col in ha_df.columns:
                        df[f'pta_{col}'] = ha_df[col]
            except Exception as e:
                print(f"pandas-ta HA failed: {e}")
        except Exception as e:
            print(f"Custom indicators section failed: {e}")
        
        # 11. Signals
        try:
            try:
                df.ta.tsignals(append=True)
            except Exception as e:
                print(f"pandas-ta TSIGNALS failed: {e}")
            
            try:
                df.ta.xsignals(append=True)
            except Exception as e:
                print(f"pandas-ta XSIGNALS failed: {e}")
        except Exception as e:
            print(f"Signal indicators section failed: {e}")
    
    # ==================== CUSTOM MANUAL INDICATORS (NON-DUPLICATES) ====================
    
    # Only add indicators not covered by TA-Lib or pandas-ta
    print("\nAdding custom manual indicators...")
    
    # Custom Price Statistics
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
    
    # Custom Weighted Moving Average (only if not in TA-Lib)
    if not talib_available:
        def calculate_wma(prices, window):
            weights = np.arange(1, window + 1)
            wma = prices.rolling(window=window, min_periods=1).apply(
                lambda x: np.dot(x, weights[-len(x):]) / weights[-len(x):].sum() if len(x) > 0 else x.mean()
            )
            return wma
        
        for window in [10, 20, 30]:
            df[f'custom_WMA_{window}'] = calculate_wma(df['Close'], window)
    
    # Fractal Adaptive Moving Average (FRAMA)
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
    
    # Custom Advanced Volume Features
    # Volume Weighted Average Price (VWAP) - Multiple Periods
    if not pandas_ta_available:
        cumulative_pv = (df['Volume'] * df['typical_price']).cumsum()
        cumulative_volume = df['Volume'].cumsum()
        df['custom_VWAP'] = cumulative_pv / cumulative_volume.replace(0, np.nan)
    
    # Rolling VWAP
    for period in [5, 10, 20, 30, 50, 100, 200]:
        rolling_pv = (df['Volume'] * df['typical_price']).rolling(window=period, min_periods=1).sum()
        rolling_volume = df['Volume'].rolling(window=period, min_periods=1).sum()
        df[f'custom_VWAP_{period}'] = rolling_pv / rolling_volume.replace(0, np.nan)
        df[f'custom_VWAP_{period}_ratio'] = df['Close'] / df[f'custom_VWAP_{period}'].replace(0, np.nan)
        df[f'custom_VWAP_{period}_distance'] = df['Close'] - df[f'custom_VWAP_{period}']
    
    # Custom Keltner Channels
    def calculate_atr_custom(df, period):
        high_low = df['High'] - df['Low']
        high_close = np.abs(df['High'] - df['Close'].shift())
        low_close = np.abs(df['Low'] - df['Close'].shift())
        ranges = pd.concat([high_low, high_close, low_close], axis=1)
        true_range = ranges.max(axis=1)
        return true_range.rolling(window=period, min_periods=1).mean()
    
    def calculate_keltner_channels(df, ema_period=20, atr_period=10, multiplier=2):
        ema = df['Close'].ewm(span=ema_period, adjust=False).mean()
        atr = calculate_atr_custom(df, atr_period)
        
        upper = ema + multiplier * atr
        lower = ema - multiplier * atr
        
        return ema, upper, lower
    
    for ema_period in [10, 20, 30, 50]:
        for multiplier in [1, 1.5, 2, 2.5, 3]:
            kc_ema, kc_upper, kc_lower = calculate_keltner_channels(df, ema_period=ema_period, multiplier=multiplier)
            df[f'custom_KC_middle_{ema_period}_{multiplier}'] = kc_ema
            df[f'custom_KC_upper_{ema_period}_{multiplier}'] = kc_upper
            df[f'custom_KC_lower_{ema_period}_{multiplier}'] = kc_lower
            df[f'custom_KC_width_{ema_period}_{multiplier}'] = kc_upper - kc_lower
            df[f'custom_KC_position_{ema_period}_{multiplier}'] = (df['Close'] - kc_lower) / (kc_upper - kc_lower).replace(0, np.nan)
    
    # Pivot Points (All Variations)
    def calculate_pivot_points(df):
        pivot = (df['High'].shift(1) + df['Low'].shift(1) + df['Close'].shift(1)) / 3
        
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
            '100': low
        }
        
        return fib_levels
    
    for period in [21, 34, 50, 89]:
        fib_levels = calculate_fib_levels(df, period)
        for level, values in fib_levels.items():
            df[f'fib_{level}_{period}'] = values
            df[f'fib_{level}_{period}_distance'] = df['Close'] - values
    
    # Smart Money Concepts (Enhanced)
    def enhanced_smc_analysis(df):
        """Advanced Smart Money Concepts"""
        
        # Market Structure Analysis
        swing_lookback = 10
        
        # Find swing points
        swing_highs = df['High'].rolling(swing_lookback*2+1, center=True).max() == df['High']
        swing_lows = df['Low'].rolling(swing_lookback*2+1, center=True).min() == df['Low']
        
        # Order Blocks (Enhanced)
        strong_move_threshold = df['Close'].pct_change().abs().rolling(20).std() * 2
        strong_bullish_move = df['Close'].pct_change() > strong_move_threshold
        strong_bearish_move = df['Close'].pct_change() < -strong_move_threshold
        
        bearish_candle = df['Close'] < df['Open']
        bullish_candle = df['Close'] > df['Open']
        
        bullish_ob = bearish_candle.shift(1) & strong_bullish_move
        bearish_ob = bullish_candle.shift(1) & strong_bearish_move
        
        # Fair Value Gap (FVG)
        fvg_bullish = (df['Low'].shift(-1) > df['High'].shift(1)) & (df['Close'] > df['Open'])
        fvg_bearish = (df['High'].shift(-1) < df['Low'].shift(1)) & (df['Close'] < df['Open'])
        
        # Premium/Discount Zones
        range_high = df['High'].rolling(50).max()
        range_low = df['Low'].rolling(50).min()
        range_mid = (range_high + range_low) / 2
        
        premium_zone = df['Close'] > range_mid + (range_high - range_mid) * 0.5
        discount_zone = df['Close'] < range_mid - (range_mid - range_low) * 0.5
        
        smc_features = {
            'smc_bullish_ob': bullish_ob.astype(int),
            'smc_bearish_ob': bearish_ob.astype(int),
            'smc_fvg_bullish': fvg_bullish.astype(int),
            'smc_fvg_bearish': fvg_bearish.astype(int),
            'smc_premium_zone': premium_zone.astype(int),
            'smc_discount_zone': discount_zone.astype(int),
            'smc_swing_high_liquidity': df['High'].rolling(50).max(),
            'smc_swing_low_liquidity': df['Low'].rolling(50).min()
        }
        
        return smc_features
    
    smc_features = enhanced_smc_analysis(df)
    for name, values in smc_features.items():
        df[name] = values
    
    # Order Flow Analysis
    def enhanced_order_flow_analysis(df):
        """Advanced Order Flow metrics"""
        
        price_move = df['Close'] - df['Open']
        typical_range = (df['High'] - df['Low']).rolling(20).mean()
        
        delta_ratio = price_move / typical_range.replace(0, np.nan)
        delta_ratio = delta_ratio.fillna(0).clip(-1, 1)
        
        buy_volume = df['Volume'] * (0.5 + 0.5 * delta_ratio)
        sell_volume = df['Volume'] * (0.5 - 0.5 * delta_ratio)
        delta = buy_volume - sell_volume
        
        cumulative_delta = delta.cumsum()
        
        order_flow_features = {
            'of_delta': delta,
            'of_cumulative_delta': cumulative_delta,
            'of_buy_volume': buy_volume,
            'of_sell_volume': sell_volume,
            'of_volume_delta_ratio': (buy_volume / sell_volume.replace(0, np.nan)).fillna(1)
        }
        
        return order_flow_features
    
    order_flow_features = enhanced_order_flow_analysis(df)
    for name, values in order_flow_features.items():
        df[name] = values
    
    # Hurst Exponent
    def calculate_hurst_exponent(prices, period=100):
        hurst_values = []
        
        for i in range(period, len(prices)):
            ts = prices.iloc[i-period:i].values
            
            lags = range(2, min(period//2, 20))
            tau = []
            
            for lag in lags:
                chunks = [ts[j:j+lag] for j in range(0, len(ts), lag)]
                chunks = [chunk for chunk in chunks if len(chunk) == lag]
                
                if not chunks:
                    continue
                
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
            
            if len(tau) > 1:
                log_lags = np.log(list(lags)[:len(tau)])
                log_tau = np.log(tau)
                hurst = np.polyfit(log_lags, log_tau, 1)[0]
                hurst_values.append(hurst)
            else:
                hurst_values.append(0.5)
        
        hurst_values = [0.5] * period + hurst_values
        return hurst_values
    
    df['hurst_exponent'] = calculate_hurst_exponent(df['Close'])
    
    # Historical Volatility
    for period in [10, 20, 30, 50, 100]:
        returns = df['Close'].pct_change()
        df[f'hist_volatility_{period}'] = returns.rolling(window=period).std() * np.sqrt(252)
    
    # Z-Score and Normal Distribution
    def calculate_bell_curve_features(df, period=50):
        price_mean = df['Close'].rolling(window=period, min_periods=1).mean()
        price_std = df['Close'].rolling(window=period, min_periods=1).std()
        
        z_score = (df['Close'] - price_mean) / price_std.replace(0, np.nan)
        
        if scipy_available:
            probability = stats.norm.cdf(z_score)
        else:
            probability = 0.5 * (1 + np.tanh(0.797884560803 * z_score))
        
        return z_score.fillna(0), probability
    
    for period in [20, 50, 100]:
        z_score, norm_probability = calculate_bell_curve_features(df, period)
        df[f'price_z_score_{period}'] = z_score
        df[f'price_norm_probability_{period}'] = norm_probability
        df[f'price_extreme_{period}'] = (abs(z_score) > 2).astype(int)
    
    # LAG FEATURES
    for lag in [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 12, 24, 48, 72, 96, 120]:
        df[f'close_lag_{lag}'] = df['Close'].shift(lag)
        df[f'return_lag_{lag}'] = df['price_change'].shift(lag)
        df[f'volume_lag_{lag}'] = df['Volume'].shift(lag)
        df[f'high_lag_{lag}'] = df['High'].shift(lag)
        df[f'low_lag_{lag}'] = df['Low'].shift(lag)
    
    # INTERACTION FEATURES
    df['price_volume_interaction'] = df['price_change'] * (df['Volume'] / df['Volume'].rolling(20).mean().replace(0, np.nan))
    df['price_volume_correlation'] = df['Close'].rolling(20).corr(df['Volume'])
    
    # Volume ratios
    df['volume_surge'] = df['Volume'] / df['Volume'].rolling(window=20, min_periods=1).mean().replace(0, np.nan)
    df['large_volume_flag'] = (df['volume_surge'] > 2).astype(int)
    df['extreme_volume_flag'] = (df['volume_surge'] > 3).astype(int)
    
    # Price features
    df['price_acceleration'] = df['price_change'].diff()
    df['price_jerk'] = df['price_acceleration'].diff()
    df['momentum_strength'] = df['price_change'].rolling(window=10, min_periods=1).mean()
    df['momentum_acceleration'] = df['momentum_strength'].diff()
    
    # Price ratios
    df['high_low_spread'] = (df['High'] - df['Low']) / df['Close']
    df['close_to_high'] = (df['High'] - df['Close']) / (df['High'] - df['Low']).replace(0, np.nan)
    df['close_to_low'] = (df['Close'] - df['Low']) / (df['High'] - df['Low']).replace(0, np.nan)
    
    # Seasonality Features (if datetime available)
    if 'datetime' in df.columns:
        dt = pd.to_datetime(df['datetime'])
        df['hour'] = dt.dt.hour
        df['day_of_week'] = dt.dt.dayofweek
        df['day_of_month'] = dt.dt.day
        df['month'] = dt.dt.month
        
        # Cyclical encoding
        df['hour_sin'] = np.sin(2 * np.pi * df['hour'] / 24)
        df['hour_cos'] = np.cos(2 * np.pi * df['hour'] / 24)
        df['day_sin'] = np.sin(2 * np.pi * df['day_of_week'] / 7)
        df['day_cos'] = np.cos(2 * np.pi * df['day_of_week'] / 7)
        df['month_sin'] = np.sin(2 * np.pi * df['month'] / 12)
        df['month_cos'] = np.cos(2 * np.pi * df['month'] / 12)
    
    # ==================== TARGET ====================
    df['target'] = df['Close'].shift(-1)
    
    # ==================== FINAL PROCESSING ====================
    
    print(f"\nCreated {len(df.columns)} features!")
    print("\nFeature categories created:")
    print("- Basic Features")
    print("- ALL TA-Lib indicators (without duplicates)")
    print("- Selected pandas-ta indicators (non-duplicates)")
    print("- Custom manual indicators (unique only)")
    print("- Price Statistics")
    print("- Advanced Volume Features")
    print("- Smart Money Concepts")
    print("- Order Flow Analysis")
    print("- Fibonacci Features")
    print("- Pivot Points")
    print("- Statistical Features")
    print("- Lag Features")
    print("- Interaction Features")
    print("- And more...")
    
    # Fill NaN values
    print("\nCleaning data...")
    # First, forward fill
    df = df.ffill()
    # Then, backward fill for any remaining NaN at the beginning
    df = df.bfill()
    # Finally, fill any remaining NaN with 0
    df = df.fillna(0)
    
    # Replace infinite values
    df = df.replace([np.inf, -np.inf], 0)
    
    # Drop rows where target is NaN (last row)
    df = df[df['target'].notna()]
    
    print(f"Data after cleaning: {len(df)} rows (removed {original_len - len(df)} rows)")
    print(f"Total features created: {len(df.columns) - 7}")  # Subtract original columns
    
    return df