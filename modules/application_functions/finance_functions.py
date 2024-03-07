import numpy as np

def get_Daily_Volatility(close,span0=20):
    # simple percentage returns
    df0=close.pct_change()
    # 20 days, a month EWM's std as boundary
    df0=df0.ewm(span=span0).std()
    df0.dropna(inplace=True)
    return df0

def yfinance_labeler(df, hold_period, upper_lower_multipliers):
    """Creates a buy/sell signal as a new column containing a 1/0 label, inspired by the triple-barrier-method.

    Parameters
    ----------
    df
        Dataframe with daily prices
    hold_period
        Time-span to evaluate the buy/sell signal
    upper_lower_multipliers
        Factor of volatility by which the price must rise or fall within the hold_period in order to generate a buy/sell signal

    Returns
    -------
    The dataframe df with a new column called "label" containing the signals.
    """

    vol_df = get_Daily_Volatility(df.Close)
    df = df.reindex(vol_df.index)
    prices = df["Close"]
    labels = []
    upper_bounds = []
    lower_bounds = []

    for row in range(0,len(vol_df)):
        if ((len(vol_df)-row) - hold_period) > 0:
            upper_bound = prices[row] + prices[row] * upper_lower_multipliers[0] * vol_df[row]
            upper_bounds.append(upper_bound)
            lower_bound = prices[row] - prices[row] * upper_lower_multipliers[1] * vol_df[row]
            lower_bounds.append(lower_bound)
            
            pot_bucket = np.array(list(prices[row+1: row+1+hold_period]))
            index_up_min = None
            index_low_min = None

            if (pot_bucket>=upper_bound).any():
                index_up_min = np.argmax(pot_bucket>=upper_bound)
            if (pot_bucket<=lower_bound).any():
                index_low_min = np.argmax(pot_bucket<=lower_bound)

            if (index_up_min is not None) & (index_low_min is not None):
                if index_up_min<index_low_min:
                    labels.append(1)
                else:
                    labels.append(0)
            elif index_up_min is not None:
                labels.append(1)
            elif index_low_min is not None:
                labels.append(0)
            else:
                labels.append(0)
        else:
            labels.append(12)
    df["vola"] = vol_df
    df["label"] = labels

    df_output = df[df["label"] != 12]
    df_output["upper_bound"] = upper_bounds
    df_output["lower_bound"] = lower_bounds

    return(df_output)

def ichimoku(full_df):
    """Creates indicator variables out of the Ichimoku Kinko Hyo system.

    Parameters
    ----------
    df_full
        Dataframe with daily prices

    Returns
    -------
    The dataframe df_full with new indicator columns.
    """

    nine_period_high = full_df['High'].rolling(window= 9).max()
    nine_period_low = full_df['Low'].rolling(window= 9).min()
    tenkan_sen = (nine_period_high + nine_period_low) /2
    full_df['tenkan_sen_perc'] = tenkan_sen.pct_change(1)
    # Kijun-sen (Base Line): (26-period high + 26-period low)/2))
    period26_high = full_df['High'].rolling(window=26).max()
    period26_low = full_df['Low'].rolling(window=26).min()
    kijun_sen = (period26_high + period26_low) / 2
    full_df['kijun_sen_perc'] = kijun_sen.pct_change(1)
    # Senkou Span A (Leading Span A): (Conversion Line + Base Line)/2))
    senkou_span_a = ((tenkan_sen + kijun_sen) / 2).shift(26)
    full_df['senkou_span_a_perc'] = senkou_span_a.pct_change(1)
    # Senkou Span B (Leading Span B): (52-period high + 52-period low)/2))
    period52_high = full_df['High'].rolling(window=52).max()
    period52_low = full_df['Low'].rolling(window=52).min()
    senkou_span_b = ((period52_high + period52_low) / 2).shift(26)
    full_df['senkou_span_b_perc'] = senkou_span_b.pct_change(1)

    var_list = [tenkan_sen, kijun_sen, senkou_span_a, senkou_span_b]
    names_list = ["tenkan_sen", "kijun_sen", "senkou_span_a", "senkou_span_b"]
    for var, name in zip(var_list,names_list):
        colname = name+"_higher_close"
        full_df[colname] = [1 if a > b else 0 for a, b in zip(var, list(full_df["Close"]))]
        
    full_df["tenkan_sen_higher_kijun_sen"] = [1 if a > b else 0 for a, b in zip(tenkan_sen, kijun_sen)]
    full_df["tenkan_sen_kijun_sen_DIFF"] = tenkan_sen - kijun_sen

    full_df["senkou_span_a_higher_senkou_span_b"] = [1 if a > b else 0 for a, b in zip(senkou_span_a, senkou_span_b)]
    full_df["senkou_span_a_senkou_span_b_DIFF"] = senkou_span_a - senkou_span_b

    full_df["tenkan_sen_higher_cloud"] = [1 if a > b else 0 for a, b in zip(tenkan_sen, [max(c, d) for c, d in zip(senkou_span_a, senkou_span_b)])]
    full_df["kijun_sen_higher_cloud"] = [1 if a > b else 0 for a, b in zip(kijun_sen, [max(c, d) for c, d in zip(senkou_span_a, senkou_span_b)])]

    full_df["tenkan_sen_cloudDIFF"] = tenkan_sen - [max(c, d) for c, d in zip(senkou_span_a, senkou_span_b)]
    full_df["kijun_sen_cloudDIFF"] = kijun_sen - [max(c, d) for c, d in zip(senkou_span_a, senkou_span_b)]

    full_df["positive_crossover"] = [1 if x >= 0 and full_df["tenkan_sen_kijun_sen_DIFF"][i - 1] < 0 else 0 for i, x in enumerate(full_df["tenkan_sen_kijun_sen_DIFF"])]
    full_df["crossover"] = [1 if (x >= 0 and full_df["tenkan_sen_kijun_sen_DIFF"][i - 1] < 0) or (x <= 0 and full_df["tenkan_sen_kijun_sen_DIFF"][i - 1] > 0) else 0 for i, x in enumerate(full_df["tenkan_sen_kijun_sen_DIFF"])]

    crossover_above = []
    for val1, val2, val3 in zip([min(c, d) for c, d in zip(tenkan_sen, kijun_sen)], [max(c, d) for c, d in zip(senkou_span_a, senkou_span_b)], full_df["crossover"]):
        if val3 == 1 and val1 > val2:
            crossover_above.append(1)
        else:
            crossover_above.append(0)

    full_df["crossover_above"] = crossover_above

    crossover_below = []
    for val1, val2, val3 in zip([min(c, d) for c, d in zip(tenkan_sen, kijun_sen)], [min(c, d) for c, d in zip(senkou_span_a, senkou_span_b)], full_df["crossover"]):
        if val3 == 1 and val1 < val2:
            crossover_below.append(1)
        else:
            crossover_below.append(0)

    full_df["crossover_below"] = crossover_below

    full_df["positive_crossover_cloud"] = [1 if x >= 0 and full_df["senkou_span_a_senkou_span_b_DIFF"][i - 1] < 0 else 0 for i, x in enumerate(full_df["senkou_span_a_senkou_span_b_DIFF"])]
    full_df["crossover_cloud"] = [1 if (x >= 0 and full_df["senkou_span_a_senkou_span_b_DIFF"][i - 1] < 0) or (x <= 0 and full_df["senkou_span_a_senkou_span_b_DIFF"][i - 1] > 0) else 0 for i, x in enumerate(full_df["senkou_span_a_senkou_span_b_DIFF"])]

    return full_df