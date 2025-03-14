import random
import pandas as pd


sp500_tickers = ['MMM', 'AOS', 'ABT', 'ABBV', 'ACN', 'ADBE', 'AMD', 'AES', 'AFL', 'A', 'APD', 'ABNB', 'AKAM', 'ALB', 'ARE', 'ALGN', 'ALLE', 'LNT', 'ALL', 'GOOGL', 'GOOG', 'MO', 'AMZN', 'AMCR', 'AEE', 'AEP', 'AXP', 'AIG', 'AMT', 'AWK', 'AMP', 'AME', 'AMGN', 'APH', 'ADI', 'ANSS', 'AON', 'APA', 'APO', 'AAPL', 'AMAT', 'APTV', 'ACGL', 'ADM', 'ANET', 'AJG', 'AIZ', 'T', 'ATO', 'ADSK', 'ADP', 'AZO', 'AVB', 'AVY', 'AXON', 'BKR', 'BALL', 'BAC', 'BAX', 'BDX', 'BRK.B', 'BBY', 'TECH', 'BIIB', 'BLK', 'BX', 'BK', 'BA', 'BKNG', 'BWA', 'BSX', 'BMY', 'AVGO', 'BR', 'BRO', 'BF.B', 'BLDR', 'BG', 'BXP', 'CHRW', 'CDNS', 'CZR', 'CPT', 'CPB', 'COF', 'CAH', 'KMX', 'CCL', 'CARR', 'CAT', 'CBOE', 'CBRE', 'CDW', 'CE', 'COR', 'CNC', 'CNP', 'CF', 'CRL', 'SCHW', 'CHTR', 'CVX', 'CMG', 'CB', 'CHD', 'CI', 'CINF', 'CTAS', 'CSCO', 'C', 'CFG', 'CLX', 'CME', 'CMS', 'KO', 'CTSH', 'CL', 'CMCSA', 'CAG', 'COP', 'ED', 'STZ', 'CEG', 'COO', 'CPRT', 'GLW', 'CPAY', 'CTVA', 'CSGP', 'COST', 'CTRA', 'CRWD', 'CCI', 'CSX', 'CMI', 'CVS', 'DHR', 'DRI', 'DVA', 'DAY', 'DECK', 'DE', 'DELL', 'DAL', 'DVN', 'DXCM', 'FANG', 'DLR', 'DFS', 'DG', 'DLTR', 'D', 'DPZ', 'DOV', 'DOW', 'DHI', 'DTE', 'DUK', 'DD', 'EMN', 'ETN', 'EBAY', 'ECL', 'EIX', 'EW', 'EA', 'ELV', 'EMR', 'ENPH', 'ETR', 'EOG', 'EPAM', 'EQT', 'EFX', 'EQIX', 'EQR', 'ERIE', 'ESS', 'EL', 'EG', 'EVRG', 'ES', 'EXC', 'EXPE', 'EXPD', 'EXR', 'XOM', 'FFIV', 'FDS', 'FICO', 'FAST', 'FRT', 'FDX', 'FIS', 'FITB', 'FSLR', 'FE', 'FI', 'FMC', 'F', 'FTNT', 'FTV', 'FOXA', 'FOX', 'BEN', 'FCX', 'GRMN', 'IT', 'GE', 'GEHC', 'GEV', 'GEN', 'GNRC', 'GD', 'GIS', 'GM', 'GPC', 'GILD', 'GPN', 'GL', 'GDDY', 'GS', 'HAL', 'HIG', 'HAS', 'HCA', 'DOC', 'HSIC', 'HSY', 'HES', 'HPE', 'HLT', 'HOLX', 'HD', 'HON', 'HRL', 'HST', 'HWM', 'HPQ', 'HUBB', 'HUM', 'HBAN', 'HII', 'IBM', 'IEX', 'IDXX', 'ITW', 'INCY', 'IR', 'PODD', 'INTC', 'ICE', 'IFF', 'IP', 'IPG', 'INTU', 'ISRG', 'IVZ', 'INVH', 'IQV', 'IRM', 'JBHT', 'JBL', 'JKHY', 'J', 'JNJ', 'JCI', 'JPM', 'JNPR', 'K', 'KVUE', 'KDP', 'KEY', 'KEYS', 'KMB', 'KIM', 'KMI', 'KKR', 'KLAC', 'KHC', 'KR', 'LHX', 'LH', 'LRCX', 'LW', 'LVS', 'LDOS', 'LEN', 'LII', 'LLY', 'LIN', 'LYV', 'LKQ', 'LMT', 'L', 'LOW', 'LULU', 'LYB', 'MTB', 'MPC', 'MKTX', 'MAR', 'MMC', 'MLM', 'MAS', 'MA', 'MTCH', 'MKC', 'MCD', 'MCK', 'MDT', 'MRK', 'META', 'MET', 'MTD', 'MGM', 'MCHP', 'MU', 'MSFT', 'MAA', 'MRNA', 'MHK', 'MOH', 'TAP', 'MDLZ', 'MPWR', 'MNST', 'MCO', 'MS', 'MOS', 'MSI', 'MSCI', 'NDAQ', 'NTAP', 'NFLX', 'NEM', 'NWSA', 'NWS', 'NEE', 'NKE', 'NI', 'NDSN', 'NSC', 'NTRS', 'NOC', 'NCLH', 'NRG', 'NUE', 'NVDA', 'NVR', 'NXPI', 'ORLY', 'OXY', 'ODFL', 'OMC', 'ON', 'OKE', 'ORCL', 'OTIS', 'PCAR', 'PKG', 'PLTR', 'PANW', 'PARA', 'PH', 'PAYX', 'PAYC', 'PYPL', 'PNR', 'PEP', 'PFE', 'PCG', 'PM', 'PSX', 'PNW', 'PNC', 'POOL', 'PPG', 'PPL', 'PFG', 'PG', 'PGR', 'PLD', 'PRU', 'PEG', 'PTC', 'PSA', 'PHM', 'PWR', 'QCOM', 'DGX', 'RL', 'RJF', 'RTX', 'O', 'REG', 'REGN', 'RF', 'RSG', 'RMD', 'RVTY', 'ROK', 'ROL', 'ROP', 'ROST', 'RCL', 'SPGI', 'CRM', 'SBAC', 'SLB', 'STX', 'SRE', 'NOW', 'SHW', 'SPG', 'SWKS', 'SJM', 'SW', 'SNA', 'SOLV', 'SO', 'LUV', 'SWK', 'SBUX', 'STT', 'STLD', 'STE', 'SYK', 'SMCI', 'SYF', 'SNPS', 'SYY', 'TMUS', 'TROW', 'TTWO', 'TPR', 'TRGP', 'TGT', 'TEL', 'TDY', 'TFX', 'TER', 'TSLA', 'TXN', 'TPL', 'TXT', 'TMO', 'TJX', 'TSCO', 'TT', 'TDG', 'TRV', 'TRMB', 'TFC', 'TYL', 'TSN', 'USB', 'UBER', 'UDR', 'ULTA', 'UNP', 'UAL', 'UPS', 'URI', 'UNH', 'UHS', 'VLO', 'VTR', 'VLTO', 'VRSN', 'VRSK', 'VZ', 'VRTX', 'VTRS', 'VICI', 'V', 'VST', 'VMC', 'WRB', 'GWW', 'WAB', 'WBA', 'WMT', 'DIS', 'WBD', 'WM', 'WAT', 'WEC', 'WFC', 'WELL', 'WST', 'WDC', 'WY', 'WMB', 'WTW', 'WDAY', 'WYNN', 'XEL', 'XYL', 'YUM', 'ZBRA', 'ZBH', 'ZTS']

def generate_scenario(n:int, seed = 42):
    # Number of stocks to select
    
    # random.seed(seed%100)

    # Randomly select n stocks from the S&P 500 tickers
    selected_tickers = random.sample(sp500_tickers, n)

    # Initialize an empty DataFrame to store the concatenated data
    concatenated_df = pd.DataFrame()

    # Iterate over the selected tickers
    for i, ticker in enumerate(selected_tickers):
        try:
            # Read the derived data for the ticker
            df = pd.read_csv(f'./data/derived/{ticker.upper()}_der_data.csv')
            # Drop rows with any NaN values
            if df.isna().any().any():
                print(f'Skipping {ticker} due to NaN values')
                return generate_scenario(10)
            
            # Rename the columns to include the stock identifier
            df = df.rename(columns={
                'return': f'S{i+1}_return',
                'ma': f'S{i+1}_ma',
                'vol': f'S{i+1}_vol',
                'rvol': f'S{i+1}_rvol'
            })
            
            # Set 'Price' as the index for joining on the same date
            df['Date'] = pd.to_datetime(df['Price'])
            df = df.set_index('Date')
            df = df.drop(columns=['Unnamed: 0', 'Price'], errors='ignore')
            # If concatenated_df is empty, initialize it with the current df
            if concatenated_df.empty:
                concatenated_df = df
            else:
                # Perform an outer join with the current df on the 'Date' index
                concatenated_df = concatenated_df.join(df, how='outer')
        
        except Exception as e:
            print(f'Error processing {ticker}: {e}')
            return generate_scenario(10)

    # Drop any columns that are duplicated due to the index or unnamed columns
    # concatenated_df = concatenated_df.loc[:, ~concatenated_df.columns.duplicated()]

    # Filter the data to leave only consecutive one year data
    # Assuming the data is sorted by date and the 'Date' column is present
    concatenated_df = concatenated_df.reset_index()
    # Randomly select a start date between 2009-01-01 and 2022-12-31
    start_date = pd.to_datetime(random.randint(pd.Timestamp('2009-01-01').value, pd.Timestamp('2022-12-31').value))
    # Calculate the end date to be one year after the start date
    end_date = start_date + pd.DateOffset(years=1) - pd.DateOffset(days=1)
    # Define the start and end date for the randomly selected year
    # start_date = f'{random_year}-01-01'
    # end_date = f'{random_year}-12-31'

    # Filter the DataFrame for the specified one-year period
    one_year_data = concatenated_df.loc[(concatenated_df['Date'] >= start_date) & (concatenated_df['Date'] <= end_date)]
    
    # Check if the number of rows is fewer than 250
    if len(one_year_data) < 250:
        # Truncate the DataFrame
        one_year_data = pd.DataFrame()
        return generate_scenario(10)
    # Read the macro data
    macro_df = pd.read_csv('./data/macro_15yr_data.csv')
    
    # Convert 'Date' column to datetime
    macro_df['Date'] = pd.to_datetime(macro_df['Date'])
    
    # Set 'Date' as the index for joining
    macro_df = macro_df.set_index('Date')
    
    # Perform an inner join with the macro data on the 'Date' index
    one_year_data = one_year_data.set_index('Date').join(macro_df, how='inner').reset_index()
    
    # Check for NA values
    if one_year_data.isna().any().any():
        return generate_scenario(10)
    
    # 데이터 정규화/표준화를 위한 코드 추가
    if not one_year_data.empty:
        # 주식 데이터 스케일 조정
        for i in range(1, n+1):
            # return과 ma는 *100
            return_col = f'S{i}_return'
            ma_col = f'S{i}_ma'
            one_year_data[return_col] *= 100
            one_year_data[ma_col] *= 100
            
            # vol도 *100
            vol_col = f'S{i}_vol'
            one_year_data[vol_col] *= 100
        
        # 데이터 셔플링
        stock_groups = []
        for i in range(1, n+1):
            group_columns = [f'S{i}_return', f'S{i}_ma', f'S{i}_vol', f'S{i}_rvol']
            stock_groups.append(group_columns)
        
        # 주식 그룹을 랜덤하게 섞음
        random.seed(seed)
        random.shuffle(stock_groups)
        
        # 섞인 순서대로 새로운 컬럼 이름 생성
        new_columns = []
        for i, group in enumerate(stock_groups, 1):
            new_names = [f'S{i}_return', f'S{i}_ma', f'S{i}_vol', f'S{i}_rvol']
            new_columns.extend(list(zip(group, new_names)))
        
        # 컬럼 이름 변경
        for old_name, new_name in new_columns:
            one_year_data = one_year_data.rename(columns={old_name: new_name + '_temp'})
        for old_name, new_name in new_columns:
            one_year_data = one_year_data.rename(columns={new_name + '_temp': new_name})
        
        # 날짜 순서대로 정렬
        one_year_data = one_year_data.sort_values('Date').reset_index(drop=True)
    
    return one_year_data
    

if __name__ == '__main__':
    # data = None
    # while data is None:
    data = generate_scenario(10, 42)
    print(data)