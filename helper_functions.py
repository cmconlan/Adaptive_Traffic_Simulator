from math import radians, sin, cos, sqrt, atan2
from matplotlib import pyplot as plt
from statsmodels.tsa.stattools import adfuller
import pandas as pd
from statsmodels.tsa.seasonal import seasonal_decompose
import matplotlib.dates as mdates
from matplotlib.ticker import MultipleLocator
import numpy as np

#From ChatGPT
def traversal_time(length_meters, speed_mph):
    # Conversion factor: 1 mile per hour = 1609.34 meters per minute
    conversion_factor = 1609.34 / 60
    
    # Calculate time in minutes
    time_minutes = length_meters / (speed_mph * conversion_factor)
    
    return time_minutes

def haversine_distance(point1, point2):
    # Radius of the Earth in kilometers
    R = 6371.0

    # Extract latitude and longitude coordinates from the point objects
    lat1, lon1 = radians(point1.y), radians(point1.x)
    lat2, lon2 = radians(point2.y), radians(point2.x)

    # Calculate the differences in latitude and longitude
    dlat = lat2 - lat1
    dlon = lon2 - lon1

    # Haversine formula to calculate distance
    a = sin(dlat / 2)**2 + cos(lat1) * cos(lat2) * sin(dlon / 2)**2
    c = 2 * atan2(sqrt(a), sqrt(1 - a))

    # Calculate the distance
    distance = R * c

    return distance


# Function to perform the Augmented Dickey-Fuller test for stationarity
def test_stationarity(timeseries, var):
    # Calculate rolling statistics
    rolmean = timeseries[var].rolling(window=30).mean()
    rolstd = timeseries[var].rolling(window=30).std()

    # Plot rolling statistics
    plt.figure(figsize=(12, 4))
    plt.plot(timeseries[var], label='Original')
    plt.plot(rolmean, label='Rolling Mean')
    plt.plot(rolstd, label='Rolling Std')
    plt.title('Rolling Mean and Standard Deviation')
    plt.legend()
    plt.show()

    # Perform Dickey-Fuller test
    result = adfuller(timeseries[var], autolag='AIC')
    print('Augmented Dickey-Fuller Test:')
    print(f'Test Statistic: {result[0]}')
    print(f'p-value: {result[1]}')
    print(f'Critical Values: {result[4]}')

def get_ts(sensor_id,start_date,end_date,cursor):
    
    dec_to_min_dict = {
    0.0 : 0,
    0.25 : 15,
    0.50 : 30,
    0.75 : 45
    }
    
    # Get time series data for sensor
    sql_query = "select * from full_data where site_ID = '{}' and yr = '2022'".format(sensor_id)
    cursor.execute(sql_query)
    result = cursor.fetchall()
    ts = pd.DataFrame(result, columns=[desc[0] for desc in cursor.description])

    # Add date/time index

    i, d = divmod((ts['time_period'] / 4), 1)
    ts['Minute'] = [*map(dec_to_min_dict.get, list(d))]

    years = []
    months = []
    days = []

    for i,r in ts.iterrows():
        years.append(r['data_date'].year)
        months.append(r['data_date'].month)
        days.append(r['data_date'].day)

    ts['Year'] = years
    ts['Month'] = months
    ts['Day'] = days
    ts['Hour'] = ts['data_hour']

    ts['Datetime'] = pd.to_datetime(ts[['Year', 'Month', 'Day', 'Hour', 'Minute']])
    ts.set_index('Datetime', inplace=True)
    ts = ts.sort_index()

    # Create a datetime index with 15-minute frequency
    datetime_index = pd.date_range(start=start_date, end=end_date, freq='15T')
    ts_columns = ['Lane 1', 'Lane 2', 'Lane 3', 'All Lanes']

    # Create all vehicles data frame with 3 lanes
    ts_all = pd.DataFrame(index = datetime_index, columns = ts_columns)

    ts_all['Lane 1'] = ts[ts['Lane'] == 'lane1']['Total_Flow']
    ts_all['Lane 2'] = ts[ts['Lane'] == 'lane2']['Total_Flow']
    ts_all['Lane 3'] = ts[ts['Lane'] == 'lane3']['Total_Flow']
    ts_all['All Lanes'] = ts.groupby(ts.index).sum('Total_Flow')['Total_Flow']

    #Add Weekend Flag
    ts_all['Weekend'] = ts_all.index.dayofweek >= 5
    
    #Interpolate missing data
    for col in ts_columns:
        variable_name = f"{col}_interpolated"
        ts_all[variable_name] = ts_all[col].interpolate(method='time')
        ts_all[variable_name].bfill(inplace=True)
    
    return ts, ts_all
    
def analyse_time_series_of_sensor(sensor_id,start_date,end_date,cursor):
    
    dec_to_min_dict = {
    0.0 : 0,
    0.25 : 15,
    0.50 : 30,
    0.75 : 45
    }
    
    # Get time series data for sensor
    sql_query = "select * from full_data where site_ID = '{}' and yr = '2022'".format(sensor_id)
    cursor.execute(sql_query)
    result = cursor.fetchall()
    ts = pd.DataFrame(result, columns=[desc[0] for desc in cursor.description])

    # Add date/time index

    i, d = divmod((ts['time_period'] / 4), 1)
    ts['Minute'] = [*map(dec_to_min_dict.get, list(d))]

    years = []
    months = []
    days = []

    for i,r in ts.iterrows():
        years.append(r['data_date'].year)
        months.append(r['data_date'].month)
        days.append(r['data_date'].day)

    ts['Year'] = years
    ts['Month'] = months
    ts['Day'] = days
    ts['Hour'] = ts['data_hour']

    ts['Datetime'] = pd.to_datetime(ts[['Year', 'Month', 'Day', 'Hour', 'Minute']])
    ts.set_index('Datetime', inplace=True)
    ts = ts.sort_index()

    # Create a datetime index with 15-minute frequency
    datetime_index = pd.date_range(start=start_date, end=end_date, freq='15T')
    ts_columns = ['Lane 1', 'Lane 2', 'Lane 3', 'All Lanes']

    # Create all vehicles data frame with 3 lanes
    ts_all = pd.DataFrame(index = datetime_index, columns = ts_columns)

    ts_all['Lane 1'] = ts[ts['Lane'] == 'lane1']['Total_Flow']
    ts_all['Lane 2'] = ts[ts['Lane'] == 'lane2']['Total_Flow']
    ts_all['Lane 3'] = ts[ts['Lane'] == 'lane3']['Total_Flow']
    ts_all['All Lanes'] = ts.groupby(ts.index).sum('Total_Flow')['Total_Flow']

    #Add Weekend Flag
    ts_all['Weekend'] = ts_all.index.dayofweek >= 5
    
    #Data Quality Checks
    # For each column - count zeros, nulls and outliers
    len_ts = len(ts_all)

    for col in ts_columns:
        print('-----------------')
        print('Next Columns : {}'.format(col))
        #Count zeros
        zero_count = (ts_all[col] == 0).sum()
        print('Count of zeros : {}'.format(zero_count))
        print('Rate of zeros : {:.1%}'.format(zero_count / len_ts))

        # Count nulls
        null_count = ts_all[col].isnull().sum()
        print('Count of nulls : {}'.format(null_count))
        print('Rate of nulls : {:.1%}'.format(null_count / len_ts))

        # Identify and count outliers (assuming values more than 2 standard deviations from the mean are outliers)
        mean_value = ts_all[col].mean()
        std_dev = ts_all[col].std()
        outlier_count = ((ts_all[col] > mean_value + 2 * std_dev) | (ts_all[col] < mean_value - 2 * std_dev)).sum()
        print('Count of outliers : {}'.format(outlier_count))
        print('Rate of outliers : {:.1%}'.format(outlier_count / len_ts))
        print('-----------------')
        print()
        print()
        
    plt.plot(ts_all['All Lanes'])
    plt.ylabel('Traffic Count')
    plt.show()
    
    #Interpolate missing data
    for col in ts_columns:
        variable_name = f"{col}_interpolated"
        ts_all[variable_name] = ts_all[col].interpolate(method='time')
        ts_all[variable_name].bfill(inplace=True)
        
        
    fig, ax = plt.subplots(2,1, figsize = [20,6])
    x_position = 0
    xlabels = []
    for i in range(1,53):
        week_data = ts_all[ts_all.index.isocalendar().week == i]
        ax[0].boxplot(week_data[week_data['Weekend'] == False]['All Lanes'].dropna(), positions=[x_position])
        x_position += 1
        xlabels.append(week_data.index[0].strftime('%m-%d'))

    ax[0].set_xticks(np.arange(len(xlabels)))
    ax[0].set_xticklabels(xlabels)
    ax[0].set_title('Weekly Boxplots (Weekday)')
    ax[0].set_xlabel('Date')
    ax[0].set_ylabel('Traffic Count')
    ax[0].xaxis.set_major_locator(MultipleLocator(2))

    x_position = 0
    xlabels = []
    for i in range(1,53):
        week_data = ts_all[ts_all.index.isocalendar().week == i]
        ax[1].boxplot(week_data[week_data['Weekend'] == True]['All Lanes'].dropna(), positions=[x_position])
        x_position += 1
        xlabels.append(week_data.index[0].strftime('%m-%d'))
    ax[1].set_title('Weekly Boxplots (Weekend)')
    ax[1].set_xticks(np.arange(len(xlabels)))
    ax[1].set_xticklabels(xlabels)
    ax[1].set_xlabel('Date')
    ax[1].set_ylabel('Traffic Count')
    ax[1].xaxis.set_major_locator(MultipleLocator(2))

    plt.tight_layout()
    plt.show()
    
    #Typical Day Analysis

    fig, ax = plt.subplots(4,3, figsize = [20,8])

    ax_across = 0
    ax_down = 0

    for m in range(1,13):
        
        x_position = 0
        xlabels = []
        for h in range(24):
            
            month_hour_data = ts_all[(ts_all.index.month == m) & (ts_all.index.hour == h)]
            ax[ax_down,ax_across].boxplot(month_hour_data[month_hour_data['Weekend'] == True]['All Lanes'].dropna(), positions=[x_position])
            x_position += 1
            xlabels.append(month_hour_data.index[0].strftime('%H:%M'))
        ax[ax_down,ax_across].set_xticks(np.arange(len(xlabels)))
        ax[ax_down,ax_across].set_xticklabels(xlabels)
        ax[ax_down,ax_across].xaxis.set_major_locator(MultipleLocator(4))
        
        ax[ax_down,ax_across].set_title(month_hour_data.index[0].strftime('%B'))
        
        if ax_across < 2:
            ax_across += 1
        else:
            ax_across = 0
            ax_down += 1

    plt.suptitle('Typical Weekend Day by Month')
    plt.tight_layout()
    plt.show()


    fig, ax = plt.subplots(4,3, figsize = [20,8])

    ax_across = 0
    ax_down = 0

    for m in range(1,13):
        
        x_position = 0
        xlabels = []
        for h in range(24):
            month_hour_data = ts_all[(ts_all.index.month == m) & (ts_all.index.hour == h)]
            ax[ax_down,ax_across].boxplot(month_hour_data[month_hour_data['Weekend'] == False]['All Lanes'].dropna(), positions=[x_position])
            x_position += 1
            xlabels.append(month_hour_data.index[0].strftime('%H:%M'))
        ax[ax_down,ax_across].set_xticks(np.arange(len(xlabels)))
        ax[ax_down,ax_across].set_xticklabels(xlabels)
        ax[ax_down,ax_across].xaxis.set_major_locator(MultipleLocator(4))
        
        ax[ax_down,ax_across].set_title(month_hour_data.index[0].strftime('%B'))
        
        if ax_across < 2:
            ax_across += 1
        else:
            ax_across = 0
            ax_down += 1

    plt.suptitle('Typical Weekday Day by Month')
    plt.tight_layout()
    plt.show()
    
    
    # Perform seasonal decomposition
    result = seasonal_decompose(ts_all['All Lanes_interpolated'], model='additive', period=365)

    # Plot the original, trend, seasonal, and residual components
    plt.figure(figsize=(12, 8))
    plt.subplot(4, 1, 1)
    plt.plot(ts_all['All Lanes'], label='Original')
    plt.legend()

    plt.subplot(4, 1, 2)
    plt.plot(result.trend, label='Trend')
    plt.legend()

    plt.subplot(4, 1, 3)
    plt.plot(result.seasonal, label='Seasonal')
    plt.legend()

    plt.subplot(4, 1, 4)
    plt.plot(result.resid, label='Residual')
    plt.legend()

    plt.suptitle('Seasonal Decomposition Analysis')
    plt.tight_layout()
    plt.show()
    
    # Call the stationarity analysis function
    test_stationarity(ts_all, 'All Lanes_interpolated')
    
    #Heatmap of flow over time
    
    #Construct heapmap

    num_days = 365
    heatmap = np.zeros((24,num_days))

    m = 1
    d = 1
    array_ind = 0
    for i in range(num_days):
        month_day_data = ts_all[(ts_all.index.month == m) & (ts_all.index.day == d)]
        if len(month_day_data) > 0:
            hourly_aggregated = month_day_data.resample('H').sum()
            heatmap[:,array_ind] = hourly_aggregated['All Lanes_interpolated'].values

            array_ind += 1
            d += 1
        else:
            m += 1
            d = 1
            month_day_data = ts_all[(ts_all.index.month == m) & (ts_all.index.day == d)]
            hourly_aggregated = month_day_data.resample('H').sum()
            heatmap[:,array_ind] = hourly_aggregated['All Lanes_interpolated'].values
            array_ind += 1
    xlabels = pd.date_range(start=start_date, periods=num_days, freq='D')
    ylabels = hourly_aggregated.sort_index( ascending=False).index.strftime('%H:%M')

    # Specify the size of the plot using figsize
    fig, ax = plt.subplots(figsize=(30, 8))

    # Plot the heatmap
    im = ax.imshow(np.flip(heatmap, axis = 0), cmap='viridis', interpolation='nearest')
    plt.colorbar(im, ax=ax)  # Add a colorbar on the side
    ax.set_title('Hourly Flow Heat Map')
    ax.set_xticks(np.arange(len(xlabels)))
    ax.set_xticklabels(xlabels.strftime('%B'))
    ax.xaxis.set_major_locator(mdates.MonthLocator(interval=1,bymonthday=15))

    ax.set_yticks(np.arange(len(ylabels)))
    ax.set_yticklabels(ylabels)
    ax.yaxis.set_major_locator(MultipleLocator(4))

    plt.show()
    
    #Plot Data Quality Over Time
    
    mean_value_l1 = ts_all['Lane 1'].mean()
    std_dev_l1 = ts_all['Lane 1'].std()

    mean_value_l2 = ts_all['Lane 2'].mean()
    std_dev_l2 = ts_all['Lane 2'].std()

    mean_value_l3 = ts_all['Lane 3'].mean()
    std_dev_l3 = ts_all['Lane 3'].std()

    mean_value_lall = ts_all['All Lanes'].mean()
    std_dev_lall = ts_all['All Lanes'].std()

    num_days = 365
    zero_counts = np.zeros((4,num_days))
    null_counts = np.zeros((4,num_days))
    outlier_counts = np.zeros((4,num_days))

    m = 1
    d = 1
    array_ind = 0

    for i in range(num_days):

        day_data = ts_all[(ts_all.index.month == m) & (ts_all.index.day == d)]
        if len(day_data) > 0:
            zero_counts[0,array_ind] = ((day_data['Lane 1'] == 0).sum() / len(day_data)) * 100
            zero_counts[1,array_ind] = ((day_data['Lane 2'] == 0).sum() / len(day_data)) * 100
            zero_counts[2,array_ind] = ((day_data['Lane 3'] == 0).sum() / len(day_data)) * 100
            zero_counts[3,array_ind] = ((day_data['All Lanes'] == 0).sum() / len(day_data)) * 100

            null_counts[0,array_ind] = (day_data['Lane 1'].isnull().sum() / len(day_data)) * 100
            null_counts[1,array_ind] = (day_data['Lane 2'].isnull().sum() / len(day_data)) * 100
            null_counts[2,array_ind] = (day_data['Lane 3'].isnull().sum() / len(day_data)) * 100
            null_counts[3,array_ind] = (day_data['All Lanes'].isnull().sum() / len(day_data)) * 100

            outlier_counts[0,array_ind] = (((day_data['Lane 1'] > mean_value_l1 + 2 * std_dev_l1) | (day_data['Lane 1'] < mean_value_l1 - 2 * std_dev_l1)).sum()) / len(day_data) * 100
            outlier_counts[1,array_ind] = (((day_data['Lane 2'] > mean_value_l2 + 2 * std_dev_l2) | (day_data['Lane 1'] < mean_value_l2 - 2 * std_dev_l2)).sum()) / len(day_data) * 100
            outlier_counts[2,array_ind] = (((day_data['Lane 3'] > mean_value_l3 + 2 * std_dev_l3) | (day_data['Lane 1'] < mean_value_l3 - 2 * std_dev_l3)).sum()) / len(day_data) * 100
            outlier_counts[3,array_ind] = (((day_data['All Lanes'] > mean_value_lall + 2 * std_dev_lall) | (day_data['All Lanes'] < mean_value_lall - 2 * std_dev_lall)).sum()) / len(day_data) * 100
            array_ind += 1
            d += 1
            
        else:
            m += 1
            d = 1
            day_data = ts_all[(ts_all.index.month == m) & (ts_all.index.day == d)]
            zero_counts[0,array_ind] = ((day_data['Lane 1'] == 0).sum() / len(day_data)) * 100
            zero_counts[1,array_ind] = ((day_data['Lane 2'] == 0).sum() / len(day_data)) * 100
            zero_counts[2,array_ind] = ((day_data['Lane 3'] == 0).sum() / len(day_data)) * 100
            zero_counts[3,array_ind] = ((day_data['All Lanes'] == 0).sum() / len(day_data)) * 100

            null_counts[0,array_ind] = (day_data['Lane 1'].isnull().sum() / len(day_data)) * 100
            null_counts[1,array_ind] = (day_data['Lane 2'].isnull().sum() / len(day_data)) * 100
            null_counts[2,array_ind] = (day_data['Lane 3'].isnull().sum() / len(day_data)) * 100
            null_counts[3,array_ind] = (day_data['All Lanes'].isnull().sum() / len(day_data)) * 100

            outlier_counts[0,array_ind] = (((day_data['Lane 1'] > mean_value_l1 + 2 * std_dev_l1) | (day_data['Lane 1'] < mean_value_l1 - 2 * std_dev_l1)).sum()) / len(day_data) * 100
            outlier_counts[1,array_ind] = (((day_data['Lane 2'] > mean_value_l2 + 2 * std_dev_l2) | (day_data['Lane 1'] < mean_value_l2 - 2 * std_dev_l2)).sum()) / len(day_data) * 100
            outlier_counts[2,array_ind] = (((day_data['Lane 3'] > mean_value_l3 + 2 * std_dev_l3) | (day_data['Lane 1'] < mean_value_l3 - 2 * std_dev_l3)).sum()) / len(day_data) * 100
            outlier_counts[3,array_ind] = (((day_data['All Lanes'] > mean_value_lall + 2 * std_dev_lall) | (day_data['All Lanes'] < mean_value_lall - 2 * std_dev_lall)).sum()) / len(day_data) * 100
            array_ind += 1
            
    xlabels = pd.date_range(start=start_date, periods=num_days, freq='D')    
    fig, ax = plt.subplots(3,1, figsize = [20,6])

    ax[0].plot(zero_counts[0],label = 'Lane 1')
    ax[0].plot(zero_counts[1],label = 'Lane 2')
    ax[0].plot(zero_counts[2],label = 'Lane 3')
    ax[0].plot(zero_counts[3],label = 'All Lanes')
    ax[0].set_xticks(np.arange(len(xlabels)))
    ax[0].set_xticklabels(xlabels.strftime('%b-%d'))
    ax[0].xaxis.set_major_locator(mdates.DayLocator(interval=15))
    ax[0].set_title('Zero Rate')
    ax[0].set_ylabel('Percent')
    ax[0].legend()

    ax[1].plot(null_counts[0],label = 'Lane 1')
    ax[1].plot(null_counts[1],label = 'Lane 2')
    ax[1].plot(null_counts[2],label = 'Lane 3')
    ax[1].plot(null_counts[3],label = 'All Lanes')
    ax[1].set_xticks(np.arange(len(xlabels)))
    ax[1].set_xticklabels(xlabels.strftime('%b-%d'))
    ax[1].xaxis.set_major_locator(mdates.DayLocator(interval=15))
    ax[1].set_title('Missing Rate')
    ax[1].set_ylabel('Percent')
    ax[1].legend()

    ax[2].plot(outlier_counts[0],label = 'Lane 1')
    ax[2].plot(outlier_counts[1],label = 'Lane 2')
    ax[2].plot(outlier_counts[2],label = 'Lane 3')
    ax[2].plot(outlier_counts[3],label = 'All Lanes')
    ax[2].set_xticks(np.arange(len(xlabels)))
    ax[2].set_xticklabels(xlabels.strftime('%b-%d'))
    ax[2].xaxis.set_major_locator(mdates.DayLocator(interval=15))
    ax[2].set_title('Outlier Rate')
    ax[2].set_ylabel('Percent')
    ax[2].legend()

    plt.tight_layout()
    plt.show()
    
    return ts, ts_all