import pandas as pd
import sklearn
from sklearn import preprocessing
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
def main():
    data = pd.read_csv("traffic_data.csv")
    new = data["date_time"].str.split(" ", n=1, expand=True)
    data["date"] = new[0]
    data["time"] = new[1]
    data.drop(columns = ["date_time"], inplace=True)
    data.loc[data['holiday'] != 'None']
    data.loc[data['date'] == '2013-01-01', 'holiday'] = 'New Years Day'
    data.loc[data['date'] == '2013-02-18', 'holiday'] = 'Washingtons Birthday'
    data.loc[data['date'] == '2013-05-27', 'holiday'] = 'Memorial Day'
    data.loc[data['date'] == '2013-07-04', 'holiday'] = 'Independence Day'
    data.loc[data['date'] == '2013-08-22', 'holiday'] = 'State Fair'
    data.loc[data['date'] == '2013-09-02', 'holiday'] = 'Labor Day'
    data.loc[data['date'] == '2013-10-14', 'holiday'] = 'Columbus Day'
    data.loc[data['date'] == '2013-11-11', 'holiday'] = 'Veterans Day'
    data.loc[data['date'] == '2013-11-28', 'holiday'] = 'Thanksgiving Day'
    data.loc[data['date'] == '2013-12-25', 'holiday'] = 'Christmas Day'
    data = data.drop(['snow_1h'], axis=1)
    data.loc[data['holiday'] != 'None', 'hol'] = 'yes'
    data.loc[data['holiday'] == 'None', 'hol'] = 'no'
    data['date'] = pd.to_datetime(data['date'])
    data['dow'] = data['date'].dt.dayofweek
    data.loc[data['dow'] == 0, 'typeofday'] = 'weekday'
    data.loc[data['dow'] == 1, 'typeofday'] = 'weekday'
    data.loc[data['dow'] == 2, 'typeofday'] = 'weekday'
    data.loc[data['dow'] == 3, 'typeofday'] = 'weekday'
    data.loc[data['dow'] == 4, 'typeofday'] = 'weekday'
    data.loc[data['dow'] == 5, 'typeofday'] = 'weekend'
    data.loc[data['dow'] == 6, 'typeofday'] = 'weekend'
    df8 = data.drop(['weather_description','date'], axis=1)
    df8['holiday'] = df8['holiday'].astype('category')
    df8['weather_main'] = df8['weather_main'].astype('category')
    df8['time'] = df8['time'].astype('category')
    df8['hol'] = df8['hol'].astype('category')
    df8['dow'] = df8['dow'].astype('category')
    df8['typeofday'] = df8['typeofday'].astype('category')
    weather_dummies = pd.get_dummies(df8.weather_main)
    df9 = pd.concat([df8, weather_dummies], axis=1)
    df9["time_cat"] = df9["time"].cat.codes
    df9["hol_cat"] = df9["hol"].cat.codes
    df9["typeofday_cat"] = df9["typeofday"].cat.codes
    df9 = df9.drop(['holiday','weather_main','time','hol','dow','typeofday'], axis=1)
    df9 = pd.DataFrame(df9)
    cols = df9.columns
    min_max_scaler = preprocessing.MinMaxScaler()
    np_scaled = min_max_scaler.fit_transform(df9)
    df_normalized = pd.DataFrame(np_scaled, columns = cols)
    columns = df_normalized.columns.tolist()
    columns = [c for c in columns if c not in ['traffic_volume']]
    target = 'traffic_volume'
    train = df_normalized.sample(frac=0.8, random_state=9876)
    test = df_normalized.loc[~df_normalized.index.isin(train.index)]
    model = LinearRegression()
    model.fit(train[columns], train[target])
    lrtestpredictions = model.predict(test[columns])
    lrtrainpredictions = model.predict(train[columns])
    model2 = RandomForestRegressor(n_estimators=100, min_samples_leaf=10, random_state=9876)
    model2.fit(train[columns], train[target])
    rftestpredictions = model2.predict(test[columns])
    rftrainpredictions = model2.predict(train[columns])
    print(mean_squared_error(lrtestpredictions, test[target])) # this is LR predictions on test data
    print(mean_squared_error(rftestpredictions, test[target])) # this is the RF predictions on test data
    print(mean_squared_error(lrtrainpredictions, train[target])) # this is LR predictions on training data
    print(mean_squared_error(rftrainpredictions, train[target])) # this is the RF predictions on training data
if __name__ == '__main__':
   main()