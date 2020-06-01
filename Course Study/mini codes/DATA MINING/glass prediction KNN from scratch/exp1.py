import pandas as pd

data=pd.read_csv(r'C:\Users\Chandresh7\Desktop\pandas-videos-master\pandas-videos-master\data\ufo.csv')
print(data.head())

#1st Method

print(data.isnull().head())
print(data.isnull())

#2nd method: Dropping missing values
print('no: of rows and columns before removing null values')
print('rows, columns: ', data.shape)
print('First 5 rows')
print(data.head())
print('---------------------------------------')

print('after null values have been removed')
print(data.dropna(subset=['City', 'Shape Reported', 'Colors Reported'], how='any').head())

print('no: of rows and columns after removing null values')
print('rows, columns', data.shape) 

print((data))


#3rd Method

print(data['Shape Reported'].value_counts(dropna='False'))
print('Since light is most frequently occured it can be replace nan values in Shape Reported')


(data['Shape Reported'].fillna(value='LIGHT', inplace=True))
print(data.head())

print(data['Colors Reported'].value_counts(dropna='False'))

print('most frequeant color is green')
(data['Colors Reported'].fillna(value='GREEN', inplace=True))
print(data.head())

print(data['City'].value_counts(dropna='False'))
print('Seattle is most repeated city')
(data['City'].fillna(value='Seattle', inplace=True))
print(data)
