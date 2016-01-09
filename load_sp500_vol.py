from pandas_datareader import data, wb
import datetime

start = datetime.datetime(1950, 1, 3)
end = datetime.datetime(2015, 12, 1)

f = data.DataReader("^GSPC", 'yahoo', start, end)

f.ix[start]