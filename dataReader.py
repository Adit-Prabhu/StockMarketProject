import pandas as pd
from datetime import datetime
import yfinance as yf

# The tech stocks we will use for this analysis
tech_list = ['AAPL', 'GOOG', 'MSFT', 'AMZN']

end = datetime.now()
start = datetime(end.year - 1, end.month, end.day)

company_list = []
company_names = ["Apple", "Google", "Microsoft", "Amazon"]

for stock in tech_list:
    globals()[stock] = yf.download(stock, start=start, end=end)
    globals()[stock]["company_name"] = company_names[tech_list.index(stock)]
    company_list.append(globals()[stock])


# Return the list of dataframes
def get_data():
    return company_list, tech_list
