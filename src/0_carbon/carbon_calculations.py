import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import netCDF4 as cdf
# import xarray as xr

FILE_DIR = ""
FILE_NAME = "AllWood_2006-2022.csv"

# Function that reads CSV file and outputs an array
def load_data(file_name):
    # Read CSV
    df = pd.read_csv(file_name)
    # Process data
    data = process_data(df)

    return data

# Function that processes the dataframe 
def process_data(df):
    # Process data
    df.loc[(df['Element']=='Export Quantity'), 'Value']*= -1
    # Organize data
    return df[['Item', 'Area', 'Year', 'Value']].groupby(['Item', 'Area', 'Year']).sum().reset_index()

# Function that gets the data slice for a specific country
def get_country_data(data, area, item):
    country_data = data[data["Area"] == area]
    return country_data

# 
def define_inflow_carbon(data, area, item, parameter_dict):
    items = country_data[country_data['Item'] == item]

    multiplier = get_multiplier(item, parameter_dict)
    for value in items['Value']:
        items['InflowC'] = value / 410.0 * multiplier

    print(items)

    # carbon_fraction = data[(data['Area'] == area) & (data['Item'] == item)]
    # carbon_fraction['Inflow_c'] = carbon_fraction['Value']/410*0.5

    # return carbon_fraction['Inflow_c']

# def calcualte_decay_per_year(data, half_life, area, item):
#     k = np.log(2)/half_life 
#     inflow_t0 = generate_carbon_fraction(data, area, item).mean() 
#     c_t0 = inflow_t0 / k 

#     first_order_decay = first_order_decay_sw(k=k, inflow=inflow_t0, c_t=c_t0)

#     return first_order_decay


# def generate_decay_array(data, half_life, area, item):
#     decay_array = []

#     inflow_t0 = generate_carbon_fraction(data, area, item).mean() 
#     print(inflow_t0)
    # c_t0 = inflow_t0 / k     

    # c_t = c_t0
    # inflow_c_t = inflow_t0

    # for year in nt_finland_sw['Year']:
    #     c_t = first_order_decay_sw(k=k,Inflow = inflow_c_t, c_t = c_t)
    #     inflow_c_t = nt_finland_sw[nt_finland_sw['Year']==year]["Inflow_c"].values

    #     print(c_t)


# def first_order_decay_sw(k , inflow_t0, c_t0): 
#     return np.exp(-1*k)*c_i + ((1-np.exp(-1*k))/k)*inflow

def get_multiplier(item, parameter_dict):
    parameters = parameter_dict.get(item, {}).values()
    multiplier = 1
    for value in parameters:
        multiplier *= value

    return multiplier

def main():

    parameter_dict = {
        'Sawnwood': {
            'c_fraction': 0.5
        } 
    }

    path = FILE_DIR + FILE_NAME
    data = load_data(path)

    c = get_country_data()
    define_inflow_carbon(data, 'Finland', "Sawnwood", parameter_dict)

    # print(nt_df[:5])

    # generate_decay_array(nt_df, 35, 'Finland', "Sawnwood")

    # print(nt_df)

if __name__ == '__main__':
    main()
