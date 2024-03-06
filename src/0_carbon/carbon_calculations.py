import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import netCDF4 as cdf
# import xarray as xr

FILE_DIR = ""
FILE_NAME = "AllWood_2006-2022.csv"

def first_order_decay_sw(k=k , inflow=inflow_t0, c_t=c_t0): 
    return np.exp(-1*k)*c_i + ((1-np.exp(-1*k))/k)*inflow

def get_data(file_name):
    # with xr.open_dataset(file_name) as ds:
    #     paris_masks = ds

    df = pd.read_csv(file_name)

    df.loc[(df['Element']=='Export Quantity'), 'Value']*= -1

    nt_df = df[['Item', 'Area', 'Year', 'Value']].groupby(['Item', 'Area', 'Year']).sum().reset_index()

    return nt_df

def generate_carbon_fraction(data, area, item):
    carbon_fraction = data[(data['Area'] == area)& (data['Item'] == item)]
    carbon_fraction['Inflow_c'] = carbon_fraction['Value']/410*0.5

    return nt_cf['Inflow_c']

def calcualte_decay_per_year(data, half_life, area, item):
    k = np.log(2)/half_life 
    inflow_t0 = generate_carbon_fraction(data, area, item).mean() 
    c_t0 = inflow_t0 / k 

    first_order_decay = first_order_decay_sw(k=k, inflow=inflow_t0, c_t=c_t0)

    return first_order_decay


def generate_decay_array(data, half_life, area, item):
    decay_array = []

    inflow_t0 = generate_carbon_fraction(data, area, item).mean() 
    print(inflow_t0)
    # c_t0 = inflow_t0 / k     

    # c_t = c_t0
    # inflow_c_t = inflow_t0

    # for year in nt_finland_sw['Year']:
    #     c_t = first_order_decay_sw(k=k,Inflow = inflow_c_t, c_t = c_t)
    #     inflow_c_t = nt_finland_sw[nt_finland_sw['Year']==year]["Inflow_c"].values

    #     print(c_t)


def main():
    path = FILE_DIR + FILE_NAME
    nt_df = get_data(path)

    generate_decay_array(nt_df, 35, 'Finland', "Sawnwood")

    print(nt_df)

if __name__ == '__main__':
    main()
