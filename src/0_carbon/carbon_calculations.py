import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

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

# 
def generate_inflow_carbon_values(data, area, item, parameter_dict):
    # Isolate Item Data
    filtered_data = data.loc[(data['Area'] == area) & (data['Item'] == item)].copy()

    # Product of multipliers for each item
    multiplier = get_mult(item, parameter_dict)
    
    # Alter Value
    filtered_data['InflowC'] = filtered_data['Value'] * multiplier

    # Assign the updated values back to the original DataFrame
    data.loc[(data['Area'] == area) & (data['Item'] == item), 'InflowC'] = filtered_data['InflowC'].values

# 
def get_mult(item, parameter_dict):
    mult_values = parameter_dict.get(item, {}).get('mult', {}).values()
    div_values = parameter_dict.get(item, {}).get('div', {}).values()
    
    # Calculate the product of values in the 'mult' dictionary
    product = 1
    for value in mult_values:
        product *= value
    
    # Calculate the product of values in the 'div' dictionary
    divisor = 1
    for value in div_values:
        divisor *= value
    
    # Perform the division
    result = product / divisor
    
    return result

# 
def calculate_first_order_decay_variables(data, area, item, parameter_dict):
     # Isolate Item Data
    filtered_data = data.loc[(data['Area'] == area) & (data['Item'] == item)].copy()

    k = np.log(2) / parameter_dict.get(item, {}).get('half-life')
    inflow_t = filtered_data['InflowC'].mean()
    c_t = inflow_t / k

    return k, inflow_t, c_t

# 
def calculate_first_order_decay(k, inflow, c):
    return np.exp(-1*k)*c + ((1-np.exp(-1*k))/k)*inflow

# 
def calculate_anual_first_order_decay(data, area, item, parameter_dict):
    # Isolate Item Data
    filtered_data = data.loc[(data['Area'] == area) & (data['Item'] == item)].copy()

    k, inflow_t0, c_t0 = calculate_first_order_decay_variables(data, area, item, parameter_dict)

    inflow_t = inflow_t0
    c_t = c_t0

    c_t_array = []

    for year in filtered_data['Year']:
        c_t = calculate_first_order_decay(k, inflow_t, c_t)

        # Watch out if a year has multiple inflowC values, because only the first is being selected
        inflow_t = filtered_data.loc[filtered_data['Year'] == year, "InflowC"].values[0]

        c_t_array.append(c_t)

    return c_t_array

def main():

    parameter_dict = {
        'Sawnwood': {
            'mult' : {
                'c_fraction': 0.5
            },
            'div' : {
                'dry-weight': 410.0
            },
            'half-life': 35
        },
        'Wood-based panels': {
            'mult' : {
                'c_fraction': 0.454
            },
            'div' : {
                'dry-weight': 410.0
            },
            'half-life': 25,
            'density': 0.595
        },
        'Paper and paperboard': {
            'mult' : {
                'c_fraction': 0.429,
                'relative_dry_mass': 0.9
            },
            'div' : {
                'dry-weight': 1.0
            },
            'half-life': 2
        }
    }

    path = FILE_DIR + FILE_NAME
    data = load_data(path)

    area, item = ('Finland', "Paper and paperboard")

    generate_inflow_carbon_values(data, area, item, parameter_dict)

    c_t_arr = calculate_anual_first_order_decay(data, area, item, parameter_dict)

    print(c_t_arr)


if __name__ == '__main__':
    main()
