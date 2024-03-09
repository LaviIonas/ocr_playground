import pandas as pd
import numpy as np
import os

import matplotlib.pyplot as plt 

FILE_DIR = "./"
FILE_NAME = "AllWood_2006-2022.csv"
FILE_NAME_OUTPUT = "calculations_2006-2022.csv"

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
    subset = df[['Item', 'Area', 'Year', 'Value']].groupby(['Item', 'Area', 'Year']).sum().reset_index()
    return subset

def generate_inflow_carbon_values(data, area, item, parameter_dict):
    # Isolate Item Data
    filtered_data = data.loc[(data['Area'] == area) & (data['Item'] == item)].copy()

    # Product of multipliers for each item
    multiplier = get_mult(item, parameter_dict)
    
    # Alter Value
    filtered_data['InflowC'] = filtered_data['Value'] * multiplier

    # Assign the updated values back to the original DataFrame
    data.loc[(data['Area'] == area) & (data['Item'] == item), 'InflowC'] = filtered_data['InflowC'].values

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

def calculate_first_order_decay_variables(data, area, item, parameter_dict):
     # Isolate Item Data
    filtered_data = data.loc[(data['Area'] == area) & (data['Item'] == item)].copy()

    k = np.log(2) / parameter_dict.get(item, {}).get('half-life')
    inflow_t = filtered_data['InflowC'].mean()
    c_t = inflow_t / k

    return k, inflow_t, c_t

def calculate_first_order_decay(k, inflow, c):
    return np.exp(-1*k)*c + ((1-np.exp(-1*k))/k)*inflow

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

def generate_output_cvs(file_name, data, columns):
    selected_data = data[columns]

    selected_data["c_t"] = {}
    selected_data["delta_c_t"] = {}

    filepath = os.path.join(FILE_DIR, file_name)
    selected_data.to_csv(filepath, index=False)

def read_output_csv(file_name):
    df = pd.read_csv(file_name)
    return df

def calculate_deltas(c_t):
    delta_array = []
    for i in range(len(c_t)):
        if i < (len(c_t)-1):
            val1 = c_t[i]
            val2 = c_t[i+1]

            delta = val2 - val1
            delta_array.append(delta)
        else:
            # VALUE OF 2007 !!!
            delta_array.append(c_t[i])

    return delta_array

def update_csv_from_dataframe(filename, dataframe):
    dataframe.to_csv(filename, index=False)

def generate_calculations(data, output, unique_country, unique_items, parameter_dict):
     for country in unique_country:
        for item in unique_items:
            generate_inflow_carbon_values(data, country, item, parameter_dict)
            c_t = calculate_anual_first_order_decay(data, country, item, parameter_dict)
            
            reversed_c_t = c_t[::-1]

            delta = calculate_deltas(c_t)
            reversed_delta = delta[::-1]

            output.loc[(output['Area'] == country) & (output['Item'] == item), 'c_t'] = reversed_c_t
            output.loc[(output['Area'] == country) & (output['Item'] == item), 'delta_c_t'] = reversed_delta

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
        },
        'Roundwood': {
            'mult' : {
                'c_fraction': 0.1
            },
            'div' : {
                'dry-weight': 1.0
            },
            'half-life': 100
        },
        'Wood fuel': {
            'mult' : {
                'c_fraction': 0.1
            },
            'div' : {
                'dry-weight': 1.0
            },
            'half-life': 100
        }
    }

    path = FILE_DIR + FILE_NAME
    data = load_data(path)

    unique_country = data['Area'].unique()
    unique_items = ['Paper and paperboard', 'Roundwood', 
                    'Sawnwood', 'Wood fuel', 'Wood-based panels']

    # Generate CSV
    # columns = ["Area", "Item", "Year"]
    # generate_output_cvs("calculations_2006-2022.csv", data, columns)

    output = read_output_csv(FILE_DIR + FILE_NAME_OUTPUT)

    # TEEST
    # a = generate_inflow_carbon_values(data, area, item, parameter_dict)
    # generate_calculations(data, output, unique_country, unique_items, parameter_dict)
  
    update_csv_from_dataframe(FILE_DIR + FILE_NAME_OUTPUT, output)
    output = read_output_csv(FILE_DIR + FILE_NAME_OUTPUT)
    print(output)

if __name__ == '__main__':
    main()

