import pandas as pd
import numpy as np
import os

FILE_DIR = "./"
FILE_NAME = "AllWood_2006-2022.csv"
FILE_NAME_OUTPUT = "calculations_2006-2022.csv"

PATH = FILE_DIR + FILE_NAME

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

country_areas_dict = {
    'Albania': 28748,
    'Andorra': 468,
    'Austria': 83871,
    'Belarus': 207600,
    'Belgium': 30528,
    'Bosnia and Herzegovina': 51197,
    'Bulgaria': 110879,
    'Croatia': 56594,
    'Czechia': 78866,
    'Denmark': 43094,
    'Estonia': 45339,
    'Faroe Islands': 1399,  # Area includes land and water
    'Finland': 338462,
    'France': 551695,
    'Germany': 357022,
    'Gibraltar': 6.7,  # Area is very small
    'Greece': 131957,
    'Holy See': 0.44,  # Area is very small
    'Hungary': 93030,
    'Iceland': 103000,
    'Ireland': 70273,
    'Italy': 301340,
    'Latvia': 64589,
    'Liechtenstein': 160,  # Area is very small
    'Lithuania': 65300,
    'Luxembourg': 2586,
    'Malta': 316,
    'Montenegro': 13812,
    'Netherlands': 41543,
    'North Macedonia': 25713,
    'Norway': 323802,
    'Poland': 312696,
    'Portugal': 92212,
    'Republic of Moldova': 33846,
    'Romania': 238397,
    'San Marino': 61,  # Area is very small
    'Serbia': 88361,
    'Slovakia': 49035,
    'Slovenia': 20273,
    'Spain': 505990,
    'Sweden': 450295,
    'Switzerland': 41284,
    'Ukraine': 603500,
    'United Kingdom of Great Britain and Northern Ireland': 242495}

unique_items = ['Paper and paperboard', 'Roundwood', 
                'Sawnwood', 'Wood fuel', 'Wood-based panels']

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

def generate_inflow_carbon_values(data, area, item):
    # Isolate Item Data
    filtered_data = data.loc[(data['Area'] == area) & (data['Item'] == item)].copy()

    # Product of multipliers for each item
    multiplier = get_mult(item)
    
    # Alter Value
    filtered_data['InflowC'] = filtered_data['Value'] * multiplier

    # Assign the updated values back to the original DataFrame
    data.loc[(data['Area'] == area) & (data['Item'] == item), 'InflowC'] = filtered_data['InflowC'].values

def get_mult(item):
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

def calculate_first_order_decay_variables(data, area, item):
     # Isolate Item Data
    filtered_data = data.loc[(data['Area'] == area) & (data['Item'] == item)].copy()

    k = np.log(2) / parameter_dict.get(item, {}).get('half-life')
    inflow_t = filtered_data['InflowC'].mean()
    c_t = inflow_t / k

    return k, inflow_t, c_t

def calculate_first_order_decay(k, inflow, c):
    return np.exp(-1*k)*c + ((1-np.exp(-1*k))/k)*inflow

def calculate_anual_first_order_decay(data, area, item):
    # Isolate Item Data
    filtered_data = data.loc[(data['Area'] == area) & (data['Item'] == item)].copy()

    k, inflow_t0, c_t0 = calculate_first_order_decay_variables(data, area, item)

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
    selected_data["delta_c_t_converted"] = {}

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

def convert_delta_c_t(delta, area):
    out = []
    for d in delta:
        a = d
        b = a * 1e+6
        c = b / 12
        d = c * 3.664

        e = area * 31536000 # area * seconds in year
        f = d / e

        out.append(f)

    return out

def generate_calculations(data, output):
     for country, area in country_areas_dict.items():
        for item in unique_items:
            generate_inflow_carbon_values(data, country, item)
            c_t = calculate_anual_first_order_decay(data, country, item)
        
            reversed_c_t = c_t[::-1]

            delta = calculate_deltas(c_t)
            reversed_delta = delta[::-1]

            # convert delta to => mol / (m^2 * s)
            country_area = area * 1000.0 # in m^2
            converted_delta = convert_delta_c_t(delta, country_area)
            reversed_converted_delta = converted_delta[::-1]

            output.loc[(output['Area'] == country) & (output['Item'] == item), 'c_t'] = reversed_c_t
            output.loc[(output['Area'] == country) & (output['Item'] == item), 'delta_c_t'] = reversed_delta
            output.loc[(output['Area'] == country) & (output['Item'] == item), 'delta_c_t_converted'] = reversed_converted_delta

def generate_flux_constants():
    data = load_data(PATH)

    columns = ["Area", "Item", "Year"]
    generate_output_cvs(FILE_NAME_OUTPUT, data, columns)

    output = read_output_csv(FILE_DIR + FILE_NAME_OUTPUT)
    generate_calculations(data, output)
    update_csv_from_dataframe(FILE_DIR + FILE_NAME_OUTPUT, output)
    output = read_output_csv(FILE_DIR + FILE_NAME_OUTPUT)

    return output

def get_country_flux(data,country, year):
    filtered_data = data.loc[(data['Area'] == country) & (data['Year'] == year)]
    flux = filtered_data['delta_c_t_converted'].sum()
    return flux

def main():
    data = generate_flux_constants()
    country = 'Finland'
    year = 2021
    flux = get_country_flux(data, country, year)
    print(flux)

if __name__ == '__main__':
    main()

