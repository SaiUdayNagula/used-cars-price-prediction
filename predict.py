import pickle
import numpy as np
import pandas as pd
import json
import sys

## seperating the brands and doing ordinal encoding
luxury_brands = ['Acura', 'Alfa', 'Aston', 'Audi', 'BMW', 'Bentley', 'Bugatti',
                 'Cadillac', 'Ferrari', 'Genesis', 'INFINITI', 'Jaguar', 'Lamborghini',
                 'Lexus', 'Lincoln', 'Lotus', 'Lucid', 'Maserati', 'Maybach', 'McLaren',
                 'Mercedes-Benz', 'Polestar', 'Porsche', 'Rolls-Royce', 'Tesla']

mainstream_brands = ['Buick', 'Chevrolet', 'Chrysler', 'Dodge', 'FIAT', 'Ford', 'GMC',
                     'Honda', 'Hummer', 'Hyundai', 'Jeep', 'Kia', 'Land', 'Mazda',
                     'Mercury', 'Mitsubishi', 'Nissan', 'Pontiac', 'RAM', 'Saab',
                     'Saturn', 'Scion', 'Subaru', 'Suzuki', 'Toyota', 'Volkswagen', 'Volvo']

specialty_brands = ['Karma', 'MINI', 'Rivian', 'smart']

## seperating the Models and doing ordinal encoding
luxury_models = ['C-Class', '3 Series', 'GLC', 'X3', 'E-Class', 'GLE', 'GLB', 'GLA', '5 Series', 'A3', 'MDX']

moderate_models = ['Corolla', 'Civic', 'Camry', 'F-150', 'RAV4', 'Outback', 'Wrangler', 'Accord', 'Forte', 'Tacoma',
                   'Tiguan', 'Silverado 1500', 'Forester', 'Escape', 'Sentra', 'Explorer', 'Malibu', 'Altima',
                   'CR-V', 'Jetta', 'Odyssey', 'Fusion', 'Optima', 'Prius', 'Sonata']

def encode_brands(X):
    
    X = X['Make']
    if not isinstance(X, pd.Series):
        raise ValueError("Input must be a pandas Series.", X.index, type(X['Make']))
    # Ordinal encoding for brand names
    encoded = pd.Series(1, index=X.index)  # Initialize all values to 1
    encoded[X.isin(luxury_brands)] = 4
    encoded[X.isin(mainstream_brands)] = 3
    encoded[X.isin(specialty_brands)] = 2
    
    return pd.DataFrame(encoded)

def encode_models(X):
    X = X['Model']
    # Ordinal encoding for Model Names
    encoded = pd.Series(1, index=X.index)  # Initialize all values to 1
    encoded[X.isin(luxury_models)] = 3
    encoded[X.isin(moderate_models)] = 2
    
    return pd.DataFrame(encoded)

pipe = pickle.load(open('pipe.pkl','rb'))

features = json.loads(sys.argv[1])

features = np.array(features,dtype=object).reshape(1,6)

# Convert the input data to a DataFrame with appropriate column names
columns = ['Make', 'Model', 'year', 'miles', 'Exterior_colour', 'Accident_Y/N']
test_input_df = pd.DataFrame(features, columns=columns)

# Assuming 'pipe' is your pre-trained pipeline, you can now make a prediction
prediction = pipe.predict(test_input_df)

print(prediction[0])


