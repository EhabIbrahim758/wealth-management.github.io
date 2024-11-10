from flask import Flask, request, jsonify
import pandas as pd
import pickle
from flask import Flask, request, jsonify
import os 
import re
import json 
import pandas as pd
import pickle
import numpy as np
from collections import defaultdict

def map_risk_to_investment_allocation(risk_category):
    investment_allocation_mapping = {
        "No Risk": {"Deposit": 0.60, "Bonds": 0.40, "Equities": 0.00},
        "Low Risk": {"Deposit": 0.30, "Bonds": 0.60, "Equities": 0.10},
        "Medium Risk": {"Deposit": 0.20, "Bonds": 0.40, "Equities": 0.40},
        "High Risk": {"Deposit": 0.00, "Bonds": 0.20, "Equities": 0.80}
    }

    return investment_allocation_mapping.get(risk_category, {"Deposit": 0.00, "Bonds": 0.00, "Equities": 0.00})



def map_risk_to_investment_allocation_2(risk_category , assets):

    investment_allocation_mapping = {}

    if assets == 'Bonds&Equity' :
        asset1 = "Bonds"
        asset2 = "Equities"
        investment_allocation_mapping = {
            "No Risk": {asset1: 0.90, asset2: 0.10},
            "Low Risk": {asset1: 0.80, asset2: 0.20},
            "Medium Risk": {asset1: 0.60, asset2: 0.40},
            "High Risk": {asset1: 0.20, asset2: 0.80}
        }

    elif assets == 'Deposit&Bonds' :
        asset1 = "Deposit"
        asset2 = "Bonds"
        investment_allocation_mapping = {
            "No Risk": {asset1: 0.80, asset2: 0.20},
            "Low Risk": {asset1: 0.60, asset2: 0.40},
            "Medium Risk": {asset1: 0.40, asset2: 0.60},
            "High Risk": {asset1: 0.20, asset2: 0.80}
        }
    elif assets == 'Equity&Deposit' :
        asset1 = "Deposit"
        asset2 = "Equities"
        investment_allocation_mapping = {
            "No Risk": {asset1: 0.90, asset2: 0.10},
            "Low Risk": {asset1: 0.80, asset2: 0.20},
            "Medium Risk": {asset1: 0.60, asset2: 0.40},
            "High Risk": {asset1: 0.20, asset2: 0.80}
        }

    return investment_allocation_mapping.get(risk_category, {asset1: 0.00, asset2: 0.00})


def lists_are_equal(list1, list2):
    return len(list1) == len(list2) and all(a == b for a, b in zip(list1, list2))

def find_matching_cluster(new_data, df, clusters_column, categorical_cols):
    for cluster in range(1, max(df[clusters_column]) + 1):
        cluster_data = df[df[clusters_column] == cluster]
        cluster_encoded = cluster_data[categorical_cols].values
        #print(new_data)
        if any(lists_are_equal(cluster_vec, new_data[0]) for cluster_vec in cluster_encoded):
            return cluster
    return None

def get_samples_from_clusters(df, clusters_column, cluster_number):
    return df[df[clusters_column] == cluster_number] if cluster_number is not None else None

#convert time durations
# Function to convert time durations to numeric values
def convert_duration_to_months(duration):
    duration_mapping = {
        "3-6 Months": 3,
        "6-12 Months": 6,
        "1 years": 12,
        "2 years": 24,
        "3 years": 36,
        "4 years": 48,
        "5 years": 60,
        "6 years": 72,
        "8 years": 96,
        "10 years": 120,
        "20 years": 240,
        "30 years": 360,
    }
    return duration_mapping.get(duration, 0)  # Return 0 if duration is not found in mapping



# Function to convert amount choices to their corresponding max values
def convert_amount_choice_to_max_amount(amount_choice):
    amount_mapping = {
        '<1M':   1000000,
        '1-3M':  3000000,
        '3-5M':  5000000,
        '5-10M': 10000000,
        '>10M':  10000000
    }
    return amount_mapping.get(amount_choice, 0)  # Return 0 if amount choice is not found in mapping

#find_best_Deposit
def find_best_Deposit(user_data, Deposit_df):
    matching_Deposit = []  # List to store matching bonds

    for i in range(len(Deposit_df)):
      user_attributes = (user_data['Currency']  ,user_data['Islamic Products'])
      Deposit_attributes = (Deposit_df['Currency'][i],Deposit_df['Islamic'][i] )

      # Convert durations to numeric values
      tenor_months = convert_duration_to_months(Deposit_df['Tenor'][i])
      inv_sigh_months = convert_duration_to_months(user_data['Inv Sigh'])



      if user_data['Currency'] =="ANY":
        user_attributes_ = (user_data['Islamic Products'])
        Deposit_attributes_ = (Deposit_df['Islamic'][i] )

        # Compare user attributes with bond attributes
        if user_attributes_ == Deposit_attributes_ and tenor_months >= inv_sigh_months:
            matching_Deposit.append(Deposit_df['Name'][i])  # Add matching  name to the list
      else:
        # Compare user attributes with bond attributes
        if user_attributes == Deposit_attributes and tenor_months >= inv_sigh_months:
          matching_Deposit.append(Deposit_df['Name'][i])  # Add matching name to the list


    return matching_Deposit

#find_best_bonds
def find_best_bonds(user_data, bond_data):
    matching_bonds = []  # List to store matching bonds
    for i in range(len(bond_data)):
      user_attributes = (user_data['Currency'] ,user_data['Investment Sector']  ,user_data['Islamic Products'] ,user_data['Geography'])
      bond_attributes = (bond_data['Currency'][i] ,bond_data['Type'][i] ,bond_data['Islamic'][i] ,bond_data['Country'][i])
      # Convert durations to numeric values
      tenor_months = convert_duration_to_months(bond_data['Tenor'][i])
      # Assuming user_data is a single row from a DataFrame (like 'sample')
      inv_sigh_months = convert_duration_to_months(user_data['Inv Sigh'])
      amount_min = convert_amount_choice_to_max_amount(user_data['Amount'])


      if user_data['Currency'] =="ANY":
        user_attributes_ = (user_data['Investment Sector']  ,user_data['Islamic Products'] ,user_data['Geography'])
        bond_attributes_ = (bond_data['Type'][i] ,bond_data['Islamic'][i] ,bond_data['Country'][i])
        # Compare user attributes with bond attributes
        if user_attributes_ == bond_attributes_ and tenor_months >= inv_sigh_months and amount_min >=  bond_data['Minimum Buy'][i] :
            matching_bonds.append(bond_data['Name'][i])  # Add matching bond's name to the list
      else:
        # Compare user attributes with bond attributes
        if user_attributes == bond_attributes and tenor_months >= inv_sigh_months and amount_min >=  bond_data['Minimum Buy'][i] :
            matching_bonds.append(bond_data['Name'][i])  # Add matching bond's name to the list


    return matching_bonds



#find_best_stocks
def find_best_stocks(user_data, Stocks_df):
    matching_Stocks = []  # List to store matching bonds
    for i in range(len(Stocks_df)):
      user_attributes = (user_data['Currency'] ,user_data['Investment Sector']  ,user_data['Islamic Products'])
      Stocks_attributes = (Stocks_df['Currency'][i] ,Stocks_df['Type'][i] ,Stocks_df['Islamic'][i] )


      if user_data['Currency'] =="ANY":
        user_attributes_ = (user_data['Investment Sector']  ,user_data['Islamic Products'])
        Stocks_attributes_ = (Stocks_df['Type'][i] ,Stocks_df['Islamic'][i] )
        # Compare user attributes with bond attributes
        if user_attributes_ == Stocks_attributes_ :
            matching_Stocks.append(Stocks_df['Name'][i])  # Add matching bond's name to the list
      else:
        # Compare user attributes with bond attributes
        if user_attributes == Stocks_attributes :
            matching_Stocks.append(Stocks_df['Name'][i])  # Add matching bond's name to the list

    return matching_Stocks

def best_Allocation(sample):
  Allocation_values = list(sample["Allocation"].values())
  Allocation_key = list(sample["Allocation"].keys())
  Inv_Products = sample["Inv Products"]
  print(Allocation_values)



  # Define a mapping from DataFrame column names to clustering model column names
  column_mapping = {
      "Currency": "Currency",
      "Investment Sector": "Type",
      "Islamic Products": "Islamic",
      "Geography": "Country"
  }

  # Map the sample columns to the clustering model columns
  sample_mapped = {model_col: sample[df_col] for df_col, model_col in column_mapping.items()}


  if Allocation_values == [1.0, 0.0, 0.0]:
    # Specify the categorical columns
    categorical_cols = ['Currency', 'Islamic']
    # Encode the sample
    sample_encoded = np.array([[encoders_deposit[col].transform([sample_mapped[col]])[0] for col in categorical_cols]])

    # Get clusters with full matching data points based on matching
    matching_Deposit_clusters = find_matching_cluster(sample_encoded, deposit_df_cluster, 'Cluster', categorical_cols)
    #print(matching_Bonds_clusters)
    samples_from_matching_Deposit_clusters = get_samples_from_clusters(deposit_df, 'Cluster', matching_Deposit_clusters)

    # Find the best matching Deposit for each user
    best_Deposit = {}
    if samples_from_matching_Deposit_clusters is not None:
      matching_Deposit= find_best_Deposit(sample, samples_from_matching_Deposit_clusters.reset_index(drop=True))
      best_Deposit['best_Deposit']= matching_Deposit  # Extend the list with matching bonds for the current user
      return best_Deposit if matching_Deposit else None

    else:
      
      return None
    

  elif Allocation_values == [0.0, 1.0, 0.0]:
    # Specify the categorical columns
    categorical_cols = ["Currency", "Type", "Islamic", "Country"]
    # Encode the sample
    sample_encoded = np.array([[encoders_bonds[col].transform([sample_mapped[col]])[0] for col in categorical_cols]])

    # Get clusters with full matching data points based on cosine similarity
    matching_Bonds_clusters = find_matching_cluster(sample_encoded, bonds_df_cluster, 'Cluster', categorical_cols)
    #print(matching_Bonds_clusters)
    samples_from_matching_Bonds_clusters = get_samples_from_clusters(bonds_df, 'Cluster', matching_Bonds_clusters)
    # Find the best matching bonds for each user
    best_bond = {}
    #print(df)
    if samples_from_matching_Bonds_clusters is not None:
      matching_bonds= find_best_bonds(sample, samples_from_matching_Bonds_clusters.reset_index(drop=True))
      best_bond['best_bonds']= matching_bonds  # Extend the list with matching bonds for the current user
      return best_bond if matching_bonds else None

    else:
      return  None

  elif Allocation_values == [0.0, 0.0, 1.0]:
    # Specify the categorical columns
    categorical_cols = ["Currency", "Type", "Islamic", ]
    # Encode the sample
    sample_encoded = np.array([[encoders_stocks[col].transform([sample_mapped[col]])[0] for col in categorical_cols]])

    # Get clusters with full matching data points based on cosine similarity
    matching_Stocks_clusters = find_matching_cluster(sample_encoded, stocks_df_cluster, 'Cluster', categorical_cols)
    samples_from_matching_Stocks_clusters = get_samples_from_clusters(stocks_df, 'Cluster', matching_Stocks_clusters)

    # Find the best matching stocks for each user
    best_stocks = {}
    if samples_from_matching_Stocks_clusters is not None:
      matching_stocks= find_best_stocks(sample, samples_from_matching_Stocks_clusters.reset_index(drop=True))
      best_stocks['best_stocks'] = matching_stocks  # Extend the list with matching bonds for the current user
      return best_stocks if matching_stocks else None
    else:
      return None

  elif Allocation_values == [0.0, 0.0, 0.0]:
    return None

#['ANY', 'Bonds', 'Equities', 'Cash' , 'Bonds&Equity', 'Equity&Deposit', 'Deposit&Bonds']

  if Inv_Products == "ANY" or 'Bonds&Equity' or 'Equity&Deposit' or 'Deposit&Bonds':
    best_all_Allocation={}
    if "Allocation" in sample:
        allocation_dict = sample["Allocation"]
        try:
            if allocation_dict['Deposit'] != 0.0:
                # Specify the categorical columns
                categorical_cols = ['Currency', 'Islamic']
                # Encode the sample
                sample_encoded = np.array([[encoders_deposit[col].transform([sample_mapped[col]])[0] for col in categorical_cols]])

                # Get clusters with full matching data points based on cosine similarity
                matching_Deposit_clusters = find_matching_cluster(sample_encoded, deposit_df_cluster, 'Cluster', categorical_cols)
                samples_from_matching_Deposit_clusters = get_samples_from_clusters(deposit_df, 'Cluster', matching_Deposit_clusters)

                # Find the best matching Deposit for each user
                matching_Deposit = find_best_Deposit(sample, samples_from_matching_Deposit_clusters.reset_index(drop=True))
                best_all_Allocation["best_Deposit"] = matching_Deposit
        except KeyError:
            pass

        try:
            if allocation_dict['Bonds'] != 0.0:
                # Specify the categorical columns
                categorical_cols = ["Currency", "Type", "Islamic", "Country"]
                # Encode the sample
                sample_encoded = np.array([[encoders_bonds[col].transform([sample_mapped[col]])[0] for col in categorical_cols]])

                # Get clusters with full matching data points based on cosine similarity
                matching_Bonds_clusters = find_matching_cluster(sample_encoded, bonds_df_cluster, 'Cluster', categorical_cols)
                samples_from_matching_Bonds_clusters = get_samples_from_clusters(bonds_df, 'Cluster', matching_Bonds_clusters)
                if samples_from_matching_Bonds_clusters is not None:
                  # Find the best matching bonds for each user
                  matching_bonds = find_best_bonds(sample, samples_from_matching_Bonds_clusters.reset_index(drop=True))
                  best_all_Allocation["best_bonds"] = matching_bonds

        except KeyError:
            pass

        try:
            if allocation_dict['Equities'] != 0.0:
                # Specify the categorical columns
                categorical_cols = ["Currency", "Type", "Islamic", ]
                # Encode the sample
                sample_encoded = np.array([[encoders_stocks[col].transform([sample_mapped[col]])[0] for col in categorical_cols]])

                # Get clusters with full matching data points based on cosine similarity
                matching_Stocks_clusters = find_matching_cluster(sample_encoded, stocks_df_cluster, 'Cluster', categorical_cols)
                #print(matching_Bonds_clusters)
                samples_from_matching_Stocks_clusters = get_samples_from_clusters(stocks_df, 'Cluster', matching_Stocks_clusters)
                if samples_from_matching_Stocks_clusters is not None:
                  # Find the best matching stocks for each user
                  matching_stocks = find_best_stocks(sample, samples_from_matching_Stocks_clusters.reset_index(drop=True))
                  best_all_Allocation["best_stocks"] = matching_stocks
        except KeyError:
            pass

    return best_all_Allocation if any(best_all_Allocation.values()) else None

def best_Allocation_2(df):
  Allocation_values = list(df["Allocation"].values())
  Allocation_key = list(df["Allocation"].keys())
  Inv_Products = df["Inv Products"]
  if Allocation_values == [1.0, 0.0, 0.0]:
    # Find the best matching Deposit for each user
    best_Deposit = {}
    matching_Deposit= find_best_Deposit(df, deposit_df)
    best_Deposit['best_Deposit']=matching_Deposit  # Extend the list with matching bonds for the current user
    return best_Deposit if matching_Deposit else None

  elif Allocation_values == [0.0, 1.0, 0.0]:
    # Find the best matching bonds for each user
    best_bond = {}
    matching_bonds= find_best_bonds(df, bonds_df)
    best_bond['best_bonds']= matching_bonds  # Extend the list with matching bonds for the current user
    return best_bond if matching_bonds else None

  elif Allocation_values == [0.0, 0.0, 1.0]:
    # Find the best matching stocks for each user
    best_stocks = {}
    matching_stocks= find_best_stocks(df, stocks_df)
    best_stocks['best_stocks'] = matching_stocks  # Extend the list with matching bonds for the current user
    return best_stocks if matching_stocks else None

  elif Allocation_values == [0.0, 0.0, 0.0]:
    return None

#['ANY', 'Bonds', 'Equities', 'Cash' , 'Bonds&Equity', 'Equity&Deposit', 'Deposit&Bonds']

  if Inv_Products == "ANY" or 'Bonds&Equity' or 'Equity&Deposit' or 'Deposit&Bonds':
    best_all_Allocation={}
    if "Allocation" in df:
        allocation_dict = df["Allocation"]
        try:
            if allocation_dict['Deposit'] != 0.0:
                # Find the best matching Deposit for each user
                matching_Deposit = find_best_Deposit(df, deposit_df)
                best_all_Allocation["best_Deposit"] = matching_Deposit
        except KeyError:
            pass

        try:
            if allocation_dict['Bonds'] != 0.0:
                # Find the best matching bonds for each user
                matching_bonds = find_best_bonds(df, bonds_df)
                best_all_Allocation["best_bonds"] = matching_bonds
        except KeyError:
            pass

        try:
            if allocation_dict['Equities'] != 0.0:
                # Find the best matching stocks for each user
                matching_stocks = find_best_stocks(df, stocks_df)
                best_all_Allocation["best_stocks"] = matching_stocks
        except KeyError:
            pass

    return best_all_Allocation if any(best_all_Allocation.values()) else None



def generate_samples_based_on_choices(sample):# if Geography==ANY generate columns for each item & if Currency==ANY generate columns for each item
    modified_samples = []

    if sample['Geography'] == 'ANY':
        for geography in geography_choices:
            modified_sample = sample.copy()
            modified_sample['Geography'] = geography
            modified_samples.append(modified_sample)
    elif sample['Currency'] == 'ANY':
        for currency in currency_choices:
            modified_sample = sample.copy()
            modified_sample['Currency'] = currency
            modified_samples.append(modified_sample)

    return modified_samples


def merge_allocations(allocations_list):
    merged_allocations = defaultdict(set)  # Using set to avoid repetitions

    for allocation in allocations_list:
        
        for key, items in allocation.items():
            if items is not None:  # Checking if the list is not None
                merged_allocations[key].update(items)  # Adding items to the set

    # Convert sets back to lists for final output
    final_allocations = {k: list(v) for k, v in merged_allocations.items()}
    return final_allocations


def get_allocation(sample):

  if (sample['Geography'] == 'ANY' and sample['Currency'] != 'ANY') or (sample['Geography'] != 'ANY' and sample['Currency'] == 'ANY'):
      all_allocations = []
      samples = generate_samples_based_on_choices(sample)
      for modified_sample in samples:
          #print(modified_sample)
          matching_Allocation = best_Allocation(modified_sample)
          all_allocations.append(matching_Allocation)

      # Merge the allocations into a single dictionary
      merged_allocations = merge_allocations(all_allocations)
      return merged_allocations

  elif (sample['Geography'] == 'ANY' and sample['Currency'] == 'ANY') :
    matching_Allocation = best_Allocation_2(sample)
    return matching_Allocation

  else :
    matching_Allocation = best_Allocation(sample)
    return matching_Allocation

def get_percentages(age, expertise, appetite, goals):
    age_percentage_mapping = {
        "20-35": 0.90,
        "35-50": 0.65,
        "50-65": 0.35,
        ">65": 0.00
    }

    expertise_percentage_mapping = {
        "No Knowledge": 0.00,
        "Beginner": 0.30,
        "Medium": 0.65,
        "Expert": 0.90
    }

    appetite_percentage_mapping = {
        "No Risk": 0.00,
        "Low Risk": 0.30,
        "Medium Risk": 0.65,
        "High Risk": 0.90,
        "I don’t know": 0.10
    }

    goals_percentage_mapping = {
        "Grow but No Risk": 0.00,
        "Grow but Low Risk": 0.30,
        "Grow with Medium Risk": 0.65,
        "Grow As much as possible": 0.90
    }

    age_percentage = age_percentage_mapping.get(age, 0.00)
    expertise_percentage = expertise_percentage_mapping.get(expertise, 0.00)
    appetite_percentage = appetite_percentage_mapping.get(appetite, 0.00)
    goals_percentage = goals_percentage_mapping.get(goals, 0.00)

    return age_percentage, expertise_percentage, appetite_percentage, goals_percentage



def calculate_risk_score(age_percent, expertise_percent, appetite_percent, goals_percent):
    # Define the weights for each feature
    weight_age = 0.10
    weight_expertise = 0.20
    weight_appetite = 0.40
    weight_goals = 0.30

    # Calculate the risk score based on the formula
    risk_score = (
        weight_age * age_percent +
        weight_expertise * expertise_percent +
        weight_appetite * appetite_percent +
        weight_goals * goals_percent
    ) * 10

    # Determine the risk category based on the calculated risk score
    if 0 <= risk_score < 3:
        risk_category = "No Risk"
    elif 3 <= risk_score < 5:
        risk_category = "Low Risk"
    elif 5 <= risk_score < 8:
        risk_category = "Medium Risk"
    elif 8 <= risk_score <= 10:
        risk_category = "High Risk"
    else:
        risk_category = "Undefined"

    return risk_score, risk_category

def risk_category(age_input, expertise_input, appetite_input, goals_input ):
  age_percentage, expertise_percentage, appetite_percentage, goals_percentage = get_percentages(age_input, expertise_input, appetite_input, goals_input)


  # Calculate risk scores for the example samples
  risk_score1, risk_category1 = calculate_risk_score(age_percentage, expertise_percentage, appetite_percentage, goals_percentage)

  return risk_score1 , risk_category1


# Function to calculate risk score and risk category based on row
def calculate_row_risk(row):
    risk_score, risk_cat = risk_category(row["Age"], row["Financial Expertise"], row["Risk Appetite"], row["Financial Goal"])
    return pd.Series([risk_score, risk_cat])


# Specify the paths to your files
current_dir = os.getcwd()

bonds_df_path = os.path.join(current_dir, 'data/clusters data/Bonds/Bonds_df.csv')
bonds_df_cluster_path = os.path.join(current_dir, 'data/clusters data/Bonds/Bonds_df_cluster.csv')
encoders_bonds_path = os.path.join(current_dir, 'data/clusters data/Bonds/Bonds_encoders.pkl')

# Load the CSV files
bonds_df = pd.read_csv(bonds_df_path)
bonds_df_cluster = pd.read_csv(bonds_df_cluster_path)

# Load the pickle file
with open(encoders_bonds_path, 'rb') as file:
    encoders_bonds = pickle.load(file)


# Define the file paths
stocks_df_path = os.path.join(current_dir, 'data/clusters data/Stocks/Stocks_df.csv')
stocks_df_cluster_path = os.path.join(current_dir, 'data/clusters data/Stocks/Stocks_df_cluster.csv')
encoders_stocks_path = os.path.join(current_dir, 'data/clusters data/Stocks/encoders_Stocks.pkl')

# Load the CSV files
stocks_df = pd.read_csv(stocks_df_path)
stocks_df_cluster = pd.read_csv(stocks_df_cluster_path)

# Load the pickle file
with open(encoders_stocks_path, 'rb') as file:
    encoders_stocks = pickle.load(file)



# Load the CSV files
deposit_df_path = os.path.join(current_dir, 'data/clusters data/Deposit/Deposit_df.csv')
deposit_df_cluster_path = os.path.join(current_dir, 'data/clusters data/Deposit/Deposit_df_cluster.csv')
encoders_deposit_path = os.path.join(current_dir, 'data/clusters data/Deposit/Deposit_encoders.pkl')

# Ensure the paths are correct and accessible
deposit_df = pd.read_csv(deposit_df_path)
deposit_df_cluster = pd.read_csv(deposit_df_cluster_path)

# Load the pickle file
with open(encoders_deposit_path, 'rb') as file:
    encoders_deposit = pickle.load(file)


geography_choices = ['FRANCE', 'UAE', 'EUROPE']
currency_choices = ['AED', 'USD', 'EUR']
investment_sector_choices = [ 'Automobile', 'Technology', 'Pharmaceutical', 'RealEstate']
islamic_products_choices = ['Islamic_Product', 'non_Islamic_Product']

# inv_geo_choices = ['ANY', 'Local', 'Regional', 'US', 'International']
inv_sigh_choices = ['3-6 Months', '6-12 Months', '3 years']
age_choices = ['20-35', '35-50', '50-65', '>65']
risk_appetite_choices = ['No Risk', 'Low Risk', 'Medium Risk', 'High Risk', 'I don’t know']
financial_expertise_choices = ['No Knowledge', 'Beginner', 'Medium', 'Expert']
inv_products_choices = ['ANY', 'Bonds', 'Equities', 'Cash' , 'Bonds&Equity', 'Equity&Deposit', 'Deposit&Bonds']
amount_choices = ['<1M', '1-3M', '3-5M', '5-10M', '>10M']
financial_goal_choices = ['Grow but No Risk', 'Grow but Low Risk', 'Grow with Medium Risk', 'Grow As much as possible']


def inferance(sample):
  Risk_Score, Risk_category = calculate_row_risk(sample)
  sample["Risk Score"] = Risk_Score
  sample["Risk category"] = Risk_category
  # Initialize the 'Allocation' column for the sample
  sample["Allocation"] = {}

  if sample["Inv Products"] == 'ANY':
      allocation = map_risk_to_investment_allocation(sample["Risk category"])
      sample["Allocation"] = allocation

  elif sample["Inv Products"] in ["Bonds&Equity", "Equity&Deposit", "Deposit&Bonds"]:
      allocation = map_risk_to_investment_allocation_2(sample["Risk category"], sample["Inv Products"])
      sample["Allocation"] = allocation

  elif sample["Inv Products"] == "Bonds":
      sample["Allocation"] = {"Deposit": 0.00, "Bonds": 1.00, "Equities": 0.00}

  elif sample["Inv Products"] == "Equities":
      sample["Allocation"] = {"Deposit": 0.00, "Bonds": 0.00, "Equities": 1.00}

  elif sample["Inv Products"] == "Cash":
      sample["Allocation"] = {"Deposit": 1.00, "Bonds": 0.00, "Equities": 0.00}


  matching_Allocation = get_allocation(sample)

  # Convert sample to DataFrame if it's not already
  if isinstance(sample, pd.Series):
      sample = sample.to_frame().transpose()

  sample["Best Selection Items"] = None

  # Add matching_Allocation to the sample DataFrame
  sample['Best Allocation'] = [matching_Allocation]
  if matching_Allocation == None:
    return sample


  rating_dict = {"AAA":12, "AA+":11,"AA":10,"AA-":9,"BBB":8,"BB+":7,"BB":6,"BB-":5,"CCC":4,"CC+":3,"CC":2,"CC-":1}
  for i in range(len(sample)):
      best_allocations = {"best_Deposit": [], "best_bonds": [], "best_stocks": []}
      for key, values in sample["Best Allocation"][i].items():
          best_yield = 0  # Initialize best yield
          best_rating = 0  # Initialize best rating

          if key == 'best_Deposit':
              for item in values:
                  deposit_item = deposit_df[deposit_df['Name'] == item]
                  if not deposit_item.empty:
                      yield_value = float(deposit_item["Yield"].values)

                      if yield_value > best_yield :
                          best_allocations["best_Deposit"] = [deposit_item.to_string(index=False, header=False)]
                          best_yield = yield_value

                      elif (yield_value == best_yield ):
                          best_allocations["best_Deposit"].append(deposit_item.to_string(index=False, header=False))
                          best_yield = yield_value

          elif key == 'best_bonds':
              for item in values:
                  bonds_item = bonds_df[bonds_df['Name'] == item]
                  if not bonds_item.empty:
                      yield_value = float(bonds_item["Yield"].values)
                      rating = int(rating_dict[bonds_item["Rating"].values[0]])

                      if yield_value > best_yield or (yield_value == best_yield and rating > best_rating):
                          best_allocations["best_bonds"] = [bonds_item.to_string(index=False, header=False)]
                          best_yield = yield_value
                          best_rating = rating

                      elif yield_value == best_yield and rating == best_rating:
                          best_allocations["best_bonds"].append(bonds_item.to_string(index=False, header=False))

          elif key == 'best_stocks':
              for item in values:
                  stocks_item = stocks_df[stocks_df['Name'] == item]
                  if not stocks_item.empty:
                      yield_value = float(stocks_item["Yield"].values)
                      rating = int(rating_dict[stocks_item["Rating"].values[0]])

                      if yield_value > best_yield or (yield_value == best_yield and rating > best_rating):
                          best_allocations["best_stocks"] = [stocks_item.to_string(index=False, header=False)]
                          best_yield = yield_value
                          best_rating = rating

                      elif yield_value == best_yield and rating == best_rating:
                          best_allocations["best_stocks"].append(stocks_item.to_string(index=False, header=False))
      # Filter out empty items from best_allocations
      best_allocations_filtered = {key: value for key, value in best_allocations.items() if value}
      if isinstance(sample, dict):
          # Convert dictionary to DataFrame
          sample = pd.DataFrame([sample])
      elif isinstance(sample, pd.Series):
          # Convert Series to DataFrame
          sample = sample.to_frame().T

      # Add non-empty items to the df_cleaned DataFrame
      sample.at[i, "Best Selection Items"] = best_allocations_filtered
      return sample

def answer(final_sample):
    final_sample = final_sample.squeeze()
    risk_category = final_sample['Risk category']

    # Access allocations as a dictionary with default values set to 0
    allocation = final_sample["Allocation"]
    deposit_allocation = allocation.get("Deposit", None)
    equities_allocation = allocation.get("Equities", None)
    bonds_allocation = allocation.get("Bonds", None)
    #sentence = f"Based on the customer’s profile, the customer choose a risk appetite of {final_sample['Risk Appetite']}, based on his input we calculated a risk score of {'%.3f' % final_sample['Risk Score']} which mean he is a {risk_category} customer, then the asset allocation would be "
    selection_Items = final_sample["Best Selection Items"]
    #print(selection_Items)
    best_Deposit = None
    best_bonds = None
    best_equities = None
    if selection_Items:
        best_Deposit = selection_Items.get("best_Deposit", None)
        best_bonds = selection_Items.get("best_bonds", None)
        best_equities = selection_Items.get("best_stocks", None)
    #sentence = f"Based on the customer’s profile, the customer chose a risk appetite of {final_sample['Risk Appetite']}. Based on this input, we calculated a risk score of {'%.3f' % final_sample['Risk Score']}, which means they are a {risk_category} customer. The asset allocation would be{f' {deposit_allocation}% deposit' if deposit_allocation not in (None, 0.0) else ''}{f', {equities_allocation}% equities' if equities_allocation not in (None, 0.0) else ''}{f', {bonds_allocation}% bonds' if bonds_allocation not in (None, 0.0) else ''}."
    sentence = f"Based on the customer’s profile, the customer choose a risk appetite of {final_sample['Risk Appetite']}, based on his input we calculated a risk score of {'%.3f' % final_sample['Risk Score']} which mean he is a {risk_category} customer, then the asset allocation would be{f' {deposit_allocation*100}% deposit,' if deposit_allocation not in (None, 0.0) else ''}{f' {equities_allocation*100}% equities,' if equities_allocation not in (None, 0.0) else ''}{f' {bonds_allocation*100}% bonds' if bonds_allocation not in (None, 0.0) else ''}."
    sentence += f"\nBest assets selection for this customer are\n"
    if selection_Items == None :

        items =''
        if deposit_allocation != 0:
          items += f"Deposit,"

        if equities_allocation != 0:
            items += f"Equities,"

        if bonds_allocation != 0:
            items += f"Bonds"
        sentence += f"\nThere are no matching {items} for this customer since there is not a {items} that is {'Islamic ' if final_sample['Islamic Products'] == 'Islamic_Product' else 'non-Islamic '}Product,{' in ' + final_sample['Currency'] + ' currency,' if final_sample['Currency'] != 'ANY' else ''}{' in ' + final_sample['Geography'] + ' geography,' if final_sample['Geography'] != 'ANY' else ''} in {final_sample['Investment Sector']} sector, exceeds {final_sample['Inv Sigh']}, and minimum buy less than customer’s amount."
        return sentence


    else :
      if bonds_allocation != None:
          selection_Items = final_sample["Best Selection Items"].get("best_bonds", 0)
          if selection_Items:
              bonds_name =''
              for best_bonds in selection_Items:
                  bonds_name +=best_bonds.split()[0]+' '
                  yield_percentage = best_bonds.split()[4]
                  yield_percentage = yield_percentage.rstrip('0').rstrip('.')  # Remove trailing zeros and the dot
              if len(bonds_name.split()) > 1:
                  bonds_phrase = "are"
                  bnd ='bonds'
              else:
                  bonds_phrase = "is"
                  bnd ='bond'
              sentence += f"\n{','.join(bonds_name.split())} {bonds_phrase} the best matching {bnd} for this customer, since it is {'Islamic ' if best_bonds.split()[7] == 'Islamic_Product' else 'non-Islamic '}Product, in {best_bonds.split()[5]} currency{', in ' + final_sample['Geography'] + ' geography' if final_sample['Geography'] != 'ANY' else ''}, in {best_bonds.split()[6]} sector, exceeds {final_sample['Inv Sigh']}, minimum buy less than customer’s amount, and the best yield of {yield_percentage}% and rating of {best_bonds.split()[-2]} over all other bonds."
          else:
            if final_sample['Inv Products'] not in ["Cash", "Equities"]:
              sentence += f"\nThere are no matching bonds for this customer, since there is not a bond that is {'Islamic ' if final_sample['Islamic Products'] == 'Islamic_Product' else 'non-Islamic '}Product,{' in ' + final_sample['Currency'] + ' currency,' if final_sample['Currency'] != 'ANY' else ''}{' in ' + final_sample['Geography'] + ' geography,' if final_sample['Geography'] != 'ANY' else ''} in {final_sample['Investment Sector']} sector, exceeds {final_sample['Inv Sigh']}, and minimum buy less than customer’s amount."

      if deposit_allocation != None:
          selection_Items =final_sample["Best Selection Items"].get("best_Deposit", 0)

          if selection_Items:
              Deposit_name = ''
              for best_Deposit in selection_Items:
                  Deposit_name +=best_Deposit.split()[0]+' '
                  yield_percentage = best_Deposit.split()[2]
                  yield_percentage = yield_percentage.rstrip('0').rstrip('.')  # Remove trailing zeros and the dot
              if len(Deposit_name.split()) > 1:
                  Deposit_phrase = "are"
                  dep ='deposits'
              else:
                  Deposit_phrase = "is"
                  dep ='deposit'
              sentence += f"\n{','.join(Deposit_name.split())} {Deposit_phrase} the best matching {dep} for this customer, since it is {'Islamic ' if best_Deposit.split()[6] == 'Islamic_Product' else 'non-Islamic '}Product{', in ' + final_sample['Currency'] + ' currency' if final_sample['Currency'] != 'ANY' else ''}, exceeds {final_sample['Inv Sigh']}, and minimum buy less than customer’s amount, and the best yield over all other deposit of {yield_percentage}%."

          else:
              if final_sample['Inv Products'] not in ["Bonds", "Equities"]:
                sentence += f"\nThere are no matching deposits for this customer, since there is not a deposit that is {'Islamic ' if final_sample['Islamic Products'] == 'Islamic_Product' else 'non-Islamic '}Product{', in ' + final_sample['Currency'] + ' currency' if final_sample['Currency'] != 'ANY' else ''}, exceeds {final_sample['Inv Sigh']}, and minimum buy less than customer’s amount."

      if equities_allocation != None:
          selection_Items = final_sample["Best Selection Items"].get("best_stocks", 0)

          if selection_Items:
              stocks_name = ''
              for best_stocks in selection_Items:
                  stocks_name +=best_stocks.split()[0]+' '
                  yield_percentage = best_stocks.split()[2]
                  yield_percentage = yield_percentage.rstrip('0').rstrip('.')  # Remove trailing zeros and the dot
              if len(stocks_name.split()) > 1:
                  stocks_phrase = "are"
                  eque ='equities'
              else:
                  stocks_phrase = "is"
                  eque ='equity'

              sentence += f"\n{','.join(stocks_name.split())} {stocks_phrase} the best matching {eque} for this customer, since it is {'Islamic ' if best_stocks.split()[5] == 'Islamic_Product' else 'non-Islamic '}Product{', in ' + final_sample['Currency'] + ' currency' if final_sample['Currency'] != 'ANY' else ''}, in {best_stocks.split()[4]} sector, and the best yield of {yield_percentage}% and rating of {best_stocks.split()[-2]} over all other equities."

          else:
            if final_sample['Inv Products'] not in ["Cash", "Bonds"]:
              sentence += f"\nThere are no matching equities for this customer, since there is not an equity that is {'Islamic ' if final_sample['Islamic Products'] == 'Islamic_Product' else 'non-Islamic '}Product,{' in ' + final_sample['Currency'] + ' currency,' if final_sample['Currency'] != 'ANY' else ''} in {final_sample['Investment Sector']} sector."
      return sentence



mapping = {
    'age_group': 'Age',
    'geography': 'Geography',
    'currency': 'Currency',
    'risk_appetite': 'Risk Appetite',
    'investment_horizon': 'Inv Sigh',
    'financial_expertise': 'Financial Expertise',
    'investment_sector': 'Investment Sector',
    'investment_products': 'Inv Products',
    'investment_amount': 'Amount',
    'financial_goal': 'Financial Goal',
    'islamic_compliance': 'Islamic Products', 
    'islamic_compliance_preference': 'Islamic Products', 
    'product_type': 'Islamic Products'
 }


def preprocess_input(text):
    # Adjusted approach to handle extraction and conversion of the dictionary from the text

    # Extract the substring that contains the dictionary content
    extracted_content = re.search(r"\{(.*?)\}", text, re.DOTALL)
    
    if extracted_content:
        # Further process the extracted content into a proper dictionary
        dict_content = extracted_content.group(1).strip()
        # Convert the string representation of the dictionary into an actual dictionary
        # This requires properly formatting the string to replace single quotes with double quotes for valid JSON
        dict_content = "{" + dict_content + "}"
        dict_content = dict_content.replace("'", '"')
        result_dict = json.loads(dict_content)

        return result_dict
"""
def convert_input(input_dict, mapping):
    converted = {}
    for key, value in input_dict.items():
        # Map the key using the mapping dictionary
        if key in mapping:
            new_key = mapping[key]
            # Here you can add more specific handling for different keys if needed
            # if 'islamic' in key or key == '':
            #     value = value.replace('non-Islamic Product', 'non_Islamic_Product')
            converted[new_key] = value

    if converted['Islamic Products'] == 'non-Islamic Product':
        converted['Islamic Products'] = 'non_Islamic_Product'
    return converted"""


def convert_input(input_dict):

    if input_dict['Islamic Products'].lower() == 'non-islamic product' or input_dict['Islamic Products'].lower() == 'non_islamic product':
        input_dict['Islamic Products'] = 'non_Islamic_Product'
    if input_dict['Investment Sector'] == 'Real Estate':
        input_dict['Investment Sector'] = 'RealEstate'
    if input_dict['Islamic Products'].lower() == 'islamic product':
        input_dict['Islamic Products'] = 'Islamic_Product'

    
        
    return input_dict

# app = Flask(__name__)
# @app.route('/api/inference', methods=['GET'])
def get_inference(data):
    # print(data)
    # print('='*10)
    data = preprocess_input(data)
    # print(data)
    # print(type(data))
    # print('='*10)
    
    # data = {'age_group': '>65', 'geography': 'EUROPE', 'currency': 'ANY', 'risk_appetite': 'High Risk', 'investment_horizon': '3 years', 'financial_expertise': 'Expert', 'investment_sector': 'Real Estate', 'investment_products': 'Cash', 'investment_amount': '>10M', 'financial_goal': 'Grow As much as possible', 'islamic_compliance': 'non-Islamic Product'}
    data= convert_input(data)
    
    sample = pd.DataFrame([data])
    # print(sample)
    # print('='*10)
    sample_series = sample.iloc[0]
    # print(sample_series)
    # print('='*10)
    best_assets = inferance(sample_series)
    result = answer(best_assets)
    # result = 'There are no matching equities for you, since there is not an equity that is'
    return result


# if __name__ == '__main__':
#     app.run()


#http://127.0.0.1:5000/api/inference