#! /usr/bin/env python3
# load data

import pandas as pd
import re
import sklearn as sk
from sklearn.preprocessing import LabelBinarizer
from sklearn.preprocessing import OneHotEncoder
import numpy as np

# 42 headers including 'income' in the last column and 'instance weight' in the middle 
# Prof. Ruiz says to remove the instance weight, but it's easier to remove by name so I'm leaving it for now.
attribute_descriptions = ['age', 'class of worker', 'detailed industry recode', 'detailed occupation recode', 'education', 'wage per hour', 'enroll in edu inst last wk', 'marital stat', 'major industry code', 'major occupation code', 'race', 'hispanic origin', 'sex', 'member of a labor union', 'reason for unemployment', 'full or part time employment stat', 'capital gains', 'capital losses', 'dividends from stocks', 'tax filer stat', 'region of previous residence', 'state of previous residence', 'detailed household and family stat', 'detailed household summary in household', 'instance weight', 'migration code-change in msa', 'migration code-change in reg', 'migration code-move within reg', 'live in this house 1 year ago', 'migration prev res in sunbelt', 'num persons worked for employer', 'family members under 18', 'country of birth father', 'country of birth mother', 'country of birth self', 'citizenship', 'own business or self employed', 'fill inc questionnaire for veteran\'s admin', 'veterans benefits', 'weeks worked in year', 'year', 'income']
attribute_names = ['AGE', 'CLSWKR', 'DTIND', 'DTOCC', 'HGA', 'HRSPAY', 'HSCOL', 'MARITL', 'MJIND', 'MJOCC', 'RACE', 'REORGN', 'SEX', 'UNMEM', 'UNTYPE', 'WKSTAT', 'CAPGAIN', 'CAPLOSS', 'DIVVAL', 'FILESTAT', 'GRINREG', 'GRINST', 'HHDFMX', 'HHDREL', 'MARSUPWT', 'MIGMTR1', 'MIGMTR3', 'MIGMTR4', 'MIGSAME', 'MIGSUN', 'NOEMP', 'PARENT', 'PEFNTVTY', 'PEMNTVTY', 'PENATVTY', 'PRCITSHP', 'SEOTR', 'VETQVA', 'VETYN', 'WKSWORK', 'YEAR', 'INCOME']
attribute_names = [x.lower() for x in attribute_names]
#print(attribute_names)
print('Number of attributes descriptions incl. \'instance weight\': {}'.format(len(attribute_descriptions)))
print('Number of attributes names incl. \'instance weight\': {}'.format(len(attribute_names)))
#print(len(feature_descriptions))

data = pd.read_csv('data/census-income.data', header=None)
data.columns = attribute_descriptions
#print(data)
# Remove period from the income attribute
data.loc[:, 'income'] = data.loc[:, 'income'].str.replace('.', '')
# Remove starting spaces
data = data.replace(to_replace=r'^\ *', value='', regex=True)
# Remove instance weight
data = data.drop(['instance weight'], axis=1)
print('Header titles')
print(data.columns.values)

# Preprocessing
# NOTE: Must use OneHotEncoding, DecisionTreeClassifier does not support categorical encoding. >:(
# Binary Encoding of sex (Female:0, Male:1)
lb = LabelBinarizer()
data.loc[:, 'sex'] = lb.fit_transform(data.loc[:, 'sex'])
# Binary Encoding of income (- 50000:0, 50000+:1)
lb = LabelBinarizer()
data.loc[:, 'income'] = lb.fit_transform(data.loc[:, 'income'])

# Saving data before onehot
data.to_csv('clean_data_not_hot.csv', index = False)

# OneHotEncoding
onehot_list_combine = []

#CLSWKR_onehot_values = ['Not in universe', 'Federal government', 'Local government', 'Never worked', 'Private', 'Self-employed-incorporated', 'Self-employed-not incorporated', 'State government', 'Without pay']
CLSWKR_onehot_values = data.loc[:, 'class of worker'].unique()
#print(CLSWKR_onehot_values)
CLSWKR_onehot_names = ['CLSWKR: ' + x for x in CLSWKR_onehot_values]
#print(CLSWKR_onehot_names)
onehot_list_combine.extend(CLSWKR_onehot_names)

#print(DTIND_onehot_values)
#DTIND_onehot_values = [0, 40, 44, 2, 43, 47, 48, 1, 11, 19, 24, 25, 32, 33, 34, 35, 36, 37, 38, 39, 4, 42, 45, 5, 15, 16, 22, 29, 31, 50, 14, 17, 18, 28, 3, 30, 41, 46, 51, 12, 13, 21, 23, 26, 6, 7, 9, 49, 27, 8, 10, 20]
DTIND_onehot_values = data.loc[:, 'detailed industry recode'].unique()
DTIND_onehot_names = ['DTIND: ' + str(x) for x in DTIND_onehot_values]
#print(DTIND_onehot_names)
onehot_list_combine.extend(DTIND_onehot_names)

#DTOCC_onehot_values = [0, 12, 31, 44, 19, 32, 10, 23, 26, 28, 29, 42, 40, 34, 14, 36, 38, 2, 20, 25, 37, 41, 27, 24, 30, 43, 33, 16, 45, 17, 35, 22, 18, 39, 3, 15, 13, 46, 8, 21, 9, 4, 6, 5, 1, 11, 7]
DTOCC_onehot_values = data.loc[:, 'detailed occupation recode'].unique()
DTOCC_onehot_names = ['DTOCC: ' + str(x) for x in DTOCC_onehot_values]
onehot_list_combine.extend(DTOCC_onehot_names)

#HGA_onehot_values = ['Children', '7th and 8th grade', '9th grade', '10th grade', 'High school graduate', '11th grade', '12th grade no diploma', '5th or 6th grade', 'Less than 1st grade', 'Bachelors degree(BA AB BS)', '1st 2nd 3rd or 4th grade', 'Some college but no degree', 'Masters degree(MA MS MEng MEd MSW MBA)', 'Associates degree-occup /vocational', 'Associates degree-academic program', 'Doctorate degree(PhD EdD)', 'Prof school degree (MD DDS DVM LLB JD)']
HGA_onehot_values = data.loc[:, 'education'].unique()
HGA_onehot_names = ['HGA: ' + x for x in HGA_onehot_values]
onehot_list_combine.extend(HGA_onehot_names)

#HSCOL_onehot_values = ['Not in universe', 'High school', 'College or university']
HSCOL_onehot_values = data.loc[:, 'enroll in edu inst last wk'].unique()
HSCOL_onehot_names = ['HSCOL: ' + x for x in HSCOL_onehot_values] 
onehot_list_combine.extend(HSCOL_onehot_names)

#MARITL_onehot_values = ['Never married', 'Married-civilian spouse present', 'Married-spouse absent', 'Separated', 'Divorced', 'Widowed', 'Married-A F spouse present']
MARITL_onehot_values = data.loc[:, 'marital stat'].unique()
MARITL_onehot_names = ['MARITL: ' + x for x in MARITL_onehot_values] 
onehot_list_combine.extend(MARITL_onehot_names)

#MJIND_onehot_values = ['Not in universe or children', 'Entertainment', 'Social services', 'Agriculture', 'Education', 'Public administration', 'Manufacturing-durable goods', 'Manufacturing-nondurable goods', 'Wholesale trade', 'Retail trade', 'Finance insurance and real estate', 'Private household services', 'Business and repair services', 'Personal services except private HH', 'Construction', 'Medical except hospital', 'Other professional services', 'Transportation', 'Utilities and sanitary services', 'Mining', 'Communications', 'Hospital services', 'Forestry and fisheries', 'Armed Forces']
MJIND_onehot_values = data.loc[:, 'major industry code'].unique()
MJIND_onehot_names = ['MJIND: ' + x for x in MJIND_onehot_values] 
onehot_list_combine.extend(MJIND_onehot_names)

#MJOCC_onehot_values = ['Not in universe', 'Professional specialty', 'Other service', 'Farming forestry and fishing', 'Sales', 'Adm support including clerical', 'Protective services', 'Handlers equip cleaners etc ', 'Precision production craft & repair', 'Technicians and related support', 'Machine operators assmblrs & inspctrs', 'Transportation and material moving', 'Executive admin and managerial', 'Private household services', 'Armed Forces']
MJOCC_onehot_values = data.loc[:, 'major occupation code'].unique()
MJOCC_onehot_names = ['MJOCC: ' + x for x in MJOCC_onehot_values] 
onehot_list_combine.extend(MJOCC_onehot_names)


#RACE_onehot_values = ['White', 'Black', 'Other', 'Amer Indian Aleut or Eskimo', 'Asian or Pacific Islander']
RACE_onehot_values = data.loc[:, 'race'].unique()
RACE_onehot_names = ['RACE: ' + x for x in RACE_onehot_values] 
onehot_list_combine.extend(RACE_onehot_names)

#REORGN_onehot_values = ['Mexican (Mexicano)', 'Mexican-American', 'Puerto Rican', 'Central or South American', 'All other', 'Other Spanish', 'Chicano', 'Cuban', 'Do not know', 'NA']
REORGN_onehot_values = data.loc[:, 'hispanic origin'].unique()
REORGN_onehot_names = ['REORGN: ' + x for x in REORGN_onehot_values] 
onehot_list_combine.extend(REORGN_onehot_names)

#UNMEM_onehot_values = ['Not in universe', 'No', 'Yes']
UNMEM_onehot_values = data.loc[:, 'member of a labor union'].unique()
UNMEM_onehot_names = ['UNMEM: ' + x for x in UNMEM_onehot_values] 
onehot_list_combine.extend(UNMEM_onehot_names)

#UNTYPE_onehot_values = ['Not in universe', 'Re-entrant', 'Job loser - on layoff', 'New entrant', 'Job leaver', 'Other job loser']
UNTYPE_onehot_values = data.loc[:, 'reason for unemployment'].unique()
UNTYPE_onehot_names = ['UNTYPE: ' + x for x in UNTYPE_onehot_values] 
onehot_list_combine.extend(UNTYPE_onehot_names)

#WKSTAT_onehot_values = ['Children or Armed Forces', 'Full-time schedules', 'Unemployed part- time', 'Not in labor force', 'Unemployed full-time', 'PT for non-econ reasons usually FT', 'PT for econ reasons usually PT', 'PT for econ reasons usually FT']
WKSTAT_onehot_values = data.loc[:, 'full or part time employment stat'].unique()
WKSTAT_onehot_names = ['WKSTAT: ' + x for x in WKSTAT_onehot_values] 
onehot_list_combine.extend(WKSTAT_onehot_names)

#FILESTAT_onehot_values = ['Nonfiler', 'Joint one under 65 & one 65+', 'Joint both under 65', 'Single', 'Head of household', 'Joint both 65+']
FILESTAT_onehot_values = data.loc[:, 'tax filer stat'].unique()
FILESTAT_onehot_names = ['FILESTAT: ' + x for x in FILESTAT_onehot_values] 
onehot_list_combine.extend(FILESTAT_onehot_names)

#GRINREG_onehot_values = ['Not in universe', 'South', 'Northeast', 'West', 'Midwest', 'Abroad']
GRINREG_onehot_values = data.loc[:, 'region of previous residence'].unique()
GRINREG_onehot_names = ['GRINREG: ' + x for x in GRINREG_onehot_values] 
onehot_list_combine.extend(GRINREG_onehot_names)


# Missing Values
#GRINST_onehot_values = ['Not in universe', 'Utah', 'Michigan', 'North Carolina', 'North Dakota', 'Virginia', 'Vermont', 'Wyoming', 'West Virginia', 'Pennsylvania', 'Abroad', 'Oregon', 'California', 'Iowa', 'Florida', 'Arkansas', 'Texas', 'South Carolina', 'Arizona', 'Indiana', 'Tennessee', 'Maine', 'Alaska', 'Ohio', 'Montana', 'Nebraska', 'Mississippi', 'District of Columbia', 'Minnesota', 'Illinois', 'Kentucky', 'Delaware', 'Colorado', 'Maryland', 'Wisconsin', 'New Hampshire', 'Nevada', 'New York', 'Georgia', 'Oklahoma', 'New Mexico', 'South Dakota', 'Missouri', 'Kansas', 'Connecticut', 'Louisiana', 'Alabama', 'Massachusetts', 'Idaho', 'New Jersey', '?']
GRINST_onehot_values = data.loc[:, 'state of previous residence'].unique()
GRINST_onehot_names = ['GRINST: ' + x for x in GRINST_onehot_values] 
onehot_list_combine.extend(GRINST_onehot_names)

#HHDFMX_onehot_values = ['Child <18 never marr not in subfamily', 'Other Rel <18 never marr child of subfamily RP', 'Other Rel <18 never marr not in subfamily', 'Grandchild <18 never marr child of subfamily RP', 'Grandchild <18 never marr not in subfamily', 'Secondary individual', 'In group quarters', 'Child under 18 of RP of unrel subfamily', 'RP of unrelated subfamily', 'Spouse of householder', 'Householder', 'Other Rel <18 never married RP of subfamily', 'Grandchild <18 never marr RP of subfamily', 'Child <18 never marr RP of subfamily', 'Child <18 ever marr not in subfamily', 'Other Rel <18 ever marr RP of subfamily', 'Child <18 ever marr RP of subfamily', 'Nonfamily householder', 'Child <18 spouse of subfamily RP', 'Other Rel <18 spouse of subfamily RP', 'Other Rel <18 ever marr not in subfamily', 'Grandchild <18 ever marr not in subfamily', 'Child 18+ never marr Not in a subfamily', 'Grandchild 18+ never marr not in subfamily', 'Child 18+ ever marr RP of subfamily', 'Other Rel 18+ never marr not in subfamily', 'Child 18+ never marr RP of subfamily', 'Other Rel 18+ ever marr RP of subfamily', 'Other Rel 18+ never marr RP of subfamily', 'Other Rel 18+ spouse of subfamily RP', 'Other Rel 18+ ever marr not in subfamily', 'Child 18+ ever marr Not in a subfamily', 'Grandchild 18+ ever marr not in subfamily', 'Child 18+ spouse of subfamily RP', 'Spouse of RP of unrelated subfamily', 'Grandchild 18+ ever marr RP of subfamily', 'Grandchild 18+ never marr RP of subfamily', 'Grandchild 18+ spouse of subfamily RP']
HHDFMX_onehot_values = data.loc[:, 'detailed household and family stat'].unique()
HHDFMX_onehot_names = ['HHDFMX: ' + x for x in HHDFMX_onehot_values] 
onehot_list_combine.extend(HHDFMX_onehot_names)

#HHDREL_onehot_values = ['Child under 18 never married', 'Other relative of householder', 'Nonrelative of householder', 'Spouse of householder', 'Householder', 'Child under 18 ever married', 'Group Quarters- Secondary individual', 'Child 18 or older']
HHDREL_onehot_values = data.loc[:, 'detailed household summary in household'].unique()
HHDREL_onehot_names = ['HHDREL: ' + x for x in HHDREL_onehot_values] 
onehot_list_combine.extend(HHDREL_onehot_names)

# Missing Values
#MIGMTR1_onehot_values = ['Not in universe', 'Nonmover', 'MSA to MSA', 'NonMSA to nonMSA', 'MSA to nonMSA', 'NonMSA to MSA', 'Abroad to MSA', 'Not identifiable', 'Abroad to nonMSA', '?']
MIGMTR1_onehot_values = data.loc[:, 'migration code-change in msa'].unique()
MIGMTR1_onehot_names = ['MIGMTR1: ' + x for x in MIGMTR1_onehot_values] 
onehot_list_combine.extend(MIGMTR1_onehot_names)

# Missing Values
#MIGMTR3_onehot_values = ['Not in universe', 'Nonmover', 'Same county', 'Different county same state', 'Different state same division', 'Abroad', 'Different region', 'Different division same region', '?']
MIGMTR3_onehot_values = data.loc[:, 'migration code-change in reg'].unique()
MIGMTR3_onehot_names = ['MIGMTR3: ' + x for x in MIGMTR3_onehot_values] 
onehot_list_combine.extend(MIGMTR3_onehot_names)


# Missing Values
#MIGMTR4_onehot_values = ['Not in universe', 'Nonmover', 'Same county', 'Different county same state', 'Different state in West', 'Abroad', 'Different state in Midwest', 'Different state in South', 'Different state in Northeast', '?']
MIGMTR4_onehot_values = data.loc[:, 'migration code-move within reg'].unique()
MIGMTR4_onehot_names = ['MIGMTR4: ' + x for x in MIGMTR4_onehot_values] 
onehot_list_combine.extend(MIGMTR4_onehot_names)

#MIGSAME_onehot_values = ['Not in universe under 1 year old', 'Yes', 'No']
MIGSAME_onehot_values = data.loc[:, 'live in this house 1 year ago'].unique()
MIGSAME_onehot_names = ['MIGSAME: ' + x for x in MIGSAME_onehot_values] 
onehot_list_combine.extend(MIGSAME_onehot_names)

# Missing Values
#MIGSUN_onehot_values = ['Not in universe', 'Yes', 'No', '?']
MIGSUN_onehot_values = data.loc[:, 'migration prev res in sunbelt'].unique()
MIGSUN_onehot_names = ['MIGSUN: ' + x for x in MIGSUN_onehot_values] 
onehot_list_combine.extend(MIGSUN_onehot_names)

#PARENT_onehot_values = ['Both parents present', 'Neither parent present', 'Mother only present', 'Father only present', 'Not in universe']
PARENT_onehot_values = data.loc[:, 'family members under 18'].unique()
PARENT_onehot_names = ['PARENT: ' + x for x in PARENT_onehot_values] 
onehot_list_combine.extend(PARENT_onehot_names)

# Missing Values
#PEFNTVTY_onehot_values = ['Mexico', 'United-States', 'Puerto-Rico', 'Dominican-Republic', 'Jamaica', 'Cuba', 'Portugal', 'Nicaragua', 'Peru', 'Ecuador', 'Guatemala', 'Philippines', 'Canada', 'Columbia', 'El-Salvador', 'Japan', 'England', 'Trinadad&Tobago', 'Honduras', 'Germany', 'Taiwan', 'Outlying-U S (Guam USVI etc)', 'India', 'Vietnam', 'China', 'Hong Kong', 'Cambodia', 'France', 'Laos', 'Haiti', 'South Korea', 'Iran', 'Greece', 'Italy', 'Poland', 'Thailand', 'Yugoslavia', 'Holand-Netherlands', 'Ireland', 'Scotland', 'Hungary', 'Panama', '?']
PEFNTVTY_onehot_values = data.loc[:, 'country of birth father'].unique()
PEFNTVTY_onehot_names = ['PEFNTVTY: ' + x for x in PEFNTVTY_onehot_values] 
onehot_list_combine.extend(PEFNTVTY_onehot_names)

#PEMNTVTY_onehot_values = ['India', 'Mexico', 'United-States', 'Puerto-Rico', 'Dominican-Republic', 'England', 'Honduras', 'Peru', 'Guatemala', 'Columbia', 'El-Salvador', 'Philippines', 'France', 'Ecuador', 'Nicaragua', 'Cuba', 'Outlying-U S (Guam USVI etc)', 'Jamaica', 'South Korea', 'China', 'Germany', 'Yugoslavia', 'Canada', 'Vietnam', 'Japan', 'Cambodia', 'Ireland', 'Laos', 'Haiti', 'Portugal', 'Taiwan', 'Holand-Netherlands', 'Greece', 'Italy', 'Poland', 'Thailand', 'Trinadad&Tobago', 'Hungary', 'Panama', 'Hong Kong', 'Scotland', 'Iran']
PEMNTVTY_onehot_values = data.loc[:, 'country of birth mother'].unique()
PEMNTVTY_onehot_names = ['PEMNTVTY: ' + x for x in PEMNTVTY_onehot_values] 
onehot_list_combine.extend(PEMNTVTY_onehot_names)

# Missing Values
#PENATVTY_onehot_values = ['United-States', 'Mexico', 'Puerto-Rico', 'Peru', 'Canada', 'South Korea', 'India', 'Japan', 'Haiti', 'El-Salvador', 'Dominican-Republic', 'Portugal', 'Columbia', 'England', 'Thailand', 'Cuba', 'Laos', 'Panama', 'China', 'Germany', 'Vietnam', 'Italy', 'Honduras', 'Outlying-U S (Guam USVI etc)', 'Hungary', 'Philippines', 'Poland', 'Ecuador', 'Iran', 'Guatemala', 'Holand-Netherlands', 'Taiwan', 'Nicaragua', 'France', 'Jamaica', 'Scotland', 'Yugoslavia', 'Hong Kong', 'Trinadad&Tobago', 'Greece', 'Cambodia', 'Ireland', '?']
#onehot_categories = ['class of worker', 'detailed industry recode', 'detailed occupation recode', 'education', 'enroll in edu inst last wk', 'marital stat', 'major industry code', 'major occupation code', 'race', 'hispanic origin', 'member of a labor union', 'reason for unemployment', 'full or part time employment stat', 'tax filer stat', 'region of previous residence', 'state of previous residence', 'detailed household and family stat', 'detailed household summary in household', 'migration code-change in msa', 'migration code-change in reg', 'migration code-move within reg', 'live in this house 1 year ago', 'migration prev res in sunbelt', 'family members under 18', 'country of birth father', 'country of birth mother', 'country of birth self', 'citizenship', 'own business or self employed', 'fill inc questionnaire for veteran\'s admin', 'veterans benefits', 'year']
PENATVTY_onehot_values = data.loc[:, 'country of birth self'].unique()
PENATVTY_onehot_names = ['PENATVTY: ' + x for x in PENATVTY_onehot_values] 
onehot_list_combine.extend(PENATVTY_onehot_names)

#PRCITSHP_onehot_values = ['Native- Born in the United States', 'Foreign born- Not a citizen of U S ', 'Native- Born in Puerto Rico or U S Outlying', 'Native- Born abroad of American Parent(s)', 'Foreign born- U S citizen by naturalization']
PRCITSHP_onehot_values = data.loc[:, 'citizenship'].unique()
PRCITSHP_onehot_names = ['PRCITSHP: ' + x for x in PRCITSHP_onehot_values] 
onehot_list_combine.extend(PRCITSHP_onehot_names)

#SEOTR_onehot_values = [0, 2, 1]
SEOTR_onehot_values = data.loc[:, 'own business or self employed'].unique()
SEOTR_onehot_names = ['SEOTR: ' + str(x) for x in SEOTR_onehot_values]
onehot_list_combine.extend(SEOTR_onehot_names)

#VETQVA_onehot_values = ['Not in universe', 'Yes', 'No']
VETQVA_onehot_values = data.loc[:, 'fill inc questionnaire for veteran\'s admin'].unique()
VETQVA_onehot_names = ['VETQVA: ' + x for x in VETQVA_onehot_values] 
onehot_list_combine.extend(VETQVA_onehot_names)

#VETYN_onehot_values = [0, 2, 1]
VETYN_onehot_values = data.loc[:, 'veterans benefits'].unique()
VETYN_onehot_names = ['VETYN: ' + str(x) for x in VETYN_onehot_values]
onehot_list_combine.extend(VETYN_onehot_names)

#YEAR_onehot_values = [94, 95]
YEAR_onehot_values = data.loc[:, 'year'].unique()
YEAR_onehot_names = ['YEAR: ' + str(x) for x in YEAR_onehot_values]
onehot_list_combine.extend(YEAR_onehot_names)

#print(onehot_list_combine)
print('Number of OneHotAttributes: {}'.format(len(onehot_list_combine)))
onehot_categories = ['class of worker', 'detailed industry recode', 'detailed occupation recode', 'education', 'enroll in edu inst last wk', 'marital stat', 'major industry code', 'major occupation code', 'race', 'hispanic origin', 'member of a labor union', 'reason for unemployment', 'full or part time employment stat', 'tax filer stat', 'region of previous residence', 'state of previous residence', 'detailed household and family stat', 'detailed household summary in household', 'migration code-change in msa', 'migration code-change in reg', 'migration code-move within reg', 'live in this house 1 year ago', 'migration prev res in sunbelt', 'family members under 18', 'country of birth father', 'country of birth mother', 'country of birth self', 'citizenship', 'own business or self employed', 'fill inc questionnaire for veteran\'s admin', 'veterans benefits', 'year']

# IDK what to do about sex and income
categorical_features_array = data.loc[:, onehot_categories].to_numpy()
#print(categorical_features_array)
#onehot_list_combine = CLSWKR_onehot_names + DTIND_onehot_names + DTOCC_onehot_names + HGA_onehot_names + HSCOL_onehot_names + MARITL_onehot_values + MJIND_onehot_values + MJOCC_onehot_names + RACE_onehot_names + REORGN_onehot_names + UNMEM_onehot_names + UNTYPE_onehot_names + WKSTAT_onehot_names + FILESTAT_onehot_names + GRINREG_onehot_names + GRINST_onehot_names + HHDFMX_onehot_names + HHDREL_onehot_names + MIGMTR1_onehot_names + MIGMTR3_onehot_names + MIGMTR4_onehot_names + MIGSAME_onehot_names + MIGSUN_onehot_names + PARENT_onehot_names + PEFNTVTY_onehot_names + PEMNTVTY_onehot_names + PENATVTY_onehot_names + PRCITSHP_onehot_names + SEOTR_onehot_names + VETQVA_onehot_names + VETYN_onehot_names + YEAR_onehot_names

onehot = sk.preprocessing.OneHotEncoder(categories=[CLSWKR_onehot_values, DTIND_onehot_values, DTOCC_onehot_values, HGA_onehot_values, HSCOL_onehot_values, MARITL_onehot_values, MJIND_onehot_values, MJOCC_onehot_values, RACE_onehot_values, REORGN_onehot_values, UNMEM_onehot_values, UNTYPE_onehot_values, WKSTAT_onehot_values, FILESTAT_onehot_values, GRINREG_onehot_values, GRINST_onehot_values, HHDFMX_onehot_values, HHDREL_onehot_values, MIGMTR1_onehot_values, MIGMTR3_onehot_values, MIGMTR4_onehot_values, MIGSAME_onehot_values, MIGSUN_onehot_values, PARENT_onehot_values, PEFNTVTY_onehot_values, PEMNTVTY_onehot_values, PENATVTY_onehot_values, PRCITSHP_onehot_values, SEOTR_onehot_values, VETQVA_onehot_values, VETYN_onehot_values, YEAR_onehot_values], sparse=False, dtype=int)
onehot_result = onehot.fit_transform(categorical_features_array)
data = data.drop(columns=onehot_categories)
#print(onehot_list_combine)
for i in range(len(onehot_list_combine)):
    data[onehot_list_combine[i]] = onehot_result[:, i]
print(data)

# Write the data to a csv
data.to_csv('clean_census_income.csv', index=False)


# creating and saving log-census data

def log_func(x):
	return np.log(x+.01)

mod_data = data.copy()
mod_data['dividends from stocks'] = mod_data['dividends from stocks'].apply(log_func)
mod_data['wage per hour'] = mod_data['wage per hour'].apply(log_func)
mod_data['capital gains'] = mod_data['capital gains'].apply(log_func)
mod_data['capital losses'] = mod_data['capital losses'].apply(log_func)

print(mod_data['dividends from stocks'])
print(data['dividends from stocks'])
print(mod_data)
mod_data.to_csv('log_cci.csv', index = False)

# https://stackoverflow.com/questions/27928275/find-p-value-significance-in-scikit-learn-linearregression#42677750
