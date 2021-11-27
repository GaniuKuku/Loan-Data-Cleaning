#!/usr/bin/env python
# coding: utf-8

# # Importing Packages

# In[437]:


import numpy as np


# In[438]:


np.set_printoptions(suppress = True, linewidth = 100, precision = 2)


# # Importing the Dataset

# In[439]:


# Using genfromtxt so that str can convert to float
raw_data= np.genfromtxt("loan-data.csv", delimiter = ';', 
                        skip_header = 1, 
                        autostrip = True)
raw_data


# # Checking for Missing Values

# In[440]:


np.isnan(raw_data).sum()


# In[441]:


# Creating two variables
temp_fill = np.nanmax(raw_data) + 1 # filler for all missing entries of the dataset
temp_mean = np.nanmean(raw_data, axis = 0)  # hold the means for every column


# In[442]:


# A column of only strings is automatically filled with NaNs


# In[443]:


# Displaying columns with text values
temp_mean


# In[444]:


# 8 columns store strings, any column with mean of NaN contains no number


# In[445]:


# Extracting the min and max value for each numeric column
temporary_stats = np.array([np.nanmin(raw_data, axis = 0),
                           temp_mean,
                           np.nanmax(raw_data, axis = 0)])


# In[446]:


temporary_stats


# # Splitting the Data

# In[447]:


# Create a variabe for only text data
col_str = np.argwhere(np.isnan(temp_mean)).squeeze()


# In[448]:


col_str


# In[449]:


# Create a variable for only numeric data
col_num = np.argwhere(np.isnan(temp_mean) == False).squeeze()
col_num


# Re-importing the Data

# In[450]:


loan_data_str = np.genfromtxt("loan-data.csv",
                                  delimiter = ';',
                                  skip_header = 1,
                                  autostrip = True, 
                                  usecols = col_str,
                                  dtype = np.str)
loan_data_str


# In[451]:


loan_data_num = np.genfromtxt("loan-data.csv",
                                  delimiter = ';',
                                  autostrip = True,
                                  skip_header = 1,
                                  usecols = col_num,
                                  filling_values = temp_fill)
loan_data_num


# # Getting Columns Names

# In[452]:


header_full = np.genfromtxt("loan-data.csv",
                            delimiter = ';',
                            autostrip = True,
                            skip_footer = raw_data.shape[0],
                            dtype = np.str)
header_full


# In[453]:


# Splitting columns names into two variables
header_strings, header_numeric = header_full[col_str], header_full[col_num]


# In[454]:


header_numeric  # Columns with intergers


# In[455]:


header_strings  # Columns with strings


# # Preprocessing String Dataset

# Creating Checkpoints

# In[456]:


def checkpoint(file_name, checkpoint_header, checkpoint_data):
    np.savez(file_name, header = checkpoint_header, data = checkpoint_data)
    checkpoint_variable = np.load(file_name + ".npz")
    return(checkpoint_variable)


# In[457]:


checkpoint_test = checkpoint("checkpoint-test", header_strings, loan_data_str)


# Manipulating Strings Columns

# In[458]:


# Preprocessing each columns


# Issue Date

# In[459]:


header_strings[0] = "issue_date" # Changing column name


# In[460]:


loan_data_str


# In[461]:


loan_data_str[:,0]


# In[462]:


np.unique(loan_data_str[:,0])


# In[463]:


# column contains missing data
# all loans are from 2015


# In[464]:


# removing '-15'


# In[465]:


np.chararray.strip(loan_data_str[:,0], "-15")


# In[466]:


loan_data_str[:,0] = np.chararray.strip(loan_data_str[:,0], "-15")


# In[467]:


np.unique(loan_data_str[:,0])


# In[468]:


# Representing month values as integers
months = np.array(['', 'Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec'])


# In[469]:


# Using for loop and setting missing value to 0
for i in range(13):
        loan_data_str[:,0] = np.where(loan_data_str[:,0] == months[i],
                                          i,
                                          loan_data_str[:,0])


# In[470]:


np.unique(loan_data_str[:,0])


# Loan Status

# In[471]:


header_strings


# In[472]:


loan_data_str[:,1]


# In[473]:


np.unique(loan_data_str[:,1])


# In[474]:


# Contains missing values
# We only care if the candidate is in stable financial condition
# We need to split all possible values into group ('good', 'bad')


# In[475]:


# 'current', 'Fully Paid', 'In Grace Period', 'Issued' are good
# 'missing values', 'Charged off', 'Default' are bad


# In[476]:


# good = 1, bad = 0


# In[477]:


status_bad = np.array(['','Charged Off','Default','Late (31-120 days)'])


# In[478]:


loan_data_str[:,1] = np.where(np.isin(loan_data_str[:,1], status_bad),0,1)


# In[479]:


np.unique(loan_data_str[:,1])


# Term

# In[480]:


header_strings


# In[481]:


np.unique(loan_data_str[:,2])


# In[482]:


# contain missing value
# stripping 'months'
loan_data_str[:,2] = np.chararray.strip(loan_data_str[:,2], " months")
loan_data_str[:,2]


# In[483]:


header_strings[2] = "term_months"


# In[484]:


loan_data_str[:,2] = np.where(loan_data_str[:,2] == '', 
                                  '60', 
                                  loan_data_str[:,2])
loan_data_str[:,2]

# When we have missing data in CRM, we assume the worst


# In[485]:


np.unique(loan_data_str[:,2])


# In[ ]:





# Grade and Subgrade

# In[486]:


header_strings


# In[487]:


(loan_data_str[:,3])


# In[488]:


np.unique(loan_data_str[:,3])


# In[489]:


np.unique(loan_data_str[:,4])


# In[490]:


# contain missing values


# In[491]:


# Using Grade to fill Subgrade
for i in np.unique(loan_data_str[:,3])[1:]:
    loan_data_str[:,4] = np.where((loan_data_str[:,4] == '') & (loan_data_str[:,3] == i),
                                      i + '5',
                                      loan_data_str[:,4])


# In[492]:


np.unique(loan_data_str[:,4], return_counts = True)


# In[493]:


# There still rows with neither subgrade nor grade
# checking for the missing value
np.unique(loan_data_str[:,4], return_counts = True)


# In[494]:


loan_data_str[:,4] = np.where(loan_data_str[:,4] == '',
                                  'H1',
                                  loan_data_str[:,4])


# In[495]:


np.unique(loan_data_str[:,4])


# In[496]:


# Removing Grade
loan_data_str = np.delete(loan_data_str, 3, axis = 1)


# In[497]:


loan_data_str[:,3]


# In[498]:


header_strings = np.delete(header_strings, 3)


# In[499]:


header_strings[3]


# # Converting Subgrade
# 

# In[500]:


np.unique(loan_data_str[:,3])


# In[501]:


keys = list(np.unique(loan_data_str[:,3]))                         
values = list(range(1, np.unique(loan_data_str[:,3]).shape[0] + 1)) 
dict_sub_grade = dict(zip(keys, values))


# In[502]:


dict_sub_grade


# In[503]:


for i in np.unique(loan_data_str[:,3]):
        loan_data_str[:,3] = np.where(loan_data_str[:,3] == i, 
                                          dict_sub_grade[i],
                                          loan_data_str[:,3])


# In[504]:


np.unique(loan_data_str[:,3])


# Verification Status

# In[505]:


header_strings


# In[506]:


(loan_data_str[:,4])


# In[507]:


np.unique(loan_data_str[:,4])


# In[508]:


loan_data_str[:,4] = np.where((loan_data_str[:,4] == '') | (loan_data_str[:,4] == 'Not Verified'), 0, 1)


# In[509]:


np.unique(loan_data_str[:,4])


# URL

# In[510]:


loan_data_str[:,5]


# In[511]:


# Removing repeating part using strip function
np.chararray.strip(loan_data_str[:,5], "https://www.lendingclub.com/browse/loanDetail.action?loan_id=")


# In[512]:


loan_data_str[:,5] = np.chararray.strip(loan_data_str[:,5], "https://www.lendingclub.com/browse/loanDetail.action?loan_id=")


# In[513]:


header_full


# In[514]:


loan_data_num[:,0].astype(dtype = np.int32)


# In[515]:


loan_data_str[:,5].astype(dtype = np.int32)


# In[516]:


np.array_equal(loan_data_num[:,0].astype(dtype = np.int32), loan_data_str[:,5].astype(dtype = np.int32))


# In[517]:


loan_data_str = np.delete(loan_data_str, 5, axis = 1)
header_strings = np.delete(header_strings, 5)


# In[518]:


loan_data_str[:,5]


# In[519]:


header_strings


# In[520]:


loan_data_num[:,0]


# In[521]:


header_numeric


# State Address

# In[522]:


header_strings


# In[523]:


header_strings[5] = "state_address"


# In[524]:


states_names, states_count = np.unique(loan_data_str[:,5], return_counts = True)
states_count_sorted = np.argsort(-states_count)
states_names[states_count_sorted], states_count[states_count_sorted]


# In[525]:


loan_data_str[:,5] = np.where(loan_data_str[:,5] == '', 
                                  0, 
                                  loan_data_str[:,5])


# In[526]:


states_west = np.array(['WA', 'OR','CA','NV','ID','MT', 'WY','UT','CO', 'AZ','NM','HI','AK'])
states_south = np.array(['TX','OK','AR','LA','MS','AL','TN','KY','FL','GA','SC','NC','VA','WV','MD','DE','DC'])
states_midwest = np.array(['ND','SD','NE','KS','MN','IA','MO','WI','IL','IN','MI','OH'])
states_east = np.array(['PA','NY','NJ','CT','MA','VT','NH','ME','RI'])


# In[527]:


loan_data_str[:,5] = np.where(np.isin(loan_data_str[:,5], states_west), 1, loan_data_str[:,5])
loan_data_str[:,5] = np.where(np.isin(loan_data_str[:,5], states_south), 2, loan_data_str[:,5])
loan_data_str[:,5] = np.where(np.isin(loan_data_str[:,5], states_midwest), 3, loan_data_str[:,5])
loan_data_str[:,5] = np.where(np.isin(loan_data_str[:,5], states_east), 4, loan_data_str[:,5])


# In[528]:


np.unique(loan_data_str[:,5])


# # Converting to Numbers

# In[529]:


loan_data_str


# In[530]:


loan_data_str = loan_data_str.astype(np.int)


# In[531]:


loan_data_str


# Checkpoint for Strings

# In[532]:


checkpoint_strings = checkpoint("Checkpoint-Strings", header_strings, loan_data_str)


# In[533]:


checkpoint_strings["header"]


# In[534]:


checkpoint_strings["data"]


# In[535]:


np.array_equal(checkpoint_strings['data'], loan_data_str)


# # Manipulating Numeric Data

# In[536]:


loan_data_num


# In[537]:


np.isnan(loan_data_num).sum()


# Substituting 'Filler' Values

# In[538]:


header_numeric


# ID

# In[539]:


temp_fill


# In[540]:


# We must check whether any of the elements equal the temp_fill
np.isin(loan_data_num[:,0], temp_fill)


# In[541]:


np.isin(loan_data_num[:,0], temp_fill).sum()


# In[542]:


header_numeric


# Temporary Stats

# In[543]:


temporary_stats[:, col_num]


# Funded Amount

# In[544]:


# setting filler value equal to minimum
loan_data_num[:,2]


# In[545]:


loan_data_num[:,2] = np.where(loan_data_num[:,2] == temp_fill, 
                                  temporary_stats[0, col_num[2]],
                                  loan_data_num[:,2])
loan_data_num[:,2]


# In[546]:


temporary_stats[0,col_num[3]]


# Loan Amount, Interest Rate, Total Payment, Installment

# In[547]:


header_numeric


# In[548]:


for i in [1,3,4,5]:
    loan_data_num[:,i] = np.where(loan_data_num[:,i] == temp_fill,
                                      temporary_stats[2, col_num[i]],
                                      loan_data_num[:,i])


# In[549]:


loan_data_num


# # Currency Conversion

# The Exchange Rate

# In[550]:


EUR_USD = np.genfromtxt("EUR-USD.csv", 
                        delimiter = ',', 
                        autostrip = True, 
                        skip_header = 1, 
                        usecols = 3)
EUR_USD


# In[551]:


loan_data_str[:,0]


# In[552]:


exchange_rate = loan_data_str[:,0]

for i in range(1,13):
    exchange_rate = np.where(exchange_rate == i,
                             EUR_USD[i-1],
                             exchange_rate)    

exchange_rate = np.where(exchange_rate == 0,
                         np.mean(EUR_USD),
                         exchange_rate)

exchange_rate


# In[553]:


exchange_rate.shape


# In[554]:


loan_data_num.shape


# In[555]:


exchange_rate = np.reshape(exchange_rate, (10000,1))


# In[556]:


loan_data_num = np.hstack((loan_data_num, exchange_rate))


# In[557]:


header_numeric = np.concatenate((header_numeric, np.array(['exchange_rate'])))
header_numeric


# USD to EUR

# In[558]:


header_numeric


# In[559]:


columns_dollar = np.array([1,2,4,5])


# In[560]:


loan_data_num[:,6]


# In[561]:


for i in columns_dollar:
    loan_data_num = np.hstack((loan_data_num, np.reshape(loan_data_num[:,i] / loan_data_num[:,6], (10000,1))))


# In[562]:


loan_data_num


# Expanding the header

# In[563]:


header_additional = np.array([column_name + '_EUR' for column_name in header_numeric[columns_dollar]])


# In[564]:


header_additional


# In[565]:


header_numeric = np.concatenate((header_numeric, header_additional))


# In[566]:


header_numeric


# In[567]:


header_numeric[columns_dollar] = np.array([column_name + '_USD' for column_name in header_numeric[columns_dollar]])


# In[568]:


header_numeric


# In[569]:


# We will rearrange the columns so that each EUR column follows it corresponding USD column
columns_index_order = [0,1,7,2,8,3,4,9,5,10,6]


# In[570]:


header_numeric = header_numeric[columns_index_order]


# In[571]:


loan_data_num


# In[572]:


loan_data_num = loan_data_num[:,columns_index_order]


# Interest Rate

# In[573]:


header_numeric


# In[574]:


loan_data_num[:,5]


# In[575]:


loan_data_num[:,5] = loan_data_num[:,5]/100


# In[576]:


loan_data_num[:,5]


# Numeric Checkpoint

# In[577]:


checkpoint_numeric = checkpoint("Checkpoint-Numeric", header_numeric, loan_data_num)


# In[578]:


checkpoint_numeric['header'], checkpoint_numeric['data']


# # Creating the "Complete" Dataset

# In[579]:


checkpoint_strings['data'].shape


# In[580]:


checkpoint_numeric['data'].shape


# In[581]:


loan_data = np.hstack((checkpoint_numeric['data'], checkpoint_strings['data']))


# In[582]:


loan_data


# In[583]:


# checking for missing values
np.isnan(loan_data).sum()


# In[584]:


header_full = np.concatenate((checkpoint_numeric['header'], checkpoint_strings['header']))


# # Sorting the New Dataset

# In[585]:


loan_data = loan_data[np.argsort(loan_data[:,0])]


# In[586]:


loan_data


# In[587]:


np.argsort(loan_data[:,0])


# # Storing the Dataset

# In[589]:


loan_data = np.vstack((header_full, loan_data))


# In[590]:


np.savetxt("loan-data-preprocessed.csv", 
           loan_data, 
           fmt = '%s',
           delimiter = ',')


# In[ ]:





# We've succesfully
# 
# Cleaned the dataset,
# preprocess the dataset, and
# prepare the dataset for futher analysis

# In[ ]:




