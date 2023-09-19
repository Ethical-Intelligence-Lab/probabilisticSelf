import csv
import pandas as pd


file = open('qualtrics.csv')
csvreader = csv.reader(file)
rows = []
col_names = None
data = {'logic': {'genders': [], 'ages': [], 'duration': []}, 'contingency': {'genders': [], 'ages': [], 'duration': []},
        'shuffle': {'genders': [], 'ages': [], 'duration': []}, 'change_agent': {'genders': [], 'ages': [], 'duration': []}
        }

attention = {'logic': {'passed': [], 'failed': [], 'finished_game_and_passed': [], 'finished_game_and_failed': []},
             'contingency': {'passed': [], 'failed': [], 'finished_game_and_passed': [],
                             'finished_game_and_failed': []},
             'shuffle': {'passed': [], 'failed': [], 'finished_game_and_passed': [], 'finished_game_and_failed': []},
             'change_agent': {'passed': [], 'failed': [], 'finished_game_and_passed': [],
                              'finished_game_and_failed': []}
             }

comprehension = {'logic': {'passed': [], 'failed': [], 'finished_game_and_passed': [], 'finished_game_and_failed': []},
                 'contingency': {'passed': [], 'failed': [], 'finished_game_and_passed': [],
                                 'finished_game_and_failed': []},
                 'shuffle': {'passed': [], 'failed': [], 'finished_game_and_passed': [],
                             'finished_game_and_failed': []},
                 'change_agent': {'passed': [], 'failed': [], 'finished_game_and_passed': [],
                                  'finished_game_and_failed': []}
                 }

for i, row in enumerate(csvreader):
    if i == 0:
        col_names = row

    if row != '\n' and row[-1] != "unknown" and row[-1] != "game_type":
        data[row[-1]]['genders'].append(row[23])
        data[row[-1]]['ages'].append(int(row[27]))
        data[row[-1]]['duration'].append(int(row[5]))

file.close()

for w in attention.keys():
    print("****** " + w + " ******")
    print("Female proportion: " + str(data[w]['genders'].count('Female') / len(data[w]['genders'])))
    print("Avg age: " + str(sum(data[w]['ages']) / len(data[w]['ages'])))
    print("Avg Duration (mins): " + str(sum(data[w]['duration']) / len(data[w]['duration'])/ 60))

##### Logic Perturbed #####
print("****** Logic Perturbed ******")
data = pd.read_csv('Logic Perturbed.csv')
data = data[(data['finished_game'] == 'yes') & (data['workerId'] != 'A_WORKER_ID')]
print("Recruited: " , len(data))
print("Female proportion: ", len(data[(data['Q4'] =='Female')]) / len(data))
print("Mean age: ", (data['Q6'].astype(int).mean()))
print("Mean Duration: ", (data['Duration (in seconds)'].astype(int).mean()) / 60)

##### Contigency Perturbed #####
print("****** Contingency perturbed ******")
data = pd.read_csv('Contingency Perturbed.csv')
data = data[(data['finished_game'] == 'yes')]
print("Recruited: " , len(data))
print("Female proportion: ,", len(data[(data['Q4'] == 'Female')]) / len(data))
print("Mean age: ", (data['Q6'].astype(int).mean()))
print("Mean Duration: ", (data['Duration (in seconds)'].astype(int).mean()) / 60)

##### Self Centering #####
print("****** Contingency Self Centering ******")
data = pd.read_csv('Contingency - Self Centering.csv')
data = data[(data['finished_game'] == 'yes') & (data['workerId'] != 'ABCD1435')]
print("Recruited: " , len(data))
print("Female proportion: ,", len(data[(data['Q4'] == 'Female')]) / len(data))
print("Mean age: ", (data['Q6'].astype(int).mean()))
print("Mean Duration: ", (data['Duration (in seconds)'].astype(int).mean()) / 60)

##### Shuffle Keys #####
print("****** Shuffle Keys Self Centering ******")
data = pd.read_csv('Shuffle Keys - Self Centering.csv')
data = data[(data['finished_game'] == 'yes') & (data['workerId'] != 'A_WORKER_ID') & (data['workerId'] != '') & (data['workerId'].notnull())]
print("Recruited: " , len(data))
print("Female proportion: ,", len(data[(data['Q4'] == 'Female')]) / len(data))
print("Mean age: ", (data['Q6'].astype(int).mean()))
print("Mean Duration: ", (data['Duration (in seconds)'].astype(int).mean()) / 60)


##### Perturbed Switching Embodiments #####
print("****** Switching Embodiments Perturbed ******")
data = pd.read_csv('Switching Embodiments Perturbed.csv')
data = data[(data['finished_game'] == 'yes') & (data['workerId'] != 'A_WORKER_ID') & (data['workerId'] != '') & (data['workerId'].notnull())]
print("Recruited: " , len(data))
print("Female proportion: ,", len(data[(data['gender'] == '2')]) / len(data))
print("Mean age: ", (data['age'].astype(int).mean()))
print("Mean Duration: ", (data['Duration (in seconds)'].astype(int).mean()) / 60)

##### Switching Embodiments self centering #####
print("****** Switching Embodiments Self Centering ******")
data = pd.read_csv('Switching Embodiments - Self Centering.csv')
data = data[(data['finished_game'] == 'yes') & (data['workerId'] != 'A_WORKER_ID') & (data['workerId'] != '') & (data['workerId'].notnull()) & (data['att1'] == 'Paul') & (data['att2'] == 'Purple')]
print("Recruited: " , len(data))
print("Female proportion: ,", len(data[(data['gender'] == 'Female')]) / len(data))
print("Mean age: ", (data['age'].astype(int).mean()))
print("Mean Duration: ", (data['Duration (in seconds)'].astype(int).mean()) / 60)



"""
def reformat():
    # Read qualtrics csv file.  Add a column called game_type based on the workerId's in the workers dictionary, i.e., if the workerId is in the logic workers, then the game_type is logic, etc.
    import pandas as pd
    import csv
    import random, string

    # Read csv into dataframe
    df = pd.read_csv('Switching Embodiments - Self Centering.csv')

    # Remove all rows where the workerId is 'A_WORKER_ID'
    df = df[df['workerId'] != 'A_WORKER_ID']

    # Remove all rows where the workerId is 'ABCD1435'
    df = df[df['workerId'] != 'ABCD1435']

    # Remove all rows where the workerId is 'A_WORKER_ID'
    df = df[df['workerId'] != '']

    # Remove all rows where the workerId is null
    df = df[df['workerId'].notnull()]

    # Make all rows in IPAddress, LocationLatitude, LocationLongitude, and Q14,  as 'Anonymized'.
    df['IPAddress'] = 'Anonymized'
    df['LocationLatitude'] = 'Anonymized'
    df['LocationLongitude'] = 'Anonymized'
    df['workerId'] = 'Anonymized'
    df['mturk_id'] = 'Anonymized'
    

    # Write dataframe to csv
    df.to_csv('Switching Embodiments - Self Centering.csv', index=False)
"""