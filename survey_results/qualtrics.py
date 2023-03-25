import csv
import pandas as pd

workers = {'logic': ["A2APG8MSLJ6G2K",
                     "A3S3WYVCVWW8IZ",
                     "A248QG4DPULP46",
                     "A3V2XCDF45VN9X",
                     "AP5TSGEVCJLOS",
                     "AMELYCC59JKB0",
                     "A1PBRKFHSF1OF8",
                     "A3QNCW3LR2AVOV",
                     "A2ZDEERVRN5AMC",
                     "A1NVJB5O4H7ZCJ",
                     "A26LOVXF4QZZCO",
                     "AFM65NU0UXIGP",
                     "AOS2PVHT2HYTL",
                     "A39K5MR4MCSC56",
                     "ASC9DUCC64M3P",
                     "A2JDYN6QM8M5UN",
                     "A3IMB1JAMHP1KK",
                     "APGX2WZ59OWDN",
                     "A12FTSX85NQ8N9",
                     "A1DZMZTXWOM9MR"],
           'contingency': ["A2G43KS55YGYQE",
                           "A8KX1HFH8NE2Q",
                           "A23KAJRDVCVGOE",
                           "A2POU9TTW177VH",
                           "A2EPBSY0VPI38S",
                           "A1QVX30ZORPQGX",
                           "AETIZKQNUSBLB",
                           "A1I7H6RDJS4EKN",
                           "A3BU8UL4W258UU",
                           "A2F1AA15HG0FRU",
                           "A1901T07YJX1OD",
                           "ADJ9I7ZBFYFH7",
                           "A5NE8TWS8ZV7B",
                           "A3Q1EZDNIUK41P",
                           "A320QA9HJFUOZO",
                           "AVI7K876BV3QL",
                           "A98E8M4QLI9RS",
                           "A1G34TESSXG1R8",
                           "A3CWEIEY3TGJFX",
                           "A37JC45Y9GLSA7"],
           'shuffle': ['AR1IWBDA7MC86', 'AF2BJOFFCZQWY', 'A3R5OJR60C6004', 'A8C3WNWRBWUXO', 'A2ZL5GAZK6S0H1',
                       'A3FGT6EU39C6S4', 'A2YZPA1SIVXL23', 'A3EA4SHCLJ1UZQ', 'AOGXFLXT1BNE9', 'A3KF6O09H04SP7',
                       'A1W7I6FN183I8F', 'A1K8QNLYYYX21W', 'A1AMGHYG5PT0L2', 'AIA7RZSAONU5M', 'AT0COJ1G23ZB0',
                       'A1AZGP16G9TURZ', 'A1IFIK8J49WBER', 'A1969Q0R4Y0E3J', 'A2YC6PEMIRSOAA', 'A2E0LU8V4EUX5C'],
           'change_agent': ['ANQ0RLFEZ17W0', 'AD3V6XGQWRD6E', 'A2DVV59R1CQU6T', 'A3L2XKXABNO0N5', 'AZ69TBTDH7AZS',
                            'A3GVUYY3TWRPZT', 'A24LUXW1DB1QI5', 'A3RMV9ZGFJ0HHF', 'A13WTEQ06V3B6D', 'AD1WGUMVD6KED',
                            'A5WWHKD82I8UE', 'A14EYTLSMJRPUK', 'A1YHIQHLLLQIIQ', 'A2A07J1P6YEW6Z', 'A23BWWRR7J5XLS',
                            'A2615YW1YERQBO', 'A1R8A8BK2VN7RH', 'A2I4PRZ9IZMKON']}
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

    for w in workers.keys():
        if row != '\n' and row[20] in workers[w]:
            data[w]['genders'].append(row[23])
            data[w]['ages'].append(int(row[27]))
            data[w]['duration'].append(int(row[5]))

file.close()

for w in workers.keys():
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
