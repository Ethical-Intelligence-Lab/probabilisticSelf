# Iterate through all files in a directory and rename them
# Specifically: 
# (1) rename all 'contingency_game_shuffled' strings with 'switching_mappings_game'
# (2) rename all 'change_agent_game' strings with 'swithcing_embodiments_game'
# (3) rename all '_harder' with '_self_finding'

import os

# Get the current working directory
path = os.getcwd()

# Get all files in the directory
files = os.listdir(path)

# Iterate through all files
for file in files:
    old_file = file
    new_file = file
    if 'contingency_game_shuffled' in file:
        new_file = file.replace('contingency_game_shuffled', 'switching_mappings_game')
        os.rename(file, new_file)
        file = new_file


    if 'change_agent_game' in file:
        new_file = file.replace('change_agent_game', 'switching_embodiments_game')
        os.rename(file, new_file)
        file = new_file

    if 'change_agent' in file:
        new_file = file.replace('change_agent', 'switching_embodiments')
        os.rename(file, new_file)
        file = new_file

    if '_harder' in file:
        new_file = file.replace('_harder', '_self_finding')
        os.rename(file, new_file)
        file = new_file

    # Print the old and new file names
    if old_file != new_file: print(old_file, '->', new_file)

# Print a message to indicate that the renaming is done
print('Renaming is done!')