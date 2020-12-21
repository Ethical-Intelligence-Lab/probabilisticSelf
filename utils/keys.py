import numpy as np

def key_converter(key_pressed):    
    if (key_pressed == 'w') | (key_pressed == 1):
        print('UP!')
        return 1
    elif (key_pressed == 's') | (key_pressed == 2):
        print('DOWN!')
        return 2
    elif (key_pressed == 'a') | (key_pressed == 3):
        print('LEFT!')
        return 3
    elif (key_pressed == 'd') | (key_pressed == 4):
        print('RIGHT!')
        return 4

def euclid(arr1, arr2):
    ans = np.sqrt((arr2[1]-arr1[1])**2 + (arr2[0]-arr1[1])**2)
    return ans