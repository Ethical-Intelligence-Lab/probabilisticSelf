import numpy as np

def key_converter(key_pressed):    
    if (key_pressed == 'w') | (key_pressed == 0):
        print('UP!')
        return 0
    elif (key_pressed == 's') | (key_pressed == 1):
        print('DOWN!')
        return 1
    elif (key_pressed == 'a') | (key_pressed == 2):
        print('LEFT!')
        return 2
    elif (key_pressed == 'd') | (key_pressed == 3):
        print('RIGHT!')
        return 3

def euclid(arr1, arr2):
    ans = np.sqrt((arr2[1]-arr1[1])**2 + (arr2[0]-arr1[1])**2)
    return ans