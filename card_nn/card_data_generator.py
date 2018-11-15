import pandas as pd
import os
import random
import sys

def main(count):
    # Gives different random numbers every minute
    random.seed()
    dataDict = { 'hole_1': [], 'hole_2': [], 'com_1': [], 'com_2': [], 'com_3': [],
        'com_4': [], 'com_5': [], 'round': []}

    for i in range(count):
        round = random.randint(0,3)
        if 'round' in dataDict:
            dataDict['round'].append(round)
        else:
            dataDict['round'] = [round]

        # handCards = [random.randint(1,52), random.randint(1,52)]
        # handCards = str(random.randint(1,52)) + ', ' + str(random.randint(1,52))
        dataDict['hole_1'].append(random.randint(1,52))
        dataDict['hole_2'].append(random.randint(1,52))
        # dataDict['com_1'].append(random.randint(1,52))
        # dataDict['com_2'].append(random.randint(1,52))
        # dataDict['com_3'].append(random.randint(1,52))
        # dataDict['com_4'].append(random.randint(1,52))
        # dataDict['com_5'].append(random.randint(1,52))

        # if 'com_1' not in dataDict:
        #     dataDict['com_1'] = []
        #     dataDict['com_2'] = []
        #     dataDict['com_3'] = []
        #     dataDict['com_4'] = []
        #     dataDict['com_5'] = []
        # # tableCards = ''
        # building the table cards
        if round == 0:
            dataDict['com_1'].append(0)
            dataDict['com_2'].append(0)
            dataDict['com_3'].append(0)
            dataDict['com_4'].append(0)
            dataDict['com_5'].append(0)
        elif round == 1:
            dataDict['com_1'].append(random.randint(1,52))
            dataDict['com_2'].append(random.randint(1,52))
            dataDict['com_3'].append(random.randint(1,52))
            dataDict['com_4'].append(0)
            dataDict['com_5'].append(0)
        elif round == 2:
            dataDict['com_1'].append(random.randint(1,52))
            dataDict['com_2'].append(random.randint(1,52))
            dataDict['com_3'].append(random.randint(1,52))
            dataDict['com_4'].append(random.randint(1,52))
            dataDict['com_5'].append(0)
        elif round == 3:
            dataDict['com_1'].append(random.randint(1,52))
            dataDict['com_2'].append(random.randint(1,52))
            dataDict['com_3'].append(random.randint(1,52))
            dataDict['com_4'].append(random.randint(1,52))
            dataDict['com_5'].append(random.randint(1,52))

        # print('tableCards: ' + tableCards)
        # if 'table' in dataDict:
        #     dataDict['table'].append(tableCards)
        # else:
        #     dataDict['table'] = [tableCards]

    # print(dataDict)
    # if len(dataDict['com_5']) != len(dataDict['hole_1']):
    #     print("mismatch! com: ")
    df = pd.DataFrame(data=dataDict)
    print(df)
    dir = os.path.dirname(os.path.abspath(__file__))
    filename = '/card_data.csv'
    print('sending to ' + dir + filename)
    df.to_csv(dir + filename)

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("data count not mentioned. Expected format: python <filename> <data count>")
    else:
       main(int(sys.argv[1]))
