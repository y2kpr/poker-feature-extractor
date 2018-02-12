import pandas as pd
import random

# Gives different random numbers every minute
random.seed()
dataDict = {}

for i in range(1000000):
    round = random.randint(0,3)
    # print('round is ' + str(round))
    if 'round' in dataDict:
        dataDict['round'].append(round)
    else:
        dataDict['round'] = [round]

    # handCards = [random.randint(1,52), random.randint(1,52)]
    # handCards = str(random.randint(1,52)) + ', ' + str(random.randint(1,52))
    if 'hole_1' in dataDict:
        dataDict['hole_1'].append(random.randint(1,52))
        dataDict['hole_2'].append(random.randint(1,52))
    else:
        dataDict['hole_1'] = [random.randint(1,52)]
        dataDict['hole_2'] = [random.randint(1,52)]

    if 'community_1' not in dataDict:
        dataDict['community_1'] = []
        dataDict['community_2'] = []
        dataDict['community_3'] = []
        dataDict['community_4'] = []
        dataDict['community_5'] = []
    # tableCards = ''
    # building the table cards
    if round == 0:
        dataDict['community_1'].append(0)
        dataDict['community_2'].append(0)
        dataDict['community_3'].append(0)
        dataDict['community_4'].append(0)
        dataDict['community_5'].append(0)
    elif round == 1:
        dataDict['community_1'].append(random.randint(1,52))
        dataDict['community_2'].append(random.randint(1,52))
        dataDict['community_3'].append(random.randint(1,52))
        dataDict['community_4'].append(0)
        dataDict['community_5'].append(0)
    elif round == 2:
        dataDict['community_1'].append(random.randint(1,52))
        dataDict['community_2'].append(random.randint(1,52))
        dataDict['community_3'].append(random.randint(1,52))
        dataDict['community_4'].append(random.randint(1,52))
        dataDict['community_5'].append(0)
    elif round == 3:
        dataDict['community_1'].append(random.randint(1,52))
        dataDict['community_2'].append(random.randint(1,52))
        dataDict['community_3'].append(random.randint(1,52))
        dataDict['community_4'].append(random.randint(1,52))
        dataDict['community_5'].append(random.randint(1,52))

    # print('tableCards: ' + tableCards)
    # if 'table' in dataDict:
    #     dataDict['table'].append(tableCards)
    # else:
    #     dataDict['table'] = [tableCards]

# print(dataDict)
# if len(dataDict['community_5']) != len(dataDict['hole_1']):
#     print("mismatch! com: ")
df = pd.DataFrame(data=dataDict)
print(df)
df.to_csv('~/src/tf_playground/poker-feature-extractor/data.csv')
