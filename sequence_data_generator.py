import pandas as pd
import random

# Gives different random numbers every minute
random.seed()
dataDict = { 'sequences': [] }
NUM_ACTIONS = 5

for i in range(10000):
    seqLength = random.randint(0,9)
    # sequence = { 'call': [], 'check': [], 'bet': [], 'raise': [], 'fold': [] }
    sequence = []
    continueSequence = True
    while continueSequence:
        setOne = random.randint(0,NUM_ACTIONS - 1)
        sequenceEnded = False
        numCalls = 0
        action = []
        # # if action is call. Rule is to not exceed more than 4 calls
        # if setOne == 0:
        #     if numCalls >= 4:
        #         break
        #     numCalls += 1
        # # if action is check. Rule is to not check on bets or raises
        # if setOne == 1:
        #     if
        # print("rand is " + str(setOne))
        for i in range(0, NUM_ACTIONS):
            if i == setOne:
                action.append(1)
            else:
                action.append(0)
        sequence.append(action)
        # This check will change later
        if seqLength == 0:
            continueSequence = False
        seqLength -= 1

    dataDict['sequences'].append(sequence)

df = pd.DataFrame(data=dataDict)
print(df)
print('sending to sequence_data.csv')
df.to_csv('~/projects/poker-feature-extractor/sequence_data.csv')
