import pandas as pd
import random

# Gives different random numbers every minute
random.seed()
dataDict = { 'sequences': [] }
NUM_ACTIONS = 5

for i in range(1000):
    # seqLength = random.randint(0,9)
    # sequence = { 'call': [], 'check': [], 'bet': [], 'raise': [], 'fold': [] }
    sequence = []
    continueSequence = True
    numRounds = 0
    # avoids incrementing numRounds on a check after two checks before
    consecutiveChecks = False
    prevAction = -1
    while continueSequence:
        # action 0 is call
        # action 1 is check
        # action 2 is bet
        # action 3 is raise
        # action 4 is fold
        setOne = random.randint(0, NUM_ACTIONS - 1)
        action = []
        # if action is call. Rule 1 is to end game on 4
        # Rule 2 is to not call after check or call or as first action
        if setOne == 0:
            if prevAction == 1 or prevAction == 0 or prevAction == -1:
                continue
            numRounds += 1
            if numRounds >= 4:
                continueSequence = False

        # if action is check. Rule 1 is to not check after bets or raises
        # Rule 2 is to increment number of rounds after two consecutive checks
        elif setOne == 1:
            if prevAction == 2 or prevAction == 3:
                continue
            if prevAction == 1 and consecutiveChecks == False:
                numRounds += 1
                consecutiveChecks = True
                if numRounds >= 4:
                    continueSequence = False
            else:
                consecutiveChecks = False


        # if action is bet. Rule is to not bet after bets or raises
        elif setOne == 2:
            if prevAction == 2 or prevAction == 3:
                continue

        # if action is raise. Rule is to not raise on check or call or as first action
        elif setOne == 3:
            if prevAction == 0 or prevAction == 1 or prevAction == -1:
                continue

        # if action is fold. End sequence
        elif setOne == 4:
            continueSequence = False

        # print("action is " + str(setOne))
        for i in range(0, NUM_ACTIONS):
            if i == setOne:
                action.append(1)
            else:
                action.append(0)
        sequence.append(action)
        prevAction = setOne
        # if seqLength == 0:
        #     continueSequence = False
        # seqLength -= 1

    dataDict['sequences'].append(sequence)

df = pd.DataFrame(data=dataDict)
print(df)
print('sending to sequence_data.csv')
df.to_csv('~/projects/poker-feature-extractor/action_sequence_nn/sequence_data.csv')
