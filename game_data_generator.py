import pandas as pd
import numpy as np
import os, sys
import random
from card_nn.card_mappings import card_mappings

'''
Converts the cryptic action and card data of a single poker game into readable format
and writes it to a text file
'''
def convert_to_readable_format(action_sequence, cards_sequence):
    action_order = ['call', 'check', 'bet', 'raise', 'fold']
    for pos, action in enumerate(action_sequence):
        action_text = action_order[np.argmax(action)] + ': '
        cards = cards_sequence[pos]
        hole_cards = 'hole cards: ' + card_mappings[cards[5]] + ', ' + card_mappings[cards[6]]
        com_cards = ' community cards: '
        for i in range(5):
            if cards[i] != 0:
                com_cards += card_mappings[cards[i]] + ', '
        action_text += hole_cards + com_cards
        print(action_text)

'''
Randomly generates the next cards to put in the given round and updates the cards_to_update (in place).
'round' tells which cards to generate.

cards_to_update are expected in the order of [com_1, com_2, com_3, com_4, com_5, hole_1, hole_2]
'''
def get_next_cards(cards_to_update, round):
    # print('got cards: ' + str(cards_to_update) + ' with round: ' + str(round))
    new_cards = list(cards_to_update)
    # pre-flop
    if round == 0:
        # QUESTION: do we want empty (zero encoded) community cards in this case?
        for i in range(5):
            new_cards.append(0)
        new_cards.append(random.randint(1,52))
        new_cards.append(random.randint(1,52))
    # flop
    elif round == 1:
        for i in range(3):
            new_cards[i] = random.randint(1,52)
    # river
    elif round == 2:
        new_cards[3] = random.randint(1,52)
    # showdown
    elif round == 3:
        new_cards[4] = random.randint(1,52)
    else:
        print("ERROR: invalid round: " + str(round))

    # print('updated cards to: ' + str(new_cards))
    return new_cards

def main(count):
    # Gives different random numbers every minute
    random.seed()
    dataDict = { 'action_sequences': [],  'cards_sequences': [] }
    NUM_ACTIONS = 5
    max_action_sequence = []
    max_cards_sequence = []

    for i in range(count):
        # seqLength = random.randint(0,9)
        # sequence = { 'call': [], 'check': [], 'bet': [], 'raise': [], 'fold': [] }
        action_sequence = []
        cards_sequence = []
        cards = []
        continueSequence = True
        numRounds = 0
        cards = get_next_cards(cards, numRounds)

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
                else:
                    cards = get_next_cards(cards, numRounds)

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
                        cards = get_next_cards(cards, numRounds)
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
            # One-hot encode the action
            for i in range(0, NUM_ACTIONS):
                if i == setOne:
                    action.append(1)
                else:
                    action.append(0)
            action_sequence.append(action)
            # print('appending cards: ' + str(cards))
            cards_sequence.append(cards)
            prevAction = setOne
            # if seqLength == 0:
            #     continueSequence = False
            # seqLength -= 1

        if len(action_sequence) > len(max_action_sequence):
            max_action_sequence = action_sequence
            max_cards_sequence = cards_sequence
        # dataDict['action_sequences'].append(action_sequence)
        # dataDict['cards_sequences'].append(cards_sequence)

    # TODO: add a validation step for the data generated
    dataDict['action_sequences'].append(max_action_sequence)
    dataDict['cards_sequences'].append(max_cards_sequence)
    convert_to_readable_format(max_action_sequence, max_cards_sequence)

    df = pd.DataFrame(data=dataDict)
    # print(df)
    dir = os.path.dirname(os.path.abspath(__file__))
    print('sending to ' + dir)
    df.to_csv(dir + '/game_data.csv')

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("data count not mentioned. Expected format: python <filename> <data count>")
    else:
       main(int(sys.argv[1]))
