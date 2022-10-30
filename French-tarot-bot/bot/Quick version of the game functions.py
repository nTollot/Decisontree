from numba import boolean
from numba import float32
from numba import jit
import numpy as np


@jit(nopython=True)
def genere_coups(hand, moves_list, first_move, second_move):
    """
    Generates the available moves

    Parameters
    ----------
    hand : array
        Boolean array of size 78 which indicates the cards in hand
    moves_list : array
        Boolean array of size 78 which indicates the cards already played during this round
    first_move : int
        integer which indicates the card played by the first player. It is worth 0 if there is no first player.
    second_move : int
        integer which indicates the card played by the second player. It is worth 0 if there is no second player.

    Returns
    ----------
    available moves : array
        Boolean table of available moves
    """
    n_moves = np.sum(moves_list)

    if n_moves == 0:
        # The first player can play every card
        return hand

    elif first_move == 0:
        # The first card played is the excuse
        if n_moves == 1:
            # The second player can play any card he has
            return hand

        else:
            # The turn takes place as if the second player had started
            moves_list[0] = False
            first_move, second_move = second_move, 0
            # In this case, the second move is no longer useful
            return genere_coups(hand, moves_list, first_move, second_move)

    elif 0 < first_move < 22:
        # The first player played a trump card
        biggest_trump_card_possessed = 0
        biggest_trump_card_played = 0
        # We will determine the greatest trump card of the player, if he has one
        for i in range(1, 22):
            if moves_list[i]:
                biggest_trump_card_played = i
            if hand[i]:
                biggest_trump_card_possessed = i

        if biggest_trump_card_possessed >= biggest_trump_card_played:
            # The player has at least one trump card greater than the greatest trump card played
            available_cards = np.zeros(78, dtype=boolean)
            available_cards[0] = hand[0]
            # The Excuse can always be played
            playable_trump_cards = np.arange(biggest_trump_card_played+1, 22)
            # The only other cards he can play
            available_cards[playable_trump_cards] = hand[playable_trump_cards]
            # We know that it has at least one superior trump card
            return available_cards

        elif biggest_trump_card_possessed > 0:
            # The player has at least one trump card
            trump_cards = np.arange(0, 22)
            # He can play any trump card
            return hand[trump_cards]

        else:
            # The player has no trump card : he can play any card
            return hand
    else:
        # The first card played is a suit
        suit_range = np.arange(14*((first_move-22)//14)+22,
                               14*((first_move-22)//14)+22+14)
        # The range of the played color
        if hand[suit_range].any():
            # The player has at least one card of the suit played
            available_cards = np.zeros(78, dtype=boolean)
            available_cards[0] = hand[0]
            # The Excuse can always be played
            available_cards[suit_range] = hand[suit_range]
            return available_cards

        elif hand[np.arange(1, 22)].any():
            # The player does not have a card of the suit but has at least one trump card
            first_move = 1
            # Small trick that allows you to place yourself in a case where trump card is played first
            return genere_coups(hand, moves_list, first_move, second_move)

        else:
            # The player does not have a card of the suit and doesn't have a trump card, so he can play any card
            return hand


@jit(nopython=True)
def determine_victoire_coups(moves_list):
    """
    Determines which player wins the trick

    Parameters
    ----------
    moves_list : array
        Integer array of size 4 which indicates the cards played

    Returns
    ----------
    determine victory : int
        Integer index of the winning player
    """
    if moves_list[0] == 0:
        # The first player has played the excuse; in this case, we look at the winner among the 3 other cards
        return 1 + determine_victoire_coups(moves_list[1:])
    elif ((moves_list < 22) * (moves_list > 0)).any():
        # At least one trump card has been played during this round; it's who played the biggest trump card
        return np.argmax(moves_list*np.sign(22-moves_list))
    else:
        # No trump card was played; in this case, the one who played the highest card of the suit played first wins
        suit_inf = 14*((moves_list[0]-22)//14)+22
        suit_sup = 14*((moves_list[0]-22)//14)+22+14
        return np.argmax(moves_list*(np.sign(moves_list-suit_inf)+np.sign(suit_sup-moves_list)-1))

@jit(nopython=True)
def generate_players_hand(n_games):
    """
    Generates game initializations 

    Parameters
    ----------
    n_games : int
        Number of games to initialize

    Returns
    ----------
    hands : array
        32-bits integer array of players cards
    chien : array
        32-bits integer array of chien
    """
    hands = np.zeros((n_games, 4, 78), dtype=float32)
    chien = np.zeros((n_games, 78), dtype=float32)
    for i in range(n_games):
        single_hand = np.arange(78)
        np.random.shuffle(single_hand)
        single_chien, hand_p1, hand_p2, hand_p3, hand_p4 = single_hand[:6], single_hand[
            6:24], single_hand[24:42], single_hand[42:60], single_hand[60:78]
        chien[i,single_chien] = 1
        hands[i,0,hand_p1] = 1
        hands[i,1,hand_p2] = 1
        hands[i,2,hand_p3] = 1
        hands[i,3,hand_p4] = 1
    return hands, chien
