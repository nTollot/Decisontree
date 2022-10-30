import numba
import numpy as np

from numba import jit


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
            available_cards = np.zeros(78, dtype=numba.boolean)
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
            available_cards = np.zeros(78, dtype=numba.boolean)
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
