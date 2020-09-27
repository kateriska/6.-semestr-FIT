import logging

from dicewars.client.ai_driver import BattleCommand, EndTurnCommand


class AI:
        """Naive player agent

        This agent performs all possible moves in random order
        """

        def __init__(self, player_name, board, players_order):
            """
            Parameters
            ----------
            game : Game
            """
            self.player_name = player_name
            self.players_order = players_order
            self.logger = logging.getLogger('AI')

        def ai_turn(self, board, nb_moves_this_turn, nb_turns_this_game, time_left):
            """AI agent's turn


            """

            self.logger.debug("Looking for possible turns.")
            self.board = board
            turns = self.possible_turns()

            if turns:
                turn = turns[0]
                area_name = turn[0]
                self.logger.debug("Possible turn: {}".format(turn))
                hold_prob = turn[2]
                self.logger.debug("{0}->{1} attack and hold probabiliy {2}".format(area_name, turn[1], hold_prob))

                return BattleCommand(area_name, turn[1])

            self.logger.debug("No more plays.")
            return EndTurnCommand()






        def from_largest_region(self, board, attacks):
            players_regions = board.get_players_regions(self.player_name)
            max_region_size = max(len(region) for region in players_regions)
            max_sized_regions = [region for region in players_regions if len(region) == max_region_size]

            the_largest_region = max_sized_regions[0]
            self.logger.debug('The largest region: {}'.format(the_largest_region))
            print('The largest region: {}'.format(the_largest_region))
            return [attack for attack in attacks if attack[0].get_name() in the_largest_region]


        def get_largest_region(self):
            """Get size of the largest region, including the areas within

            Attributes
            ----------
            largest_region : list of int
                Names of areas in the largest region

            Returns
            -------
            int
                Number of areas in the largest region
            """
            self.largest_region = []

            players_regions = self.board.get_players_regions(self.player_name)
            max_region_size = max(len(region) for region in players_regions)
            max_sized_regions = [region for region in players_regions if len(region) == max_region_size]

            for region in max_sized_regions:
                for area in region:
                    self.largest_region.append(area)
            return max_region_size

        def calculate_rate_function(self):
            largest_region_size = self.get_largest_region(self)
            owned_fields = len(bot_board.get_player_areas(player_name))

            rate_func = numpy.log(0.5 * largest_region_size + 0.5 * owned_fields)

            return rate_func

        def possible_turns(self):
            """Get a list of preferred moves

            This list is sorted with respect to hold probability in descending order.
            It includes all moves that either have hold probability higher or equal to 50 %
            or have strength of eight dice.
            """
            turns = []

            for source, target in possible_attacks(self.board, self.player_name):
                area_name = source.get_name()
                atk_power = source.get_dice()
                atk_prob = probability_of_successful_attack(self.board, area_name, target.get_name())
                hold_prob = atk_prob * probability_of_holding_area(self.board, target.get_name(), atk_power - 1, self.player_name)
                if hold_prob >= 0.5 or atk_power == 8:
                    turns.append([area_name, target.get_name(), hold_prob])

            return sorted(turns, key=lambda turn: turn[2], reverse=True)
