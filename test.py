import sys
import os
import time
import traceback
import threading
from collections import deque
from queue import Queue

SINGLE_PLAYER_MODE = False
action_queue = deque()
feedback_queue = Queue()


class ActionEngine(threading.Thread):
    def __init__(self):
        super().__init__()

        self.p1_action_queue = deque()

        # Flags
        self.p1_gun_shot = False
        self.p1_vest_shot = False
        self.p1_grenade_hit = False

        if not SINGLE_PLAYER_MODE:
            self.p2_action_queue = deque()

            self.p2_gun_shot = False
            self.p2_vest_shot = False
            self.p2_grenade_hit = False

    def handle_grenade(self, player):
        print("Handling Grenade")
        if player == 1:
            self.p1_action_queue.append('grenade')
        else:
            self.p2_action_queue.append('grenade')

    def handle_shield(self, player):
        print("Handling Shield")
        if player == 1:
            self.p1_action_queue.append('shield')
        else:
            self.p2_action_queue.append('shield')

    def handle_reload(self, player):
        print("Handling Reload")
        if player == 1:
            self.p1_action_queue.append('reload')
        else:
            self.p2_action_queue.append('reload')

    def handle_logout(self, player):
        print('Handling Logout')
        if player == 1:
            self.p1_action_queue.append('logout')
        else:
            self.p2_action_queue.append('logout')

    def handle_gun_shot(self, player):
        print('Handling Gun Shot')
        if player == 1:
            self.p1_gun_shot = True
            self.p1_action_queue.append('shoot')
        else:
            self.p2_gun_shot = True
            self.p2_action_queue.append('shoot')

    def handle_vest_shot(self, player):
        print('Handling Vest Shot')
        if player == 1:
            self.p1_vest_shot = True
        else:
            self.p2_vest_shot = True

    def determine_grenade_hit(self):
        self.p2_grenade_hit = True
        self.p1_grenade_hit = True

    def run(self):
        action_data_p1, action_data_p2 = None, None
        action = [['None', True], ['None', True]]
        while True:
            if self.p1_action_queue or self.p2_action_queue:
                action_dic = {
                    "p1": "",
                    "p2": ""
                }

                if action_data_p1 is None and self.p1_action_queue:
                    action_data_p1 = self.p1_action_queue.popleft()

                    if action_data_p1 == 'shoot':
                        action[0] = [action_data_p1, self.p2_vest_shot]
                    elif action_data_p1 == 'grenade':
                        action_dic["p1"] = "check_grenade"
                        action[0] = [action_data_p1, False]
                    else:
                        action[0] = [action_data_p1, True]

                if action_data_p2 is None and self.p2_action_queue:
                    action_data_p2 = self.p2_action_queue.popleft()

                    if action_data_p2 == 'shoot':
                        action[1] = [action_data_p2, self.p1_vest_shot]
                    elif action_data_p2 == 'grenade':
                        action_dic["p2"] = "check_grenade"
                        action[1] = [action_data_p2, False]
                    else:
                        action[1] = [action_data_p2, True]

                if action_data_p1 is not None:
                    self.p1_action_queue.clear()

                if action_data_p2 is not None:
                    self.p2_action_queue.clear()

                if action_data_p1 == "grenade" or action_data_p2 == "grenade":
                    # subscribe_queue.put(json.dumps(action_dic))
                    self.determine_grenade_hit()
                    action[0][1] = self.p2_grenade_hit
                    action[1][1] = self.p1_grenade_hit
                    if action_data_p1 == "grenade":
                        action_dic["p1"] = None
                        action_data_p1 = False
                    if action_data_p2 == "grenade":
                        action_dic["p2"] = None
                        action_data_p2 = False

                if not (action_data_p1 is None or action_data_p2 is None):
                    print(action)
                    action_queue.append(action)
                    action_data_p1, action_data_p2 = None, None
                    action = [['None', True], ['None', True]]

                    self.p1_grenade_hit = False
                    self.p1_gun_shot = False
                    self.p1_vest_shot = False
                    self.p1_action_queue.clear()

                    if not SINGLE_PLAYER_MODE:
                        self.p2_gun_shot = False
                        self.p2_vest_shot = False
                        self.p2_grenade_hit = False
                        self.p2_action_queue.clear()


class GameEngine(threading.Thread):
    def __init__(self, eval_client):
        super().__init__()

        # queue to receive status from sw
        self.eval_client = eval_client
        self.p1 = self.eval_client.gamestate.player_1
        self.p2 = self.eval_client.gamestate.player_2

        self.shutdown = threading.Event()

    def reset_player(self, player):
        player.hp = 100
        player.action = "none"
        player.bullets = 6
        player.grenades = 2
        player.shield_time = 0
        player.shield_health = 0
        player.num_shield = 3
        player.num_deaths += 1

    def run(self):
        while not self.shutdown.is_set():
            try:
                if len(action_queue) != 0:
                    p1_action, p2_action = action_queue.popleft()  # [[p1_action, status], [p2_action, status]]

                    viz_action_p1, viz_action_p2 = None, None
                    print(f"P1 action data: {p1_action}")
                    print(f"P2 action data: {p2_action}")

                    self.p1.update_shield()
                    self.p2.update_shield()
                    # TODO - Need to check with chris regarding convention, [action, status] -> for grenade [grenade, true] means p1 throws grenade and hit p2?
                    # TODO - Chris: Yes thats right
                    valid_action_p1 = self.p1.action_is_valid(p1_action[0])
                    valid_action_p2 = self.p2.action_is_valid(p2_action[0])

                    self.p1.action = p1_action[0]
                    self.p2.action = p2_action[0]

                    if p1_action[0] == "logout" and p2_action[0] == "logout":
                        # send to visualizer
                        # send to eval server - eval_queue
                        data = self.eval_client.gamestate._get_data_plain_text()
                        self.eval_client.submit_to_eval()
                        break

                    if p1_action[0] == "shield":
                        if valid_action_p1 and not self.p1.check_shield():
                            viz_action_p1 = "shield"
                            self.p1.activate_shield()

                    if p2_action[0] == "shield":
                        if valid_action_p2 and not self.p2.check_shield():
                            viz_action_p2 = "shield"
                            self.p2.activate_shield()

                    if p1_action[0] == "grenade":
                        if valid_action_p1:
                            self.p1.throw_grenade()
                            if p1_action[1]:
                                viz_action_p2 = "hit_grenade"
                                self.p2.got_hit_grenade()
                                # if self.p2.check_shield():
                                #     viz_action_p2 = "hit_grenade_shield"

                    if p2_action[0] == "grenade":
                        if valid_action_p2:
                            self.p2.throw_grenade()
                            if p2_action[1]:
                                viz_action_p1 = "hit_grenade"
                                self.p1.got_hit_grenade()
                                # if self.p1.check_shield():
                                #     viz_action_p1 = "hit_grenade_shield"

                    if p1_action[0] == "shoot":
                        if valid_action_p1:
                            viz_action_p1 = "shoot"
                            self.p1.shoot()
                            if p1_action[1]:
                                viz_action_p2 = "hit_bullet"
                                self.p2.got_shot()
                                # if self.p2.check_shield():
                                #     viz_action_p2 = "hit_shield"

                    if p2_action[0] == "shoot":
                        if valid_action_p2:
                            viz_action_p2 = "shoot"
                            self.p2.shoot()
                            if p2_action[1]:
                                viz_action_p1 = "hit_bullet"
                                self.p1.got_shot()
                                # if self.p1.check_shield():
                                #     viz_action_p1 = "hit_shield"

                    if p1_action[0] == "reload":
                        if valid_action_p1:
                            self.p1.reload()
                            viz_action_p1 = "reload"

                    if p2_action[0] == "reload":
                        if valid_action_p2:
                            self.p2.reload()
                            viz_action_p2 = "reload"

                    # If health drops to 0 then everything resets except for number of deaths
                    if self.p1.hp <= 0:
                        self.reset_player(self.p1)
                    if self.p2.hp <= 0:
                        self.reset_player(self.p2)

                    print(self.eval_client.gamestate._get_data_plain_text())

                    # # gamestate to eval_server
                    # self.eval_client.submit_to_eval()
                    # # eval server to subscriber queue
                    # self.eval_client.receive_correct_ans()
                    # subscriber queue to sw/feedback queue
                    #
                    # if valid_action_p1:
                    #     self.p1.action = viz_action_p1
                    # else:
                    #     self.p1.action = "invalid"
                    #
                    # if valid_action_p2:
                    #     self.p2.action = viz_action_p2
                    # else:
                    #     self.p2.action = "invalid"

                    # laptop_queue.append(self.eval_client.gamestate._get_data_plain_text())
                    # subscribe_queue.put(self.eval_client.gamestate._get_data_plain_text())

            except KeyboardInterrupt as _:
                traceback.print_exc()


class AIModel(threading.Thread):
    def __init__(self, action_engine_model):
        super().__init__()

        self.action_engine = action_engine_model

    def run(self):
        while True:
            action = input('Action: ')
            player = int(input('Player: '))

            if action == 's':
                self.action_engine.handle_shield(player)
            elif action == 'g':
                self.action_engine.handle_grenade(player)
            elif action == 'r':
                self.action_engine.handle_reload(player)
            else:
                self.action_engine.handle_gun_shot(player)


if __name__ == '__main__':
    action_engine = ActionEngine()
    action_engine.start()

    ai = AIModel(action_engine)
    ai.start()



    # action_engine.handle_gun_shot()