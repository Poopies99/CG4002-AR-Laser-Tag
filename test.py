import sys
import os
import time
import traceback
import threading
from collections import deque

SINGLE_PLAYER_MODE = False
action_queue = deque()

class ActionEngine(threading.Thread):
    def __init__(self):
        super().__init__()

        self.p1_action_queue = deque()

        # Flags
        self.p1_gun_shot = False
        self.p1_vest_shot = False

        if not SINGLE_PLAYER_MODE:
            self.p2_action_queue = deque()

            self.p2_gun_shot = False
            self.p2_vest_shot = False

    def handle_grenade(self, player):
        print("Handling Grenade")
        if player == 1:
            # self.p1_grenade = True
            self.p1_action_queue.append('grenade')
        else:
            # self.p2_grenade = True
            self.p2_action_queue.append('grenade')

    def handle_shield(self, player):
        print("Handling Shield")
        if player == 1:
            # self.p1_shield = True
            self.p1_action_queue.append('shield')
        else:
            # self.p2_shield = True
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

    def run(self):
        while True:
            if self.p1_action_queue or self.p2_action_queue:
                time.sleep(3) # TODO - Based on AI prediction duration
                # Default
                action = [['None', True], ['None', True]]

                if self.p1_action_queue:
                    action_data = self.p1_action_queue.popleft()
                    if action_data == 'shoot':
                        action[0] = [action_data, self.p2_vest_shot]
                    elif action_data == 'grenade':
                        status = True # TODO - Check whether p2 is in frame
                        action[0] = [action_data, status]
                    elif action_data == 'reload':
                        action[0] = [action_data, True]
                    elif action_data == 'shield':
                        action[0] = [action_data, True]
                    elif action_data == 'logout':
                        action[0] = [action_data, True]

                if self.p2_action_queue:
                    action_data = self.p2_action_queue.popleft()
                    if action_data == 'shoot':
                        action[1] = [action_data, self.p1_vest_shot]
                    elif action_data == 'grenade':
                        status = True  # TODO - Check whether p1 is in frame
                        action[1] = [action_data, status]
                    elif action_data == 'reload':
                        action[1] = [action_data, True]
                    elif action_data == 'shield':
                        action[1] = [action_data, True]
                    elif action_data == 'logout':
                        action[1] = [action_data, True]

                self.p1_gun_shot = False
                self.p1_vest_shot = False

                if not SINGLE_PLAYER_MODE:
                    self.p2_gun_shot = False
                    self.p2_vest_shot = False

                print(action)

                action_queue.append(action)


class GameEngine(threading.Thread):
    def __init__(self, eval_client):
        super().__init__()

        # queue to receive status from sw
        self.eval_client = eval_client
        self.p1 = self.eval_client.gamestate.player_1
        self.p2 = self.eval_client.gamestate.player_2

        self.shutdown = threading.Event()

    def determine_grenade_hit(self):
        '''
        while True:
            print("Random")
            while not feedback_queue.empty():
                data = feedback_queue.get()
                if data == "6 hit_grenade#":
                    return True
                else:
                    return False
        '''
        return True

    # one approach is to put it in action queue and continue processing/ or do we want to wait for the grenade actions
    def random_ai_action(self, data):
        actions = ["shoot", "grenade", "shield", "reload", "invalid"]
        action_queue.append(([random.choice(actions)], ["False"]))

    def parse_action(self, player_action, player1, player2):
        if player_action[0] == "logout":
            # send to visualizer
            # send to eval server - eval_queue
            data = self.eval_client.gamestate._get_data_plain_text()
            subscribe_queue.put(data)
            # self.eval_client.submit_to_eval()
        if player_action[0] == "grenade":
            # receiving the status mqtt topic
            player1.throw_grenade()
            subscribe_queue.put(self.eval_client.gamestate._get_data_plain_text())
        elif player_action[0] == "shield":
            player1.activate_shield()
        elif player_action[0] == "shoot":
            player1.shoot()
            if player_action[1]:
                player2.got_shot()
        elif player_action[0] == "reload":
            player1.reload()
        if player_action[0] == "grenade":
            if self.determine_grenade_hit():
                player2.got_hit_grenade()

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

                    print(f"P1 action data: {p1_action}")
                    print(f"P2 action data: {p2_action}")

                    self.p1.update_shield()
                    self.p2.update_shield()

                    valid_action_p1 = self.p1.action_is_valid(p1_action[0])
                    valid_action_p2 = self.p2.action_is_valid(p2_action[0])

                    if valid_action_p1:
                        self.parse_action(p1_action, self.p1, self.p2)
                    if valid_action_p2:
                        self.parse_action(p2_action, self.p2, self.p1)

                    # If health drops to 0 then everything resets except for number of deaths
                    if self.p1.hp <= 0:
                        self.reset_player(self.p1)
                    if self.p2.hp <= 0:
                        self.reset_player(self.p2)

                    # gamestate to eval_server
                    self.eval_client.submit_to_eval()
                    # eval server to subscriber queue
                    self.eval_client.receive_correct_ans()
                    # subscriber queue to sw/feedback queue

                    if valid_action_p1 or valid_action_p2:
                        subscribe_queue.put(self.eval_client.gamestate._get_data_plain_text())
                    else:
                        self.p1.update_invalid_action()
                        self.p2.update_invali_action()
                        subscribe_queue.put(self.eval_client.gamestate._get_data_plain_text())

            except KeyboardInterrupt as _:
                traceback.print_exc()


class AIModel(threading.Thread):
    def __init__(self, action_engine_model):
        super().__init__()

        self.action_engine = action_engine_model

    def run(self):
        while True:
            action = input('Action: ')

            if action == 's':
                self.action_engine.handle_shield(1)
            elif action == 'g':
                self.action_engine.handle_grenade(1)
            else:
                self.action_engine.handle_gun_shot(1)


if __name__ == '__main__':
    action_engine = ActionEngine()
    action_engine.start()

    ai = AIModel(action_engine)
    ai.start()



    # action_engine.handle_gun_shot()