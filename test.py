import sys
import os
import time
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