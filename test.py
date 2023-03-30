import sys
import os
import time
import threading
from collections import deque



SINGLE_PLAYER_MODE = False


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

    def handle_grenade_throw(self, player):
        if player == 1:
            # self.p1_grenade = True
            self.p1_action_queue.append('grenade')
        else:
            # self.p2_grenade = True
            self.p2_action_queue.append('grenade')

    def handle_shield(self, player):
        if player == 1:
            # self.p1_shield = True
            self.p1_action_queue.append('shield')
        else:
            # self.p2_shield = True
            self.p2_action_queue.append('shield')

    def handle_reload(self, player):
        if player == 1:
            self.p1_action_queue.append('reload')
        else:
            self.p2_action_queue.append('reload')

    def handle_logout(self, player):
        if player == 1:
            self.p1_action_queue.append('logout')
        else:
            self.p2_action_queue.append('logout')

    def handle_gun_shot(self, player):
        if player == 1:
            self.p1_gun_shot = True
            self.p1_action_queue.append('shoot')
        else:
            self.p2_gun_shot = True
            self.p2_action_queue.append('shoot')

    def handle_vest_shot(self, player):
        if player == 1:
            self.p1_vest_shot = True
        else:
            self.p2_vest_shot = True

    def run(self):
        while True:
            if self.p1_action_queue or self.p2_action_queue:
                time.sleep(1)  # TODO - Based on AI prediction duration
                # Default
                action = [['None', True], ['None', True]]

                if self.p1_action_queue:
                    action_data = self.p1_action_queue.popleft()
                    if action_data == 'shoot':
                        action[0] = [action_data, self.p2_vest_shot]
                    elif action_data == 'grenade':
                        status = True  # TODO - Check whether p2 is in frame
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

if __name__ == '__main__':
    action_engine = ActionEngine()
    action_engine.start()

    action_engine.handle_gun_shot(1)
    time.sleep(0.5)
    action_engine.handle_vest_shot(2)
    action_engine.handle_grenade_throw(2)
    action_engine.handle_gun_shot(2)
    time.sleep(2)
    action_engine.handle_gun_shot(1)



    # action_engine.handle_gun_shot()