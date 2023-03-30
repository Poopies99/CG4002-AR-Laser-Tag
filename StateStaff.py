import time
import json

class Player:
    def __init__(self):
        super().__init__()

        self.max_grenades       = 2
        self.max_shields        = 3
        self.bullet_hp          = 10
        self.grenade_hp         = 30
        self.shield_max_time    = 10
        self.shield_health_max  = 30
        self.magazine_size      = 6
        self.max_hp             = 100

        self.hp             = self.max_hp
        self.action         = "none"
        self.bullets        = self.magazine_size
        self.grenades       = self.max_grenades
        self.num_shield     = self.max_shields
        self.num_deaths     = 0
        
        self.shield_time    = 0
        self.shield_health  = 0
        self.shield_timer = 0
        self.shield_status = False

    def update_json(self):
        with open('example.json', 'w') as f:
            json_object = json.loads(f.read())

        json_object["p1"]["action"] = self.action
        json_object["p1"]["hp"] = self.hp
        json_object["p1"]["action"] = self.action
        json_object["p1"]["bullets"] = self.bullets
        json_object["p1"]["grenades"] = self.grenades
        json_object["p1"]["shield_time"] = self.shield_time
        json_object["p1"]["shield_health"] = self.shield_health
        json_object["p1"]["num_deaths"] = self.num_deaths
        json_object["p1"]["num_shield"] = self.num_shield

        with open('example.json', 'w') as f:
            f.write(json.dumps(json_object))

        return json.dumps(json_object)

    def get_dict(self):
        _player = dict()
        _player["hp"]               = self.hp
        _player["action"]           = self.action
        _player["bullets"]          = self.bullets
        _player["grenades"]         = self.grenades
        _player["shield_time"]      = self.shield_time
        _player["shield_health"]    = self.shield_health
        _player["num_deaths"]       = self.num_deaths
        _player["num_shield"]       = self.num_shield
        return _player

    def initialize(self, action, bullets_remaining, grenades_remaining,
                   hp, num_deaths, num_unused_shield,
                   shield_health, shield_time_remaining):
        self.hp             = hp
        self.action         = action
        self.bullets        = bullets_remaining
        self.grenades       = grenades_remaining
        self.shield_time    = shield_time_remaining
        self.shield_health  = shield_health
        self.num_shield     = num_unused_shield
        self.num_deaths     = num_deaths

    def initialize_from_dict(self, player_dict: dict):
        self.hp             = int(player_dict['hp'])
        self.action         = player_dict['action']
        self.bullets        = int(player_dict['bullets'])
        self.grenades       = int(player_dict['grenades'])
        self.shield_time    = float(player_dict['shield_time'])
        self.shield_health  = int(player_dict['shield_health'])
        self.num_shield     = int(player_dict['num_shield'])
        self.num_deaths     = int(player_dict['num_deaths'])

    def check_hp(self):
        if self.hp <= 0:
            return "He is dead, not big souprise"
        else:
            return "He is alive"

    """
    Function checks if IR Sensor receives a Signal (to be completed)
    """
    @staticmethod
    def check_sensor():
        return True

    """
    Function checks if shield is active
    """
    def check_shield(self):
        return self.shield_status

    """
    Function checks if Player is in Screen
    """
    @staticmethod
    def check_grenade_hit():
        return True

    def shoot(self):
        self.bullets -= 1                

    def got_shot(self):
        if self.shield_status:
            self.shield_health -= 10
        else:
            self.hp -= 10
    
    def update_shield(self):
        if self.shield_status:
            self.shield_time = 10 - (float(time.time() - self.shield_timer))
            if (self.shield_time <= 0 or self.shield_health  <= 0):
                self.shield_status = False
                self.shield_time = 0
                self.shield_health = 0
                               
    def throw_grenade(self):
        self.grenades -= 1
 
    def got_hit_grenade(self):
        if not self.shield_status or not self.shield_health:
            self.hp -= self.grenade_hp
        elif self.shield_health < 30:  # Player is shielded but grenade will break through shield
            diff = self.grenade_hp - self.shield_health
            self.shield_health = 0
            self.hp -= diff

    def activate_shield(self):
        if not self.shield_active():
            self.num_shield -= 1
            self.shield_status = True
            self.shield_health = 30
            self.shield_timer = time.time()
            self.shield_time = 10 - int(time.time() - self.shield_timer)   
                        
    def reload(self):
        self.bullets = self.magazine_size