import time


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
        self.shield_time    = 0
        self.shield_health  = 0
        self.num_shield     = self.max_shields
        self.num_deaths     = 0

        self.shield_start_time = time.time()-30

    def get_string(self):
        print("Current Player HP:", self.hp)
        print("Player Action:", self.action)
        print("Number of Bullets Left:", self.bullets)
        print("Number of Grenades Left:", self.grenades)
        print("Shield Time Left:", self.shield_time)
        print("Shield Health Left:", self.shield_health)
        print("Number of Deaths:", self.num_deaths)
        print("Number of Shields Left:",  self.num_shield)

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
    @staticmethod
    def check_shield():
        return False

    """
    Function checks if Player is in Screen
    """
    @staticmethod
    def check_grenade_hit():
        return True

    def shoot(self):
        if self.bullets == 0:
            return "Please Reload"
        elif self.check_sensor():  # If True, Player has been shot
            if self.shield_health > 0:
                self.shield_health -= self.bullet_hp
            else:
                self.hp -= self.bullet_hp
        self.bullets -= 1
        return self.check_hp()

    def throw_grenade(self):
        if self.grenades == 0:
            return "No Grenades"
        elif not self.check_grenade_hit():
            if not self.shield_health:
                self.hp -= self.grenade_hp
            elif self.shield_health < 30:  # Player is shielded but grenade will break through shield
                diff = self.grenade_hp - self.shield_health
                self.shield_health = 0
                self.hp -= diff
        self.grenades -= 1
        return self.check_hp()

    def activate_shield(self):
        if self.num_shield == 0:
            return "No Shields Left"
        elif self.check_shield():
            return "Shield is Active"
        else:
            self.shield_health = self.shield_health_max
            self.shield_time = self.shield_max_time
            self.num_shield -= 1

    def reload(self):
        if self.bullets != 0:
            return "Cannot reload when there magazine is not empty"
        else:
            self.bullets = self.magazine_size
