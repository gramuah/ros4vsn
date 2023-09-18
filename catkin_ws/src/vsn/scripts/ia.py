import random


class Ia:
    def random(self):
        a = random.randrange(0, 4, 1)
        angle = random.randrange(0, 180, 1)

        if a == 0:
            movement = "Left"
        elif a == 1:
            movement = "Right"
        elif a == 2:
            movement = "Forward"
        elif a == 3:
            movement = "Stop"

        print(movement, angle)

        return movement, angle
