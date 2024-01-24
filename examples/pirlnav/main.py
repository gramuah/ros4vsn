from PIL import Image
import numpy as np
import glob

from class_model import Pirlnav


def main():
    model = Pirlnav()

    actions = []

    images = sorted(glob.glob("imgs/ep19/*"))
    goal = 1

    for i in images:

        # GPS[0, 5], COMPASS[0]
        odom = np.array([.0, .0, .0])
        img = np.array(Image.open(i))

        input_observation = {'image': img,
                             'obj_goal': goal,
                             'position': odom}

        action = model.evaluation(observation=input_observation)

        actions.append(action)

    print(actions)


if __name__ == "__main__":
    main()
