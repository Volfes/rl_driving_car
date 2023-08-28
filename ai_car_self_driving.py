import pygame
import math
import utils
import numpy as np
import random


win_point = 0
iteration = 1
move_keys = ['w', 'a', 's', 'd']
STATE_DIMENSION = 81024768  # number of state dimensions
LEARNING_RATE = 0.3
DISCOUNT_FACTOR = 0.8
EXPLORATION_PROB = 0.2

q_table = np.zeros((STATE_DIMENSION, len(move_keys)))


def main():

    def choose_action(state):
        if random.randint(0, 100)/100 < EXPLORATION_PROB and max(
                q_table[state]) < 0.05:
            random_key = move_keys[random.randint(0, 3)]
            return random_key
        bot_key = move_keys[np.argmax(q_table[state])]
        return bot_key

    def update_q_table(state, action, reward, next_state):
        current_q = q_table[state][move_keys.index(action)]
        max_next_q = np.max(q_table[next_state])
        new_q = current_q + LEARNING_RATE * (reward + DISCOUNT_FACTOR *
                                             max_next_q - current_q)
        q_table[state][move_keys.index(action)] = new_q

    GRASS = pygame.image.load('imgs/grass.jpg')
    TRACK = pygame.image.load('imgs/track-small.png')
    TRACK_BORDER = pygame.image.load('imgs/track-small-border.png')
    FINISH = pygame.image.load('imgs/finish-small-track.png')
    INTERSECTION = pygame.image.load('imgs/intersection-small-v.png')
    BOT_LEARN_INTERSECTION = pygame.image.load('imgs/intersection-small-h.png')
    BOT_LEARN_POINT = pygame.image.load('imgs/bot_learn_point.png')
    TRACK_WARNING = pygame.image.load('imgs/track_warning.png')
    AI_CAR = pygame.image.load('imgs/ferrari-scaled.png')
    TRACK_WARNING_MASK = pygame.mask.from_surface(TRACK_WARNING)
    FINISH_MASK = pygame.mask.from_surface(FINISH)
    BOT_INTERSECTION_MASK = pygame.mask.from_surface(BOT_LEARN_INTERSECTION)
    BOT_LEARN_POINT_MASK = pygame.mask.from_surface(BOT_LEARN_POINT)
    INTERSECTION_MASK = pygame.mask.from_surface(INTERSECTION)
    WIDTH, HEIGHT = TRACK.get_width(), TRACK.get_height()
    TRACK_BORDER_MASK = pygame.mask.from_surface(TRACK_BORDER)

    FINISH_POSITION = (101, 38)
    INTERSECTION_1 = (700, 710)
    INTERSECTION_2 = (690, 650)
    INTERSECTION_3 = (690, 600)
    INTERSECTION_4 = (690, 510)
    INTERSECTION_5 = (670, 420)
    INTERSECTION_6 = (645, 200)
    INTERSECTION_7 = (500, 200)
    INTERSECTION_8 = (400, 200)
    INTERSECTION_9 = (300, 150)
    INTERSECTION_10 = (100, 180)
    BOT_LEARN_POINT_POS_1 = (690, 325)
    BOT_LEARN_POINT_POS_2 = (220, 230)
    TRACK_WARNING_POS_1 = (0, 0)

    # place and draw images
    def draw(screen, images, images_intersection):
        # bottom layer - reward
        for img, pos in images_intersection:
            screen.blit(img, pos)
        # top level images
        for img, pos in images:
            screen.blit(img, pos)
        bot_car.draw(screen)

    # constant images
    images = [(TRACK_WARNING, TRACK_WARNING_POS_1),
              (GRASS, (0, 0)), (TRACK, (0, 0)),
              (FINISH, FINISH_POSITION)]

    # intersections
    images_intersection = [(INTERSECTION, INTERSECTION_4),
                           (INTERSECTION, INTERSECTION_5),
                           (INTERSECTION, INTERSECTION_10),
                           (INTERSECTION, INTERSECTION_1),
                           (INTERSECTION, INTERSECTION_2),
                           (INTERSECTION, INTERSECTION_3),
                           (BOT_LEARN_INTERSECTION, INTERSECTION_6),
                           (BOT_LEARN_INTERSECTION, INTERSECTION_7),
                           (BOT_LEARN_INTERSECTION, INTERSECTION_8),
                           (BOT_LEARN_INTERSECTION, INTERSECTION_9),
                           (BOT_LEARN_POINT, BOT_LEARN_POINT_POS_1),
                           (BOT_LEARN_POINT, BOT_LEARN_POINT_POS_2),
                           ]

    class DefaultCar:
        def __init__(self, max_vel, rotation_vel):
            self.img = self.IMG
            self.max_vel = max_vel
            self.vel = 0
            self.rotation_vel = rotation_vel
            self.angle = 0
            self.x, self.y = self.START_POS
            self.acceleration = 0.1

        def rotate(self, left=False, right=False):
            if left:
                self.angle += self.rotation_vel
            elif right:
                self.angle -= self.rotation_vel

        def draw(self, screen):
            utils.blit_rotate_center(screen, self.img, (self.x, self.y),
                                     self.angle)

        def move_forward(self):
            self.vel = min(self.vel + self.acceleration, self.max_vel)
            self.move()

        def move_backward(self):
            self.vel = max(self.vel - self.acceleration, -self.max_vel/2)
            self.vel = 0
            self.move()

        def move(self):
            radians = math.radians(self.angle)
            vertical_vel = math.cos(radians) * self.vel
            horizontal_vel = math.sin(radians) * self.vel
            self.y -= vertical_vel
            self.x -= horizontal_vel
            return radians-(2*math.pi)*int(radians/(2*math.pi))

        # given value are correlated to top left bottom of picture, it doesn't
        # change place at rotation
        def read_pixel_colour(self):
            bottom_pixel = screen.get_at((int(self.x)+7, int(self.y)+67))
            top_pixel = screen.get_at((int(self.x)+7, int(self.y)-40))
            left_pixel = screen.get_at((int(self.x)-26, int(self.y)+13))
            right_pixel = screen.get_at((int(self.x)+40, int(self.y)+13))
            return bottom_pixel, top_pixel, left_pixel, right_pixel, self.move()

        def read_axis(self):
            x_axis = self.x
            y_axis = self.y
            return x_axis, y_axis

        def collide(self, mask, x=0, y=0):
            car_mask = pygame.mask.from_surface(self.img)
            offset = (int(self.x - x), int(self.y - y))
            point_of_intersection = mask.overlap(car_mask, offset)
            return point_of_intersection

        def bounce(self):
            self.vel = -self.vel/2
            self.move()

    class BotCar(DefaultCar):
        IMG = AI_CAR
        START_POS = (810, 680)

        def reduce_speed(self):
            self.vel = max(self.vel - self.acceleration / 4, 0)
            self.move()

        def disqualification(self):
            self.vel = 0
            self.acceleration = 0

    def move_bot(bot_car):
        if bot_path[bot_seed] == 'w':
            if tpx == (254, 2, 2, 255) and bpx != (
                    254, 2, 2, 255) and lpx != (254, 2, 2, 255) and rpx != (
                    254, 2, 2, 255) and (angle_pos in (1, 2, 7, 8)):
                bot_car.move_backward()
                if angle_pos == 1:
                    bot_car.rotate(left=True)
                    bot_car.rotate(left=True)
                elif angle_pos == 2:
                    bot_car.rotate(left=True)
                    bot_car.rotate(left=True)

                elif angle_pos == 7:
                    bot_car.rotate(right=True)
                    bot_car.rotate(right=True)

                elif angle_pos == 8:
                    bot_car.rotate(right=True)
                    bot_car.rotate(right=True)

            elif lpx == (254, 2, 2, 255) and (angle_pos in (1, 2, 3, 4, 6, 7)):
                bot_car.move_backward()
                bot_car.rotate(right=True)

            elif rpx == (254, 2, 2, 255) and (angle_pos in (2, 3, 4, 6, 7, 8)):
                bot_car.move_backward()
                bot_car.rotate(left=True)

            elif bpx == (254, 2, 2, 255) and tpx != (
                    254, 2, 2, 255) and lpx != (254, 2, 2, 255) and rpx != (
                    254, 2, 2, 255) and (angle_pos in (3, 4, 5, 6)):
                bot_car.move_backward()
                if angle_pos == 3:
                    bot_car.rotate(right=True)
                    bot_car.rotate(right=True)

                elif angle_pos == 4:
                    bot_car.rotate(right=True)
                    bot_car.rotate(right=True)

                elif angle_pos == 5:
                    bot_car.rotate(left=True)
                    bot_car.rotate(left=True)

                elif angle_pos == 6:
                    bot_car.rotate(left=True)
                    bot_car.rotate(left=True)

            bot_car.move_forward()
        if bot_path[bot_seed] == 's':
            bot_car.move_backward()
        if bot_path[bot_seed] == 'a':
            bot_car.rotate(left=True)
        if bot_path[bot_seed] == 'd':
            bot_car.rotate(right=True)
        return bot_car.read_axis()

    # pygame setup
    pygame.init()
    screen = pygame.display.set_mode((WIDTH, HEIGHT))
    pygame.display.set_caption('Race')
    clock = pygame.time.Clock()
    running = True
    FPS = 60
    speed = 20
    angle_speed = 6
    bot_car = BotCar(speed, angle_speed)
    bot_points = np.array([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0])
    bot_seed = 0
    bot_path = []
    bot_state = []

    global win_point
    global iteration
    print(f"iteracja: {iteration}")
    iteration += 1
    print(f'liczba wygranych: {win_point}')
    timer = 0

    while running:
        # pygame.QUIT event means the user clicked X to close your window
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
                break

        reward = 0
        timer += 1
        prev_state = bot_car.read_axis()
        (x_axis, y_axis) = prev_state
        bpx, tpx, lpx, rpx, angle = bot_car.read_pixel_colour()
        plus_reward = False

        if angle >= 0:
            if angle < 0.785:
                angle_pos = 1
            elif angle < 1.57:
                angle_pos = 2
            elif angle < 2.355:
                angle_pos = 3
            elif angle < 3.14:
                angle_pos = 4
            elif angle < 3.925:
                angle_pos = 5
            elif angle < 4.71:
                angle_pos = 6
            elif angle < 5.495:
                angle_pos = 7
            else:
                angle_pos = 8
        else:
            if angle > -0.785:
                angle_pos = 8
            elif angle > -1.57:
                angle_pos = 7
            elif angle > -2.355:
                angle_pos = 6
            elif angle > -3.14:
                angle_pos = 5
            elif angle > -3.925:
                angle_pos = 4
            elif angle > -4.71:
                angle_pos = 3
            elif angle > -5.495:
                angle_pos = 2
            else:
                angle_pos = 1

        prev_state = int(x_axis)*1000 + int(y_axis) + angle_pos * 10_000_000
        bot_action = choose_action(prev_state)
        bot_path.append(bot_action)
        bot_state.append(prev_state)
        move_bot(bot_car)
        bot_seed += 1

        if bot_car.collide(FINISH_MASK, *FINISH_POSITION):
            win_point += 1
            plus_reward = True
            add_points(bot_path, bot_state, 1)
            play_again()

        if bot_car.collide(TRACK_WARNING_MASK, *TRACK_WARNING_POS_1):
            reward -= 0.001
        if bot_points[1] == 0 and timer > 30:
            reward -= 0.01
        if bot_points[2] == 0 and timer > 40:
            reward -= 0.01
        if bot_points[3] == 0 and timer > 50:
            reward -= 0.01
        if bot_points[4] == 0 and timer > 60:
            reward -= 0.01
        if bot_points[5] == 0 and timer > 140:
            reward -= 0.01
        if bot_points[6] == 0 and timer > 200:
            reward -= 0.01
        if bot_points[7] == 0 and timer > 300:
            reward -= 0.01
        if bot_points[8] == 0 and timer > 350:
            reward -= 0.01
        if bot_points[10] == 0 and timer > 400:
            reward -= 0.01

        if bot_car.collide(INTERSECTION_MASK, *INTERSECTION_1):
            add_points(bot_path, bot_state, -100)
            bot_car.disqualification()
            play_again()
        if bot_car.collide(INTERSECTION_MASK, *INTERSECTION_2):
            if bot_points[0] == 0:
                add_points(bot_path, bot_state, 0, 0.1)
                bot_points[0] = 1
                plus_reward = True
            if bot_points[1] == 1:
                play_again()
        if bot_car.collide(INTERSECTION_MASK, *INTERSECTION_3):
            if bot_points[1] == 0:
                add_points(bot_path, bot_state, 0, 0.1)
                plus_reward = True
                bot_points[1] = 1
            if bot_points[2] == 1:
                play_again()
        if bot_car.collide(INTERSECTION_MASK, *INTERSECTION_4):
            if bot_points[2] == 0:
                add_points(bot_path, bot_state, 0, 0.1)
                plus_reward = True
                bot_points[2] = 1
            if bot_points[3] == 1:
                play_again()
        if bot_car.collide(INTERSECTION_MASK, *INTERSECTION_5):
            if bot_points[3] == 0:
                add_points(bot_path, bot_state, 0, 0.1)
                add_points(bot_path, bot_state)
                bot_points[3] = 1
            if bot_points[5] == 1:
                play_again()
        if bot_car.collide(BOT_LEARN_POINT_MASK, *BOT_LEARN_POINT_POS_1):
            if bot_points[3] == 1 and bot_points[5] == 0 and bot_points[4] == 0:
                add_points(bot_path, bot_state, 1)
                bot_points[4] = 1
        if bot_car.collide(BOT_INTERSECTION_MASK, *INTERSECTION_6):
            if bot_points[5] == 0:
                add_points(bot_path, bot_state, 0, 0.1)
                add_points(bot_path, bot_state)
                bot_points[5] = 1
            if bot_points[6] == 1:
                play_again()
        if bot_car.collide(BOT_INTERSECTION_MASK, *INTERSECTION_7):
            if bot_points[6] == 0:
                add_points(bot_path, bot_state, 0, 0.1)
                add_points(bot_path, bot_state)
                bot_points[6] = 1
            if bot_points[7] == 1:
                play_again()
        if bot_car.collide(BOT_INTERSECTION_MASK, *INTERSECTION_8):
            if bot_points[7] == 0:
                add_points(bot_path, bot_state, 0, 0.1)
                add_points(bot_path, bot_state)
                bot_points[7] = 1
            if bot_points[8] == 1:
                play_again()
        if bot_car.collide(BOT_INTERSECTION_MASK, *INTERSECTION_9):
            if bot_points[8] == 0:
                bot_points[8] = 1
                add_points(bot_path, bot_state, 0, 0.1)
                add_points(bot_path, bot_state)
        if bot_car.collide(BOT_LEARN_POINT_MASK, *BOT_LEARN_POINT_POS_2):
            if bot_points[8] == 1 and bot_points[9] == 0 and bot_points[10] == 0:
                add_points(bot_path, bot_state, 1)
                bot_points[9] = 1
        if bot_car.collide(INTERSECTION_MASK, *INTERSECTION_10):
            if bot_points[10] == 0:
                reward += 20
                add_points(bot_path, bot_state, 2, 0.1)
                bot_points[10] = 1
        if bot_car.collide(TRACK_BORDER_MASK) is not None:
            reward = -1
            bot_car.bounce()
            play_again()

        if timer == 180 and bot_points[3] != 1:
            print('time out #1')
            add_points(bot_path, bot_state, -0.5)
            play_again()
        if timer == 400 and bot_points[7] != 1:
            print('time out #2')
            add_points(bot_path, bot_state, -0.5)
            play_again()
        if timer == 600 and bot_points[10] != 1:
            print('time out #3')
            add_points(bot_path, bot_state, -0.5)
            play_again()

        bpx, tpx, lpx, rpx, angle = bot_car.read_pixel_colour()

        if angle >= 0:
            if angle < 0.785:
                angle_pos = 1
            elif angle < 1.57:
                angle_pos = 2
            elif angle < 2.355:
                angle_pos = 3
            elif angle < 3.14:
                angle_pos = 4
            elif angle < 3.925:
                angle_pos = 5
            elif angle < 4.71:
                angle_pos = 6
            elif angle < 5.495:
                angle_pos = 7
            else:
                angle_pos = 8
        else:
            if angle > -0.785:
                angle_pos = 8
            elif angle > -1.57:
                angle_pos = 7
            elif angle > -2.355:
                angle_pos = 6
            elif angle > -3.14:
                angle_pos = 5
            elif angle > -3.925:
                angle_pos = 4
            elif angle > -4.71:
                angle_pos = 3
            elif angle > -5.495:
                angle_pos = 2
            else:
                angle_pos = 1

        new_x, new_y = bot_car.read_axis()
        new_state = int(new_x)*1000 + int(new_y) + angle_pos * 10_000_000
        update_q_table(prev_state, bot_action, reward, new_state)

        if plus_reward is True:
            add_points(bot_path, bot_state, reward=0.05)

        # RENDER YOUR GAME HERE
        draw(screen, images, images_intersection)

        # flip() the display to put your work on screen
        pygame.display.flip()
        clock.tick(FPS)  # limits FPS to given value

    pygame.quit()


def play_again():
    main()


def add_points(bot_path, bot_state, value=0, reward=0.01):
    for index_num, move in enumerate(bot_path):
        try:
            q_table[bot_state[index_num]][move_keys.index(move)] += (value +
                                                                     reward)
        except IndexError:
            print('out of range')


if __name__ == "__main__":
    main()
