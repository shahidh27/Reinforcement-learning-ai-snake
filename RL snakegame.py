import tkinter as tk
import random
import pickle
import os

# =========================
# AI SNAKE - Q LEARNING
# =========================

CELL = 20
WIDTH = 20
HEIGHT = 20

ACTIONS = [0, 1, 2]   # 0 = straight, 1 = left, 2 = right

# Directions: UP, RIGHT, DOWN, LEFT
DIRECTIONS = [
    (0, -1),
    (1, 0),
    (0, 1),
    (-1, 0)
]

Q_FILE = "snake_qtable.pkl"

# -------------------------
# Q TABLE
# -------------------------

if os.path.exists(Q_FILE):
    with open(Q_FILE, "rb") as f:
        Q = pickle.load(f)
    print("Previous learning loaded!")
else:
    Q = {}

# Exploration rate
epsilon = 1.0
epsilon_min = 0.03
epsilon_decay = 0.995

alpha = 0.1
gamma = 0.9


def get_q(state):
    if state not in Q:
        Q[state] = [0.0, 0.0, 0.0]
    return Q[state]


# -------------------------
# SNAKE GAME
# -------------------------

class SnakeGame:

    def __init__(self):
        self.reset()

    def reset(self):
        self.snake = [(10, 10), (9, 10), (8, 10)]
        self.direction = 1
        self.score = 0
        self.steps = 0
        self.steps_since_food = 0
        self.place_food()

    def place_food(self):
        while True:
            self.food = (
                random.randint(0, WIDTH - 1),
                random.randint(0, HEIGHT - 1)
            )

            if self.food not in self.snake:
                break

    def danger(self, direction):
        head_x, head_y = self.snake[0]
        dx, dy = DIRECTIONS[direction]

        nx = head_x + dx
        ny = head_y + dy

        # Wall
        if nx < 0 or nx >= WIDTH or ny < 0 or ny >= HEIGHT:
            return 1

        # Body
        if (nx, ny) in self.snake[:-1]:
            return 1

        return 0

    def get_state(self):

        head_x, head_y = self.snake[0]

        left_dir = (self.direction - 1) % 4
        straight_dir = self.direction
        right_dir = (self.direction + 1) % 4

        danger_straight = self.danger(straight_dir)
        danger_left = self.danger(left_dir)
        danger_right = self.danger(right_dir)

        food_x, food_y = self.food

        food_left = int(food_x < head_x)
        food_right = int(food_x > head_x)
        food_up = int(food_y < head_y)
        food_down = int(food_y > head_y)

        state = (
            danger_straight,
            danger_left,
            danger_right,
            food_left,
            food_right,
            food_up,
            food_down,
            self.direction
        )

        return state

    def move(self, action):

        old_distance = abs(self.snake[0][0] - self.food[0]) + \
                       abs(self.snake[0][1] - self.food[1])

        # Turn
        if action == 1:
            self.direction = (self.direction - 1) % 4

        elif action == 2:
            self.direction = (self.direction + 1) % 4

        head_x, head_y = self.snake[0]
        dx, dy = DIRECTIONS[self.direction]

        new_head = (head_x + dx, head_y + dy)

        self.steps += 1
        self.steps_since_food += 1

        # Collision
        if (
            new_head[0] < 0 or
            new_head[0] >= WIDTH or
            new_head[1] < 0 or
            new_head[1] >= HEIGHT or
            new_head in self.snake[:-1]
        ):
            return -10, True

        self.snake.insert(0, new_head)

        # Food
        if new_head == self.food:

            self.score += 1
            self.steps_since_food = 0
            self.place_food()

            return 10, False

        # Remove tail
        self.snake.pop()

        new_distance = abs(new_head[0] - self.food[0]) + \
                       abs(new_head[1] - self.food[1])

        reward = -0.1

        # Reward for moving closer
        if new_distance < old_distance:
            reward += 0.3
        else:
            reward -= 0.2

        # Prevent endless movement
        if self.steps_since_food > 100:
            return -5, True

        return reward, False


# -------------------------
# TRAINING
# -------------------------

def train(episodes=3000):

    global epsilon

    game = SnakeGame()

    best_score = 0

    print("\nAI TRAINING STARTED...\n")

    for episode in range(1, episodes + 1):

        game.reset()

        state = game.get_state()

        while True:

            q_values = get_q(state)

            # Exploration
            if random.random() < epsilon:
                action = random.choice(ACTIONS)
            else:
                action = q_values.index(max(q_values))

            reward, done = game.move(action)

            next_state = game.get_state()
            next_q = get_q(next_state)

            # Q-Learning formula
            old_value = q_values[action]

            if done:
                target = reward
            else:
                target = reward + gamma * max(next_q)

            Q[state][action] = old_value + alpha * (
                target - old_value
            )

            state = next_state

            if done:
                break

        if game.score > best_score:
            best_score = game.score

        epsilon = max(
            epsilon_min,
            epsilon * epsilon_decay
        )

        if episode % 100 == 0:
            print(
                "Episode:",
                episode,
                "| Score:",
                game.score,
                "| Best:",
                best_score,
                "| Epsilon:",
                round(epsilon, 3)
            )

    with open(Q_FILE, "wb") as f:
        pickle.dump(Q, f)

    print("\nTRAINING FINISHED!")
    print("Best Score:", best_score)
    print("Q-table saved as:", Q_FILE)


# -------------------------
# PLAY AI
# -------------------------

class AISnakeWindow:

    def __init__(self):

        self.game = SnakeGame()

        self.root = tk.Tk()
        self.root.title("AI Snake - Reinforcement Learning")

        self.canvas = tk.Canvas(
            self.root,
            width=WIDTH * CELL,
            height=HEIGHT * CELL,
            bg="black"
        )

        self.canvas.pack()

        self.info = tk.Label(
            self.root,
            text="AI Snake",
            font=("Arial", 14)
        )

        self.info.pack()

        self.play()

        self.root.mainloop()

    def draw(self):

        self.canvas.delete("all")

        # Food
        fx, fy = self.game.food

        self.canvas.create_oval(
            fx * CELL,
            fy * CELL,
            (fx + 1) * CELL,
            (fy + 1) * CELL,
            fill="red"
        )

        # Snake
        for i, (x, y) in enumerate(self.game.snake):

            if i == 0:
                color = "lime"
            else:
                color = "green"

            self.canvas.create_rectangle(
                x * CELL,
                y * CELL,
                (x + 1) * CELL,
                (y + 1) * CELL,
                fill=color
            )

        self.info.config(
            text=f"AI Score: {self.game.score}"
        )

    def play(self):

        state = self.game.get_state()

        q_values = get_q(state)

        action = q_values.index(max(q_values))

        reward, done = self.game.move(action)

        self.draw()

        if done:

            print(
                "AI Game Over | Score:",
                self.game.score
            )

            self.game.reset()

            self.root.after(300, self.play)

        else:
            self.root.after(80, self.play)


# =========================
# MAIN
# =========================

print("================================")
print("   REINFORCEMENT LEARNING SNAKE")
print("================================")

print("\n1. Training AI...")
train(3000)

print("\n2. Starting AI gameplay...")

AISnakeWindow()
