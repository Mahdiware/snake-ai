"""Unified Snake AI script: train or visualize a trained Q-table.

Usage:
    python snake_ai.py            # visualize trained agent (GUI)
    python snake_ai.py train      # train with GUI (saves snake_qtable.pickle)

Requires: pygame, numpy
"""

import argparse
import pickle
import random
import sys
from typing import Dict, Tuple

import numpy as np
import pygame

# Board settings
CELL_SIZE = 40
GRID_WIDTH = 10
GRID_HEIGHT = 10
ACTIONS = [0, 1, 2, 3]  # up, down, left, right

# Training settings
ALPHA = 0.1
GAMMA = 0.9
EPSILON_START = 1.0
EPSILON_DECAY = 0.995
MIN_EPSILON = 0.01
EPISODES = 1500
MAX_STEPS_PER_EPISODE = 400
LIVING_PENALTY = -0.05  # small negative reward each step to encourage finding food

# Render settings (shared)
BACKGROUND_COLOR = (12, 12, 12)
GRID_COLOR = (32, 32, 32)
SNAKE_COLOR = (40, 200, 64)
SNAKE_HEAD_COLOR = (80, 255, 110)
FOOD_COLOR = (235, 64, 52)
TEXT_COLOR = (230, 230, 230)
TRAIN_FPS = 30
VIEW_FPS = 15
RENDER_EVERY = 1  # render every N episodes during training


# --- Common helpers ---
def get_state(snake, food):
    head = snake[0]
    state = (
        int(head[0] == 0 or (head[0] - 1, head[1]) in snake),
        int(head[0] == GRID_WIDTH - 1 or (head[0] + 1, head[1]) in snake),
        int(head[1] == 0 or (head[0], head[1] - 1) in snake),
        int(head[1] == GRID_HEIGHT - 1 or (head[0], head[1] + 1) in snake),
        int(food[0] < head[0]),
        int(food[0] > head[0]),
        int(food[1] < head[1]),
        int(food[1] > head[1]),
    )
    return state


def spawn_food(snake):
    free = [(x, y) for x in range(GRID_WIDTH) for y in range(GRID_HEIGHT) if (x, y) not in snake]
    return random.choice(free) if free else snake[0]


def handle_events(paused):
    exit_requested = False
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            exit_requested = True
        elif event.type == pygame.KEYDOWN:
            if event.key == pygame.K_SPACE:
                paused = not paused
            elif event.key == pygame.K_ESCAPE:
                exit_requested = True
    return exit_requested, paused


# --- Drawing ---
def draw_board(screen, snake, food):
    screen.fill(BACKGROUND_COLOR)

    # Grid
    for x in range(GRID_WIDTH + 1):
        pygame.draw.line(screen, GRID_COLOR, (x * CELL_SIZE, 0), (x * CELL_SIZE, GRID_HEIGHT * CELL_SIZE), 1)
    for y in range(GRID_HEIGHT + 1):
        pygame.draw.line(screen, GRID_COLOR, (0, y * CELL_SIZE), (GRID_WIDTH * CELL_SIZE, y * CELL_SIZE), 1)

    # Food
    pygame.draw.rect(
        screen,
        FOOD_COLOR,
        (food[0] * CELL_SIZE + 3, food[1] * CELL_SIZE + 3, CELL_SIZE - 6, CELL_SIZE - 6),
        border_radius=6,
    )

    # Snake
    for i, (sx, sy) in enumerate(snake):
        color = SNAKE_HEAD_COLOR if i == 0 else SNAKE_COLOR
        pygame.draw.rect(
            screen,
            color,
            (sx * CELL_SIZE + 3, sy * CELL_SIZE + 3, CELL_SIZE - 6, CELL_SIZE - 6),
            border_radius=6,
        )


# --- Training ---
def train(render=True, episodes=EPISODES, q_save_path="snake_qtable.pickle", seed: int | None = None):
    if seed is not None:
        random.seed(seed)
        np.random.seed(seed)

    if render:
        pygame.init()
        font = pygame.font.SysFont("consolas", 20)
        screen_height = GRID_HEIGHT * CELL_SIZE + 32
        screen = pygame.display.set_mode((GRID_WIDTH * CELL_SIZE, screen_height))
        pygame.display.set_caption("Snake Q-learning (train)")
        clock = pygame.time.Clock()
    else:
        screen = None
        font = None
        clock = None

    Q: Dict[Tuple[int, ...], np.ndarray] = {}
    epsilon = EPSILON_START
    global_best = -float("inf")

    try:
        for ep in range(1, episodes + 1):
            snake = [(GRID_WIDTH // 2, GRID_HEIGHT // 2)]
            food = spawn_food(snake)
            total_reward = 0
            done = False
            step = 0
            paused = False

            while not done:
                if render:
                    exit_requested, paused = handle_events(paused)
                    if exit_requested:
                        raise KeyboardInterrupt
                    if paused:
                        clock.tick(15)
                        continue

                state = get_state(snake, food)
                if state not in Q:
                    Q[state] = np.zeros(len(ACTIONS))

                if random.random() < epsilon:
                    action = random.choice(ACTIONS)
                else:
                    action = int(np.argmax(Q[state]))

                dir_map = [(0, -1), (0, 1), (-1, 0), (1, 0)]
                direction = dir_map[action]
                new_head = (snake[0][0] + direction[0], snake[0][1] + direction[1])
                snake.insert(0, new_head)

                reward = LIVING_PENALTY
                if (
                    new_head[0] < 0
                    or new_head[0] >= GRID_WIDTH
                    or new_head[1] < 0
                    or new_head[1] >= GRID_HEIGHT
                    or new_head in snake[1:]
                ):
                    reward = -10
                    done = True
                elif new_head == food:
                    reward = 10
                    food = spawn_food(snake)
                else:
                    snake.pop()

                next_state = get_state(snake, food)
                if next_state not in Q:
                    Q[next_state] = np.zeros(len(ACTIONS))
                Q[state][action] = (1 - ALPHA) * Q[state][action] + ALPHA * (reward + GAMMA * np.max(Q[next_state]))

                total_reward += reward
                step += 1

                if step >= MAX_STEPS_PER_EPISODE:
                    done = True

                if render and ep % RENDER_EVERY == 0:
                    draw_board(screen, snake, food)
                    hud_y = GRID_HEIGHT * CELL_SIZE + 6
                    hud_text = f"Ep: {ep}  Step: {step}  Reward: {total_reward:+.2f}  ε: {epsilon:.3f}"
                    label = font.render(hud_text, True, TEXT_COLOR)
                    screen.blit(label, (6, hud_y))
                    pygame.display.flip()
                    clock.tick(TRAIN_FPS)

            epsilon = max(MIN_EPSILON, epsilon * EPSILON_DECAY)
            global_best = max(global_best, total_reward)

            if ep % 10 == 0:
                print(
                    f"Episode {ep:4d} | Reward: {total_reward:+7.2f} | Best: {global_best:+7.2f} | epsilon {epsilon:.3f}"
                )

    except KeyboardInterrupt:
        print("Exit requested, stopping training early…")
    finally:
        with open(q_save_path, "wb") as f:
            pickle.dump(Q, f)
        if render:
            pygame.quit()
        print(f"Training complete ✅  Q-table saved to {q_save_path}")


# --- Inference / viewer ---
def load_qtable(path: str = "snake_qtable.pickle"):
    try:
        with open(path, "rb") as f:
            return pickle.load(f)
    except FileNotFoundError:
        print("Q-table not found. Run training first.")
        sys.exit(1)


def draw_view(screen, font, snake, food, score, steps):
    draw_board(screen, snake, food)
    label = font.render(f"Score: {score}   Steps: {steps}   Space: pause/resume", True, TEXT_COLOR)
    screen.blit(label, (6, GRID_HEIGHT * CELL_SIZE + 4))
    pygame.display.flip()


def view(q_path="snake_qtable.pickle"):
    Q = load_qtable(q_path)

    pygame.init()
    font = pygame.font.SysFont("consolas", 22)
    height = GRID_HEIGHT * CELL_SIZE + 32
    screen = pygame.display.set_mode((GRID_WIDTH * CELL_SIZE, height))
    pygame.display.set_caption("Snake Q-table Viewer")
    clock = pygame.time.Clock()

    snake = [(GRID_WIDTH // 2, GRID_HEIGHT // 2)]
    food = spawn_food(snake)
    steps = 0
    score = 0
    paused = False

    running = True
    while running:
        exit_requested, paused = handle_events(paused)
        if exit_requested:
            break
        if paused:
            clock.tick(10)
            continue

        state = get_state(snake, food)
        action_values = Q.get(state)
        if action_values is None:
            action = random.choice(ACTIONS)
        else:
            action = int(np.argmax(action_values))

        dir_map = [(0, -1), (0, 1), (-1, 0), (1, 0)]
        direction = dir_map[action]
        new_head = (snake[0][0] + direction[0], snake[0][1] + direction[1])
        snake.insert(0, new_head)

        if (
            new_head[0] < 0
            or new_head[0] >= GRID_WIDTH
            or new_head[1] < 0
            or new_head[1] >= GRID_HEIGHT
            or new_head in snake[1:]
        ):
            running = False
        elif new_head == food:
            score += 1
            food = spawn_food(snake)
        else:
            snake.pop()

        steps += 1
        draw_view(screen, font, snake, food, score, steps)
        clock.tick(VIEW_FPS)

    pygame.quit()
    print(f"Session ended. Score: {score}, Steps: {steps}")


# --- CLI ---
def parse_args():
    parser = argparse.ArgumentParser(description="Snake AI: train or view a trained agent")
    parser.add_argument(
        "mode",
        nargs="?",
        choices=["train"],
        help="Mode: 'train' to train; default (no arg) views trained agent",
    )
    parser.add_argument("--episodes", type=int, default=EPISODES, help="Number of training episodes")
    parser.add_argument("--qtable", type=str, default="snake_qtable.pickle", help="Path to load/save Q-table")
    parser.add_argument("--no-render", action="store_true", help="Disable rendering during training (headless-safe)")
    parser.add_argument("--console", action="store_true", help="Console-only training (no Pygame loop)")
    parser.add_argument("--seed", type=int, default=None, help="Random seed for reproducible training")
    return parser.parse_args()


def main():
    args = parse_args()
    if args.mode == "train":
        render_flag = not (args.no_render or args.console)
        train(render=render_flag, episodes=args.episodes, q_save_path=args.qtable, seed=args.seed)
    else:
        view(q_path=args.qtable)


if __name__ == "__main__":
    main()
