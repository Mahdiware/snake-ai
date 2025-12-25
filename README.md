# Snake Q-learning (GUI + Console)

A minimal Snake implementation with Q-learning. One script does it all:
- Train the agent (GUI or console).
- View/play back the trained policy in a simple GUI viewer.

## Requirements
- Python 3.10+
- `pygame`
- `numpy`

Install locally:
```bash
python3 -m venv .venv
source .venv/bin/activate
pip install pygame numpy
```

## Script
`snake_ai.py` — unified entry point for training and viewing.

### Modes
- **Default (no args):** GUI viewer using the saved Q-table.
- **train:** Run training (GUI by default; console/headless optional).

### Command-line arguments
```
usage: snake_ai.py [mode] [--episodes N] [--qtable PATH] [--no-render] [--console] [--seed N]

positional arguments:
  mode                 'train' to train; omit to view the trained agent

optional arguments:
  --episodes N         Number of training episodes (default: 1500)
  --qtable PATH        Path to load/save Q-table (default: snake_qtable.pickle)
  --no-render          Disable rendering during training (headless-safe)
  --console            Console-only training (no Pygame loop; implies no render)
  --seed N             Random seed for reproducible training
```

### Examples
Train headless/fast and save table:
```bash
python snake_ai.py train --no-render --episodes 5000 --seed 123
```

Train with GUI:
```bash
python snake_ai.py train --episodes 2000
```

Console-only training:
```bash
python snake_ai.py train --console --episodes 2000 --seed 1
```

View the trained agent:
```bash
python snake_ai.py            # uses snake_qtable.pickle by default
python snake_ai.py --qtable my_q.pickle
```

## Controls (GUI modes)
- Space: pause/resume
- ESC or window close: stop

## Notes
- Grid: 10x10, cell size 40px (tweak constants at top of `snake_ai.py`).
- Rewards: +10 food, -10 death, -0.05 living penalty, episode step cap 400.
- Training prints progress every 10 episodes; Q-table saved on exit (even on Ctrl+C/ESC).
- Headless runs still import pygame (warning about pkg_resources may appear). This is cosmetic.

## Files
- `snake_ai.py` — unified trainer/viewer
- `snake_qtable.pickle` — generated after training

## Contributing
Pull requests and issues are very welcome—feel free to open one with ideas, fixes, or improvements.
