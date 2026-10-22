"""Print the environment record that every run saves as env.json."""

import json

from robust_visual_perception.environment import collect_environment

if __name__ == "__main__":
    print(json.dumps(collect_environment(), indent=2))
