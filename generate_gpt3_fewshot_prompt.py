"""Helper functions to convert json scene to prompt."""

import os
import json
from pathlib import Path
from datetime import datetime

import yaml
from absl import app
from absl import flags

from helper_data import json_scene_to_prompt

flags.DEFINE_string(
    "train_file_path",
    None,
    help="Path to the json training data.",
)

flags.DEFINE_string(
    "config_file_path",
    "./configs/gpt3_config.yaml",
    "Path to the GPT-3 config file.",
)

FLAGS = flags.FLAGS


def main(argv):
    if len(argv) > 1:
        raise app.UsageError("Too many command-line arguments.")

    if FLAGS.train_file_path is None:
        raise ValueError("Please provide a path to the training data.")

    datetime_obj = datetime.now()
    date_time_stamp = datetime_obj.strftime("%Y_%m_%d_%H_%M_%S")

    # Load config.
    with open(FLAGS.config_file_path, "r") as fconfig:
        config = yaml.safe_load(fconfig)

    with open(FLAGS.train_file_path, "r") as fh:
        data = json.load(fh)

        prompt = ""
        prompt += json_scene_to_prompt(
            data[config["FEWSHOT_PROMPT"]["key_class"]], example_id=1, is_example=True
        )
        prompt += "\n"
        prompt += json_scene_to_prompt(
            data[config["FEWSHOT_PROMPT"]["key_utility"]], example_id=2, is_example=True
        )
        prompt += "\n"
        prompt += json_scene_to_prompt(
            data[config["FEWSHOT_PROMPT"]["key_affordance"]],
            example_id=3,
            is_example=True,
        )
        prompt += "\n"
        prompt += json_scene_to_prompt(
            data[config["FEWSHOT_PROMPT"]["key_ooe"]], example_id=4, is_example=True
        )

    prompt_destination_folder = config["FEWSHOT_PROMPT"]["prompt_destination_folder"]
    Path(prompt_destination_folder).mkdir(parents=True, exist_ok=True)

    with open(
        os.path.join(
            prompt_destination_folder, f"gpt_few_shot_prompt_{date_time_stamp}.txt"
        ),
        "w",
    ) as fw:
        fw.write(prompt)


if __name__ == "__main__":
    app.run(main)
