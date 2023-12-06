import os
import subprocess

# NOTE: this part requires the scraper to be run first, which results are not included in the repository
for file in os.listdir(os.path.join("..", "scraper", "downloaded_files")):
    if file.endswith(".py"):
        subprocess.run(
            [
                "python",
                os.path.join("..", "parser", "parse_python3.py"),
                os.path.join("..", "scraper", "downloaded_files", file),
                os.path.join(
                    "..", "dataset", "single_asts", file.replace(".py", ".json")
                ),
            ]
        )

with open("dataset.json", "w") as outfile:
    for file in os.listdir("single_asts"):
        if file.endswith(".json"):
            with open(os.path.join("single_asts", file)) as infile:
                outfile.write(infile.read())
                outfile.write("\n")
