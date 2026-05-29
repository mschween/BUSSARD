import csv
import json
import time
from pathlib import Path

from run_flow import main as run_model
from utils import date_folders, set_seed


def main(config="config.json", log_dir="output/flow"):
    base_path = log_dir
    hidden_dim = 128
    epochs = 1000
    embed_type = "glove"  # "Qwen3-Embedding-8B"

    # Create log path with timestamp
    out_path = date_folders(base_dir=base_path)
    scenes = ["office", "dining_room"]
    # scenes = ["all"]

    # CSV file for tracking runtimes
    runtime_csv = Path(out_path, "runtimes.csv")

    # Load config
    with open(config, "r") as f:
        all_config = json.load(f)
    c = all_config["dataset"]

    seeds = c.get("seed", [0])

    # Open CSV file
    with open(runtime_csv, "w", newline="") as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(["seed", "scene", "runtime_seconds"])

        # Load embedding model once
        print("Loading embedding model once...")
        embed_start = time.time()
        if embed_type == "glove":
            from gensim.models import KeyedVectors

            word_embed = KeyedVectors.load_word2vec_format(
                "data/dolma_300_2024_1.2M.100_combined.txt",
                binary=False,
                no_header=True,
            )
        else:
            from llm_word_embedding import CachedLLMEmbedder

            embed_config = all_config["embeddings"][embed_type]
            word_embed = CachedLLMEmbedder(
                model_name=embed_config["model_name"],
                cache_path=f"concept_cache/{embed_type}/embeddings_cache.db",
            )
        embed_time = time.time() - embed_start
        print(f"Embedding loading time: {embed_time:.2f}s")

        for seed in seeds:
            print(f"Current seed {seed}")

            if len(seeds) > 1:
                seed_path = Path(out_path, str(seed))
            else:
                seed_path = out_path
            set_seed(seed)

            for scene in scenes:
                print(f"Running code with {scene} scene.")
                scene_path = Path(seed_path, scene)

                latent_dim = 512

                # Track runtime
                start_time = time.time()
                run_model(
                    config=config,
                    scene_name=scene,
                    seed=seed,
                    hidden_dim=hidden_dim,
                    epochs=epochs,
                    out_path=scene_path,
                    cross_check=False,
                    flow_input_dim=latent_dim,
                    mutation_rate=None,
                    word_embed=word_embed,
                    embed_type=embed_type,
                    dataset_source="sard",
                )
                runtime = time.time() - start_time

                # Add embed loading time to first scene only
                if seed == seeds[0] and scene == scenes[0]:
                    runtime += embed_time
                # Write to CSV
                writer.writerow([seed, scene, runtime])
                csvfile.flush()

                print(f"Runtime: {runtime:.2f} seconds")


if __name__ == "__main__":
    main()
