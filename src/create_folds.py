from hydra import compose, initialize

if __name__ == "__main__":
    with initialize(config_path="conf", job_name="fold_creation"):
        cfg = compose(config_name="config")