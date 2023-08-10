import argparse


def get_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    ...
    return parser.parse_args()


def main(args: argparse.Namespace):
    if args.task == "train":
        pass
    elif args.task == "test":
        pass
    else:
        raise ValueError("Illegal task option (must be either `train` or `test`).")


if __name__ == "__main__":
    main(get_args())
