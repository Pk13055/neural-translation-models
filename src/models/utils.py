"""
:brief: Utility functions used across various models

"""

def collect_args() -> argparse.Namespace:
    """Passing command line arguments"""
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', '-d', help="dataset path", type=str, required=True)
    parser.add_argument('--batch', '-b', help="Batch size", type=int, default=64)
    parser.add_argument('--epoch', '-e', help="Number of epochs", type=int, default=128)
    parser.add_argument('--latent-dim', '-l', help="Latent dimension encoding size", type=int, default=256)
    args = parser.parse_args()
    return args