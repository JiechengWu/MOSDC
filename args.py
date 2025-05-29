import argparse


def parameter_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--cuda', action='store_true', default=True, help='enables CUDA training')
    parser.add_argument("--device", type=str, default='0', help="Device: cuda:num or cpu")
    parser.add_argument("--seed", type=int, default=1024, help="Random seed  42")
    parser.add_argument("--fix_seed", action='store_true', default=True, help="xx")
    parser.add_argument("--n_repeated", type=int, default=5, help="Repeated times")

    parser.add_argument("--dataset", type=str, default="KIPAN", help="Dataset name")
    parser.add_argument("--lr", type=float, default=0.001, help="Learning rate")
    parser.add_argument("--weight_decay", type=float, default=5e-3, help="Weight decay  5e-3")
    parser.add_argument('--dropout', type=float, default=0.3, help='Dropout rate (1 - keep probability).')
    parser.add_argument("--num_epoch", type=int, default=250, help="Training epochs")
    parser.add_argument("--lambda1", type=float, default=2, help="Value of Lambda")
    parser.add_argument("--lambda2", type=float, default=50, help="Value of Lambda")
    parser.add_argument('--k', type=int, default=5, help='Number of knn neighbors.')

    parser.add_argument("--d1", type=int, default=128, help="hidden dimensions")
    parser.add_argument("--d2", type=int, default=64, help="hidden dimensions")
    parser.add_argument("--d3", type=int, default=32, help="hidden dimensions")
    parser.add_argument("--d4", type=int, default=16, help="hidden dimensions")
    parser.add_argument("--d5", type=int, default=32, help="hidden dimensions")
    parser.add_argument("--d6", type=int, default=16, help="hidden dimensions")
    parser.add_argument('--label_ratio', type=float, default=0.05, help='Ratio of Labeled Samples for Validation (default: 0.10)')
    parser.add_argument('--top_ratio', type=float, default=0.01, help='Ratio of Labeled Samples for Validation (default: 0.10)')
    parser.add_argument('--select_each_ratio', type=float, default=0.05, help='Ratio of Labeled Samples for Validation (default: 0.10)')

    args = parser.parse_args()

    return args
