import argparse

arg_lists = []
parser = argparse.ArgumentParser(description="RAM")

def get_config():
    parser = argparse.ArgumentParser(description="RAM")

    # Add arguments
    parser.add_argument('--use_gpu', type=bool, default=True, help='Use GPU if available')
    parser.add_argument('--patch_size', type=int, default=8, help='Size of the image patch')
    parser.add_argument('--glimpse_scale', type=int, default=2, help='Scale of the glimpse')
    parser.add_argument('--num_patches', type=int, default=3, help='Number of patches')
    parser.add_argument('--loc_hidden', type=int, default=128, help='Location hidden layer size')
    parser.add_argument('--glimpse_hidden', type=int, default=128, help='Glimpse hidden layer size')
    parser.add_argument('--num_glimpses', type=int, default=6, help='Number of glimpses')
    parser.add_argument('--hidden_size', type=int, default=256, help='Hidden layer size')
    parser.add_argument('--std', type=float, default=0.17, help='Standard deviation for location sampling')
    parser.add_argument('--M', type=int, default=10, help='Number of Monte Carlo samples')
    parser.add_argument('--is_train', type=bool, default=True, help='Training or testing phase')
    parser.add_argument('--epochs', type=int, default=100, help='Number of training epochs')
    parser.add_argument('--momentum', type=float, default=0.5, help='Optimizer momentum')
    parser.add_argument('--init_lr', type=float, default=0.001, help='Initial learning rate')
    parser.add_argument('--best', type=bool, default=True, help='Load the best checkpoint')
    parser.add_argument('--ckpt_dir', type=str, default='./checkpoints', help='Checkpoint directory')
    parser.add_argument('--logs_dir', type=str, default='./logs', help='Logs directory')
    parser.add_argument('--lr_patience', type=int, default=10, help='Learning rate scheduler patience')
    parser.add_argument('--train_patience', type=int, default=20, help='Training patience for early stopping')
    parser.add_argument('--use_tensorboard', type=bool, default=True, help='Use TensorBoard for logging')
    parser.add_argument('--resume', type=bool, default=False, help='Resume training from checkpoint')
    parser.add_argument('--print_freq', type=int, default=10, help='Print frequency')
    parser.add_argument('--plot_freq', type=int, default=10, help='Plot frequency')

    config, unparsed = parser.parse_known_args()
    return config, unparsed