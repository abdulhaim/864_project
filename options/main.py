import argparse

from runner_options import Runner

if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Option Critic PyTorch")

    parser.add_argument('--env', default='beavekitchen', help='ROM to run')
    parser.add_argument('--optimal-eps', type=float, default=0.05, help='Epsilon when playing optimally')
    parser.add_argument('--learning-rate',type=float, default=.0005, help='Learning rate')
    parser.add_argument('--rms-decay', type=float, default=.95, help='Decay rate for rms_prop')
    parser.add_argument('--rms-epsilon', type=float, default=.01, help='Denominator epsilson for rms_prop')
    parser.add_argument('--gamma', type=float, default=.99, help='Discount rate')
    parser.add_argument('--epsilon-start',  type=float, default=1.0, help=('Starting value for epsilon.'))
    parser.add_argument('--epsilon-min', type=float, default=.1, help='Minimum epsilon.')
    parser.add_argument('--epsilon-decay', type=float, default=0.999, help=('Number of steps to minimum epsilon.'))
    parser.add_argument('--seed', type=int, default=1, help='Random seed for numpy, torch, random.')
    parser.add_argument('--logdir', type=str, default='runs', help='Directory for logging statistics')
    parser.add_argument('--modeldir', type=str, default='models', help='Directory for saving model')
    parser.add_argument('--level', type=str, default='train_level.py', help="The level for the domain.")
    parser.add_argument('--render-train', action="store_true", help="Whether or not to render the domain during training")
    parser.add_argument('--render-eval', action="store_true", help="Whether or not to render the domain during evaluation")
    parser.add_argument('--clip-value', type=float, default=0.5, help="clip value")
    parser.add_argument('--n-episodes', type=int, default=int(2000), help='number of maximum epsisodes to take.') # bout 4 million
    parser.add_argument('--eval-checkpoint', type=str, default="", help="Checkpoint for model evaluation")
    parser.add_argument('--n-eval-episodes', type=int, default=100, help="Number of evaluation episodes")
    parser.add_argument('--num-options', type=int, default=5, help=('Number of options to create.'))
    parser.add_argument('--max_steps_ep', type=int, default=50, help='number of maximum steps per episode.')
    parser.add_argument('--cuda', type=bool, default=True, help='Enable CUDA training (recommended if possible).')
    parser.add_argument('--eval-freq', type=int, default=50, help="Number of training episodes before evaluation")
    parser.add_argument('--checkpoint-freq', type=int, default=50, help="Number of episodes before checkpointing model")
    parser.add_argument('--batch-size', type=int, default=64, help='Batch size.')
    parser.add_argument('--entropy-reg', type=float, default=0.01, help=('Regularization to increase policy entropy.'))
    parser.add_argument('--termination-reg', type=float, default=0.01, help=('Regularization to decrease termination prob.'))
    parser.add_argument('--update-frequency', type=int, default=4, help=('Number of actions before each SGD update.'))
    parser.add_argument('--freeze-interval', type=int, default=200, help=('Interval between target freezes.'))
    parser.add_argument('--max-history', type=int, default=10000, help=('Maximum number of steps stored in replay'))
    parser.add_argument('--temp', type=float, default=1, help='Action distribution softmax tempurature param.')
    parser.add_argument('--exp', type=str, default=None, help='optional experiment name')
    parser.add_argument('--delib', type=float,default=0.01,metavar='opt',help='deliberation cost (default: 0.01)')
    parser.add_argument('--warmup-size', type=float, default=1e2, help="Size of memory before updates")
    parser.add_argument('--n-eval-samples', type=int, default=5, help="Number of evaluation samples after N training episodes")

    args = parser.parse_args()
    run_obj = Runner(args)
    run_obj.run()