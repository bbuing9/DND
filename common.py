import argparse

DATA_PATH = './dataset'
CKPT_PATH = './checkpoint'

def parse_args(mode):
    assert mode in ['train', 'eval']

    parser = argparse.ArgumentParser()

    parser.add_argument("--dataset", help='dataset (news|review|imdb|etc.)',
                        required=True, type=str)
    parser.add_argument("--data_ratio", help='imbalance ratio, i.e., |min class| / |max class|',
                        default=1.0, type=float)
    parser.add_argument("--backbone", help='backbone network',
                        choices=['bert', 'roberta', 'roberta_large', 'albert'],
                        default='bert', type=str)
    parser.add_argument("--seed", help='random seed',
                        default=0, type=int)

    parser = _parse_args_train(parser)

    return parser.parse_args()


def _parse_args_train(parser):
    # ========== Training ========== #
    parser.add_argument("--train_type", help='train type (base|aug|mixup|lad2)',
                        default='base', type=str)
    parser.add_argument("--epochs", help='training epochs',
                        default=15, type=int)
    parser.add_argument("--batch_size", help='training bacth size',
                        default=8, type=int)
    parser.add_argument("--model_lr", help='learning rate for model update',
                        default=1e-5, type=float)
    parser.add_argument("--save_ckpt", help='save the best model checkpoint',
                        action='store_true')
    parser.add_argument("--pre_ckpt", help='path for the pre-trained model',
                        default=None, type=str)
    parser.add_argument("--pre_policy", help='path for the pre-trained policy',
                        default=None, type=str)

    # ========== Data Augmentations ========== #
    parser.add_argument("--mixup_alpha", help='hyper-parameter of beta distribution for Mixup augmentation',
                        default=1.0, type=float)
    parser.add_argument("--cutoff", help='length of cutoff tokens',
                        default=0.30, type=float)
    parser.add_argument("--eps", help='random noise size for r3f',
                        default=1e-5, type=float)
    parser.add_argument("--step_size", help='step size for adversarial example',
                        default=0.1, type=float)
    parser.add_argument("--policy", help='using random policy for augmentation',
                        action='store_true')

    parser.add_argument("--lambda_kl", help='weight for consistency loss',
                        default=0.0, type=float)
    
    # ========== DND ========== #
    parser.add_argument("--reweight", help='re-weighting the augmented samples',
                        action='store_true')

    parser.add_argument("--policy_lr", help='learning rate for policy update',
                        default=1e-3, type=float)
    parser.add_argument("--policy_temp", help='temperature for policy update',
                        default=0.05, type=float)
    parser.add_argument("--policy_update", help='update frequency of policy network',
                        default=1, type=int)

    parser.add_argument("--lambda_cls", help='weight for classification loss',
                        default=1.0, type=float)
    parser.add_argument("--lambda_aug", help='weight for classification loss for augmented samples',
                        default=1.0, type=float)
    parser.add_argument("--lambda_sim", help='weight for similarity loss for updating policy',
                        default=1.0, type=float)
    parser.add_argument("--lambda_recon", help='weight for masked reconstruction loss',
                        default=0.0, type=float)

    return parser

