from util import torchutils


def get_optimizer(args, model, max_step=None):
    param_groups = model.trainable_parameters()
    optimizer = torchutils.PolyOptimizer([
        {'params': param_groups[0], 'lr': args.lr, 'weight_decay': args.wt_dec},
        {'params': param_groups[1], 'lr': 10 * args.lr, 'weight_decay': args.wt_dec},
    ], lr=args.lr, weight_decay=args.lr, max_step=max_step)
    return optimizer
