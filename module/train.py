import os

import torch
from torch.nn import functional as F

from util import pyutils

from settings import NUM_CLASSES


def train_cls(train_loader, model, optimizer, max_step, args):
    avg_meter = pyutils.AverageMeter('loss')
    timer = pyutils.Timer("Session started: ")
    loader_iter = iter(train_loader)
    for iteration in range(args.start_iteration, args.max_iters):
        try:
            img_id, img, label = next(loader_iter)
        except:
            loader_iter = iter(train_loader)
            img_id, img, label = next(loader_iter)
        img = img.cuda(non_blocking=True)
        label = label.cuda(non_blocking=True)
        pred, cam = model(img)

        # Classification loss
        loss = F.multilabel_soft_margin_loss(pred, label)
        avg_meter.add({'loss': loss.item()})

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if (optimizer.global_step-1) % 50 == 0:
            timer.update_progress(optimizer.global_step / max_step)

            print('Iter:%5d/%5d' % (iteration, args.max_iters),
                  'Loss:%.4f' % (avg_meter.pop('loss')),
                  'Rem:%s' % (timer.get_est_remain()),
                  'lr: %.4f' % (optimizer.param_groups[0]['lr']), flush=True)

            state = {
                'iteration': iteration,
                'state_dict': model.state_dict(),
                'optimizer' : optimizer.state_dict(),
                'lr' : optimizer.param_groups[0]['lr']
                }
            torch.save(state, os.path.join(args.log_folder, 'checkpoint_cls.pth'))

        timer.reset_stage()

    torch.save(model.state_dict(), os.path.join(args.log_folder, f'{NUM_CLASSES}_final_cls.pth'))
