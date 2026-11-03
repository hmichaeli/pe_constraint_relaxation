import math
from typing import Iterable, Optional
from af_ops import UpSampleAF
import torch
import torch.nn.functional as F
from timm.data import Mixup
from timm.utils import accuracy, ModelEma
from tqdm import tqdm
import utils
import itertools
import torch.distributed as dist


@torch.no_grad()
def evaluate_adversarial_crop_shift(data_loader, model, device, inp_size, max_shift, use_amp=False):
    """
    Evaluate model robustness against adversarial crop shifts.
    This function assesses how consistently a model predicts the correct class when
    the input image is shifted by various offsets. It first evaluates on center crops
    as a baseline, then tests increasingly larger shifts to determine robustness.
    A sample is considered robust at shift level N if it's correctly classified for
    all possible shifts up to N pixels in any direction.
    :param data_loader: DataLoader providing batches of expanded images (larger than inp_size to allow for shifts)
    :param model: Neural network model to evaluate
    :param device: Device to run computation on (e.g., 'cuda' or 'cpu')
    :param inp_size: Input size the model expects for images
    :param max_shift: Maximum pixel shift to test in any direction, relative to center crop
    :param use_amp: Whether to use automatic mixed precision for inference
    :return: Dictionary with accuracy and robustness metrics for different shift magnitudes
    """

    criterion = torch.nn.CrossEntropyLoss()
    num_batches = len(data_loader)

    metric_logger = utils.MetricLogger(delimiter="  ")
    header = 'Test_consistency_metrics:'

    def forwrad(images, target):
        images = images.to(device, non_blocking=True)
        target = target.to(device, non_blocking=True)

        # compute output
        if use_amp:
            with torch.cuda.amp.autocast():
                output = model(images)
                loss = criterion(output, target)
        else:
            output = model(images).cpu()

        return output


    # switch to evaluation mode
    model.eval()
    for batch_idx, batch in enumerate(metric_logger.log_every(data_loader, 2, header)):
        expanded_images = batch[0].to(device) # size inp + 2 * max_shift
        target = batch[-1].to(device)
        batch_size = expanded_images.shape[0]

        assert expanded_images.shape[2] == expanded_images.shape[3], "images should be square"
        assert expanded_images.shape[2] >= inp_size + (2 * max_shift), f"expanded_images.shape[2] = {expanded_images.shape[2]} < {inp_size} + 2 * {max_shift}"

        # Baseline eval - center crop
        center_crop_offset = (expanded_images.shape[2] - inp_size) // 2
        images = expanded_images[:, :, center_crop_offset: center_crop_offset + inp_size, center_crop_offset: center_crop_offset + inp_size]

        output_labels = forwrad(images, target)
        target = target.to(output_labels.device)
        acc1, acc5 = accuracy(output_labels, target, topk=(1, 5))

        metric_logger.meters['acc1'].update(acc1.item(), n=batch_size)
        metric_logger.meters['acc5'].update(acc5.item(), n=batch_size)

        correct = torch.argmax(output_labels, dim=1) == target

        # initialize robust_flags with baseline result.
        robust_flags = correct

        for shift in range(1, max_shift + 1):
            # get all relevant shifts offsets for the current shift value (avoid duplicate computations)
            shift_list = get_shifts(shift, max_shift, max_shift)

            # update robust flags with new shifts results
            for shift_x, shift_y in shift_list:
                # crop image at inp_size, starting from shift_x, shift_y
                shifted_images = expanded_images[:, :, shift_x: inp_size+shift_x, shift_y:inp_size+shift_y]
                shifted_output_labels = forwrad(shifted_images, target)
                correct_shift = torch.argmax(shifted_output_labels, dim=1) == target

                robust_flags = robust_flags & correct_shift

            shift_robust_score = 100 * torch.sum(robust_flags) / len(robust_flags)
            metric_logger.meters[f'shift_robust_score_{shift}'].update(shift_robust_score.item(), n=batch_size)
        
        # synchronize every batch
        dist.barrier()

    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    print('* Acc@1 {top1.global_avg:.3f} Acc@5 {top5.global_avg:.3f}'
          .format(top1=metric_logger.acc1, top5=metric_logger.acc5))

    for shift in range(1, max_shift + 1):
        print(f'* adversarail_crop_shift_robust_score max_shift {shift} : {metric_logger.meters[f"shift_robust_score_{shift}"].global_avg:.3f} \n')

    res_dict = {f"adversarial_crop_shift_{k}": meter.global_avg for k, meter in metric_logger.meters.items()}
    return res_dict

@torch.no_grad()
def evaluate_adversarial_cyclic_shift(data_loader, model, device, max_shift: int, upsample: int, use_amp=False):
    """
    Evaluate model robustness against adversarial cyclic shifts.
    This function assesses how consistently a model predicts the correct class when
    the input image is shifted by various offsets. It first evaluates on center crops
    as a baseline, then tests increasingly larger shifts to determine robustness.
    A sample is considered robust at shift level N if it's correctly classified for
    all possible shifts up to N pixels in any direction.
    :param data_loader: DataLoader providing batches of expanded images (larger than inp_size to allow for shifts)
    :param model: Neural network model to evaluate
    :param device: Device to run computation on (e.g., 'cuda' or 'cpu')
    :param max_shift: Maximum pixel shift to test in any direction, relative to center crop
    :param upsample: Upsample factor for the images. This is used to evaluate fractional shifts.
    e.g. if upsample = 2 and max_shift = 2, the function will evaluate shifts of [0.5, 1] pixels.
    :param use_amp: Whether to use automatic mixed precision for inference
    :return: Dictionary with accuracy and robustness"
    """

    # switch to evaluation mode
    model.eval()

    metric_logger = utils.MetricLogger(delimiter="  ")
    header = 'Adversarial Cyclic Shift:'

    def forwrad(images, target):
        images = images.to(device, non_blocking=True)
        target = target.to(device, non_blocking=True)

        # compute output
        if use_amp:
            with torch.cuda.amp.autocast():
                output = model(images)
        else:
            output = model(images).cpu()

        return output


    up_layer = None
    if upsample > 1:
        up_layer = UpSampleAF(up=upsample).to(device)


    for batch_idx, batch in enumerate(metric_logger.log_every(data_loader, 2, header)):

        # Baseline eval - center crop
        images = batch[0].to(device)
        target = batch[1]
        batch_size = images.shape[0]

        output_labels = forwrad(images, target)
        target = target.to(output_labels.device)
        acc1, acc5 = accuracy(output_labels, target, topk=(1, 5))

        metric_logger.meters['acc1'].update(acc1.item(), n=batch_size)
        metric_logger.meters['acc5'].update(acc5.item(), n=batch_size)

        correct = torch.argmax(output_labels, dim=1) == target

        # initialize robust_flags with baseline result.
        robust_flags = correct

        # upsample if needed
        if upsample > 1:
            images = up_layer(images)

        for shift in range(1, max_shift + 1):
            # get all relevant shifts offsets for the current shift value (avoid duplicate computations)
            shift_list = get_shifts(shift, max_shift, max_shift)

            # update robust flags with new shifts results
            for shift_x, shift_y in shift_list:

                # shift image batch
                shifted_images = torch.roll(images, shifts=(-shift_x, -shift_y), dims=(2, 3))
                if upsample > 1:
                    # downsample shifted image to original size
                    shifted_images = shifted_images[:,:,::upsample, ::upsample]

                shifted_output_labels = forwrad(shifted_images, target)
                correct_shift = torch.argmax(shifted_output_labels, dim=1) == target

                robust_flags = robust_flags & correct_shift

            shift_robust_score = 100 * torch.sum(robust_flags) / len(robust_flags)
            metric_logger.meters[f'shift_robust_score_{shift}'].update(shift_robust_score.item(), n=batch_size)

    dist.barrier()  # synchronize all processes
    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    print('* Acc@1 {top1.global_avg:.3f} Acc@5 {top5.global_avg:.3f}'
          .format(top1=metric_logger.acc1, top5=metric_logger.acc5))

    for shift in range(1, max_shift + 1):
        print(f'* adversarial_cyclic_shift_robust_score max_shift {shift} : {metric_logger.meters[f"shift_robust_score_{shift}"].global_avg:.3f} \n')

    res_dict = {f"adversarial_cyclic_shift_{k}_{upsample}": meter.global_avg for k, meter in metric_logger.meters.items()}
    return res_dict



def get_shifts(s, x_c, y_c):
    '''
    input: a center offset (x_c,y_c) (integer)
    input: shift s (integer)
    output: a list of all shift tuples indicating offsetes (x,y) with L1 distance at most s from the center in at least one direction,
    but not less in both directions
    '''
    shifts = set()
    # left column:
    for row in range(-s, s+1):
        shifts.add((x_c - s, y_c + row))
    # right column:
    for row in range(-s, s+1):
        shifts.add((x_c + s, y_c + row))
    # top row:
    for col in range(-s+1, s):
        shifts.add((x_c + col, y_c - s))
    # bottom row:
    for col in range(-s+1, s):
        shifts.add((x_c + col, y_c + s))
    return list(shifts)



def _build_frac_grid(dx: float, dy: float, H: int, W: int, device,
                     align_corners: bool = True):
    """Return a normalised grid that translates the inner (H‑2)×(W‑2) crop
    by (dx, dy) pixels (positive → right/down)."""
    iy = torch.arange(1, H - 1, device=device) + dy
    ix = torch.arange(1, W - 1, device=device) + dx

    if align_corners:
        gy = -1. + 2. * iy / (H - 1)
        gx = -1. + 2. * ix / (W - 1)
    else:  # half‑pixel reference frame
        gy = -1. + 2. * (iy + 0.5) / H
        gx = -1. + 2. * (ix + 0.5) / W

    gy, gx = torch.meshgrid(gy, gx, indexing='ij')
    return torch.stack((gx, gy), dim=-1).unsqueeze(0)  # (1, H‑2, W‑2, 2)


@torch.no_grad()
def evaluate_adversarial_bilinear_fractional_shift(
        data_loader,
        model,
        device,
        inp_size: int,
        max_shift: int,          # number of fractional steps (m)
        use_amp: bool = False,
        align_corners: bool = True):
    """
    Robustness to *fractional* translations in (‑1,1) px with step 1/max_shift.
    The input batch is expected to be (B,C,inp_size+2,inp_size+2) – i.e. a 1‑px
    border so that any ±1‑pixel shift keeps the valid region in‑bounds.
    """

    fractions = [k / max_shift for k in range(-max_shift, max_shift + 1)]
    # Pre‑build grids for every (dx,dy) needed
    H = W = inp_size + 2
    grid_bank = {
        (dx, dy): _build_frac_grid(dx, dy, H, W, device, align_corners)
        for dx in fractions for dy in fractions
    }

    def forward_pass(img):
        if use_amp:
            with torch.cuda.amp.autocast():
                return model(img)
        return model(img)

    criterion = torch.nn.CrossEntropyLoss()
    metric_logger = utils.MetricLogger(delimiter="  ")
    header = 'Adversarial Fractional Shift:'

    # ------------------------------------------------------------------
    model.eval()
    for images_full, target in metric_logger.log_every(data_loader, 2, header):
        images_full = images_full.to(device)          # (B,C,H,W) with border
        target = target.to(device)
        B = images_full.size(0)

        # Center crop (baseline)
        images_c = images_full[:, :, 1:-1, 1:-1]
        logits_c = forward_pass(images_c)
        acc1, acc5 = accuracy(logits_c, target, topk=(1, 5))
        metric_logger.meters['acc1'].update(acc1.item(), n=B)
        metric_logger.meters['acc5'].update(acc5.item(), n=B)

        robust_flags = (logits_c.argmax(1) == target)

        # Fractional shifts
        for s in range(1, max_shift + 1):
            frac = s / max_shift
            # collect shifts lying exactly on the "ring" max(|dx|,|dy|) == frac
            shifts_lvl = [
                (dx, dy) for dx in fractions for dy in fractions
                if max(abs(dx), abs(dy)) == frac
            ]
            for dx, dy in shifts_lvl:
                grid = grid_bank[(dx, dy)].to(device)
                shifted = F.grid_sample(
                    images_full, grid.repeat(B, 1, 1, 1),
                    mode='bilinear', align_corners=align_corners
                )
                preds = forward_pass(shifted).argmax(1)
                robust_flags &= (preds == target)

            score = 100. * robust_flags.float().mean()
            metric_logger.meters[f'shift_frac_robust_{frac:.3f}'].update(
                score.item(), n=B)
        
        dist.barrier()  # synchronize all processes

    metric_logger.synchronize_between_processes()
    print('* Acc@1 {top1.global_avg:.3f}  Acc@5 {top5.global_avg:.3f}'
          .format(top1=metric_logger.acc1, top5=metric_logger.acc5))
    for s in range(1, max_shift + 1):
        frac = s / max_shift
        print(f'* adversarial_fractional_shift_robust_score {frac:.3f}px : '
              f'{metric_logger.meters[f"shift_frac_robust_{frac:.3f}"].global_avg:.3f}')

    # Return dict keyed by meter names (same style as existing evaluators)
    return {
        f'adversarial_fractional_shift_{k}': m.global_avg
        for k, m in metric_logger.meters.items()
    }



if __name__ == '__main__':
    print(get_shifts(1, 2, 2))
    print(get_shifts(2, 2, 2))
