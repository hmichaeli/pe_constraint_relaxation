###############################################################################
###############################################################################

from af_ops import subpixel_shift
import torch
import numpy as np
import utils
from timm.utils import accuracy


@torch.no_grad()
def evaluate_consistency_metrics(data_loader, model, device, use_amp=False, max_shift=32, shift=None, print_freq: int = 500):
    '''
    Evaluate shift-consistency metrics on the model (integer and fractional (1/2) shifts)
    '''
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
            output = model(images)
            loss = criterion(output, target)
        # images = images.cpu()

        return output, loss

    # switch to evaluation mode
    model.eval()
    for batch in metric_logger.log_every(data_loader, print_freq, header):
        images = batch[0]
        target = batch[-1]

        if shift is None:
            shift = np.random.randint(-max_shift, max_shift, 2)

        shifted_images = torch.roll(images, shifts=(shift[0], shift[1]), dims=(2, 3))
        # sub_pix_shifted_images = subpixel_shift(images)  # shift 0.5 pixel
        sub_pix_shifted_images = subpixel_shift(images.to(device), up=2,
                                                shift_x=int(2*(shift[0]+0.5)),
                                                shift_y=int(2 * (shift[1] + 0.5)))  # shift + 0.5 pixel

        # regular images
        output_labels, loss = forwrad(images, target)
        target = target.to(output_labels.device)
        acc1, acc5 = accuracy(output_labels, target, topk=(1, 5))

        batch_size = images.shape[0]
        metric_logger.update(loss=loss.item())
        metric_logger.meters['acc1'].update(acc1.item(), n=batch_size)
        metric_logger.meters['acc5'].update(acc5.item(), n=batch_size)


        shifted_output_labels, _ = forwrad(shifted_images, target)

        consistent_labels = (torch.argmax(output_labels, dim=1) == torch.argmax(shifted_output_labels, dim=1)).tolist()
        consistency = np.sum(consistent_labels) / len(consistent_labels)

        pred1 = torch.gather(torch.softmax(output_labels, dim=1), 1,
                             torch.argmax(output_labels, 1, keepdim=True)).detach().cpu().numpy()
        pred2 = torch.gather(torch.softmax(shifted_output_labels, dim=1), 1,
                             torch.argmax(output_labels, 1, keepdim=True)).detach().cpu().numpy()
        mac = np.sum(np.abs(pred1 - pred2)) / pred1.shape[0]  # Mean absolute change

        tmp_accurate_labels = (torch.argmax(shifted_output_labels, dim=1) == target).tolist()
        tmp_accuracy_top1 = 100 * np.sum(tmp_accurate_labels) / len(tmp_accurate_labels)


        metric_logger.meters['shift_consistency'].update(consistency.item(), n=batch_size)
        metric_logger.meters['shift_mac'].update(mac.item(), n=batch_size)
        metric_logger.meters['shift_accuracy'].update(tmp_accuracy_top1.item(), n=batch_size)

        sub_pix_shifted_output_labels, _ = forwrad(sub_pix_shifted_images, target)

        consistent_labels = (
                    torch.argmax(output_labels, dim=1) == torch.argmax(sub_pix_shifted_output_labels, dim=1)).tolist()

        consistency = np.sum(consistent_labels) / len(consistent_labels)

        pred1 = torch.gather(torch.softmax(output_labels, dim=1), 1,
                             torch.argmax(output_labels, 1, keepdim=True)).detach().cpu().numpy()
        pred2 = torch.gather(torch.softmax(sub_pix_shifted_output_labels, dim=1), 1,
                             torch.argmax(output_labels, 1, keepdim=True)).detach().cpu().numpy()
        mac = np.sum(np.abs(pred1 - pred2)) / pred1.shape[0]  # Mean absolute change

        tmp_accurate_labels = (torch.argmax(sub_pix_shifted_output_labels, dim=1) == target).tolist()
        tmp_accuracy_top1 = 100 * np.sum(tmp_accurate_labels) / len(tmp_accurate_labels)


        metric_logger.meters['sub_pix_shift_consistency'].update(consistency.item(), n=batch_size)
        metric_logger.meters['sub_pix_shift_mac'].update(mac.item(), n=batch_size)
        metric_logger.meters['sub_pix_shift_accuracy'].update(tmp_accuracy_top1.item(), n=batch_size)


    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    print('* Acc@1 {top1.global_avg:.3f} Acc@5 {top5.global_avg:.3f} loss {losses.global_avg:.3f}'
          .format(top1=metric_logger.acc1, top5=metric_logger.acc5, losses=metric_logger.loss))
    res_dict = {k: meter.global_avg for k, meter in metric_logger.meters.items()}
    res_dict['shift_accuracy_degredation'] = 100 * (res_dict['acc1'] - res_dict['shift_accuracy']) / res_dict['acc1']
    res_dict['subpix_shift_accuracy_degredation'] = 100 * (res_dict['acc1'] - res_dict['sub_pix_shift_accuracy']) / res_dict['acc1']

    return res_dict
