# --------------------------------------------------------
# Swin Transformer
# Copyright (c) 2021 Microsoft
# Licensed under The MIT License [see LICENSE for details]
# Written by Ze Liu
# --------------------------------------------------------

import os
import torch
import torch.distributed as dist
import numpy as np
import sklearn.metrics as sk_metrics

try:
    # noinspection PyUnresolvedReferences
    # from apex import amp
    from torch.cuda import amp
except ImportError:
    amp = None


def load_checkpoint(config, model, optimizer, lr_scheduler, logger):
    logger.info(f"==============> Resuming form {config.MODEL.RESUME}....................")
    checkpoint = torch.load(config.MODEL.RESUME, map_location='cpu')
    msg = model.load_state_dict(checkpoint['model'], strict=False)
    logger.info(msg)
    max_accuracy = 0.0
    if not config.EVAL_MODE and 'optimizer' in checkpoint and 'lr_scheduler' in checkpoint and 'epoch' in checkpoint:
        optimizer.load_state_dict(checkpoint['optimizer'])
        lr_scheduler.load_state_dict(checkpoint['lr_scheduler'])
        config.defrost()
        config.TRAIN.START_EPOCH = checkpoint['epoch'] + 1
        config.freeze()
        if 'amp' in checkpoint and config.AMP_OPT_LEVEL != "O0" and checkpoint['config'].AMP_OPT_LEVEL != "O0":
            amp.load_state_dict(checkpoint['amp'])
        logger.info(f"=> loaded successfully '{config.MODEL.RESUME}' (epoch {checkpoint['epoch']})")
        if 'max_accuracy' in checkpoint:
            max_accuracy = checkpoint['max_accuracy']

    del checkpoint
    torch.cuda.empty_cache()
    return max_accuracy


def save_checkpoint(config, epoch, model, max_accuracy, optimizer, lr_scheduler, logger):
    save_state = {'model': model.state_dict(),
                  'optimizer': optimizer.state_dict(),
                  'lr_scheduler': lr_scheduler.state_dict(),
                  'max_accuracy': max_accuracy,
                  'epoch': epoch,
                  'config': config}
    if config.AMP_OPT_LEVEL != "O0":
        save_state['amp'] = amp.state_dict()

    save_path = os.path.join(config.OUTPUT, f'ckpt_epoch_{epoch}.pth')
    logger.info(f"{save_path} saving......")
    torch.save(save_state, save_path)
    logger.info(f"{save_path} saved !!!")


def get_grad_norm(parameters, norm_type=2):
    if isinstance(parameters, torch.Tensor):
        parameters = [parameters]
    parameters = list(filter(lambda p: p.grad is not None, parameters))
    norm_type = float(norm_type)
    total_norm = 0
    for p in parameters:
        param_norm = p.grad.data.norm(norm_type)
        total_norm += param_norm.item() ** norm_type
    total_norm = total_norm ** (1. / norm_type)
    return total_norm


def auto_resume_helper(output_dir):
    checkpoints = os.listdir(output_dir)
    checkpoints = [ckpt for ckpt in checkpoints if ckpt.endswith('pth')]
    print(f"All checkpoints founded in {output_dir}: {checkpoints}")
    if len(checkpoints) > 0:
        latest_checkpoint = max([os.path.join(output_dir, d) for d in checkpoints], key=os.path.getmtime)
        print(f"The latest checkpoint founded: {latest_checkpoint}")
        resume_file = latest_checkpoint
    else:
        resume_file = None
    return resume_file


def compute_AUCs_global(all_genuine_preds, all_forge_preds):
    y_true = np.ones(len(all_genuine_preds) + len(all_forge_preds))
    y_true[len(all_genuine_preds):] = 0
    y_scores = np.concatenate([all_genuine_preds, all_forge_preds])
    auc_global = sk_metrics.roc_auc_score(y_true, y_scores)
    return auc_global

def compute_AUCs(genuine_preds, forge_preds):
    """ Compute the area under the curve for the classifiers

    Parameters
    ----------
    genuine_preds: list of np.ndarray
        A list of predictions of genuine signatures (each element on the list is the
        prediction for one user)
    skilled_preds: list of np.ndarray
        A list of predictions of skilled forgeries (each element on the list is the
        prediction for one user)

    Returns
    -------
    list
        The list of AUCs (one per user)
    float
        The mean AUC

    """
    aucs = []
    for thisRealPreds, thisforgePreds in zip(genuine_preds, forge_preds):
        y_true = np.ones(len(thisRealPreds) + len(thisforgePreds))
        y_true[len(thisRealPreds):] = 0
        y_scores = np.concatenate([thisRealPreds, thisforgePreds])
        aucs.append(sk_metrics.roc_auc_score(y_true, y_scores))
    meanAUC = np.mean(aucs)
    return aucs, meanAUC.item()


def compute_EER(all_genuine_preds,
                all_forge_preds):
    """ Calculate Equal Error Rate with a global decision threshold.

    Parameters
    ----------
    all_genuine_preds: np.ndarray
        Scores for genuine predictions of all users
    all_skilled_preds: np.ndarray
    Scores for skilled forgery predictions of all users

    Returns
    -------
    float:
        The Equal Error Rate
    float:
        The optimum global threshold (a posteriori)

    """

    all_preds = np.concatenate([all_genuine_preds, all_forge_preds])
    all_ys = np.concatenate([np.ones_like(all_genuine_preds), np.ones_like(all_forge_preds) * 0])
    fpr, tpr, thresholds = sk_metrics.roc_curve(all_ys, all_preds)

    # Select the threshold closest to (FPR = 1 - TPR)
    t = thresholds[sorted(enumerate(abs(fpr - (1 - tpr))), key=lambda x: x[1])[0][0]]
    genuineErrors = 1 - np.mean(all_genuine_preds >= t).item()
    skilledErrors = 1 - np.mean(all_forge_preds < t).item()
    EER = (genuineErrors + skilledErrors) / 2.0
    return EER, t


def calculate_EER_user_thresholds(genuine_preds, forge_preds):
    """ Calculate Equal Error Rate with a decision threshold specific for each user

    Parameters
    ----------
    genuine_preds: list of np.ndarray
        A list of predictions of genuine signatures (each element on the list is the
        prediction for one user)
    skilled_preds: list of np.ndarray
        A list of predictions of skilled forgeries (each element on the list is the
        prediction for one user)

    Returns
    -------
    float
        The Equal Error Rate when using user-specific thresholds

    """
    all_genuine_errors = []
    all_skilled_errors = []
    all_thresh = []
    nRealPreds = 0
    nSkilledPreds = 0

    for this_real_preds, this_skilled_preds in zip(genuine_preds, forge_preds):
        # Calculate user AUC
        y_true = np.ones(len(this_real_preds) + len(this_skilled_preds))
        y_true[len(this_real_preds):] = 0
        y_scores = np.concatenate([this_real_preds, this_skilled_preds])

        # Calculate user threshold
        fpr, tpr, thresholds = sk_metrics.roc_curve(y_true, y_scores)
        # Select the threshold closest to (FPR = 1 - TPR).
        t = thresholds[sorted(enumerate(abs(fpr - (1 - tpr))), key=lambda x: x[1])[0][0]]
        all_thresh.append(t)
        genuineErrors = np.sum(this_real_preds < t)
        skilledErrors = np.sum(this_skilled_preds >= t)

        all_genuine_errors.append(genuineErrors)
        all_skilled_errors.append(skilledErrors)

        nRealPreds += len(this_real_preds)
        nSkilledPreds += len(this_skilled_preds)

    genuineErrors = float(np.sum(all_genuine_errors)) / nRealPreds
    skilledErrors = float(np.sum(all_skilled_errors)) / nSkilledPreds

    # Errors should be nearly equal, up to a small rounding error since we have few examples per user.
    EER = (genuineErrors + skilledErrors) / 2.0
    return EER, all_thresh


def compute_AUC_EER(predict, label):
    """ Calculate AUC and Equal Error Rate with a global decision threshold.

    Parameters
    ----------
    predict: np.ndarray
        Scores for predictions of all users
    label: np.ndarray
    labels of all users

    Returns
    -------
    float:
        AUC
    float:
        The Equal Error Rate
    float:
        The optimum global threshold (a posteriori)

    """
    predict = predict.cpu().detach().numpy()
    label = label.cpu().detach().numpy()
    aucs = sk_metrics.roc_auc_score(label, predict)
    fpr, tpr, thresholds = sk_metrics.roc_curve(label, predict, pos_label=1)
    # Select the threshold closest to (FPR = 1 - TPR)
    t = thresholds[sorted(enumerate(abs(fpr - (1 - tpr))), key=lambda x: x[1])[0][0]]
    genuineErrors = 1 - np.mean(predict[label == 1] >= t).item()
    skilledErrors = 1 - np.mean(predict[label == 0] < t).item()
    EER = (genuineErrors + skilledErrors) / 2.0
    return aucs, EER, t


def compute_accuracy_global(all_genuine_preds, all_forge_preds, global_thresh):
    y_true = np.ones(len(all_genuine_preds) + len(all_forge_preds))
    y_true[len(all_genuine_preds):] = 0
    preds = np.concatenate([all_genuine_preds, all_forge_preds])
    length = np.array(len(all_genuine_preds) + len(all_forge_preds))
    preds[preds >= global_thresh] = 1
    preds[preds < global_thresh] = 0
    accuracy_global = np.sum(preds == y_true) / length
    return accuracy_global


def compute_accuracy(genuine_preds, forge_preds, thresh):
    accuracys = []
    for (user_genuinepreds, user_forgepreds, user_thresh) in zip(genuine_preds, forge_preds, thresh):
        length = np.array(len(user_genuinepreds) + len(user_forgepreds))
        y_true = np.ones(len(user_genuinepreds) + len(user_forgepreds))
        y_true[len(user_genuinepreds):] = 0
        preds = np.concatenate((user_genuinepreds, user_forgepreds), axis=0)
        preds[preds >= user_thresh] = 1
        preds[preds < user_thresh] = 0
        accuracy = np.sum(preds == y_true) / length
        accuracys.append(accuracy)
    return np.mean(accuracys)


def compute_metrics(genuine_preds, forge_preds):
    """ Compute metrics given the predictions (scores) of genuine signatures,
    random forgeries and skilled forgeries.

    Parameters
    ----------
    genuine_preds: list of np.ndarray
        A list of predictions of genuine signatures (each element on the list is the
        prediction for one user)
    random_preds: list of np.ndarray
        A list of predictions of random forgeries (each element on the list is the
        prediction for one user)
    skilled_preds: list of np.ndarray
        A list of predictions of skilled forgeries (each element on the list is the
        prediction for one user)
    global_threshold: float
        The global threshold used to compute false acceptance and false rejection rates

    Returns
    -------
    dict
        A dictionary containing:
        'FRR': false rejection rate
        'FAR_random': false acceptance rate for random forgeries
        'FAR_skilled': false acceptance rate for skilled forgeries
        'mean_AUC': mean Area Under the Curve (average of AUC for each user)
        'EER': Equal Error Rate using a global threshold
        'EER_userthresholds': Equal Error Rate using user-specific thresholds
        'auc_list': the list of AUCs (one per user)
        'global_threshold': the optimum global threshold (used in EER)
    """
    all_genuine_preds = np.concatenate(genuine_preds)
    all_forge_preds = np.concatenate(forge_preds)

    aucs, meanAUC = compute_AUCs(genuine_preds, forge_preds)
    AUC_global = compute_AUCs_global(all_genuine_preds, all_forge_preds)
    EER, global_threshold = compute_EER(all_genuine_preds, all_forge_preds)
    EER_userthresholds, thresh_user = calculate_EER_user_thresholds(genuine_preds, forge_preds)
    accuracy = compute_accuracy(genuine_preds, forge_preds, thresh_user)
    accuracy_global = compute_accuracy_global(all_genuine_preds, all_forge_preds, global_threshold)
    all_metrics = {
        'AUC_user': meanAUC,
        'AUC_global':AUC_global,
        'EER_global': EER,
        'EER_user': EER_userthresholds,
        'Accuracy_user': accuracy,
        'Accuracy_global': accuracy_global,
        'auc_list': aucs,
        'global_threshold': global_threshold}
    return all_metrics