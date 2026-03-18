import os
import copy
import torch
import torch.nn as nn
from torch.nn import functional as F

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def cross_loss(feature_1, feature_2, args):
    labels = torch.cat([torch.arange(len(feature_1)) for _ in range(2)], dim=0)
    labels = (labels.unsqueeze(0) == labels.unsqueeze(1)).float()
    labels = labels.to(device)

    feature_1 = F.normalize(feature_1, dim=1)
    feature_2 = F.normalize(feature_2, dim=1)
    features = torch.cat((feature_1, feature_2), dim=0)  # (2*B, Feat_dim)

    similarity_matrix = torch.matmul(features, features.T)  # (2*B, 2*B)

    # discard the main diagonal from both: labels and similarities matrix
    mask = torch.eye(labels.shape[0], dtype=torch.bool).to(device)
    labels = labels[~mask].view(labels.shape[0], -1)
    similarity_matrix = similarity_matrix[~mask].view(similarity_matrix.shape[0], -1)  # (2*B, 2*B - 1)

    # select and combine multiple positives
    positives = similarity_matrix[labels.bool()].view(labels.shape[0], -1)  # (2*B, 1)
    negatives = similarity_matrix[~labels.bool()].view(similarity_matrix.shape[0], -1)  # (2*B, 2*(B - 1))

    logits = torch.cat([positives, negatives], dim=1)
    labels = torch.zeros(logits.shape[0], dtype=torch.long).to(device)

    logits = logits / args.temperature

    return nn.CrossEntropyLoss()(logits, labels)

def mcc_loss(sk_feature, photo_feature):
    l1 = nn.L1Loss()
    sk_feature = F.normalize(sk_feature).float()
    photo_feature = F.normalize(photo_feature).float()
    
    sk2sk_sim = sk_feature @ sk_feature.t()
    ph2ph_sim = photo_feature @ photo_feature.t()
    
    mcc_sk = torch.tensor(0.1).float()
    mcc_ph = torch.tensor(0).float()
    
    loss_mcc_sk = l1(sk2sk_sim.mean(), mcc_sk.to(device)) * 4
    loss_mcc_ph = l1(ph2ph_sim.mean(),mcc_ph.to(device)) * 8
    
    return loss_mcc_sk + loss_mcc_ph

def nt_xent(features_view1: torch.Tensor, features_view2: torch.Tensor):
    """
    NT-Xent for SimCLR
    features_view1, features_view2: (B, D)
    """
    features_view1 = F.normalize(features_view1)
    features_view2 = F.normalize(features_view2)
    
    temperature = 0.07
    B, D = features_view1.shape
    device = features_view1.device

    z = torch.cat([features_view1, features_view2], dim=0)

    logits = z @ z.t()                              # (2B, 2B)
    mask = torch.eye(2 * B, dtype=torch.bool, device=device)
    logits = logits.masked_fill(mask, float('-inf'))

    logits = logits / temperature

    labels = torch.cat([
        torch.arange(B, 2*B, device=device),
        torch.arange(0, B, device=device)
    ], dim=0).long()

    loss = F.cross_entropy(logits, labels)
    return loss


def loss_fn(args, model, features, mode='train'):
    photo_features_norm, sk_feature_norm, photo_aug_features, sk_aug_features, \
        neg_features, label, pos_logits, sk_logits, photo_features, sk_features, \
            ph_txt_feature, sk_txt_feature = features

    label = label.to(pos_logits.device)
    # loss_mcc = mcc_loss(photo_features, sk_features)
    
    loss_ce_photo = F.cross_entropy(pos_logits, label)
    loss_ce_sk = F.cross_entropy(sk_logits, label)
    loss_cls = loss_ce_photo + loss_ce_sk
    
    # loss_distill_photo = nt_xent(photo_features, photo_aug_features)
    # loss_distill_sk = nt_xent(sk_features, sk_aug_features)
    
    loss_distill_photo = cross_loss(photo_features, photo_aug_features, args)
    loss_distill_sk = cross_loss(sk_features, sk_aug_features, args)
    loss_distill = loss_distill_photo + loss_distill_sk
    
    # photo_aug_features = photo_aug_features / photo_aug_features.norm(dim=-1, keepdim=True)
    # sk_aug_features = sk_aug_features / sk_aug_features.norm(dim=-1, keepdim=True)
    # logit_scale = model.logit_scale.exp()
    
    # ph_aug_logits = logit_scale * photo_aug_features @ ph_txt_feature.t()
    # sk_aug_logits = logit_scale * sk_aug_features @ sk_txt_feature.t()
    
    # loss_ce_ph_aug = F.cross_entropy(ph_aug_logits, label)
    # loss_ce_sk_aug = F.cross_entropy(sk_aug_logits, label)
    # loss_cls_aug = loss_ce_ph_aug + loss_ce_sk_aug
    
    distance_fn = lambda x, y: 1.0 - F.cosine_similarity(x, y)
    triplet = nn.TripletMarginWithDistanceLoss(
            distance_function=distance_fn, margin=0.3)
    loss_triplet = triplet(sk_feature_norm, photo_features_norm, neg_features)
    
    # loss_photo_skt = cross_loss(photo_features, sk_features, args)
    nt_xent_loss = nt_xent(photo_features, sk_features)
    
    total_loss = (
        loss_cls \
        + loss_triplet \
        + loss_distill \
        # + loss_cls_aug \
        # + loss_photo_skt \
        + nt_xent_loss
    )
    
    return total_loss