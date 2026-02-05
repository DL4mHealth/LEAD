import torch
import torch.nn.functional as F
import numpy as np
import pdb
from typing import Dict, Tuple


# compute the different contrastive loss

def ts2vec_loss(z1, z2, alpha=0.5):
    loss = torch.tensor(0., device=z1.device)
    # counter for loop number
    d = 0
    # shorter the length of time sequence each loop
    while z1.size(1) > 1:
        loss += alpha * instance_loss(z1, z2)
        loss += alpha * temporal_loss(z1, z2)
        d += 1
        z1 = F.max_pool1d(z1.transpose(1, 2), kernel_size=2).transpose(1, 2)
        z2 = F.max_pool1d(z2.transpose(1, 2), kernel_size=2).transpose(1, 2)
        return loss / d


def instance_loss(z1, z2):
    B, T, C = z1.size(0), z1.size(1), z1.size(2)
    if B == 1:
        return z1.new_tensor(0.)
    z = torch.cat([z1, z2], dim=0)  # 2B x T x C
    z = z.transpose(0, 1)  # T x 2B x C
    sim = torch.matmul(z, z.transpose(1, 2))  # T x 2B x 2B
    logits = torch.tril(sim, diagonal=-1)[:, :, :-1]  # T 2B x (2B-1), left-down side, remove last zero column
    logits += torch.triu(sim, diagonal=1)[:, :, 1:]  # T x 2B x (2B-1), right-up side, remove first zero column
    logits = -F.log_softmax(logits, dim=-1)  # log softmax do dividing and log

    i = torch.arange(B, device=z1.device)
    # take all timestamps by [:,x,x]
    # logits[:, i, B + i - 1] : right-up, takes r_i and r_j
    # logits[:, B + i, i] : down-left, takes r_i_prime and r_j_prime
    loss = (logits[:, i, B + i - 1].mean() + logits[:, B + i, i].mean()) / 2
    return loss


def temporal_loss(z1, z2):
    B, T, C = z1.size(0), z1.size(1), z1.size(2)
    if T == 1:
        return z1.new_tensor(0.)
    z = torch.cat([z1, z2], dim=1)  # B x 2T x C
    sim = torch.matmul(z, z.transpose(1, 2))  # B x 2T x 2T
    logits = torch.tril(sim, diagonal=-1)[:, :, :-1]  # B x 2T x (2T-1)
    logits += torch.triu(sim, diagonal=1)[:, :, 1:]
    logits = -F.log_softmax(logits, dim=-1)

    t = torch.arange(T, device=z1.device)
    # take all samples by [:,x,x]
    loss = (logits[:, t, T + t - 1].mean() + logits[:, T + t, t].mean()) / 2
    return loss


def moco_loss(q, k, queue, T=0.07):
    device = q.device
    # positive
    l_pos = torch.einsum('nc,nc->n', [q, k]).unsqueeze(-1)
    # negative from queue
    l_neg = torch.einsum('nc,ck->nk', [q, queue])

    logits = torch.cat([l_pos, l_neg], dim=1)
    logits /= T
    labels = torch.zeros(logits.shape[0], dtype=torch.long).to(device)

    loss = F.cross_entropy(logits, labels)
    return loss


def simclr_loss(z1, z2):
    """
    Computes a sample-level contrastive loss for embeddings z1 and z2 as describe in SimCLR framework.

    Args:
        z1 (torch.Tensor): Embeddings from view 1, shape [B, H].
        z2 (torch.Tensor): Embeddings from view 2, shape [B, H].

    Returns:
        torch.Tensor: The computed contrastive loss.
    """
    # Normalize embeddings
    z1 = F.normalize(z1, dim=1)
    z2 = F.normalize(z2, dim=1)

    # Compute similarity matrix
    sim_matrix = torch.mm(z1, z2.T)
    sim_matrix_exp = torch.exp(sim_matrix / 0.1)

    # Diagonal elements (positive pairs)
    diag_elements = torch.diag(sim_matrix_exp)
    triu_sum = sim_matrix_exp.sum(dim=1)
    tril_sum = sim_matrix_exp.sum(dim=0)

    # Loss terms for diagonal
    loss_diag1 = -torch.mean(torch.log(diag_elements / triu_sum))
    loss_diag2 = -torch.mean(torch.log(diag_elements / tril_sum))
    loss = loss_diag1 + loss_diag2
    loss_terms = 2

    # Final loss normalization
    return loss / loss_terms if loss_terms > 0 else 0


def id_loss(z1, z2, id):
    '''
    Computes a contrastive loss for embeddings z1 and z2 based on subject ID pairing.

    Args:
        z1 (torch.Tensor): Embeddings from view 1, shape [B, H].
        z2 (torch.Tensor): Embeddings from view 2, shape [B, H].
        id (torch.Tensor): Subject IDs corresponding to embeddings, shape [B].

    Returns:
        torch.Tensor: The computed contrastive loss.
    '''
    # Ensure all tensors are on the same device
    device = z1.device

    # Normalize embeddings
    z1 = F.normalize(z1, dim=1)
    z2 = F.normalize(z2, dim=1)

    # Compute similarity matrix
    sim_matrix = torch.mm(z1, z2.T)
    sim_matrix_exp = torch.exp(sim_matrix / 0.1)

    # Convert IDs to a boolean matrix for positive pairs
    id_matrix = id.unsqueeze(1) == id.unsqueeze(0)  # Boolean matrix for matching IDs

    # Get upper and lower triangle indices
    rows1, cols1 = torch.triu_indices(id.size(0), id.size(0), offset=1, device=device)
    rows2, cols2 = torch.tril_indices(id.size(0), id.size(0), offset=-1, device=device)

    triu_sum = sim_matrix_exp.sum(dim=1)
    tril_sum = sim_matrix_exp.sum(dim=0)

    loss = 0
    loss_terms = 0

    # Upper triangle positive pairs
    upper_mask = id_matrix[rows1, cols1].to(device)  # Ensure mask is on the correct device
    if upper_mask.any():
        selected_rows = rows1[upper_mask]
        selected_cols = cols1[upper_mask]
        triu_elements = sim_matrix_exp[selected_rows.to(device), selected_cols.to(device)]  # Move indices to correct device
        loss_triu = -torch.mean(torch.log(triu_elements / triu_sum[selected_rows]))
        loss += loss_triu
        loss_terms += 1

    # Lower triangle positive pairs
    lower_mask = id_matrix[rows2, cols2].to(device)  # Ensure mask is on the correct device
    if lower_mask.any():
        selected_rows = rows2[lower_mask]
        selected_cols = cols2[lower_mask]
        tril_elements = sim_matrix_exp[selected_rows.to(device), selected_cols.to(device)]  # Move indices to correct device
        loss_tril = -torch.mean(torch.log(tril_elements / tril_sum[selected_cols]))
        loss += loss_tril
        loss_terms += 1

    # Final loss normalization
    return loss / loss_terms if loss_terms > 0 else 0


def simclr_id_loss(z1, z2, id, lambda1=0.25, lambda2=0.75):
    '''
    Computes a contrastive loss for embeddings z1 and z2 based on the SimCLR framework and subject ID pairing.

    Args:
        z1 (torch.Tensor): Embeddings from view 1, shape [B, H].
        z2 (torch.Tensor): Embeddings from view 2, shape [B, H].
        id (torch.Tensor): Subject IDs corresponding to embeddings, shape [B].

    Returns:
        torch.Tensor: The computed contrastive loss.
    '''
    # Ensure all tensors are on the same device
    device = z1.device

    # Normalize embeddings
    z1 = F.normalize(z1, dim=1)
    z2 = F.normalize(z2, dim=1)

    # Compute similarity matrix
    sim_matrix = torch.mm(z1, z2.T)
    sim_matrix_exp = torch.exp(sim_matrix / 0.1)

    # Convert IDs to a boolean matrix for positive pairs
    id_matrix = id.unsqueeze(1) == id.unsqueeze(0)  # Boolean matrix for matching IDs

    # Get upper and lower triangle indices
    rows1, cols1 = torch.triu_indices(id.size(0), id.size(0), offset=1, device=device)
    rows2, cols2 = torch.tril_indices(id.size(0), id.size(0), offset=-1, device=device)

    # Diagonal elements (positive pairs)
    diag_elements = torch.diag(sim_matrix_exp)
    triu_sum = sim_matrix_exp.sum(dim=1)
    tril_sum = sim_matrix_exp.sum(dim=0)

    # Loss terms for diagonal
    loss_diag1 = -torch.mean(torch.log(diag_elements / triu_sum))
    loss_diag2 = -torch.mean(torch.log(diag_elements / tril_sum))
    sample_loss = (loss_diag1 + loss_diag2) / 2

    subject_loss = 0
    subject_loss_terms = 0
    # Upper triangle positive pairs
    upper_mask = id_matrix[rows1, cols1].to(device)  # Ensure mask is on the correct device
    if upper_mask.any():
        selected_rows = rows1[upper_mask]
        selected_cols = cols1[upper_mask]
        triu_elements = sim_matrix_exp[selected_rows.to(device), selected_cols.to(device)]  # Move indices to correct device
        loss_triu = -torch.mean(torch.log(triu_elements / triu_sum[selected_rows]))
        subject_loss += loss_triu
        subject_loss_terms += 1

    # Lower triangle positive pairs
    lower_mask = id_matrix[rows2, cols2].to(device)  # Ensure mask is on the correct device
    if lower_mask.any():
        selected_rows = rows2[lower_mask]
        selected_cols = cols2[lower_mask]
        tril_elements = sim_matrix_exp[selected_rows.to(device), selected_cols.to(device)]  # Move indices to correct device
        loss_tril = -torch.mean(torch.log(tril_elements / tril_sum[selected_cols]))
        subject_loss += loss_tril
        subject_loss_terms += 1

    subject_loss /= subject_loss_terms  # Average over upper and lower triangle losses
    return lambda1 * sample_loss + lambda2 * subject_loss


def subject_ce_loss(
    logits: torch.Tensor,            # (B, C)
    labels: torch.Tensor,            # (B,)
    subject_ids: torch.Tensor,       # (B,)
    alpha: float = 1.0,              # weight for sample CE
    beta:  float = 1.0,              # weight for subject CE
):
    device = logits.device

    # -------- 1) sample-level CE --------
    sample_ce = F.cross_entropy(logits, labels, reduction='mean')

    # -------- 2) subject-level CE --------
    uniq_sids = torch.unique(subject_ids)
    subj_mean_logits, subj_labels = [], []
    for sid in uniq_sids:
        m = (subject_ids == sid)
        z_mean = logits[m].mean(0)   # keep gradient here
        lbl = labels[m][0]           # same subject → same label
        subj_mean_logits.append(z_mean.unsqueeze(0))
        subj_labels.append(lbl.unsqueeze(0))

    subj_mean_logits = torch.cat(subj_mean_logits, dim=0)  # (S_b, C)
    subj_labels = torch.cat(subj_labels, dim=0)            # (S_b,)

    subject_ce = F.cross_entropy(subj_mean_logits, subj_labels, reduction='mean')

    # -------- 3) total loss --------
    return alpha * sample_ce + beta * subject_ce

