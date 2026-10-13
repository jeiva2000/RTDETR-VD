import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.ops import generalized_box_iou
from scipy.optimize import linear_sum_assignment

class DiceLoss(nn.Module):
    def __init__(self, smooth=1e-6):
        super(DiceLoss, self).__init__()
        self.smooth = smooth

    def forward(self, inputs, targets):
        """
        inputs: predicciones del modelo (logits o probabilidades) con forma (N, C, H, W)
        targets: máscaras de verdad (one-hot o índices) con forma (N, H, W) o (N, C, H, W)
        """
        # Aplanar tensores
        inputs = inputs.contiguous().view(-1)
        targets = targets.contiguous().view(-1)

        # Calcular coeficiente de Dice
        intersection = (inputs * targets).sum()
        dice = (2. * intersection + self.smooth) / (inputs.sum() + targets.sum() + self.smooth)

        return 1 - dice

def xcycwh_to_xyxy(boxes):
    x1 = boxes[..., 0] - boxes[..., 2] / 2
    y1 = boxes[..., 1] - boxes[..., 3] / 2
    x2 = boxes[..., 0] + boxes[..., 2] / 2
    y2 = boxes[..., 1] + boxes[..., 3] / 2
    return torch.stack([x1, y1, x2, y2], dim=-1)


def xyxy_to_xcycwh(boxes):
    x1, y1, x2, y2 = boxes.unbind(-1)
    xc = (x1 + x2) / 2
    yc = (y1 + y2) / 2
    w = x2 - x1
    h = y2 - y1
    return torch.stack([xc, yc, w, h], dim=-1)

def xyxy_to_xywh(boxes):
    x1, y1, x2, y2 = boxes.unbind(-1)
    w = x2 - x1
    h = y2 - y1
    return torch.stack([x1, y1, w, h], dim=-1)

def giou_l1_with_ids(boxes_pred, boxes_gt, ids_pred, ids_target, lambda_l1=5.0, lambda_giou=2.0):
    """
    boxes_pred: (N_pred, 4) en formato cx,cy,w,h normalizado [0,1]
    boxes_gt:   (N_gt, 4) mismo formato
    ids_pred:   (N_pred,)
    ids_target: (N_gt,)
    """
    boxes_pred_xyxy = xcycwh_to_xyxy(boxes_pred)  
    boxes_gt_xyxy   = xcycwh_to_xyxy(boxes_gt)      

    giou_matrix = generalized_box_iou(boxes_pred_xyxy, boxes_gt_xyxy)  # [N_pred, N_gt]
    match_matrix = ids_pred[:, None] == ids_target[None, :]            # [N_pred, N_gt]
    indexs = torch.nonzero(match_matrix)

    if len(indexs) == 0:
        return torch.tensor(0.0, device=boxes_pred.device, requires_grad=True)

    # ---- GIoU loss ----
    giou_scores = 1.0 - giou_matrix[indexs[:, 0], indexs[:, 1]]
    loss_giou = giou_scores.mean()

    # ---- L1 loss ----
    matched_pred = boxes_pred[indexs[:, 0]]
    matched_gt   = boxes_gt[indexs[:, 1]]
    loss_l1 = F.l1_loss(matched_pred, matched_gt, reduction="mean")

    # ---- combinar ----
    #loss = lambda_l1 * loss_l1 + lambda_giou * loss_giou
    #cambio
    loss = (lambda_l1 * loss_l1 + lambda_giou * loss_giou) * (len(indexs) / max(len(ids_target), 1))

    return loss




class SupConLoss(nn.Module):
    def __init__(self, temperature=0.07):
        super(SupConLoss, self).__init__()
        self.temperature = temperature

    def forward(self, features, labels):
        """
        Args:
            features: tensor de forma [batch_size, dim]
            labels: tensor de forma [batch_size]
        Returns:
            loss: escalar
        """
        #print('labels:',labels)
        device = features.device
        #batch_size = features.shape[0]
        features = F.normalize(features, dim=1)  # normalización L2
        #print('labels shape:',labels)

        # Similaridad de coseno
        similarity_matrix = torch.matmul(features, features.T) / self.temperature
        #print('sim loss:',similarity_matrix)
        # Crear máscara de positivos (misma clase, excluir self-pairs)
        labels = labels.contiguous().view(-1, 1)
        mask = torch.eq(labels, labels.T).float().to(device)
        mask.fill_diagonal_(0)

        # Estabilidad numérica
        logits_max, _ = torch.max(similarity_matrix, dim=1, keepdim=True)
        logits = similarity_matrix - logits_max.detach()

        # Máscara para exclusión de sí mismos
        logits_mask = torch.ones_like(mask).fill_diagonal_(0)

        # Log-probabilidades
        exp_logits = torch.exp(logits) * logits_mask
        log_prob = logits - torch.log(exp_logits.sum(1, keepdim=True) + 1e-12)

        # Calcular pérdida
        #print('log_prob:',log_prob)
        #print('mask:',mask)
        mean_log_prob_pos = (mask * log_prob).sum(1) / (mask.sum(1) + 1e-12)

        loss = -mean_log_prob_pos.mean()
        return loss

"""
class SupConLossHardNegatives(torch.nn.Module):
    def __init__(self, temperature=0.07, hard_k=5, sim_threshold=0.4, penalty_weight=2.0):#0.5
        super().__init__()
        self.temperature = temperature
        self.hard_k = hard_k
        self.sim_threshold = sim_threshold
        self.penalty_weight = penalty_weight

    def forward(self, features, labels):
        
        device = features.device
        batch_size = features.shape[0]

        features = F.normalize(features, dim=1)
        
        # Cosine similarity matrix
        sim_matrix = torch.matmul(features, features.T) / self.temperature

        # Mask for positives (same label, exclude self)
        labels = labels.contiguous().view(-1, 1)
        mask_pos = torch.eq(labels, labels.T).float().to(device)
        mask_self = torch.eye(batch_size).to(device)
        mask_pos = mask_pos * (1 - mask_self)

        # For numerical stability
        sim_matrix = sim_matrix - torch.max(sim_matrix, dim=1, keepdim=True)[0]

        # Compute log prob for positives
        exp_sim = torch.exp(sim_matrix) * (1 - mask_self)
        log_prob = sim_matrix - torch.log(exp_sim.sum(dim=1, keepdim=True) + 1e-8)

        # Positive loss
        mean_log_prob_pos = (mask_pos * log_prob).sum(1) / (mask_pos.sum(1) + 1e-8)
        loss_pos = -mean_log_prob_pos.mean()

        # ----------- Hard Negative Penalty ------------
        # Select top-k most similar negatives (sim > threshold)
        mask_neg = 1 - mask_pos - mask_self
        sim_negatives = sim_matrix * mask_neg

        hard_neg_penalty = 0.0
        for i in range(batch_size):
            # Get top-k most similar negatives
            sim_row = sim_negatives[i]
            topk_sim, _ = torch.topk(sim_row, k=min(self.hard_k, (sim_row > -1e6).sum().item()))
            # Penalize if any are above threshold
            #penalty = F.relu(topk_sim - self.sim_threshold).mean()
            penalty = ((topk_sim - self.sim_threshold).clamp(min=0) ** 2).mean()
            hard_neg_penalty += penalty

        hard_neg_penalty /= batch_size
        total_loss = loss_pos + self.penalty_weight * hard_neg_penalty

        return total_loss
"""

class SupConLossHardNegatives(torch.nn.Module):
    def __init__(self, temperature=0.07, sim_threshold=0.2, penalty_weight=2.0):
        super().__init__()
        self.temperature = temperature
        self.sim_threshold = sim_threshold
        self.penalty_weight = penalty_weight

    def forward(self, features, labels):
        """
        features: (N, D) tensor of L2-normalized embeddings
        labels: (N,) tensor of identity labels
        """
        device = features.device
        batch_size = features.shape[0]

        features = F.normalize(features, dim=1)
        sim_matrix = torch.matmul(features, features.T) / self.temperature

        # Masks
        labels = labels.contiguous().view(-1, 1)
        mask_pos = torch.eq(labels, labels.T).float().to(device)
        mask_self = torch.eye(batch_size).to(device)
        mask_pos = mask_pos * (1 - mask_self)
        mask_neg = (1 - mask_pos - mask_self)

        # For numerical stability
        sim_matrix = sim_matrix - torch.max(sim_matrix, dim=1, keepdim=True)[0]

        # Log prob
        exp_sim = torch.exp(sim_matrix) * (1 - mask_self)
        log_prob = sim_matrix - torch.log(exp_sim.sum(dim=1, keepdim=True) + 1e-8)

        # Positive loss
        mean_log_prob_pos = (mask_pos * log_prob).sum(1) / (mask_pos.sum(1) + 1e-8)
        loss_pos = -mean_log_prob_pos.mean()

        # ---------- Hard Negative Penalty ----------
        hard_neg_penalty = 0.0
        for i in range(batch_size):
            sim_row = sim_matrix[i]
            neg_mask = mask_neg[i].bool()
            high_sim_neg = sim_row[neg_mask] > self.sim_threshold
            if high_sim_neg.any():
                penalty = ((sim_row[neg_mask][high_sim_neg] - self.sim_threshold) ** 2).mean()
                hard_neg_penalty += penalty
        hard_neg_penalty /= batch_size

        total_loss = loss_pos + self.penalty_weight * hard_neg_penalty
        return total_loss


def attention_entropy(attn_weights, eps=1e-8):
    entropy = - (attn_weights * torch.log(attn_weights + eps)).sum(dim=-1)  # shape: (num_heads, num_queries)
    return entropy.mean()


def weighted_bce_loss(update_gate, occ_mask, weight_0=10.0, weight_1=1.0):
    """
    Aplica BCE con pesos por clase para compensar desbalance de clases.
    
    Args:
        update_gate: Tensor de salida con `sigmoid` aplicado, forma (B, M, 1) o (B, M)
        occ_mask: Tensor binario de ground truth, forma (B, M)
        weight_0: Peso para la clase 0 (ocultos / no visibles)
        weight_1: Peso para la clase 1 (visibles)
    
    Returns:
        Escalar con la pérdida ponderada.
    """
    if update_gate.dim() == 3:
        update_gate = update_gate.squeeze(-1)  # (B, M)
    
    assert update_gate.shape == occ_mask.shape, "Shapes no coinciden"

    # Define pesos por muestra
    weights = torch.where(occ_mask == 1, weight_1, weight_0)

    # BCE por elemento (ambos deben estar en [0,1])
    bce = nn.BCELoss(reduction='none')
    loss = bce(update_gate, occ_mask)

    # Aplica pesos y reduce
    weighted_loss = (loss * weights).mean()
    return weighted_loss


def focal_loss(logits, targets, alpha=0.25, gamma=2.0, reduction="mean"):
    """
    logits: (B, N) salida cruda de la MLP de presencia
    targets: (B, N) etiquetas binarias 0/1
    alpha: peso de clase positiva (puedes poner 0.25 o 0.5)
    gamma: factor de enfoque (2.0 es común)
    """
    bce_loss = F.binary_cross_entropy_with_logits(
        logits, targets.float(), reduction="none"
    )
    probs = torch.sigmoid(logits)
    pt = torch.where(targets == 1, probs, 1 - probs)  # prob. de predicción correcta
    focal_weight = (alpha * targets + (1 - alpha) * (1 - targets)) * (1 - pt) ** gamma
    loss = focal_weight * bce_loss

    if reduction == "mean":
        return loss.mean()
    elif reduction == "sum":
        return loss.sum()
    else:
        return loss


def build_bipartite_adjacency(A):
    N, M = A.shape
    device = A.device
    top = torch.cat([torch.zeros(N, N, device=device), A], dim=1)
    bottom = torch.cat([A.T, torch.zeros(M, M, device=device)], dim=1)
    B = torch.cat([top, bottom], dim=0)
    return B

def compute_laplacian_bipartite(A, normalized=True):
    """
    A: matriz de adyacencia bipartita (N x M)
    Retorna matriz Laplaciana normalizada de tamaño (N+M) x (N+M).
    """
    B = build_bipartite_adjacency(A)
    N = B.shape[0]
    device = B.device
    I = torch.eye(N, device=device)
    
    d = torch.sum(B, dim=1)
    D = torch.diag(d)
    
    if normalized:
        d_inv_sqrt = torch.pow(d + 1e-10, -0.5)
        D_inv_sqrt = torch.diag(d_inv_sqrt)
        L = I - D_inv_sqrt @ B @ D_inv_sqrt
    else:
        L = D - B
    return L

def compute_lambda2_bipartite(A, normalized=True):
    L = compute_laplacian_bipartite(A, normalized)
    eigvals = torch.linalg.eigvalsh(L)
    lambda_2 = eigvals[1]
    return lambda_2.item()


def spectral_loss(A, lambda2_target=0.5, normalized=True):
    """
    Calcula la pérdida basada en el segundo menor autovalor de la Laplaciana de A.
    lambda2_target: valor deseado mínimo para lambda_2.
    """
    L = compute_laplacian_bipartite(A, normalized=normalized)
    # Autovalores en orden ascendente
    eigvals = torch.linalg.eigvalsh(L)  # Para matriz simétrica
    
    lambda_2 = eigvals[1]  # segundo menor valor propio (lambda_1 es 0)
    
    loss = torch.relu(lambda2_target - lambda_2)  # penaliza si lambda_2 < target
    return loss, lambda_2


def loss_dispersion(A):
    """
    Mide la dispersión de cada detección sobre múltiples identidades (entropía por fila).
    A: Tensor de atención suave [n_detections, n_identities]
    """
    A_soft = F.softmax(A, dim=1)
    entropy = - (A_soft * torch.log(A_soft + 1e-8)).sum(dim=1)
    return -entropy.mean()  # Queremos maximizar entropía → minimizar negativo


def loss_collapse(A):
    """
    Mide si las detecciones colapsan sobre un solo nodo (entropía por columna).
    A: Tensor de atención suave [n_detections, n_identities]
    """
    A_soft = F.softmax(A, dim=1)
    identity_weights = A_soft.mean(dim=0)  # Promedio de activación por identidad
    entropy = - (identity_weights * torch.log(identity_weights + 1e-8)).sum()
    return -entropy  # Queremos maximizar entropía → minimizar negativo


def loss_expansor(A, lambda_disp=1.0, lambda_collapse=1.0):
    """
    Combinación de dispersión y colapso para promover expansión.
    A: Tensor [n_detections, n_identities]
    """
    return lambda_disp * loss_dispersion(A) + lambda_collapse * loss_collapse(A)


class CheegerSpectralLoss(nn.Module):
    def __init__(self,
                 lambda2_target=0.5,
                 cheeger_target=0.5,
                 dominance_threshold=0.5,
                 dominance_weight=1.0,
                 entropy_weight=0.1,
                 cheeger_weight=1.0,
                 lambda2_weight=1.0,
                 symmetric_mode='AAT'):  # puedes elegir 'AAT' o 'ATA'
        super().__init__()
        self.lambda2_target = lambda2_target
        self.cheeger_target = cheeger_target
        self.dominance_threshold = dominance_threshold
        self.dominance_weight = dominance_weight
        self.entropy_weight = entropy_weight
        self.cheeger_weight = cheeger_weight
        self.lambda2_weight = lambda2_weight
        self.symmetric_mode = symmetric_mode

    def forward(self, A):
        device = A.device

        # Si A no es cuadrada, hacerla cuadrada con A*A^T o A^T*A
        if A.shape[0] != A.shape[1]:
            if self.symmetric_mode == 'AAT':
                A = A @ A.T  # (N, N)
            elif self.symmetric_mode == 'ATA':
                A = A.T @ A  # (M, M)
            else:
                raise ValueError(f"Unknown symmetric_mode: {self.symmetric_mode}")

        N = A.shape[0]

        D = torch.diag(A.sum(dim=1))
        L = D - A

        try:
            lambda_vals, _ = torch.linalg.eigh(L)
        except RuntimeError:
            lambda_vals = torch.linalg.eigvalsh(L)

        lambda2 = lambda_vals[1]

        lambda2_loss = (lambda2 - self.lambda2_target) ** 2

        cheeger_approx = torch.sqrt(2 * lambda2 + 1e-8)
        cheeger_loss = F.relu(self.cheeger_target - cheeger_approx)

        degree = A.sum(dim=0)
        degree_frac = degree / (degree.sum() + 1e-8)
        max_frac = degree_frac.max()
        dominance_penalty = F.relu(max_frac - self.dominance_threshold)

        entropy = -(degree_frac * (degree_frac + 1e-8).log()).sum()
        entropy_loss = self.entropy_weight * entropy

        total_loss = (
            self.lambda2_weight * lambda2_loss +
            self.cheeger_weight * cheeger_loss +
            self.dominance_weight * dominance_penalty +
            entropy_loss
        )

        stats = {
            'lambda_2': lambda2.item(),
            'cheeger_loss': cheeger_loss.item(),
            'dominance_penalty': dominance_penalty.item(),
            'entropy_loss': entropy_loss.item(),
            'lambda2_loss': lambda2_loss.item(),
            'total_loss': total_loss.item()
        }

        return total_loss, stats

class SpectralLoss(nn.Module):
    def __init__(self, k=10):
        """
        k: número de autovalores principales del espectro que se comparan.
        """
        super(SpectralLoss, self).__init__()
        self.k = k

    def forward(self, A1: torch.Tensor, A2: torch.Tensor):
        """
        A1, A2: matrices de adyacencia (batch_size, N, N)
        Retorna: pérdida L2 entre los espectros (batch)
        """
        assert A1.shape == A2.shape, "Las matrices de adyacencia deben tener la misma forma"
        B, N, _ = A1.shape
        loss = 0.0

        for b in range(B):
            L1 = self.normalized_laplacian(A1[b])
            L2 = self.normalized_laplacian(A2[b])

            # Autovalores
            evals1 = torch.linalg.eigvalsh(L1)
            evals2 = torch.linalg.eigvalsh(L2)

            # Tomamos los k menores (más significativos) y los ordenamos
            topk1 = torch.sort(evals1)[:self.k][0]
            topk2 = torch.sort(evals2)[:self.k][0]

            loss += torch.nn.functional.mse_loss(topk1, topk2)

        return loss / B

    def normalized_laplacian(self, A):
        """
        Calcula el Laplaciano normalizado: L = I - D^{-1/2} A D^{-1/2}
        """
        I = torch.eye(A.size(0), device=A.device)
        D = torch.diag(A.sum(dim=1))
        D_inv_sqrt = torch.diag(torch.pow(D.diag() + 1e-6, -0.5))
        L = I - D_inv_sqrt @ A @ D_inv_sqrt
        return L

def spectral_graph_loss_with_collapse(A, eps=1e-5, alpha=1.0, beta=1.0, gamma=1.0):
    """
    Penaliza desconexión, débil conectividad (fiedler bajo) y colapso estructural (filas redundantes),
    ignorando filas completamente cero en la parte de colapso.
    """
    A = A.clone()
    A.fill_diagonal_(0)

    # Laplaciano
    D = torch.diag(A.sum(dim=1))
    L = D - A

    eigvals = torch.linalg.eigvalsh(L)
    num_zero_eigvals = (eigvals < eps).sum()
    fiedler_val = eigvals[1]

    loss_disconnect = (num_zero_eigvals - 1).float() ** 2
    #loss_fiedler = 1.0 / (fiedler_val + eps)
    loss_fiedler = torch.relu(1.0 - fiedler_val)

    # ---- Penalización por colapso estructural (solo con filas no-cero) ----
    mask = (A.norm(p=1, dim=1) > 0)  # Selecciona solo filas no-cero
    A_valid = A[mask]

    if A_valid.shape[0] > 1:
        row_norms = F.normalize(A_valid, dim=1)
        sim_matrix = row_norms @ row_norms.T
        N_valid = A_valid.shape[0]
        redundancy = (sim_matrix.sum() - N_valid) / (N_valid * (N_valid - 1))
        loss_collapse = redundancy
    else:
        loss_collapse = torch.tensor(0.0, device=A.device)

    # ---- Pérdida total ----
    loss = alpha * loss_disconnect + beta * loss_fiedler + gamma * loss_collapse
    return loss


def loss_colapse_general(A):
    """
    Penaliza que muchas filas de A sean redundantes o colapsadas.
    A puede ser de tamaño (n x m), no necesariamente cuadrada.
    """
    # Evita dividir por cero: normalizá solo si las filas no son nulas
    row_norms = torch.nn.functional.normalize(A, dim=1, eps=1e-6)
    sim_matrix = row_norms @ row_norms.T  # similitud coseno entre filas

    # Elimina la diagonal (autosemejanza)
    n = A.shape[0]
    redundancy = (sim_matrix.sum() - n) / (n * (n - 1) + 1e-6)  # promedio de similitud entre filas distintas

    return redundancy.clamp(min=0.0)  # evitamos valores negativos


def spectral_loss(A_hat, A_true, alpha=1.0, k=None):
    """
    Calcula la pérdida espectral entre una matriz de adyacencia predicha y la real.
    
    Args:
        A_hat (Tensor): Matriz de adyacencia predicha (N x N), debe ser simétrica.
        A_true (Tensor): Matriz de adyacencia real (N x N), debe ser simétrica.
        alpha (float): Peso relativo de la pérdida de eigenvectores.
        k (int or None): Número de eigenvalores/eigenvectores a considerar (por defecto: todos).
    
    Returns:
        loss (Tensor): Pérdida escalar total.
    """
    # Forzar simetría
    A_hat = (A_hat + A_hat.T) / 2
    A_true = (A_true + A_true.T) / 2
    A_true = A_true + 1e-5 * torch.eye(A_true.shape[-1], device=A_true.device)

    # Eigen descomposición (diferenciable)
    λ_hat, v_hat = torch.linalg.eigh(A_hat)
    #print('A true:',A_true)
    λ_true, v_true = torch.linalg.eigh(A_true)

    # Orden descendente por valor propio
    λ_hat = λ_hat.flip(-1)
    λ_true = λ_true.flip(-1)
    v_hat = v_hat.flip(-1)
    v_true = v_true.flip(-1)

    # Selección de top-k (si aplica)
    if k is not None:
        λ_hat = λ_hat[:k]
        λ_true = λ_true[:k]
        v_hat = v_hat[:, :k]
        v_true = v_true[:, :k]

    # Pérdida entre eigenvalores
    loss_val = F.mse_loss(λ_hat, λ_true)

    # Similaridad entre eigenvectores (comparación columna a columna)
    cos_sim = torch.sum(v_hat * v_true, dim=0)  # tamaño (k,)
    loss_vec = torch.mean(1 - torch.abs(cos_sim))  # evitar ambigüedad de signo

    # Pérdida total
    return loss_val + alpha * loss_vec


def resistance_distance(adj: torch.Tensor) -> torch.Tensor:
    """
    Calcula la matriz de distancias de resistencia entre todos los pares de nodos.
    Args:
        adj: [N, N] matriz de adyacencia (simétrica, sin pesos negativos)
    Returns:
        R: [N, N] matriz de resistencias efectivas
    """
    N = adj.size(0)
    deg = torch.diag(adj.sum(dim=-1))
    L = deg - adj  # Laplaciano

    # pseudoinversa (simétrica y semidefinida positiva)
    L_pinv = torch.linalg.pinv(L)

    diag = torch.diag(L_pinv).unsqueeze(0)  # [1, N]
    R = diag.T + diag - 2 * L_pinv
    return R


def resistance_loss(adj_pred: torch.Tensor, adj_gt: torch.Tensor):
    """
    Pérdida espectral de resistencias entre grafo predicho y real.
    
    Args:
        adj_pred: [N, N] matriz de adyacencia predicha
        adj_gt:   [N, N] matriz de adyacencia real
    Returns:
        loss: escalar
    """
    R_pred = resistance_distance(adj_pred)
    R_gt   = resistance_distance(adj_gt)

    # comparar solo pares distintos (i != j), opcional
    mask = ~torch.eye(adj_pred.size(0), dtype=bool, device=adj_pred.device)

    loss = torch.mean((R_pred[mask] - R_gt[mask])**2)
    return loss