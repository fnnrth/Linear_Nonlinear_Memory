import torch

def regularization_loss_old(model, tau, M_reg):
    """
    Compute the regularization loss for the given model.
    Args:
        model: The model object with attributes M, L, W, A, h.
        tau: Regularization strength (float).
        M_reg: Number of units to regularize (int).
    Returns:
        The combined regularization loss (torch scalar).
    """
    M = model.M
    P = model.L
    linear = M - P  # Number of linear units
    reg_linear_end = min(M_reg, linear)
    reg_nonlinear_start = max(0, M_reg - linear)
    # ---- Regularization for Linear Units (only W) ----
    if reg_linear_end > 0:
        reg_A_linear = torch.sum((torch.diag(model.W)[:reg_linear_end] - 1) ** 2)
        W_off_diag_linear = model.W[:reg_linear_end, :]
        reg_W_linear = torch.sum(W_off_diag_linear ** 2) - torch.sum(W_off_diag_linear.diag() ** 2)
    else:
        reg_A_linear = 0
        reg_W_linear = 0
    # ---- Regularization for Nonlinear Units (W + A) ----
    if reg_nonlinear_start > 0:
        reg_A_nonlinear = torch.sum(
            (torch.diag(model.A)[:reg_nonlinear_start] + torch.diag(model.W)[linear:linear + reg_nonlinear_start] - 1) ** 2
        )
        W_off_diag_nonlinear = model.W[linear:linear + reg_nonlinear_start, :]
        reg_W_nonlinear = torch.sum(W_off_diag_nonlinear ** 2) - torch.sum(W_off_diag_nonlinear.diag() ** 2)
    else:
        reg_A_nonlinear = 0
        reg_W_nonlinear = 0
    # ---- Bias regularization for all units ----
    reg_h = torch.sum(model.h[:M_reg] ** 2)
    # Combine all regularization terms
    reg_loss = tau * (reg_A_linear + reg_W_linear + reg_A_nonlinear + reg_W_nonlinear + reg_h)
    return reg_loss

def regularization_loss(model, tau, M_reg):
    """
    Works for any (M, L, M_reg) combination.
    """
    M, L = model.M, model.L
    linear = M - L
    loss = 0.0
    for i in range(M_reg):
        # effective diagonal term 𝛁A_ii
        if i < linear:                        # linear unit
            diag_eff = model.W[i, i]
        else:                                 # nonlinear unit
            diag_eff = model.W[i, i] + model.A[i - linear]
        loss += (diag_eff - 1) ** 2           # first sum
        # off-diagonal W
        loss += torch.sum(model.W[i, : ]**2) - model.W[i, i]**2

        # bias term
        loss += model.h[i]**2
    return tau * loss



