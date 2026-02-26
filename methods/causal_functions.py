import numpy as np

def get_Neyman_ATE(y, t):
    """
    Compute the Neyman ATE estimate w.r.t. a binary treatment.
    
    Parameters
    ----------
    y: array-like of shape (n_samples,)
       Vector of observed responses
    t: array-like of shape (n_samples,)
       Vector of treatment assignments taking values in {0,1}
    
    Returns
    -------
    ATE_: float
       The Neyman ATE estimate.
    """
    y0_obs = y[t==0]
    y1_obs = y[t==1]
    ATE_ = y1_obs.mean() - y0_obs.mean()
    return ATE_


def get_Neyman_var(y, t):
    """
    Compute the Neyman ATE variance estimate w.r.t. a binary treatment.
    
    Parameters
    ----------
    y: array-like of shape (n_samples,)
       Vector of observed responses
    t: array-like of shape (n_samples,)
       Vector of treatment assignments taking values in {0,1}
    
    Returns
    -------
    V_Neyman: float
       The Neyman ATE sampling variance estimate.
    """
    N_c = (1-t).sum()
    N_t = t.sum()
    y0_obs = y[t==0]
    y1_obs = y[t==1]
    
    var_c_ = np.var(y0_obs, ddof=1)
    var_t_ = np.var(y1_obs, ddof=1)
    V_Neyman = var_c_/N_c + var_t_/N_t
    
    return V_Neyman

def get_subgroup_CATE(y, t, subgroup_indicator):
    """
    Compute the Neyman estimate for the CATE for a subgroup (treatment effect 
    averaged over the subgroup.)
    
    Parameters
    ----------
    y: array-like of shape (n_samples,)
        Vector of observed responses
    t: array-like of shape (n_samples,)
        Vector of treatment assignments taking values in {0,1}
    subgroup_indicator: array-like of shape (n_samples,)
        Indicator vector for a subgroup
    
    Returns
    -------
    CATE_: float
           The CATE for the subgroup defined by the indicator.
    """
    
    # Compute y, t vectors for the subgroup
    y_sg = y[subgroup_indicator]
    t_sg = t[subgroup_indicator]
    CATE_ = get_Neyman_ATE(y_sg, t_sg)
    
    return CATE_

def get_subgroup_CATE_std(y, t, subgroup_indicator):
    """
    Compute the std estimate for the Neyman CATE estimator for a subgroup 
    (treatment effect averaged over the subgroup.)
    
    Parameters
    ----------
    y: array-like of shape (n_samples,)
        Vector of observed responses
    t: array-like of shape (n_samples,)
        Vector of treatment assignments taking values in {0,1}
    subgroup_indicator: array-like of shape (n_samples,)
        Indicator vector for a subgroup
    
    Returns
    -------
    CATE_: float
           The CATE for the subgroup defined by the indicator.
    """
    
    # Compute y, t vectors for the subgroup
    y_sg = y[subgroup_indicator]
    t_sg = t[subgroup_indicator]
    CATE_std_ = np.sqrt(get_Neyman_var(y_sg, t_sg))
    
    return CATE_std_


def get_subgroup_t_statistic(y, t, subgroup_indicator, 
                             dataset_indicator = None):
    """
    Compute the t-statistic for a subgroup. t-statistic to be defined in paper.
    
    Parameters
    ----------
    y: array-like of shape (n_samples,)
        Vector of observed responses
    t: array-like of shape (n_samples,)
        Vector of treatment assignments taking values in {0,1}
    subgroup_indicator: array-like of shape (n_samples,)
        Indicator vector for a subgroup
    datasset_indicator: array-like of shape (n_samples,)
        Indicator vector for the validation set / training set if we would
        like to compute the t-statistic w.r.t these sets
    
    Returns
    -------
    t_statistic: float
           The t_statistic for the subgroup defined by the indicator.
    """
    
    # Restrict the dataset if so desired
    if dataset_indicator is not None:
        assert len(dataset_indicator) == len(y)
        y = y[dataset_indicator]
        t = t[dataset_indicator]
        subgroup_indicator = subgroup_indicator[dataset_indicator]
    else:
        pass
    
    # Compute Neyman estimates for the ATE and subgroup CATE
    ATE_ = get_Neyman_ATE(y, t)
    CATE_ = get_subgroup_CATE(y, t, subgroup_indicator)
    
    # Compute y, t vectors for both the subgroup and its complement.
    y_sg = y[subgroup_indicator]
    t_sg = t[subgroup_indicator]
    y_rest = y[~subgroup_indicator]
    t_rest = t[~subgroup_indicator]
    
    # Compute y_obs vectors and their lengths for both the subgroup and
    # its complement
    y0_obs_sg = y_sg[t_sg==0]
    y1_obs_sg = y_sg[t_sg==1]
    y0_obs_rest = y_rest[t_rest==0]
    y1_obs_rest = y_rest[t_rest==1]
    N_c_sg = len(y0_obs_sg)
    N_t_sg = len(y1_obs_sg)
    N_c_rest = len(y0_obs_rest)
    N_t_rest = len(y1_obs_rest)
    N_c = N_c_sg + N_c_rest
    N_t = N_t_sg + N_t_rest
    
    # Guard against zero counts; if any denominator would be zero, return NaN
    if (N_c == 0 or N_t == 0 or
        N_c_sg == 0 or N_t_sg == 0 or
        N_c_rest == 0 or N_t_rest == 0):
        return np.nan

    # Try computing the variance estimate, if it fails (e.g., small sample size), return NaN
    try:
        var_sg_ = (
            np.var(y0_obs_sg, ddof=1) * N_c_sg * (1 / N_c_sg - 1 / N_c) ** 2
            + np.var(y1_obs_sg, ddof=1) * N_t_sg * (1 / N_t_sg - 1 / N_t) ** 2
        )
        var_rest_ = (
            np.var(y0_obs_rest, ddof=1) * N_c_rest / (N_c ** 2)
            + np.var(y1_obs_rest, ddof=1) * N_t_rest / (N_t ** 2)
        )
    except Exception:
        return np.nan

    var_total = var_sg_ + var_rest_
    if not np.isfinite(var_total) or var_total <= 0:
        return np.nan

    t_statistic = (CATE_ - ATE_) / np.sqrt(var_total)
    
    return t_statistic


def get_relative_risk(y, t):
    """
    Compute the plug-in estimate for relative risk of a treatment.
    
    Parameters
    ----------
    y: array-like of shape (n_samples,)
       Vector of observed responses
    t: array-like of shape (n_samples,)
       Vector of treatment assignments taking values in {0,1}
    
    Returns
    -------
    RR_: float
       The plug-in estimate for relative risk.
    """
    Y0_obs = y[t==0]
    Y1_obs = y[t==1]
    n0 = len(Y0_obs)
    n1 = len(Y1_obs)
    # If either group is empty, RR is undefined
    if n0 == 0 or n1 == 0:
        return np.nan

    # Treat y as binary event indicator; count events
    x0 = Y0_obs.sum()
    x1 = Y1_obs.sum()
    f0 = n0 - x0  # non-events in control
    f1 = n1 - x1  # non-events in treated

    # Apply Haldane–Anscombe correction when any cell is zero to avoid division by zero
    if (x0 == 0) or (x1 == 0) or (f0 == 0) or (f1 == 0):
        x0 = x0 + 0.5
        x1 = x1 + 0.5
        n0 = n0 + 1.0
        n1 = n1 + 1.0

    p0 = x0 / n0
    p1 = x1 / n1
    RR_ = p1 / p0

    return float(RR_)

def get_relative_risk_CI(y, t):
    """
    Compute a 95% CI for the plug-in estimate for relative risk of a treatment.
    
    Parameters
    ----------
    y: array-like of shape (n_samples,)
       Vector of observed responses
    t: array-like of shape (n_samples,)
       Vector of treatment assignments taking values in {0,1}
    
    Returns
    -------
    RR_CI: array-like of shape (2,)
       A 95% CI for the plug-in estimate for relative risk.
    """
    # Build counts
    Y0_obs = y[t==0]
    Y1_obs = y[t==1]
    n0 = len(Y0_obs)
    n1 = len(Y1_obs)

    # If either group is empty, CI is undefined
    if n0 == 0 or n1 == 0:
        return np.array([np.nan, np.nan], dtype=float)

    x0 = Y0_obs.sum()
    x1 = Y1_obs.sum()
    f0 = n0 - x0
    f1 = n1 - x1

    # Apply Haldane–Anscombe correction when any cell is zero
    if (x0 == 0) or (x1 == 0) or (f0 == 0) or (f1 == 0):
        x0 = x0 + 0.5
        x1 = x1 + 0.5
        n0 = n0 + 1.0
        n1 = n1 + 1.0

    # Compute RR and its log-variance using delta method
    p0 = x0 / n0
    p1 = x1 / n1
    RR_ = p1 / p0

    # Guard against numerical issues
    if RR_ <= 0 or not np.isfinite(RR_):
        return np.array([np.nan, np.nan], dtype=float)

    log_RR_var = 1.0/x0 - 1.0/n0 + 1.0/x1 - 1.0/n1
    if not np.isfinite(log_RR_var) or log_RR_var < 0:
        return np.array([np.nan, np.nan], dtype=float)

    z = 1.96
    log_RR = np.log(RR_)
    log_RR_CI = np.array((log_RR - z * np.sqrt(log_RR_var),
                          log_RR + z * np.sqrt(log_RR_var)))
    RR_CI = np.exp(log_RR_CI)

    return RR_CI