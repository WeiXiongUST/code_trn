   negative_approx_kl = log_prob - old_log_prob
    ratio = torch.exp(negative_approx_kl)
    ppo_kl = verl_F.masked_mean(-negative_approx_kl, eos_mask)

    #pg_losses = -advantages * ratio
    pg_losses = -advantages * log_prob
    pg_losses2 = -advantages * torch.clamp(ratio, 1.0 - cliprange, 1.0 + cliprange)

    #pg_loss = verl_F.masked_mean(torch.max(pg_losses, pg_losses2), eos_mask)

    pg_loss = verl_F.masked_mean(pg_losses, eos_mask)
    pg_clipfrac = verl_F.masked_mean(torch.gt(pg_losses2, pg_losses).float(), eos_mask)
    return pg_loss, pg_clipfrac, ppo_kl
