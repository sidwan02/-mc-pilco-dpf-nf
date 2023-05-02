def nll_loss(particles_update_nf, jac, prior_distribution):
    # Step 1: Calculate the log-likelihood of the particles_update_nf under the prior distribution
    log_likelihood = prior_distribution.log_prob(particles_update_nf)
    
    # Step 2: Subtract the Jacobian (log_det) from the log-likelihood
    log_likelihood_with_jac = log_likelihood - jac

    # Step 3: Negate the difference to get the negative log-likelihood
    nll = -log_likelihood_with_jac.mean()

    return nll