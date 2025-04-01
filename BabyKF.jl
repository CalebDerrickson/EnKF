function BabyKF(xf, y, H, R, infl, rho, observe_index, localize=false)
    num_states, N = size(xf)
    num_observation = size(y, 1)

    # Since Julia 0.7 sqrt of a matrix is sqrtm in Matlab
    y_perturbed = repeat(y, 1, N) + sqrt.(R) * randn(num_observation, N)

    xfm = mean(xf, dims=2) # dims=2 might be wrong?
    xf_dev = (1 / sqrt(N-1)) * (xf - repeat(xfm, 1, N))
    xf = sqrt(N - 1) * infl * xf_dev + repeat(xfm, 1, N)

    observable = H * xf
    inn = y_perturbed - observable

    if localize
        rhoPH = rho[:, observe_index]
        rhoHPHt = rho[observe_index, observe_index]
    else
        # Implement VecchiaMLE here. H is the linear mapping from state space to observation locations (xyGrid -> ptGrid). 
        #n = cld(sqrt(N)), k = 10, samples = ensemble (xf), number_of_samples, ptGrid  
        rhoPH = ones(num_states, num_observation)
        rhoHPHt = ones(num_observation, num_observation)
    end
    
    zb = observable
    zbm = mean(zb, dims=2) # dims=2 might be wrong?
    Zb = (1 / sqrt(N - 1)) * (zb .- repeat(zbm, 1, N))
    
    K = (rhoHPHt .* (Zb * Zb') + R) \ inn
    xa = xf + rhoPH .* (xf_dev * Zb') * K
    return xa
end

