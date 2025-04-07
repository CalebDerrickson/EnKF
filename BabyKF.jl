function BabyKF(xf, y, H, R, infl, rho, observe_index, localize=false)
    num_states, N = size(xf)
    num_observation = size(y, 1)

    # Since Julia 0.7 sqrt of a matrix is sqrtm in Matlab
    y_perturbed = repeat(y, 1, N) + sqrt.(R) * randn(num_observation, N)

    xfm = mean(xf, dims=2)
    xf_dev = (1 / sqrt(N-1)) * (xf - repeat(xfm, 1, N))
    xf = sqrt(N - 1) * infl * xf_dev + repeat(xfm, 1, N)

    observable = H * xf
    inn = y_perturbed - observable
    
    if localize
        rhoPH = rho[:, observe_index]
        rhoHPHt = rho[observe_index, observe_index]

        println("rhoPH size: ", size(rhoPH))
        println("rhoHPHt size: ", size(rhoHPHt))
        println("zb size: ", size(observable))

        zb = observable
        zbm = mean(zb, dims=2) 
        Zb = (1 / sqrt(N - 1)) * (zb .- repeat(zbm, 1, N))
        
        K = (rhoHPHt .* (Zb * Zb') + R) \ inn
        xa = xf + rhoPH .* (xf_dev * Zb') * K
        return xa

    else
        # Implement VecchiaMLE here. H is the linear mapping from state space to observation locations (xyGrid -> ptGrid). 
        #n = cld(sqrt(N)), k = 10, samples = ensemble (xf), number_of_samples, ptGrid  
        ptGrid = VecchiaMLE.generate_safe_xyGrid(N)
        obs_pts = ptGrid[observe_index]

        zb = observable .- mean(observable, dims=2)
        input = VecchiaMLEInput(size(observable, 1), 10, zb', size(zb, 2), 5, 1; ptGrid=obs_pts)
        _, L = VecchiaMLE_Run(input)
        L_inv = inv(L)

        K = (L_inv' * L_inv + R) \ inn
        xa = xf + (xf_dev * L_inv') * K
    end
    

    return xf
end

