function BabyKF(xf, y, H, R, infl, rho, ptGrid::AbstractVector, observe_index::AbstractVector, localize, k)
    num_states, N = size(xf) # 2500 x 50 
    num_observation = size(y, 1)

    y_perturbed = repeat(y, 1, N) + sqrt.(R) * randn(num_observation, N)

    xfm = mean(xf, dims=2)
    xf_dev = (1 / sqrt(N-1)) * (xf - repeat(xfm, 1, N))
    xf = sqrt(N - 1) * infl * xf_dev + repeat(xfm, 1, N)

    observable = xf[observe_index, :]
    inn = y_perturbed - observable
    
    if localize
        rhoPH = rho[:, observe_index]
        rhoHPHt = rho[observe_index, observe_index]

        zb = observable
        zbm = mean(zb, dims=2) 
        Zb = (1 / sqrt(N - 1)) * (zb .- repeat(zbm, 1, N))

        K = (rhoHPHt .* (Zb * Zb') + R) \ inn
        xa = xf + rhoPH .* (xf_dev * Zb') * K

    else

        xf_mat = Matrix{Float64}(xf')
        n = Int(sqrt(size(xf_mat, 2)))
        
        # Idea is making the samples mean zero. Seems better, but idk how it changes the calculations
        #xfm = mean(xf_mat, dims = 2)
        #xf_mat .-= repeat(xfm, 1, num_states)
        
        input = VecchiaMLEInput(n, k, xf_mat, N, 5, 1; ptGrid=ptGrid)
        #d , L = VecchiaMLE_Run(input)
        L = Matrix{Float64}(I, N*N, N*N)
        #println("\ngradient: ", d.normed_grad_value)
        #println("constran: ", d.normed_constraint_value)
        #println("argmax: ", argmax(L), "indmax: ", L[argmax(L)])

        t1 = H' * (R \ H)

        # Inner part of the Kalman filter product
        # I.e., ( Rₖ - Hₖ(Bₖ⁻¹ + HₖᵀRₖ⁻¹Hₖ)⁻¹Hₖᵀ )

        inner = H * ((L*L' + t1) \ H')
        inner .= R - inner
        
        # outer part of Kalman filter product
        # I.e., BₖHₖᵀRₖ⁻¹(inner)Rₖ⁻¹
        inner .= (R*R) \ inner
        
        # Now need to map from 100 x 100 to 2500 x 100. 
        K = zeros(num_states, num_observation)
        #K = view(temp, :, observe_index)
        #fill!(K, 0.0)

        view(K, observe_index, :) .= inner

        K .= L \ K
        K .= L' \ K 
        

        # now to apply the Kalman filter
        xa = xf + K * inn
    end
    
    return xa
end

