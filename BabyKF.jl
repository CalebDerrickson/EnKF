function BabyKF(xf, y, H, R, infl, rho, ptGrid::AbstractVector, observe_index::AbstractVector, localize, k)#, rep, L)
    num_states, N = size(xf) # 2500 x 50 
    num_observation = size(y, 1)

    y_perturbed = repeat(y, 1, N) + sqrt.(R) * randn(num_observation, N)

    xfm = mean(xf, dims=2)
    xf_dev = (1 / sqrt(N-1)) * (xf - repeat(xfm, 1, N))
    #xf = sqrt(N - 1) * infl * xf_dev + repeat(xfm, 1, N)

    # Have to transpose them since that's how VecciaMLE parses them. 
    xf_mat = Matrix{Float64}(xf')
    xfm = mean(xf_mat, dims = 2)
    xf_mat .-= repeat(xfm, 1, num_states)

    inn = y_perturbed .- view(xf, observe_index, :)
    
    if localize
        rhoPH = rho[:, observe_index]
        rhoHPHt = rho[observe_index, observe_index]

        zb = xf[observe_index, :]
        zbm = mean(zb, dims=2) 
        Zb = (1 / sqrt(N - 1)) * (zb .- repeat(zbm, 1, N))

        K = (rhoHPHt .* (Zb * Zb') + R) \ inn
        xf .+= rhoPH .* (xf_dev * Zb') * K

    else
        # First check rank of samples matrix
        #println("Samples rank: ", rank(xf_mat))
        
        n = Int(sqrt(size(xf_mat, 2)))

        input = VecchiaMLEInput(n, k, xf_mat, N, 5, 1; ptGrid=ptGrid)
        d, L = VecchiaMLE_Run(input)
        
        println("\ngradient: ", d.normed_grad_value)
        println("constran: ", d.normed_constraint_value)
        println("argmax: ", argmax(L), "indmax: ", L[argmax(L)])
        
        rep = L' \ (L \ H')
        xf_mat .+= (rep * ((view(rep, observe_index, :).+R) \ inn))'
        return xf_mat'
        #temp = spzeros(N*N, N*N)
        ## Sherman Morrison Woodburry on Kalman filter
        #view(temp, observe_index, observe_index) .= spdiagm(0 => 1.0 ./ diag(R))
        ## Inner part of the Kalman filter product
        ## I.e., ( Rₖ - Hₖ(Bₖ⁻¹ + HₖᵀRₖ⁻¹Hₖ)⁻¹Hₖᵀ )
        #inner = R .- H \ ((L * L' .+ temp) \ H')
        ## outer part of Kalman filter product
        ## I.e., BₖHₖᵀRₖ⁻¹(inner)Rₖ⁻¹
        #inner .= (temp.^2) * inner
        ## Now need to map from 100 x 100 to 2500 x 100. 
        #K = spzeros(num_states, num_observation)
        ##K = view(temp, :, observe_index)
        ##fill!(K, 0.0)
        #view(K, observe_index, :) .= inner
        #K .= L' \ (L \ K) 
        # now to apply the Kalman filter
        #xf .+= K * inn 
    end
    
    return xf
end

