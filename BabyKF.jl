function BabyKF(xf, y, H, R, infl, rho, ptGrid::AbstractVector, observe_index::AbstractVector, localize::Bool, k, rep, L)
    num_states, N = size(xf) # 2500 x 50 
    num_observation = size(y, 1)

    y_perturbed = repeat(y, 1, N) + sqrt.(R) * (randn(num_observation, N) .- 0.5)

    xfm = mean(xf, dims=2)
    xf_dev = (1 / sqrt(N-1)) * (xf .- repeat(xfm, 1, N))
    xf .= sqrt(N - 1) * sqrt(infl) * xf_dev + repeat(xfm, 1, N)
    inn = y_perturbed .- view(xf, observe_index, :)
    println("Ensemble rank: ", rank(xf_mat))

    if localize
        rhoPH = rho[:, observe_index]
        rhoHPHt = rho[observe_index, observe_index]

        zb = xf[observe_index, :]
        zbm = mean(zb, dims=2) 
        Zb = (1 / sqrt(N - 1)) * (zb .- repeat(zbm, 1, N))

        K = (rhoHPHt .* (Zb * Zb') + R) \ inn
        xf .+= rhoPH .* (xf_dev * Zb') * K

        return xf

    else
        # Have to transpose them since that's how VecciaMLE parses them. 
        xf_mat = Matrix{Float64}(xf')
        xfm = mean(xf_mat, dims = 2)
        xf_mat .-= repeat(xfm, 1, num_states)
        xf_mat ./= sqrt(N-1)

        n = Int(sqrt(size(xf_mat, 2)))

        input = VecchiaMLEInput(n, k, xf_mat, N, 3, 1; ptGrid=ptGrid)
        L .= VecchiaMLE_Run(input)[2]
        L .*= sqrt(N-1)

        rep .= L' \ (L \ H')
        xf .+= (rep * ((view(rep, observe_index, :).+R) \ inn))
        return xf
    end
    
    
end

