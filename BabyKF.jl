function BabyKF(xf, y, H, R, infl, rho, observe_index::AbstractVector, localize, k)
    num_states, N = size(xf)
    num_observation = size(y, 1)

    # Since Julia 0.7 sqrt of a matrix is sqrtm in Matlab
    y_perturbed = repeat(y, 1, N) + sqrt.(R) * randn(num_observation, N)

    xfm = mean(xf, dims=2)
    xf_dev = (1 / sqrt(N-1)) * (xf - repeat(xfm, 1, N))
    #xf = sqrt(N - 1) * infl * xf_dev + repeat(xfm, 1, N)

    observable = H * xf
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
        Number_of_Samples = size(xf_mat, 1)


        ptGrid = VecchiaMLE.generate_safe_xyGrid(n)
        input = VecchiaMLEInput(n, k, xf_mat, Number_of_Samples, 5, 1; ptGrid=ptGrid)
        d, L = VecchiaMLE_Run(input)
        R_inv = inv(R)
        #println("L cond: ", cond(L))

        #println("constraint: ", d.normed_constraint_value)
        #println("gradient: ", d.normed_grad_value)

        temp = zeros(N^2, N^2)
        t1 = H' * R_inv * H
        #view(temp, observe_index, observe_index) .= R_inv


        # Inner part of the Kalman filter product
        # I.e., ( Rₖ - Hₖ(Bₖ⁻¹ + HₖᵀRₖ⁻¹Hₖ)⁻¹Hₖᵀ )
        # The inv() is really ugly, and needs to be changed.
        inner = H * ((L*L' + t1) \ H')
        inner .= R - inner
        #println("inner cond after inverse: ", cond(inner))
        
        # outer part of Kalman filter product
        # I.e., BₖHₖᵀRₖ⁻¹(inner)Rₖ⁻¹
        inner = R_inv * inner * R_inv

        # check condition number of inner
        #println("inner cond: ", cond(inner))

        # Now need to map from 100 x 100 to 2500 x 100. 
        outer = view(temp, :, observe_index)
        fill!(outer, 0.0)

        view(outer, observe_index, :) .= inner

        outer = L \ outer
        outer = L' \ outer 
        
        K = outer
        #println("K cond: ", cond(K))

        # now to apply the Kalman filter
        xa = xf + K * inn
    end
    

    return xa
end

