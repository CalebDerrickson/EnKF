function integro(u, N, dt, kernel)

    u = reshape(u, N*N, 1)
    # Define the nonlinearity f(N)
    r = 0.0
    K = 1.0

    # Does not allow for just linear term at r = 0
    # f(s) = r .* s .* (1 .- s./ K)
    
    # Allows for linearity at r = 0
    f(s) = (1 + r) .* s .- (r / K) .* s.^2
    
    return u .+ dt .* (kernel * f(u))

end