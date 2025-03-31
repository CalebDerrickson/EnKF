function gaspari_cohn(r)::Float64

    ra = r

    if 0 <= ra <= 1
        return -0.25 * ra^5 + 0.5 * ra^4 + 0.625 * ra^3 - (5/3) * ra^2 + 1.0
    elseif ra < 1.0 && ra <= 2.0
        return (1.0 / 12) * ra^5 - 0.5 * ra^4 + 0.625 * ra^3 + (5/3) * ra^2 - 5 * ra + 4.0 - (2/3) / ra
    else
        return 0.0
    end
    
end


function cal_rho(loc_rad::Float64, num_states::Int, f, Nx, Ny, Lx, Ly)::Matrix{Float64}

    x = LinRange(0.0, Lx, Nx)
    y = LinRange(0.0, Ly, Ny)

    X = [x for _ in y, x in x]
    Y = [y for y in y, _ in x]
    
    rho = zeros(num_states, num_states)

    for i in 1:num_states
        for j in 1:num_states
            dx = min(abs(X[i] - X[j]), Lx - abs(X[i] - X[j]))
            dy = min(abs(Y[i] - Y[j]), Ly - abs(Y[i] - Y[j]))
            d = sqrt(dx^2 + dy^2)

            rad = d * (1.0 / loc_rad)
            rho[i, j] = f(rad)
        end
    end
    return rho
end