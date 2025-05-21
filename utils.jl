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


function cal_rho(loc_rad::Float64, num_states::Int, f, N, Lx, Ly)::Matrix{Float64}

    x = LinRange(0.0, Lx, N)
    y = LinRange(0.0, Ly, N)

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

struct ADParams
    N::Int               # grid size in each direction
    dx::Float64          # grid spacing in x
    dy::Float64          # grid spacing in y
    cx::Float64          # advection speed in x
    cy::Float64          # advection speed in y
    nu::Float64          # diffusion coefficient
    dU::Matrix{Float64}  # pre‐allocated N×N buffer for ∂U/∂t
end

@enum Strategy begin
    localization
    OneVecchia
    TwoVecchia
end

function getKLDivergence(xf, L)
    mat = (1.0 / (size(xf, 1) - 1)) * sum(xf[i, :] * xf[i, :]' for i in size(xf, 1))
    return VecchiaMLE.KLDivergence(mat, L)
end


struct CSCPattern
    rowval::Vector{Int}
    colval::Vector{Int}
    colptr::Vector{Int}
end

mutable struct PatternCache
    L::Union{Nothing, CSCPattern}
    S::Union{Nothing, CSCPattern}
end

function get_csc_pattern(A::SparseMatrixCSC)
    rows = A.rowval
    colptr = A.colptr
    col_indices = Vector{Int}(undef, length(rows))
    for j in 1:length(colptr)-1
        for k in colptr[j]:colptr[j+1]-1
            col_indices[k] = j
        end
    end
    return (rows = rows, cols = col_indices, colptr = colptr)
end

function cache_pattern!(mat::SparseMatrixCSC, which::Symbol, PATTERN_CACHE::PatternCache)
    I, J, P = get_csc_pattern(mat)
    pattern = CSCPattern(I, J, P)
    if which === :L
        PATTERN_CACHE.L = pattern
    elseif which === :S
        PATTERN_CACHE.S = pattern
    else
        error("Unknown matrix identifier: $which")
    end
end