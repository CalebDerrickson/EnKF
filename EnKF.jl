using LinearAlgebra, Statistics, Distributions
using VecchiaMLE
using DifferentialEquations, SparseArrays
using Plots

include("utils.jl")
include("BabyKF.jl")
include("forward_euler.jl")
include("integro.jl")
include("main.jl")

main()