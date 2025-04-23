using LinearAlgebra, Statistics, Distributions
using VecchiaMLE
using DifferentialEquations

include("utils.jl")
include("BabyKF.jl")
include("forward_euler.jl")
include("main.jl")

main()