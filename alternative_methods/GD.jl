include("../fom_interface.jl")

mutable struct GD <: FOM
    x
    iteration
end

GD() = GD(missing, missing)

function initialize(method::GD, start::Vector{Float64})
    method.x = start
    method.iteration = 0
end

function update(method::GD, value::Float64, gradient::Vector{Float64})
    method.iteration += 1

    method.x = method.x - gradient

    return method.x
end

function guarantee(method::GD)

    return 1/(2*method.iteration + 1)
end

function methodTitle(method::GD)
    return "GD"
end

