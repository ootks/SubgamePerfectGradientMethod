# Abstract superclass for a number of first order methods
# Should implement
# initialize(method, start::Vector{Float64})
# update(method, value::Float64, gradient::Vector{Float64})
# guarantee(method)
# methodTitle(method)
abstract type FOM end

# Abstract superclass for first order oracles
# Should implement
# (q::Oracle)(x::Vector{Float64})
# gradient(q::Oracle, x::Vector{Float64})
# smoothness(q::Oracle)
abstract type Oracle end
