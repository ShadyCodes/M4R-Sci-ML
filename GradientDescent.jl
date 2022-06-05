using ForwardDiff, LinearAlgebra
import ForwardDiff: gradient

### The below approach does not work with Zygote as Zygote does not support mutating arrays
function banded_eigvals(x) # x is a vector of polynomial coefficients
    range = -10.0:10.0
    f = zeros(eltype(x), length(range)) # vector of 0s for the diagonal
    for i = 1:length(x)
        f .+= x[i] .* range .^ (i - 1)  # at each iteration, add on the contribution of the (i-1)th term
    end
    M = SymTridiagonal(f, ones(eltype(x), length(range) - 1)) # generate the potential matrix
    eigvals(M)
end

function update_weights(x, y, lrate)
    function loss(x)
        return norm(banded_eigvals(x) - convert(Vector{eltype(x)}, y))
    end
    x .-= gradient(loss, x) .* lrate
end

y = banded_eigvals([0,0,1])
x = [0.0,0.0,0.0]
for i = 1:999
    x = update_weights(x, y, 0.001)
end



### Reverse mode attempts
using Zygote, SpecialMatrices, ChainRules, ChainRulesCore
include("symtridiagonal.jl")

# Pullback for the Polynomial constructor (in case of future use)
function ChainRulesCore.rrule(::Type{<:Polynomial}, x::AbstractVector)
    p = Polynomial(x)
    function Polynomial_pullback(p̄)
        Δp = unthunk(p̄)
        Δx = Δp.coeffs
        return (NoTangent(), Δx)
    end
    return (p, Polynomial_pullback)
end

# Pullback for the Vandermonde constructor
function ChainRulesCore.rrule(::Type{Vandermonde}, c::AbstractVector{T}) where T
    V = Vandermonde(c)
    function Vandermonde_pullback(V̄)
        ΔV = unthunk(V̄)
        Δc = ΔV[:,2]
        return (NoTangent(), Δc)
    end
    return (V, Vandermonde_pullback)
end


# Closest attempt at function to generate banded matrix from polynomial coefficients
function banded_eigvals_rev(x)
    f = collect(-10.0:10.0) # set the range
    V = Vandermonde(x)   # construct Vandermonde matrix using SpecialMatrices.jl
    M = # what next?
    eigvals(M)  # compute the eigenvalues
end

function update_weights_rev(x, y, lrate)
    function loss(x)
        return norm(banded_eigvals_rev(x) - y)  # norm of the difference between the eigenvalues and the target
    end
    x .-= Zygote.gradient(loss, x) .* lrate
end

y = banded_eigvals([0,0,1])
x = [0.0,0.0,0.0]
for i = 1:999
    x = update_weights_rev(x, y, 0.001)
end

function test2(x)
    V = Vandermonde(x)
    norm(V)
end

Zygote.gradient(test2, [0.,0.,4.])