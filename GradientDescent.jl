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


### Reverse mode
using Zygote, ChainRules, ChainRulesCore
include("symtridiagonal.jl")


function banded_eigvals_rev(x)
    f = -10.0:10.0 # set the range
    V = f .^ (0:length(x)-1)'
    M = V*x
    M = SymTridiagonal(M, ones(length(f)-1))
    eigvals(M)  # compute the eigenvalues
end

function update_weights_rev(x, y, lrate)
    function loss(x)
        return norm(banded_eigvals_rev(x) - y)
    end
    x .-= Zygote.gradient(loss, x)[1] .* lrate
end

y = banded_eigvals([0,0,1])
x = [0.0,0.0,0.0]
for i = 1:999
    x = update_weights_rev(x, y, 0.001)
end