using ForwardDiff, LinearAlgebra
import ForwardDiff: gradient

function banded_eigvals(x) # x is a vector of polynomial coefficients
    range = -10.0:10
    f = zeros(eltype(x), length(range))
    for i = 1:length(x)
        f .+= x[i] .* range .^ (i - 1)
    end
    M = SymTridiagonal(f, ones(eltype(x), length(range) - 1))
    eigvals(M)
end

function update_weights(x, y, lrate)
    function loss(x)
        norm(banded_eigvals(x) - convert(Vector{eltype(x)}, y))
    end
    x .-= gradient(loss, x) .* lrate
end

y = banded_eigvals([0,0,1])
x = [0.0,0.0,0.0]
for i = 1:1000
    x = update_weights(x, y, 0.001)
    println(norm(banded_eigvals(x)-y))
end
println(x)