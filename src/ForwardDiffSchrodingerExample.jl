using ForwardDiff, LinearAlgebra
import ForwardDiff: derivative, gradient, hessian, Dual, jacobian

function banded_eigvals(x) # x is a vector of polynomial coefficients
    range = -10.0:10
    f = zeros(eltype(x), length(range))
    for i = 1:length(x)
        f .+= x[i] .* range .^ (i - 1)
    end
    M = SymTridiagonal(f, ones(eltype(x), length(range) - 1))
    eigvals(M)
end

banded_eigvals([0.0, 1, 2])

jacobian(banded_eigvals, [0.0, 1, 2])
h = 0.00001

(banded_eigvals([0.0, 1 + h, 2]) - banded_eigvals([0.0, 1, 2])) / h


(banded_eigvals([0 + h, 1, 2]) - banded_eigvals([0, 1, 2])) / h
(banded_eigvals([0, 1 + h, 2]) - banded_eigvals([0, 1, 2])) / h
(banded_eigvals([0, 1, 2 + h]) - banded_eigvals([0, 1, 2])) / h

