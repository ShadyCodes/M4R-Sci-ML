using ForwardDiff, LinearAlgebra
import ForwardDiff: derivative, gradient, hessian, Dual, jacobian

function banded_det(x) # x is a vector of polynomial coefficients
    range = -10.0:10
    function f
        
    end
    for i = 1:length(x)
        f .+= x[i]*range.^(i-1)
    end
    M = SymTridiagonal(f, ones(length(range)-1))
    det(M) 
end

jacobian(banded_det, [0,1,2])