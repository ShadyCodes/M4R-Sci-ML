using Zygote, LinearAlgebra, ChainRules, ChainRulesCore

function ChainRulesCore.rrule(::Type{SymTridiagonal}, dv::AbstractVector, ev::AbstractVector)
    SymTridiagonal_pullback(Δdv, Δev) = NoTangent(), SymTridiagonal(Δdv, Δev)
    return SymTridiagonal(dv,ev), SymTridiagonal_pullback
end


A = c -> Symmetric([c[1] c[1]; 0 c[2]+2c[3]]);
λ = c -> eigvals(A(c))

Zygote.jacobian(λ, [0.1,0.2,0.3])


A = c -> SymTridiagonal([c[1],c[1],c[2]+2c[3], c[2]], ones(3));
λ = c -> eigvals(A(c))
Zygote.jacobian(A, [0.1,0.2,0.3])