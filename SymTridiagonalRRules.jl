using Zygote, LinearAlgebra, ChainRules, ChainRulesCore, ChainRulesTestUtils

function ChainRulesCore.frule((_, Δdv, Δev), ::Type{SymTridiagonal}, dv::V, ev::V) where V <: AbstractVector
    return SymTridiagonal(dv,ev), SymTridiagonal(Δdv, Δev)
end

ChainRulesTestUtils.test_frule(SymTridiagonal, [0.1,0.2,0.3],ones(2))

function ChainRulesCore.rrule(::Type{SymTridiagonal}, dv::V, ev::V) where V <: AbstractVector
    Ω = SymTridiagonal(dv,ev)
<<<<<<< HEAD
    function SymTridiagonal_pullback(Ω̄)
        ΔΩ = unthunk(Ω̄)
        Δdv = @thunk(diag(ΔΩ))
        Δev = @thunk(diag(ΔΩ,1))
=======
    function SymTridiagonal_pullback(ΔΩ)
        Δdv = @thunk(diag(unthunk(ΔΩ)))
        Δev = @thunk(diag(unthunk(ΔΩ),1))
>>>>>>> cec3e631feacd4d5aafaaef6f6a136106fc203b2
        return (NoTangent(), Δdv, Δev)
    end
    return (Ω, SymTridiagonal_pullback)
end

test_rrule(SymTridiagonal, [0.1,0.2,0.3],ones(2))

#A = c -> SymTridiagonal([c[1],c[1],c[2]+2c[3], c[2]], ones(3));
#λ = c -> eigvals(A(c))
#Zygote.jacobian(A, [0.1,0.2,0.3])
#Zygote.jacobian(λ, [0.1,0.2,0.3])