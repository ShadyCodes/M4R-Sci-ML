using LinearAlgebra, ChainRules, ChainRulesCore

function ChainRulesCore.frule((_, Δdv, Δev), ::Type{SymTridiagonal}, dv::V, ev::V) where V <: AbstractVector
    return SymTridiagonal(dv,ev), SymTridiagonal(Δdv, Δev)
end

function ChainRulesCore.rrule(::Type{SymTridiagonal}, dv::V, ev::V) where V <: AbstractVector
    Ω = SymTridiagonal(dv,ev)
    function SymTridiagonal_pullback(Ω̄)
        ΔΩ = unthunk(Ω̄)
        Δdv = @thunk(diag(ΔΩ))
        Δev = @thunk(diag(ΔΩ,1))
        return (NoTangent(), Δdv, Δev)
    end
    return (Ω, SymTridiagonal_pullback)
end

function ChainRulesCore.rrule(
    ::typeof(eigvals),
    A::SymTridiagonal;
    kwargs...
)
    F, eigen_back = rrule(eigen, A; kwargs...)
    λ = F.values
    function eigvals_pullback(Δλ)
        ∂F = Tangent{typeof(F)}(values = Δλ)
        _, ∂A = eigen_back(∂F)
        return (NoTangent(), ∂A)
    end
    return (λ, eigvals_pullback)
end

function  ChainRulesCore.rrule(
    ::typeof(eigen),
    A::SymTridiagonal;
    kwargs...
)
    F = eigen(A; kwargs...)
    function eigen_pullback(ΔF::Tangent)
        λ, U = F.values, F.vectors
        Δλ, ΔU = ΔF.values, ΔF.vectors
        ΔU = ΔU isa AbstractZero ? ΔU : copy(ΔU)
        ∂A = eigen_rev!(A, λ, U, Δλ, ΔU)
        return NoTangent(), ∂A
    end
    eigen_pullback(ΔF::AbstractZero) = (NoTangent(), ΔF)
    return F, eigen_pullback
end
 
function eigen_rev!(A::SymTridiagonal, λ, U, ∂λ, ∂U)
    ∂λ isa AbstractZero && ∂U isa AbstractZero && return ∂λ + ∂U
    Ā = similar(parent(A), eltype(U))
    tmp = ∂U
    Ā = Matrix(Ā)
    if ∂U isa AbstractZero
        mul!(Ā, U, real.(∂λ) .* U')
    else
        _eigen_norm_phase_rev!(∂U, A, U)
        ∂K = mul!(Ā, U', ∂U)
        ∂K ./= λ' .- λ
        ∂K[diagind(∂K)] .= real.(∂λ)
        mul!(tmp, ∂K, U')
        mul!(Ā, U, tmp)
    end
    ∂A = SymTridiagonal(diag(Ā), (diag(Ā,1) + diag(Ā,-1))/2)
    return ∂A
end