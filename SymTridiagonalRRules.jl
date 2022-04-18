using Zygote, LinearAlgebra, ChainRules, ChainRulesCore, ChainRulesTestUtils

function ChainRulesCore.frule((_, Δdv, Δev), ::Type{SymTridiagonal}, dv::V, ev::V) where V <: AbstractVector
    return SymTridiagonal(dv,ev), SymTridiagonal(Δdv, Δev)
end

test_frule(SymTridiagonal, [0.1,0.2,0.3],ones(2))

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

test_rrule(SymTridiagonal, [0.1,0.2,0.3],ones(2))

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
<<<<<<< HEAD
    Ā = Matrix(Ā)
    if ∂U isa AbstractZero
=======
    if ∂U isa AbstractZero
        Ā = Matrix(Ā)
>>>>>>> 8073e681e2a78245a2981a77983e8a42a250dbb1
        mul!(Ā, U, real.(∂λ) .* U')
    else
        _eigen_norm_phase_rev!(∂U, A, U)
        ∂K = mul!(Ā, U', ∂U)
        ∂K ./= λ' .- λ
        ∂K[diagind(∂K)] .= real.(∂λ)
        mul!(tmp, ∂K, U')
        mul!(Ā, U, tmp)
    end
<<<<<<< HEAD
    ∂A = SymTridiagonal(diag(Ā), (diag(Ā,1) + diag(Ā,-1))/2)
=======
    ∂A = SymTridiagonal(diag(Ā), diag(Ā,1))
>>>>>>> 8073e681e2a78245a2981a77983e8a42a250dbb1
    return ∂A
end

function _eigen_norm_phase_rev!(∂V, A::SymTridiagonal, V)
    ϵ = sqrt(eps(real(eltype(V))))
    @inbounds for i in axes(V, 2)
<<<<<<< HEAD
        k = size(A,1)
        v = @view V[:, i]
        vₖ = real(v[k])
        if abs(vₖ) > ϵ
            ∂v = @view ∂V[:, i]
            ∂c = dot(v, ∂v)
            ∂v[k] -= im * (imag(∂c) / vₖ)
=======
        v = @view V[:, i]
        vₖ = real(v[1])
        if abs(vₖ) > ϵ
            ∂v = @view ∂V[:, i]
            ∂c = dot(v, ∂v)
            ∂v[1] -= im * (imag(∂c) / vₖ)
>>>>>>> 8073e681e2a78245a2981a77983e8a42a250dbb1
        end
    end
    return ∂V
end

A = SymTridiagonal([0.1,0.2,0.3],ones(2))
test_rrule(eigvals, A)
test_rrule(eigen, A)

A = c -> SymTridiagonal([c[1],c[1],c[2]+2c[3], c[2]], ones(3));
λ = c -> eigvals(A(c))
Zygote.jacobian(A, [0.1,0.2,0.3])
Zygote.jacobian(λ, [0.1,0.2,0.3])   