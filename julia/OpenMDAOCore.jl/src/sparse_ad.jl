# Sparse type definitions.
#
# This file declares the sparse-related *types* — `SparseADExplicitComp` and
# `PerturbedDenseSparsityDetector` — plus trivial field accessors that do not
# require `SparseArrays` or `SparseMatrixColorings`.
#
# All *functionality* that actually needs `SparseArrays`/`SparseMatrixColorings`
# (constructors, `compute_partials!`, `setup_partials`, sparsity detection,
# `ca2strdict_sparse`, `get_rows_cols_dict_from_sparsity`, the sparse overloads
# of `_maybe_nonzeros`, etc.) lives in the `OpenMDAOCoreSparseMatrixColoringsExt`
# extension, which loads only when both weakdeps are available.
#
# Splitting types-from-functionality mirrors the OpenMDAO4Core.jl pattern (where
# `SparseFlavor` lives in the main package and its methods live in the
# extension). The benefit: `using OpenMDAOCore: SparseADExplicitComp,
# PerturbedDenseSparsityDetector` always works, and the types can be used for
# `isa`/`typeof` dispatch even without the extension loaded.

"""
    SparseADExplicitComp{InPlace,TAD,TCompute,TX,TY,TJ,TPrep,TXCS,TYCS} <: AbstractADExplicitComp{InPlace}

An `<:AbstractADExplicitComp` for sparse Jacobians.

# Fields
* `ad_backend::TAD`: `<:ADTypes.AutoSparse` automatic differentation "backend" library
* `compute_adable::TCompute`: function of the form `compute_adable(Y, X)` compatible with DifferentiationInterface.jl that performs the desired computation, where `Y` and `X` are `ComponentVector`s of outputs and inputs, respectively
* `X_ca::ComponentVector`: `ComponentVector` of inputs
* `Y_ca::ComponentVector`: `ComponentVector` of outputs
* `J_ca_sparse::ComponentMatrix`: Sparse `ComponentMatrix` of the Jacobian of `Y_ca` with respect to `X_ca`
* `units_dict::Dict{Symbol,String}`: mapping of variable names to units. Can be an empty `Dict` if units are not desired.
* `tags_dict::Dict{Symbol,Vector{String}`: mapping of variable names to `Vector`s of `String`s specifing variable tags.
* `shape_by_conn_dict::Dict{Symbol,Bool}`: mapping of variable names to `Bool` indicating if the variable shape should be determined dynamically by a connection.
* `prep::DifferentiationInterface.JacobianPrep`: `DifferentiationInterface.jl` "preparation" object
* `rcdict`: `Dict{Tuple{Symbol,Sympol}, Tuple{Vector{Int}, Vector{Int}}` mapping sub-Jacobians of the form `(:output_name, :input_name)` to `Vector`s of non-zero row and column indices (1-based)
* `X_ca::ComponentVector`: `ComplexF64` version of `X_ca` (for the complex-step method)
* `Y_ca::ComponentVector`: `ComplexF64` version of `Y_ca` (for the complex-step method)
"""
struct SparseADExplicitComp{InPlace,TAD,TCompute,TX,TY,TJ,TPrep,TXCS,TYCS} <: AbstractADExplicitComp{InPlace}
    ad_backend::TAD
    compute_adable::TCompute
    X_ca::TX
    Y_ca::TY
    J_ca_sparse::TJ
    prep::TPrep
    rcdict::Dict{Tuple{Symbol,Symbol}, Tuple{Vector{Int},Vector{Int}}}
    units_dict::Dict{Symbol,String}
    tags_dict::Dict{Symbol,Vector{String}}
    shape_by_conn_dict::Dict{Symbol,Bool}
    copy_shape_dict::Dict{Symbol,Symbol}
    X_ca_cs::TXCS
    Y_ca_cs::TYCS

    function SparseADExplicitComp{InPlace}(ad_backend, compute_adable, X_ca, Y_ca, J_ca_sparse, prep, rcdict, units_dict, tags_dict, shape_by_conn_dict, copy_shape_dict, X_ca_cs, Y_ca_cs) where {InPlace}
        return new{
                                    InPlace, typeof(ad_backend), typeof(compute_adable), typeof(X_ca), typeof(Y_ca),
                                    typeof(J_ca_sparse), typeof(prep),
                                    typeof(X_ca_cs), typeof(Y_ca_cs)}(ad_backend,
                                                              compute_adable, X_ca,
                                                              Y_ca, J_ca_sparse,
                                                              prep, rcdict,
                                                              units_dict,
                                                              tags_dict,
                                                              shape_by_conn_dict,
                                                              copy_shape_dict,
                                                              X_ca_cs, Y_ca_cs)
    end
end

function SparseADExplicitComp{false}(ad_backend, compute_adable, X_ca, J_ca_sparse, prep, rcdict, units_dict, tags_dict, shape_by_conn_dict, copy_shape_dict, X_ca_cs)
    Y_ca = nothing
    Y_ca_cs = nothing
    return SparseADExplicitComp{false}(ad_backend, compute_adable, X_ca, Y_ca, J_ca_sparse, prep, rcdict, units_dict, tags_dict, shape_by_conn_dict, copy_shape_dict, X_ca_cs, Y_ca_cs)
end

# Trivial field accessors that do not require SparseArrays/SparseMatrixColorings.
# (The `compute_partials!`/`setup_partials`/constructor *methods* for this type
# live in the `OpenMDAOCoreSparseMatrixColoringsExt` extension.)
get_rows_cols_dict(comp::SparseADExplicitComp) = comp.rcdict
get_jacobian_ca(comp::SparseADExplicitComp) = comp.J_ca_sparse

has_setup_partials(self::SparseADExplicitComp) = true
has_compute_partials(self::SparseADExplicitComp) = true
has_compute_jacvec_product(self::SparseADExplicitComp) = false

function get_rows_cols_dict_from_sparsity end
function ca2strdict_sparse end

"""
    PerturbedDenseSparsityDetector

Tweaked version of [`DenseSparsityDetector`](https://gdalle.github.io/DifferentiationInterface.jl/DifferentiationInterface/stable/api/#DifferentiationInterface.DenseSparsityDetector) sparsity pattern detector satisfying the [detection API](https://sciml.github.io/ADTypes.jl/stable/#Sparse-AD) of [ADTypes.jl](https://github.com/SciML/ADTypes.jl) that evaluates the Jacobian multiple times using a perturbed input vector.
Specifically, input vector `x` will be perturbed via

```julia
    x_perturb = (1 .+ rel_x_perturb.*perturb1).*x .+ perturb2.*abs_x_perturb
```

where `perturb1` and `perturb2` are random `Vector`s of numbers ranging from `-0.5` to `0.5`, and `rel_x_perturb` and `abs_x_perturb` are relative and absolute perturbation magnitudes specified by the user.

All of the caveats associated with the performance of `DenseSparsityDetector` apply to `PerturbedDenseSparsityDetector`, since it essentially does the same thing as `DenseSparsityDetector` multiple times.
The nonzeros in a Jacobian or Hessian are detected by computing the relevant matrix with _dense_ AD, and thresholding the entries with a given tolerance (which can be numerically inaccurate).
This process can be very slow, and should only be used if its output can be exploited multiple times to compute many sparse matrices.

!!! danger
    In general, the sparsity pattern you obtain can depend on the provided input `x`. If you want to reuse the pattern, make sure that it is input-agnostic.
    Perturbing the input vector should hopefully guard against getting "unlucky" and finding zero Jacobian entries that aren't actually zero for all `x`, but is of course problem-dependent.

# Fields

- `backend::AbstractADType` is the dense AD backend used under the hood
- `atol::Float64` is the minimum magnitude of a matrix entry to be considered nonzero
- `nevals::Int=3` is the number of times the Jacobian will be evaluated using the perturbed input `x`
- `rel_x_perturb=0.001`: is the relative magnitude of the `x` perturbation.

# Constructor

    PerturbedDenseSparsityDetector(backend; atol, method=:iterative, nevals=3, rel_x_perturb=0.001, abs_x_perturb=0.0001)

The keyword argument `method::Symbol` can be either:

- `:iterative`: compute the matrix in a sequence of matrix-vector products (memory-efficient)
- `:direct`: compute the matrix all at once (memory-hungry but sometimes faster).

Note that the constructor is type-unstable because `method` ends up being a type parameter of the `PerturbedDenseSparsityDetector` object (this is not part of the API and might change).

"""
struct PerturbedDenseSparsityDetector{method,B,TRelXPerturb,TAbsXPerturb} <: ADTypes.AbstractSparsityDetector
    backend::B
    atol::Float64
    nevals::Int
    rel_x_perturb::TRelXPerturb
    abs_x_perturb::TAbsXPerturb
end

function Base.show(io::IO, detector::PerturbedDenseSparsityDetector{method}) where {method}
    (; backend, atol, nevals, rel_x_perturb, abs_x_perturb) = detector
    return print(
        io,
        PerturbedDenseSparsityDetector,
        "(",
        repr(backend; context=io),
        "; atol=$atol, method=",
        repr(method; context=io),
        "nevals=$nevals, rel_x_perturb=$rel_x_perturb", "abs_x_perturb=$abs_x_perturb",
        ")",
    )
end

function PerturbedDenseSparsityDetector(
    backend::ADTypes.AbstractADType; atol::Float64, method::Symbol=:iterative, nevals=3, rel_x_perturb=0.001, abs_x_perturb=0.0001
)
    if !(method in (:iterative, :direct))
        throw(
            ArgumentError("The keyword `method` must be either `:iterative` or `:direct`.")
        )
    end

    if nevals < 1
        throw(
            ArgumentError("The keyword `nevals` should be > 0")
        )
    end

    return PerturbedDenseSparsityDetector{method,typeof(backend),typeof(rel_x_perturb),typeof(abs_x_perturb)}(backend, atol, nevals, rel_x_perturb, abs_x_perturb)
end
