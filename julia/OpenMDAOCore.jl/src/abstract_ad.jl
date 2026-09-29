# Derivative "flavors" — the strategy used to compute derivatives of an
# `ADExplicitComp`. Following the OpenMDAO4Core.jl design: the flavor is a
# singleton type carried as the first type parameter of `ADExplicitComp`, and
# all flavor-specific behavior (constructors, `compute_partials!`,
# `compute_jacvec_product!`, accessors) dispatches on it.

"""
    DerivativeFlavor

Abstract supertype for the derivative computation strategy of an
[`ADExplicitComp`](@ref).

Concrete subtypes: [`DenseFlavor`](@ref), [`SparseFlavor`](@ref),
[`MatrixFreeForwardFlavor`](@ref), [`MatrixFreeReverseFlavor`](@ref).
"""
abstract type DerivativeFlavor end

"""
    AssembledFlavor <: DerivativeFlavor

Abstract supertype for derivative flavors that assemble and store a Jacobian
matrix.

Concrete subtypes: [`DenseFlavor`](@ref), [`SparseFlavor`](@ref).
"""
abstract type AssembledFlavor <: DerivativeFlavor end

"""
    DenseFlavor <: AssembledFlavor

Derivative flavor that computes and stores the full dense Jacobian matrix via
`DifferentiationInterface.jacobian!`.
"""
struct DenseFlavor <: AssembledFlavor end

"""
    MatrixFreeFlavor <: DerivativeFlavor

Abstract supertype for matrix-free derivative flavors that compute
Jacobian-vector products without materializing the Jacobian.

Concrete subtypes: [`MatrixFreeForwardFlavor`](@ref),
[`MatrixFreeReverseFlavor`](@ref).
"""
abstract type MatrixFreeFlavor <: DerivativeFlavor end

"""
    MatrixFreeForwardFlavor <: MatrixFreeFlavor

Derivative flavor that computes Jacobian-vector products (JVPs) via
`DifferentiationInterface.pushforward!`. Only `compute_jacvec_product!` with
`mode="fwd"` is supported.
"""
struct MatrixFreeForwardFlavor <: MatrixFreeFlavor end

"""
    MatrixFreeReverseFlavor <: MatrixFreeFlavor

Derivative flavor that computes vector-Jacobian products (VJPs) via
`DifferentiationInterface.pullback!`. Only `compute_jacvec_product!` with
`mode="rev"` is supported.
"""
struct MatrixFreeReverseFlavor <: MatrixFreeFlavor end

"""
    SparseFlavor <: AssembledFlavor

Derivative flavor that computes and stores the sparse Jacobian using graph
coloring, via `DifferentiationInterface.jacobian!` with an `ADTypes.AutoSparse`
backend.

The `SparseFlavor` *type* and the `SparseDerivPrep` prep struct are declared
here in the main package, but the sparse *functionality* (constructors,
`compute_partials!`, sparsity detection, etc.) is provided by the
`OpenMDAOCoreSparseMatrixColoringsExt` extension, which loads when both
`SparseArrays` and `SparseMatrixColorings` are available.
"""
struct SparseFlavor <: AssembledFlavor end

# ---------------------------------------------------------------------------
# Flavor-specific derivative-prep structs (the `deriv_prep` field of
# `ADExplicitComp`). Only the *types* live here; their construction and the
# methods that read them are defined alongside the flavor-specific code.
# ---------------------------------------------------------------------------

"""
    DenseDerivPrep{TJ, TPrep}

Derivative preparation data for [`DenseFlavor`](@ref) components.

# Fields
* `J_ca::TJ`: dense `ComponentMatrix` storing the Jacobian of outputs w.r.t. inputs
* `prep::TPrep`: `DifferentiationInterface.JacobianPrep`, or `nothing` if prep was skipped
"""
struct DenseDerivPrep{TJ,TPrep}
    J_ca::TJ
    prep::TPrep
end

"""
    MatrixFreeDerivPrep{TdX, TdY, TPrep}

Derivative preparation data for [`MatrixFreeForwardFlavor`](@ref) and
[`MatrixFreeReverseFlavor`](@ref) components.

# Fields
* `dX_ca::TdX`: input tangent/cotangent `ComponentVector` buffer
* `dY_ca::TdY`: output tangent/cotangent `ComponentVector` buffer
* `prep::TPrep`: `DifferentiationInterface.PushforwardPrep` (for
  [`MatrixFreeForwardFlavor`](@ref)) or `PullbackPrep` (for
  [`MatrixFreeReverseFlavor`](@ref)), or their `No*Prep` skip-variants
"""
struct MatrixFreeDerivPrep{TdX,TdY,TPrep}
    dX_ca::TdX
    dY_ca::TdY
    prep::TPrep
end

"""
    SparseDerivPrep{TJ, TPrep}

Derivative preparation data for [`SparseFlavor`](@ref) components.

# Fields
* `J_ca_sparse::TJ`: sparse `ComponentMatrix` (SparseMatrixCSC-backed) storing the
  Jacobian sparsity pattern and values
* `prep::TPrep`: `DifferentiationInterface.JacobianPrep` with graph coloring
  information
* `rcdict`: `Dict{Tuple{Symbol,Symbol}, Tuple{Vector{Int},Vector{Int}}}` mapping
  each `(output_name, input_name)` sub-Jacobian to its nonzero row and column
  indices (1-based)
"""
struct SparseDerivPrep{TJ,TPrep}
    J_ca_sparse::TJ
    prep::TPrep
    rcdict::Dict{Tuple{Symbol,Symbol}, Tuple{Vector{Int},Vector{Int}}}
end

# ---------------------------------------------------------------------------
# The unified explicit AD component.
# ---------------------------------------------------------------------------

"""
    ADExplicitComp{F<:DerivativeFlavor, InPlace, TAD, TCompute, TX, TY, TDP, TXCS, TYCS}
        <: AbstractExplicitComp

An explicit AD component parameterized by derivative flavor `F`.

# Type parameters
* `F<:DerivativeFlavor`: [`DenseFlavor`](@ref), [`MatrixFreeForwardFlavor`](@ref),
  [`MatrixFreeReverseFlavor`](@ref), or [`SparseFlavor`](@ref)
* `InPlace::Bool`: `true` for in-place functions `f!(Y, X, params)`, `false` for
  out-of-place `Y = f(X, params)`
* `TAD`: the AD backend type (`<:ADTypes.AbstractADType`)

# Fields
* `ad_backend`: `<:ADTypes.AbstractADType` automatic differentiation backend
* `compute_adable`: closure compatible with DifferentiationInterface.jl
* `X_ca`: `ComponentVector` of inputs (`Float64`)
* `Y_ca`: `ComponentVector` of outputs (`Float64`); `nothing` for out-of-place
* `deriv_prep`: flavor-specific prep data ([`DenseDerivPrep`](@ref),
  [`MatrixFreeDerivPrep`](@ref), or [`SparseDerivPrep`](@ref))
* `units_dict`: `Dict{Symbol,String}` mapping variable names to OpenMDAO units
* `tags_dict`: `Dict{Symbol,Vector{String}}` mapping variable names to tags
* `shape_by_conn_dict`: `Dict{Symbol,Bool}` for connection-determined shapes
* `copy_shape_dict`: `Dict{Symbol,Symbol}` for shape-copying
* `X_ca_cs`: `ComplexF64` copy of `X_ca` (for Python-side complex-step)
* `Y_ca_cs`: `ComplexF64` copy of `Y_ca` (for Python-side complex-step); `nothing` for out-of-place
"""
struct ADExplicitComp{F<:DerivativeFlavor, InPlace, TAD, TCompute, TX, TY, TDP, TXCS, TYCS} <: AbstractExplicitComp
    ad_backend::TAD
    compute_adable::TCompute
    X_ca::TX
    Y_ca::TY            # nothing for out-of-place
    deriv_prep::TDP
    units_dict::Dict{Symbol,String}
    tags_dict::Dict{Symbol,Vector{String}}
    shape_by_conn_dict::Dict{Symbol,Bool}
    copy_shape_dict::Dict{Symbol,Symbol}
    X_ca_cs::TXCS
    Y_ca_cs::TYCS       # nothing for out-of-place

    function ADExplicitComp{F, InPlace}(
            ad_backend, compute_adable, X_ca, Y_ca, deriv_prep,
            units_dict, tags_dict, shape_by_conn_dict, copy_shape_dict,
            X_ca_cs, Y_ca_cs
        ) where {F<:DerivativeFlavor, InPlace}
        return new{F, InPlace,
                   typeof(ad_backend), typeof(compute_adable),
                   typeof(X_ca), typeof(Y_ca),
                   typeof(deriv_prep),
                   typeof(X_ca_cs), typeof(Y_ca_cs)}(
            ad_backend, compute_adable, X_ca, Y_ca, deriv_prep,
            units_dict, tags_dict, shape_by_conn_dict, copy_shape_dict,
            X_ca_cs, Y_ca_cs)
    end
end

# ---------------------------------------------------------------------------
# Shared accessors (work for any flavor — read only common fields).
# ---------------------------------------------------------------------------

get_callback(comp::ADExplicitComp) = comp.compute_adable
get_backend(comp::ADExplicitComp) = comp.ad_backend
get_prep(comp::ADExplicitComp) = comp.deriv_prep.prep

get_input_ca(::Type{Float64}, comp::ADExplicitComp) = comp.X_ca
get_input_ca(::Type{ComplexF64}, comp::ADExplicitComp) = comp.X_ca_cs
get_input_ca(::Type{Any}, comp::ADExplicitComp) = comp.X_ca
get_input_ca(comp::ADExplicitComp) = get_input_ca(Float64, comp)

get_output_ca(::Type{Float64}, comp::ADExplicitComp{<:DerivativeFlavor, true}) = comp.Y_ca
get_output_ca(::Type{ComplexF64}, comp::ADExplicitComp{<:DerivativeFlavor, true}) = comp.Y_ca_cs
get_output_ca(::Type{Any}, comp::ADExplicitComp{<:DerivativeFlavor, true}) = comp.Y_ca

get_output_ca(::Type{Float64}, comp::ADExplicitComp{<:DerivativeFlavor, false}) = get_callback(comp)(get_input_ca(Float64, comp))
get_output_ca(::Type{ComplexF64}, comp::ADExplicitComp{<:DerivativeFlavor, false}) = get_callback(comp)(get_input_ca(ComplexF64, comp))
get_output_ca(::Type{Any}, comp::ADExplicitComp{<:DerivativeFlavor, false}) = get_callback(comp)(get_input_ca(Float64, comp))

get_output_ca(comp::ADExplicitComp) = get_output_ca(Float64, comp)

get_units(comp::ADExplicitComp, varname) = get(comp.units_dict, varname, "unitless")
get_tags(comp::ADExplicitComp, varname) = get(comp.tags_dict, varname, Vector{String}())

# Return the `copy_shape` target for `varname` as a `String` (or `nothing` if unset).
# `copy_shape_dict` maps a variable's `Symbol` key to another `Symbol` key whose shape
# should be copied; `VarData.copy_shape` expects a `String`.
function get_copy_shape(comp::ADExplicitComp, varname)
    cs = get(comp.copy_shape_dict, varname, nothing)
    return cs === nothing ? nothing : string(cs)
end

function get_input_var_data(self::ADExplicitComp)
    ca = get_input_ca(self)
    return [VarData(string(k);
                    shape=size(ca[k]),
                    val=ca[k],
                    units=get_units(self, k),
                    tags=get_tags(self, k),
                    shape_by_conn=get(self.shape_by_conn_dict, k, false),
                    copy_shape=get_copy_shape(self, k)) for k in keys(ca)]
end

function get_output_var_data(self::ADExplicitComp)
    ca = get_output_ca(self)
    return [VarData(string(k);
                    shape=size(ca[k]),
                    val=ca[k],
                    units=get_units(self, k),
                    tags=get_tags(self, k),
                    shape_by_conn=get(self.shape_by_conn_dict, k, false),
                    copy_shape=get_copy_shape(self, k)) for k in keys(ca)]
end

function OpenMDAOCore.setup(self::ADExplicitComp)
    input_data = get_input_var_data(self)
    output_data = get_output_var_data(self)

    return input_data, output_data, Vector{PartialsData}()
end

function OpenMDAOCore.compute!(self::ADExplicitComp{<:DerivativeFlavor, true}, inputs, outputs)
    # I used to try to do eltype(valtype(inputs)) but that would return `Any` if there were arrays and scalars in `inputs`.
    TF = eltype(inputs[first(keys(inputs))])

    # Copy the inputs into the input `ComponentArray`.
    X_ca = get_input_ca(TF, self)
    for iname in keys(X_ca)
        # This works even if `X_ca[iname]` is a scalar, because of the `@view`!
        @view(X_ca[iname]) .= inputs[string(iname)]
    end

    # Call the actual function.
    Y_ca = get_output_ca(TF, self)
    f! = get_callback(self)
    f!(Y_ca, X_ca)

    # Copy the output `ComponentArray` to the outputs.
    for oname in keys(Y_ca)
        oname_str = string(oname)
        if typeof(outputs[oname_str]) <: AbstractArray
            outputs[oname_str] .= @view(Y_ca[oname])
        else
            outputs[oname_str] = only(Y_ca[oname])
        end
    end

    return nothing
end

function OpenMDAOCore.compute!(self::ADExplicitComp{<:DerivativeFlavor, false}, inputs, outputs)
    # I used to try to do TF = eltype(valtype(inputs)) but that would return `Any` if there were arrays and scalars in `inputs`.
    TF = eltype(inputs[first(keys(inputs))])
    # Copy the inputs into the input `ComponentArray`.
    X_ca = get_input_ca(TF, self)
    for iname in keys(X_ca)
        # This works even if `X_ca[iname]` is a scalar, because of the `@view`!
        @view(X_ca[iname]) .= inputs[string(iname)]
    end

    # Call the actual function.
    f = get_callback(self)
    Y_ca = f(X_ca)

    # Copy the output `ComponentArray` to the outputs.
    for oname in keys(Y_ca)
        # This requires that each output is at least a vector.
        oname_str = string(oname)
        if typeof(outputs[oname_str]) <: AbstractArray
            outputs[oname_str] .= @view(Y_ca[oname])
        else
            outputs[oname_str] = only(Y_ca[oname])
        end
    end

    return nothing
end

# ---------------------------------------------------------------------------
# Flavor-dispatch accessors (trivial field reads).
# ---------------------------------------------------------------------------

# Dense: read the dense Jacobian ComponentMatrix.
get_jacobian_ca(comp::ADExplicitComp{DenseFlavor}) = comp.deriv_prep.J_ca

# Sparse: read the sparse Jacobian ComponentMatrix. (The sparse *methods* live
# in the extension, but this accessor is a trivial field read and is fine here.)
get_jacobian_ca(comp::ADExplicitComp{SparseFlavor}) = comp.deriv_prep.J_ca_sparse

# Matrix-free: read the tangent/cotangent buffers.
get_dinput_ca(comp::ADExplicitComp{<:MatrixFreeFlavor}) = comp.deriv_prep.dX_ca
get_doutput_ca(comp::ADExplicitComp{<:MatrixFreeFlavor}) = comp.deriv_prep.dY_ca

# Sparse: read the row/col index dict.
get_rows_cols_dict(comp::ADExplicitComp{SparseFlavor}) = comp.deriv_prep.rcdict

# ---------------------------------------------------------------------------
# `has_*` introspection — flavor-dispatched.
# ---------------------------------------------------------------------------

# Dense and Sparse: support setup_partials + compute_partials, not jacvec.
has_setup_partials(self::ADExplicitComp{DenseFlavor}) = true
has_compute_partials(self::ADExplicitComp{DenseFlavor}) = true
has_compute_jacvec_product(self::ADExplicitComp{DenseFlavor}) = false

has_setup_partials(self::ADExplicitComp{SparseFlavor}) = true
has_compute_partials(self::ADExplicitComp{SparseFlavor}) = true
has_compute_jacvec_product(self::ADExplicitComp{SparseFlavor}) = false

# Matrix-free: support setup_partials + jacvec, not compute_partials.
has_setup_partials(self::ADExplicitComp{<:MatrixFreeFlavor}) = true
has_compute_partials(self::ADExplicitComp{<:MatrixFreeFlavor}) = false
has_compute_jacvec_product(self::ADExplicitComp{<:MatrixFreeFlavor}) = true
