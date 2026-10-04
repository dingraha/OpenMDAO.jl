# Dense-flavor implicit AD component: constructors and methods.
#
# Note: the prep builders from `dense_ad.jl` (`_get_dense_prep_stuff`) are
# reused for implicit components: the implicit `compute_adable(R, YX)`
# closures already have DifferentiationInterface's in-place (`f!(y, x)`) or
# out-of-place (`y = f(x)`) forms, so we pass `R_ca` as the "output" argument
# (`Y_ca`) and the combined `YX_ca` as the "input" argument (`X_ca`).

"""
    ADImplicitComp(::DenseFlavor, ::Val{true}, ad_backend, f!, Y_ca::ComponentVector, X_ca::ComponentVector; params=nothing, force_skip_prep=false)

Create an in-place [`DenseFlavor`](@ref) [`ADImplicitComp`](@ref).

# Positional Arguments
* `ad_backend`: `<:ADTypes.AbstractADType` automatic differentation "backend" library
* `f!`: function of the form `f!(R_ca, Y_ca, X_ca, params)` which writes residuals to `R_ca` using states/outputs `Y_ca`, inputs `X_ca` and, optionally, parameters `params`.
* `Y_ca`: `ComponentVector` of states/outputs
* `X_ca`: `ComponentVector` of inputs

# Keyword Arguments
* `params`: parameters passed to the fourth argument to `f!`. Could be anything, or `nothing`, but the derivatives of `R_ca` with respect to `params` will not be calculated
* `force_skip_prep`: if true, defer creating internal arrays and other structs until the user calls `update_prep`
"""
function ADImplicitComp(::DenseFlavor, ::Val{true}, ad_backend::TAD, f!, Y_ca::ComponentVector, X_ca::ComponentVector; params=nothing, force_skip_prep=false) where {TAD<:ADTypes.AbstractADType}
    # Check for name collisions between output and input keys.
    common_keys = intersect(keys(Y_ca), keys(X_ca))
    if !isempty(common_keys)
        throw(ArgumentError("State and input ComponentVectors share the following key(s): $(collect(common_keys)). State and input names must be distinct."))
    end

    # Build the concatenated YX vector: outputs first, then inputs.
    YX_ca = ComponentVector(; (k => Y_ca[k] for k in keys(Y_ca))..., (k => X_ca[k] for k in keys(X_ca))...)

    # Precompute integer index ranges for Y and X within YX so the closure uses only
    # plain range indexing — compatible with all AD backends including ReverseDiff.
    Y_range = 1:length(Y_ca)
    X_range = length(Y_ca)+1:length(YX_ca)
    Y_axes = getaxes(Y_ca)
    X_axes = getaxes(X_ca)

    # Create a closure that captures params and reconstructs Y and X as ComponentVectors.
    # https://docs.julialang.org/en/v1/manual/performance-tips/#man-performance-captured
    compute_adable = let params=params, Y_range=Y_range, X_range=X_range, Y_axes=Y_axes, X_axes=X_axes
        (R, YX) -> begin
            Y = ComponentArray(@view(YX[Y_range]), Y_axes)
            X = ComponentArray(@view(YX[X_range]), X_axes)
            f!(R, Y, X, params)
            return nothing
        end
    end

    # Derive the residual vector from Y_ca (same structure).
    R_ca = similar(Y_ca)

    deriv_prep, YX_ca_cs, R_ca_cs = _get_dense_prep_stuff(ad_backend, compute_adable, R_ca, YX_ca, force_skip_prep)

    return ADImplicitComp{DenseFlavor, true}(ad_backend, compute_adable, YX_ca, R_ca, YX_ca_cs, R_ca_cs, deriv_prep, Y_range, X_range, Y_axes, X_axes)
end

"""
    ADImplicitComp(::DenseFlavor, ::Val{false}, ad_backend, f, Y_ca::ComponentVector, X_ca::ComponentVector; params=nothing, force_skip_prep=false)

Create an out-of-place [`DenseFlavor`](@ref) [`ADImplicitComp`](@ref).

# Positional Arguments
* `ad_backend`: `<:ADTypes.AbstractADType` automatic differentation "backend" library
* `f`: function of the form `R_ca = f(Y_ca, X_ca, params)` which returns residuals `R_ca` using states/outputs `Y_ca`, inputs `X_ca` and, optionally, parameters `params`.
* `Y_ca`: `ComponentVector` of states/outputs
* `X_ca`: `ComponentVector` of inputs

# Keyword Arguments
* `params`: parameters passed to the third argument to `f`. Could be anything, or `nothing`, but the derivatives of `R_ca` with respect to `params` will not be calculated
* `force_skip_prep`: if true, defer creating internal arrays and other structs until the user calls `update_prep`
"""
function ADImplicitComp(::DenseFlavor, ::Val{false}, ad_backend::TAD, f, Y_ca::ComponentVector, X_ca::ComponentVector; params=nothing, force_skip_prep=false) where {TAD<:ADTypes.AbstractADType}
    # Check for name collisions between output and input keys.
    common_keys = intersect(keys(Y_ca), keys(X_ca))
    if !isempty(common_keys)
        throw(ArgumentError("State and input ComponentVectors share the following key(s): $(collect(common_keys)). State and input names must be distinct."))
    end

    # Build the concatenated YX vector: outputs first, then inputs.
    YX_ca = ComponentVector(; (k => Y_ca[k] for k in keys(Y_ca))..., (k => X_ca[k] for k in keys(X_ca))...)

    # Precompute integer index ranges for Y and X within YX so the closure uses only
    # plain range indexing — compatible with all AD backends including ReverseDiff.
    Y_range = 1:length(Y_ca)
    X_range = length(Y_ca)+1:length(YX_ca)
    Y_axes = getaxes(Y_ca)
    X_axes = getaxes(X_ca)

    # Create a closure that captures params and reconstructs Y and X as ComponentVectors.
    # https://docs.julialang.org/en/v1/manual/performance-tips/#man-performance-captured
    compute_adable = let params=params, Y_range=Y_range, X_range=X_range, Y_axes=Y_axes, X_axes=X_axes
        YX -> f(ComponentArray(@view(YX[Y_range]), Y_axes), ComponentArray(@view(YX[X_range]), X_axes), params)
    end

    deriv_prep, YX_ca_cs = _get_dense_prep_stuff(ad_backend, compute_adable, YX_ca, force_skip_prep)

    R_ca = nothing
    R_ca_cs = nothing

    return ADImplicitComp{DenseFlavor, false}(ad_backend, compute_adable, YX_ca, R_ca, YX_ca_cs, R_ca_cs, deriv_prep, Y_range, X_range, Y_axes, X_axes)
end

# Scatter the entries of the dense (R, YX) Jacobian `J_ca` into the `partials`
# dict, reshaping each sub-Jacobian as needed. Sub-Jacobian keys that are not
# present in `partials` (i.e., OpenMDAO didn't ask for them) are skipped.
function _scatter_implicit_partials!(J_ca, okeys, ikeys, partials)
    R_axis, YX_axis = getaxes(J_ca)
    for rname in keys(R_axis)
        rstr = string(rname)

        # dR/dY block
        for uname in okeys
            ustr = string(uname)
            Jsub_in = @view(J_ca[rname, uname])
            local Jsub_out
            try
                Jsub_out = partials[rstr, ustr]
            catch e
                isa(e, KeyError) || rethrow()
            else
                Jsub_out .= reshape(Jsub_in, size(Jsub_out))
            end
        end

        # dR/dX block
        for iname in ikeys
            istr = string(iname)
            Jsub_in = @view(J_ca[rname, iname])
            local Jsub_out
            try
                Jsub_out = partials[rstr, istr]
            catch e
                isa(e, KeyError) || rethrow()
            else
                Jsub_out .= reshape(Jsub_in, size(Jsub_out))
            end
        end
    end
    return nothing
end

function linearize!(comp::ADImplicitComp{DenseFlavor, true}, inputs, outputs, partials)
    YX_ca = get_combined_ca(comp)
    okeys = output_keys(comp)
    ikeys = input_keys(comp)
    
    # Copy outputs (states) and inputs into the combined YX vector
    for uname in okeys
        @view(YX_ca[uname]) .= outputs[string(uname)]
    end
    for iname in ikeys
        @view(YX_ca[iname]) .= inputs[string(iname)]
    end

    # Single jacobian! call for the full [dR/dY | dR/dX] block.
    f! = get_callback(comp)
    R_ca = get_residual_ca(comp)
    J_ca = get_jacobian_ca(comp)
    prep = get_prep(comp)
    ad_backend = get_backend(comp)
    if prep === nothing
        DifferentiationInterface.jacobian!(f!, R_ca, J_ca, ad_backend, YX_ca)
    else
        DifferentiationInterface.jacobian!(f!, R_ca, J_ca, prep, ad_backend, YX_ca)
    end

    _scatter_implicit_partials!(J_ca, okeys, ikeys, partials)
    return nothing
end

function linearize!(comp::ADImplicitComp{DenseFlavor, false}, inputs, outputs, partials)
    YX_ca = get_combined_ca(comp)
    okeys = output_keys(comp)
    ikeys = input_keys(comp)
    
    # Copy outputs (states) and inputs into the combined YX vector
    for uname in okeys
        @view(YX_ca[uname]) .= outputs[string(uname)]
    end
    for iname in ikeys
        @view(YX_ca[iname]) .= inputs[string(iname)]
    end

    # Single jacobian! call for the full [dR/dY | dR/dX] block.
    f = get_callback(comp)
    J_ca = get_jacobian_ca(comp)
    prep = get_prep(comp)
    ad_backend = get_backend(comp)
    if prep === nothing
        DifferentiationInterface.jacobian!(f, J_ca, ad_backend, YX_ca)
    else
        DifferentiationInterface.jacobian!(f, J_ca, prep, ad_backend, YX_ca)
    end

    _scatter_implicit_partials!(J_ca, okeys, ikeys, partials)
    return nothing
end
