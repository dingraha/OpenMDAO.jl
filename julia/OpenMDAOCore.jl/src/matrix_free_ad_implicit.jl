# Matrix-free-flavor implicit AD component: constructors and methods.
#
# Note: the prep builders from `matrix_free_ad.jl`
# (`_get_matrix_free_forward_prep_in_place`, etc.) are reused for implicit
# components: the implicit `compute_adable(R, YX)` closures already have
# DifferentiationInterface's in-place (`f!(y, x)`) or out-of-place
# (`y = f(x)`) forms, so we pass `R_ca` as the "output" argument (`Y_ca`) and
# the combined `YX_ca` as the "input" argument (`X_ca`). In the resulting
# `MatrixFreeDerivPrep`, `dX_ca` holds the combined dYX tangent/cotangent
# buffer and `dY_ca` holds the residual tangent/cotangent buffer dR.

"""
    ADImplicitComp(::MatrixFreeForwardFlavor, ::Val{true}, ad_backend, f!, Y_ca, X_ca; params=nothing, force_skip_prep=false)

Create an in-place [`MatrixFreeForwardFlavor`](@ref) [`ADImplicitComp`](@ref).
Derivatives are computed via `DifferentiationInterface.pushforward!`.
"""
function ADImplicitComp(::MatrixFreeForwardFlavor, ::Val{true}, ad_backend::TAD, f!, Y_ca::ComponentVector, X_ca::ComponentVector;
        params=nothing, units_dict=Dict{Symbol,String}(), tags_dict=Dict{Symbol,Vector{String}}(), force_skip_prep=false) where {TAD<:ADTypes.AbstractADType}
    common_keys = intersect(keys(Y_ca), keys(X_ca))
    if !isempty(common_keys)
        throw(ArgumentError("State and input ComponentVectors share the following key(s): $(collect(common_keys)). State and input names must be distinct."))
    end

    YX_ca = ComponentVector(; (k => Y_ca[k] for k in keys(Y_ca))..., (k => X_ca[k] for k in keys(X_ca))...)
    Y_range = 1:length(Y_ca)
    X_range = length(Y_ca)+1:length(YX_ca)
    Y_axes = getaxes(Y_ca)
    X_axes = getaxes(X_ca)

    compute_adable = let params=params, Y_range=Y_range, X_range=X_range, Y_axes=Y_axes, X_axes=X_axes
        (R, YX) -> begin
            Y = ComponentArray(@view(YX[Y_range]), Y_axes)
            X = ComponentArray(@view(YX[X_range]), X_axes)
            f!(R, Y, X, params)
            return nothing
        end
    end

    R_ca = similar(Y_ca)

    deriv_prep, YX_ca_cs, R_ca_cs = _get_matrix_free_forward_prep_in_place(
        ad_backend, compute_adable, R_ca, YX_ca, force_skip_prep)

    return ADImplicitComp{MatrixFreeForwardFlavor, true}(ad_backend, compute_adable, YX_ca, R_ca, YX_ca_cs, R_ca_cs, deriv_prep, units_dict, tags_dict, Y_range, X_range, Y_axes, X_axes)
end

"""
    ADImplicitComp(::MatrixFreeForwardFlavor, ::Val{false}, ad_backend, f, Y_ca, X_ca; params=nothing, force_skip_prep=false)

Create an out-of-place [`MatrixFreeForwardFlavor`](@ref) [`ADImplicitComp`](@ref).
Derivatives are computed via `DifferentiationInterface.pushforward!`.
"""
function ADImplicitComp(::MatrixFreeForwardFlavor, ::Val{false}, ad_backend::TAD, f, Y_ca::ComponentVector, X_ca::ComponentVector;
        params=nothing, units_dict=Dict{Symbol,String}(), tags_dict=Dict{Symbol,Vector{String}}(), force_skip_prep=false) where {TAD<:ADTypes.AbstractADType}
    common_keys = intersect(keys(Y_ca), keys(X_ca))
    if !isempty(common_keys)
        throw(ArgumentError("State and input ComponentVectors share the following key(s): $(collect(common_keys)). State and input names must be distinct."))
    end

    YX_ca = ComponentVector(; (k => Y_ca[k] for k in keys(Y_ca))..., (k => X_ca[k] for k in keys(X_ca))...)
    Y_range = 1:length(Y_ca)
    X_range = length(Y_ca)+1:length(YX_ca)
    Y_axes = getaxes(Y_ca)
    X_axes = getaxes(X_ca)

    compute_adable = let params=params, Y_range=Y_range, X_range=X_range, Y_axes=Y_axes, X_axes=X_axes
        YX -> f(ComponentArray(@view(YX[Y_range]), Y_axes), ComponentArray(@view(YX[X_range]), X_axes), params)
    end

    deriv_prep, YX_ca_cs = _get_matrix_free_forward_prep_out_of_place(
        ad_backend, compute_adable, compute_adable(YX_ca), YX_ca, force_skip_prep)

    R_ca = R_ca_cs = nothing

    return ADImplicitComp{MatrixFreeForwardFlavor, false}(ad_backend, compute_adable, YX_ca, R_ca, YX_ca_cs, R_ca_cs, deriv_prep, units_dict, tags_dict, Y_range, X_range, Y_axes, X_axes)
end

"""
    ADImplicitComp(::MatrixFreeReverseFlavor, ::Val{true}, ad_backend, f!, Y_ca, X_ca; params=nothing, force_skip_prep=false)

Create an in-place [`MatrixFreeReverseFlavor`](@ref) [`ADImplicitComp`](@ref).
Derivatives are computed via `DifferentiationInterface.pullback!`.
"""
function ADImplicitComp(::MatrixFreeReverseFlavor, ::Val{true}, ad_backend::TAD, f!, Y_ca::ComponentVector, X_ca::ComponentVector;
        params=nothing, units_dict=Dict{Symbol,String}(), tags_dict=Dict{Symbol,Vector{String}}(), force_skip_prep=false) where {TAD<:ADTypes.AbstractADType}
    common_keys = intersect(keys(Y_ca), keys(X_ca))
    if !isempty(common_keys)
        throw(ArgumentError("State and input ComponentVectors share the following key(s): $(collect(common_keys)). State and input names must be distinct."))
    end

    YX_ca = ComponentVector(; (k => Y_ca[k] for k in keys(Y_ca))..., (k => X_ca[k] for k in keys(X_ca))...)
    Y_range = 1:length(Y_ca)
    X_range = length(Y_ca)+1:length(YX_ca)
    Y_axes = getaxes(Y_ca)
    X_axes = getaxes(X_ca)

    compute_adable = let params=params, Y_range=Y_range, X_range=X_range, Y_axes=Y_axes, X_axes=X_axes
        (R, YX) -> begin
            Y = ComponentArray(@view(YX[Y_range]), Y_axes)
            X = ComponentArray(@view(YX[X_range]), X_axes)
            f!(R, Y, X, params)
            return nothing
        end
    end

    R_ca = similar(Y_ca)

    deriv_prep, YX_ca_cs, R_ca_cs = _get_matrix_free_reverse_prep_in_place(
        ad_backend, compute_adable, R_ca, YX_ca, force_skip_prep)

    return ADImplicitComp{MatrixFreeReverseFlavor, true}(ad_backend, compute_adable, YX_ca, R_ca, YX_ca_cs, R_ca_cs, deriv_prep, units_dict, tags_dict, Y_range, X_range, Y_axes, X_axes)
end

"""
    ADImplicitComp(::MatrixFreeReverseFlavor, ::Val{false}, ad_backend, f, Y_ca, X_ca; params=nothing, force_skip_prep=false)

Create an out-of-place [`MatrixFreeReverseFlavor`](@ref) [`ADImplicitComp`](@ref).
Derivatives are computed via `DifferentiationInterface.pullback!`.
"""
function ADImplicitComp(::MatrixFreeReverseFlavor, ::Val{false}, ad_backend::TAD, f, Y_ca::ComponentVector, X_ca::ComponentVector;
        params=nothing, units_dict=Dict{Symbol,String}(), tags_dict=Dict{Symbol,Vector{String}}(), force_skip_prep=false) where {TAD<:ADTypes.AbstractADType}
    common_keys = intersect(keys(Y_ca), keys(X_ca))
    if !isempty(common_keys)
        throw(ArgumentError("State and input ComponentVectors share the following key(s): $(collect(common_keys)). State and input names must be distinct."))
    end

    YX_ca = ComponentVector(; (k => Y_ca[k] for k in keys(Y_ca))..., (k => X_ca[k] for k in keys(X_ca))...)
    Y_range = 1:length(Y_ca)
    X_range = length(Y_ca)+1:length(YX_ca)
    Y_axes = getaxes(Y_ca)
    X_axes = getaxes(X_ca)

    compute_adable = let params=params, Y_range=Y_range, X_range=X_range, Y_axes=Y_axes, X_axes=X_axes
        YX -> f(ComponentArray(@view(YX[Y_range]), Y_axes), ComponentArray(@view(YX[X_range]), X_axes), params)
    end

    deriv_prep, YX_ca_cs = _get_matrix_free_reverse_prep_out_of_place(
        ad_backend, compute_adable, compute_adable(YX_ca), YX_ca, force_skip_prep)

    R_ca = R_ca_cs = nothing

    return ADImplicitComp{MatrixFreeReverseFlavor, false}(ad_backend, compute_adable, YX_ca, R_ca, YX_ca_cs, R_ca_cs, deriv_prep, units_dict, tags_dict, Y_range, X_range, Y_axes, X_axes)
end

function _compute_implicit_pushforward!(comp::ADImplicitComp{MatrixFreeForwardFlavor, true}, inputs, outputs, dinputs, doutputs, dresids)
    YX_ca = get_combined_ca(comp)
    for uname in output_keys(comp)
        @view(YX_ca[uname]) .= outputs[string(uname)]
    end
    for iname in input_keys(comp)
        @view(YX_ca[iname]) .= inputs[string(iname)]
    end

    # Assemble the YX tangent from doutputs (Y part) and dinputs (X part).
    dYX_ca = get_dcombined_ca(comp)
    for uname in output_keys(comp)
        ustr = string(uname)
        @view(dYX_ca[uname]) .= doutputs[ustr]
    end
    for iname in input_keys(comp)
        istr = string(iname)
        @view(dYX_ca[iname]) .= dinputs[istr]
    end

    R_ca = get_residual_ca(comp)
    dR_ca = get_dresidual_ca(comp)
    compute_adable = get_callback(comp)
    prep = get_prep(comp)
    backend = get_backend(comp)

    DifferentiationInterface.pushforward!(compute_adable, R_ca, (dR_ca,), prep, backend, YX_ca, (dYX_ca,))

    for rname in keys(dR_ca)
        rstr = string(rname)
        if typeof(dresids[rstr]) <: AbstractArray
            dresids[rstr] .+= @view(dR_ca[rname])
        else
            dresids[rstr] += only(dR_ca[rname])
        end
    end

    return nothing
end

function _compute_implicit_pushforward!(comp::ADImplicitComp{MatrixFreeForwardFlavor, false}, inputs, outputs, dinputs, doutputs, dresids)
    YX_ca = get_combined_ca(comp)
    for uname in output_keys(comp)
        @view(YX_ca[uname]) .= outputs[string(uname)]
    end
    for iname in input_keys(comp)
        @view(YX_ca[iname]) .= inputs[string(iname)]
    end

    dYX_ca = get_dcombined_ca(comp)
    for uname in output_keys(comp)
        ustr = string(uname)
        @view(dYX_ca[uname]) .= doutputs[ustr]
    end
    for iname in input_keys(comp)
        istr = string(iname)
        @view(dYX_ca[iname]) .= dinputs[istr]
    end

    dR_ca = get_dresidual_ca(comp)
    compute_adable = get_callback(comp)
    prep = get_prep(comp)
    backend = get_backend(comp)

    DifferentiationInterface.pushforward!(compute_adable, (dR_ca,), prep, backend, YX_ca, (dYX_ca,))

    for rname in keys(dR_ca)
        rstr = string(rname)
        if typeof(dresids[rstr]) <: AbstractArray
            dresids[rstr] .+= @view(dR_ca[rname])
        else
            dresids[rstr] += only(dR_ca[rname])
        end
    end

    return nothing
end

function _compute_implicit_pullback!(comp::ADImplicitComp{MatrixFreeReverseFlavor, true}, inputs, outputs, dinputs, doutputs, dresids)
    YX_ca = get_combined_ca(comp)
    for uname in output_keys(comp)
        @view(YX_ca[uname]) .= outputs[string(uname)]
    end
    for iname in input_keys(comp)
        @view(YX_ca[iname]) .= inputs[string(iname)]
    end

    # Load the residual cotangent seed from dresids.
    dR_ca = get_dresidual_ca(comp)
    for rname in keys(dR_ca)
        rstr = string(rname)
        @view(dR_ca[rname]) .= dresids[rstr]
    end

    R_ca = get_residual_ca(comp)
    dYX_ca = get_dcombined_ca(comp)
    compute_adable = get_callback(comp)
    backend = get_backend(comp)
    prep = get_prep(comp)

    DifferentiationInterface.pullback!(compute_adable, R_ca, (dYX_ca,), prep, backend, YX_ca, (dR_ca,))

    # Scatter dYX cotangents back: Y part → doutputs, X part → dinputs.
    for uname in output_keys(comp)
        ustr = string(uname)
        if typeof(doutputs[ustr]) <: AbstractArray
            doutputs[ustr] .+= @view(dYX_ca[uname])
        else
            doutputs[ustr] += only(dYX_ca[uname])
        end
    end
    for iname in input_keys(comp)
        istr = string(iname)
        if typeof(dinputs[istr]) <: AbstractArray
            dinputs[istr] .+= @view(dYX_ca[iname])
        else
            dinputs[istr] += only(dYX_ca[iname])
        end
    end

    return nothing
end

function _compute_implicit_pullback!(comp::ADImplicitComp{MatrixFreeReverseFlavor, false}, inputs, outputs, dinputs, doutputs, dresids)
    YX_ca = get_combined_ca(comp)
    for uname in output_keys(comp)
        @view(YX_ca[uname]) .= outputs[string(uname)]
    end
    for iname in input_keys(comp)
        @view(YX_ca[iname]) .= inputs[string(iname)]
    end

    dR_ca = get_dresidual_ca(comp)
    for rname in keys(dR_ca)
        rstr = string(rname)
        @view(dR_ca[rname]) .= dresids[rstr]
    end

    dYX_ca = get_dcombined_ca(comp)
    compute_adable = get_callback(comp)
    backend = get_backend(comp)
    prep = get_prep(comp)

    DifferentiationInterface.pullback!(compute_adable, (dYX_ca,), prep, backend, YX_ca, (dR_ca,))

    for uname in output_keys(comp)
        ustr = string(uname)
        if typeof(doutputs[ustr]) <: AbstractArray
            doutputs[ustr] .+= @view(dYX_ca[uname])
        else
            doutputs[ustr] += only(dYX_ca[uname])
        end
    end
    for iname in input_keys(comp)
        istr = string(iname)
        if typeof(dinputs[istr]) <: AbstractArray
            dinputs[istr] .+= @view(dYX_ca[iname])
        else
            dinputs[istr] += only(dYX_ca[iname])
        end
    end

    return nothing
end

function apply_linear!(comp::ADImplicitComp{MatrixFreeForwardFlavor}, inputs, outputs, d_inputs, d_outputs, d_residuals, mode)
    if mode == "fwd"
        _compute_implicit_pushforward!(comp, inputs, outputs, d_inputs, d_outputs, d_residuals)
    else
        throw(ArgumentError("MatrixFreeForwardFlavor only supports mode=\"fwd\""))
    end
    return nothing
end

function apply_linear!(comp::ADImplicitComp{MatrixFreeReverseFlavor}, inputs, outputs, d_inputs, d_outputs, d_residuals, mode)
    if mode == "rev"
        _compute_implicit_pullback!(comp, inputs, outputs, d_inputs, d_outputs, d_residuals)
    else
        throw(ArgumentError("MatrixFreeReverseFlavor only supports mode=\"rev\""))
    end
    return nothing
end
