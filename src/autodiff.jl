using CompositeArrays
using ForwardDiff

abstract type AbstractADExplicitComp <: AbstractExplicitComp end
abstract type AbstractADImplicitComp <: AbstractImplicitComp end

struct ExplicitAutoDiffData
    inputs_ncv
    outputs_ncv
    partials_ncm
end

struct ImplicitAutoDiffData
    inputs_outputs_ncv
    residuals_ncv
    partials_ncm
end

function declare_ad!(self::AbstractADExplicitComp)
    inputs, outputs, partials = OpenMDAO.setup(self)
    input_names = getproperty.(inputs, :name)
    input_sizes = getproperty.(inputs, :shape)
    output_names = getproperty.(outputs, :name)
    output_sizes = getproperty.(outputs, :shape)

    N = length(input_sizes[1])
    if ! all(length.(input_sizes) .== N)
        throw(ArgumentError("input sizes do not all have the same number of dimensions"))
    end
    if ! all(length.(output_sizes) .== N)
        throw(ArgumentError("all output sizes do not have the same number of dimensions as the inputs"))
    end

    # I think I should make the size inputs work if they are an array that mixes
    # tuples and scalars. Could I do that?

    inputs_ncv = NamedCompositeVector(PyArray{Float64, N}, input_sizes, input_names)
    outputs_ncv = NamedCompositeVector(Array{Float64, N}, output_sizes, output_names)
    @show output_sizes
    @show input_sizes
    # partials_ncm = NamedCompositeMatrix(PyArray{Float64, 2}, output_sizes, input_sizes, output_names, input_names)
    partials_ncm = NamedCompositeMatrix{N}(PyArray{Float64, 2}, output_sizes, input_sizes, output_names, input_names)

    # Need to initialize the outputs data in an ExplicitComp, since the
    # compute_partials! function doesn't take an outputs arg (just inputs and
    # partials).
    ddata = Dict{String, Array{Float64, N}}()
    for (name, sz) in zip(output_names, output_sizes)
        ddata[name] = fill(0.0, sz...)
    end
    update!(outputs_ncv, ddata)

    self.ad_data = ExplicitAutoDiffData(inputs_ncv, outputs_ncv, partials_ncm)

    return nothing
end

function declare_ad!(self::AbstractADImplicitComp)
    inputs, outputs, partials = OpenMDAO.setup(self)
    input_names = getproperty.(inputs, :name)
    input_sizes = getproperty.(inputs, :shape)
    output_names = getproperty.(outputs, :name)
    output_sizes = getproperty.(outputs, :shape)

    N = length(input_sizes[1])
    if ! all(length.(input_sizes) .== N)
        throw(ArgumentError("input sizes do not all have the same number of dimensions"))
    end
    if ! all(length.(output_sizes) .== N)
        throw(ArgumentError("all output sizes do not have the same number of dimensions as the inputs"))
    end

    input_output_names = cat(input_names, output_names, dims=1)
    input_output_sizes = cat(input_sizes, output_sizes, dims=1)

    inputs_outputs_ncv = NamedCompositeVector(PyArray{Float64, N}, input_output_sizes, input_output_names)
    residuals_ncv = NamedCompositeVector(Array{Float64, N}, output_sizes, output_names)
    partials_ncm = NamedCompositeMatrix(PyArray{Float64, 2}, output_sizes, input_output_sizes, output_names, input_output_names)

    # Need to initialize the residuals data in an ImplicitComp, since the
    # linearize! function doesn't take a residuals arg (just inputs, outputs,
    # and partials).
    ddata = Dict{String, Array{Float64, N}}()
    for (name, sz) in zip(output_names, output_sizes)
        ddata[name] = fill(0.0, sz...)
    end
    update!(residuals_ncv, ddata)

    self.ad_data = ImplicitAutoDiffData(inputs_outputs_ncv, residuals_ncv, partials_ncm)

    return nothing
end

function OpenMDAO.compute_partials!(self::AbstractADExplicitComp, inputs, partials)
    function compute_wrapped!(y, x)
        OpenMDAO.compute!(self, x.ddata, y.ddata)
        return nothing
    end

    update!(self.ad_data.inputs_ncv, inputs)
    update!(self.ad_data.partials_ncm, partials)

    ForwardDiff.jacobian!(self.ad_data.partials_ncm, compute_wrapped!, self.ad_data.outputs_ncv, self.ad_data.inputs_ncv)

    return nothing
end

function OpenMDAO.linearize!(self::AbstractADImplicitComp, inputs, outputs, partials)
    function apply_nonlinear_wrapped!(y, x)
        OpenMDAO.apply_nonlinear!(self, x.ddata, x.ddata, y.ddata)
        return nothing
    end

    update!(self.ad_data.inputs_outputs_ncv, inputs)
    update!(self.ad_data.inputs_outputs_ncv, outputs)
    update!(self.ad_data.partials_ncm, partials)

    ForwardDiff.jacobian!(self.ad_data.partials_ncm, apply_nonlinear_wrapped!, self.ad_data.residuals_ncv, self.ad_data.inputs_outputs_ncv)

    return nothing
end

