using ForwardDiff
include("composite_vector.jl")

x = convert(Array{Float64}, reshape(collect(1:5*2), 5, 2))
y = convert(Array{Float64}, reshape(collect(1:5*1), 5, 1))
z1 = zeros(5, 2)
z2 = zeros(5, 2)

row_sizes = size.([z1, z2])
col_sizes = size.([x, y])

cm = CompositeMatrix{Float64}(row_sizes, col_sizes)

ddata = Dict(("z1", "x") => zeros(prod(size(z1)), prod(size(x))),
             ("z1", "y") => zeros(prod(size(z1)), prod(size(y))),
             ("z2", "x") => zeros(prod(size(z2)), prod(size(x))),
             ("z2", "y") => zeros(prod(size(z2)), prod(size(y))))
@show ddata

ncm = NamedCompositeMatrix{Float64}(ddata)
@show ncm[10, 10]
@show ncm.ddata["z2", "x"]
@show ncm.data[2, 1]
ncm.data[2, 1][1] = -5.0
@show ncm.ddata["z2", "x"]
@show ncm.data[2, 1]

function compute!(inputs, outputs)
    x = inputs["x"]
    y = inputs["y"]

    @. outputs["z1"] = x*x + y*y
    @. outputs["z2"] = x + y

    return nothing
end

function compute_partials!(inputs, outputs, partials::Dict{Tuple{String, String}, Array{T, 2}}) where {T}
    # Need a CompositeVector for the inputs and the outputs, right?
    inputs_ncv = NamedCompositeVector(inputs)
    outputs_ncv = NamedCompositeVector(outputs)
    partials_ncm = NamedCompositeMatrix{T}(partials)
    # partials_ncm = NamedCompositeMatrix(partials)

    function compute_wrapped!(y, x)
        compute!(x.ddata, y.ddata)
        return nothing
    end
    
    partials = ForwardDiff.jacobian!(partials_ncm, compute_wrapped!, outputs_ncv, inputs_ncv)
    return nothing
end

foo = Dict("x" => x, "y" => y)
bar = Dict("z1" => z1, "z2" => z2)

compute_partials!(foo, bar, ddata)
@show ddata
for ((row, col), v) in ddata
    @show row, col
    @show v
end
