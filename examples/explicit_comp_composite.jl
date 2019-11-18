using OpenMDAO
using PyCall
import Base.convert
using ForwardDiff
using CompositeArrays

om = PyCall.pyimport("openmdao.api")

mutable struct SquareIt{TF} <: AbstractExplicitComp
    a::TF
    inputs::Vector{VarData{TF, 1}}
    outputs::Vector{VarData{TF, 1}}
    partials::Vector{PartialsData}
    inputs_ncv::NamedCompositeVector{TF, 1, PyArray{TF, 1}}
    outputs_ncv::NamedCompositeVector{TF, 1, Array{TF, 1}}
    partials_ncm::NamedCompositeMatrix{TF, 1, PyArray{TF, 2}}
    # function SquareIt{TF}(a) where {TF}
    #     return new(a)
    # end
    # function SquareIt{TF}(a, inputs_ncv, outputs_ncv, partials_ncm) where {TF}
    #     return new(a, inputs_ncv, outputs_ncv, partials_ncm)
    # end
end
# SquareIt(a::TF) where {TF} = SquareIt{TF}(a)

function convert(::Type{SquareIt{TF}}, po::PyObject) where {TF}
    println("in SquareIt's convert")
    # return SquareIt{TF}(po.a)
    return SquareIt(po.a)
end

function SquareIt(a::TF) where {TF}
    inputs = [
        VarData("x", (1,), 2.0),
        VarData("y", (1,), 3.0)]

    outputs = [
        VarData("z1", (1,), 2.0),
        VarData("z2", (1,), 3.0)]

    partials = [
        PartialsData("z1", "x"),
        PartialsData("z1", "y"),
        PartialsData("z2", "x"),
        PartialsData("z2", "y")]

    # inputs_ncv = NamedCompositeVector(inputs)
    # outputs_ncv = NamedCompositeVector(outputs)
    # partials_ncm = NamedCompositeMatrix(inputs, outputs)
    
    input_sizes = [(1,), (1,)]
    output_sizes = [(1,), (1,)]
    input_names = ["x", "y"]
    output_names = ["z1", "z2"]
    inputs_ncv = NamedCompositeVector(PyArray{TF, 1}, input_sizes, input_names)
    outputs_ncv = NamedCompositeVector(Array{TF, 1}, output_sizes, output_names)
    partials_ncm = NamedCompositeMatrix(PyArray{TF, 2}, output_sizes, input_sizes, output_names, input_names)

    # Need to initialize the outputs data.
    update!(outputs_ncv, Dict("z1"=>fill(0.0, 1), "z2"=>fill(0.0, 1)))

    return SquareIt(a, inputs, outputs, partials, inputs_ncv, outputs_ncv, partials_ncm)
end

function OpenMDAO.setup(self::SquareIt)
    # println("In the correct setup function")
    # inputs = [
    #     VarData("x", (1,), 2.0),
    #     VarData("y", (1,), 3.0)]

    # outputs = [
    #     VarData("z1", (1,), 2.0),
    #     VarData("z2", (1,), 3.0)]

    # partials = [
    #     PartialsData("z1", "x"),
    #     PartialsData("z1", "y"),
    #     PartialsData("z2", "x"),
    #     PartialsData("z2", "y")]

    # println("fieldnames = $(fieldnames(SquareIt))")

    # self.inputs_ncv = NamedCompositeVector(inputs)
    # self.outputs_ncv = NamedCompositeVector(outputs)
    # self.partials_ncm = NamedCompositeMatrix(inputs, outputs)

    return self.inputs, self.outputs, self.partials
end

function OpenMDAO.compute!(self::SquareIt, inputs, outputs)
    a = self.a
    x = inputs["x"]
    y = inputs["y"]

    @. outputs["z1"] = a*x*x + y*y
    @. outputs["z2"] = a*x + y

    return nothing
end

function OpenMDAO.compute_partials!(self::SquareIt, inputs, partials)
    # a = self.a
    # x = inputs["x"]
    # y = inputs["y"]

    # @. partials["z1", "x"] = 2*a*x
    # @. partials["z1", "y"] = 2*y
    # @. partials["z2", "x"] = a
    # @. partials["z2", "y"] = 1.0

    # This doesn't work, because self.inputs_ncv isn't defined on the first
    # call, so referencing it is an error. I guess this whole updating idea
    # isn't going to work. But what do I do about outputs, then? Lame. This
    # won't be a problem for the implicit components, I think, since the
    # linearize function needs the inputs and outputs.
    # update!(self.inputs_ncv, inputs)
    # update!(self.partials_ncm, partials)

    function compute_wrapped!(y, x)
        OpenMDAO.compute!(self, x.ddata, y.ddata)
        return nothing
    end

    @show typeof(inputs)
    update!(self.inputs_ncv, inputs)
    # update!(self.outputs_ncv, outputs)
    update!(self.partials_ncm, partials)

    @show self.outputs_ncv
    ForwardDiff.jacobian!(self.partials_ncm, compute_wrapped!, self.outputs_ncv, self.inputs_ncv)
    @show self.partials_ncm

    return nothing
end


prob = om.Problem()

ivc = om.IndepVarComp()
ivc.add_output("x", 2.0)
ivc.add_output("y", 3.0)
prob.model.add_subsystem("ivc", ivc, promotes=["*"])

comp = make_component(SquareIt(4.0))
prob.model.add_subsystem("square_it_comp", comp, promotes=["*"])

prob.setup()
prob.run_model()
@show prob.get_val("z1")
@show prob.get_val("z2")
@show prob.compute_totals(of=["z1", "z2"], wrt=["x", "y"])
