using OpenMDAO

inputs = [VarData("x", (1, 2), [8.0], "inch"),
          VarData("y", (2, 2), [[2.0,],], "ft"),
          VarData("z", (2, 1), [[-1.0], 
                                [-2.0]], "m")]
@show inputs
