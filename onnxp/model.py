from onnx import AttributeProto, shape_inference, numpy_helper

__all__ = ['Model']


class Variable:
    def __init__(self, name, size):
        self.name = name
        self.size = size
        self.module = None


class Module:
    def __init__(self, name, type, attributes):
        self.name = name
        self.type = type
        self.attributes = attributes
        self.weights, self.inputs, self.outputs = [], [], []


class Model:
    def __init__(self, model, remove_batchnorms=True, propagate_size=True):
        self.model = model

        self.parse_variables()
        self.parse_modules()

        if remove_batchnorms:
            self.remove_batchnorms()

        if propagate_size:
            self.propagate_size()

    def parse_variables(self):
        self.variables = {}

        for variable in self.model.graph.input:
            name = variable.name
            size = [d.dim_value for d in variable.type.tensor_type.shape.dim]
            self.variables[name] = Variable(name=name, size=size)

        for variable in self.model.graph.output:
            name = variable.name
            size = [d.dim_value for d in variable.type.tensor_type.shape.dim]
            self.variables[name] = Variable(name=name, size=size)

        self.model = shape_inference.infer_shapes(self.model)

        for variable in self.model.graph.value_info:
            name = variable.name
            size = [d.dim_value for d in variable.type.tensor_type.shape.dim]
            self.variables[name] = Variable(name=name, size=size)

    def parse_modules(self):
        self.modules = []

        weight_names = set()
        for variable in self.model.graph.initializer:
            weight_names.add(variable.name)

        for node in self.model.graph.node:
            name = node.name
            type = node.op_type.lower()

            attributes = {}
            for attribute in node.attribute:
                if attribute.type == AttributeProto.FLOAT:
                    attributes[attribute.name] = attribute.f
                elif attribute.type == AttributeProto.INT:
                    attributes[attribute.name] = attribute.i
                elif attribute.type == AttributeProto.STRING:
                    attributes[attribute.name] = attribute.s
                elif attribute.type == AttributeProto.TENSOR:
                    attributes[attribute.name] = numpy_helper.to_array(attribute.t)
                elif attribute.type == AttributeProto.GRAPH:
                    attributes[attribute.name] = attribute.g
                elif attribute.type == AttributeProto.FLOATS:
                    attributes[attribute.name] = attribute.floats
                elif attribute.type == AttributeProto.INTS:
                    attributes[attribute.name] = attribute.ints
                elif attribute.type == AttributeProto.STRINGS:
                    attributes[attribute.name] = attribute.strings
                elif attribute.type == AttributeProto.TENSORS:
                    attributes[attribute.name] = attribute.tensors
                elif attribute.type == AttributeProto.GRAPHS:
                    attributes[attribute.name] = attribute.graphs

            module = Module(name=name, type=type, attributes=attributes)
            self.modules.append(module)

            for name in node.input:
                if name not in self.variables:
                    self.variables[name] = Variable(name=name, size=None)

                if name not in weight_names:
                    module.inputs.append(self.variables[name])
                else:
                    module.weights.append(self.variables[name])

            for name in node.output:
                if name not in self.variables:
                    self.variables[name] = Variable(name=name, size=None)

                self.variables[name].module = module
                module.outputs.append(self.variables[name])

    def remove_batchnorms(self):
        for module in list(self.modules):
            if module.type == 'batchnormalization' and module.inputs[0].module.type in ['conv', 'gemm']:
                module.inputs[0].module.outputs = module.outputs

                if len(module.inputs[0].module.weights) == 1:
                    module.inputs[0].module.weights.append(module.weights[1])

                self.modules.remove(module)

    def propagate_size(self):
        batch_size = None
        for variable in self.variables.values():
            if variable.module is not None and variable.module.type in ['conv', 'gemm'] and variable.size:
                batch_size = variable.size[0]

        if batch_size is None:
            return

        modified = True
        while modified:
            modified = False

            for module in self.modules:
                if module.type == 'gemm':
                    if not module.inputs[0].size:
                        module.inputs[0].size = [batch_size, module.weights[0].size[1]]
                        modified = True

                    if not module.outputs[0].size:
                        module.outputs[0].size = [batch_size, module.weights[0].size[0]]
                        modified = True

                if module.type in ['relu', 'dropout']:
                    if not module.inputs[0].size and module.outputs[0].size:
                        module.inputs[0].size = module.outputs[0].size
                        modified = True

                    if not module.outputs[0].size and module.inputs[0].size:
                        module.outputs[0].size = module.inputs[0].size
                        modified = True
