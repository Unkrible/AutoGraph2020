from collections import OrderedDict


class MacroSearchSpace:
    def __init__(self, cell_space=None, num_of_layers=2):
        self.num_of_layers = num_of_layers
        if cell_space is not None:
            self.cell_space = cell_space
        else:
            self.cell_space = OrderedDict([
                ("attention_type", ["gat", "gcn", "cos", "const", "gat_sym", 'linear', 'generalized_linear']),
                ('aggregator_type', ["sum", "mean", "max", "mlp", ]),
                ('activate_function', ["sigmoid", "tanh", "relu", "linear",
                                      "softplus", "leaky_relu", "relu6", "elu"]),
                ('number_of_heads', [1, 2, 4, 6, 8, 16]),
                ('hidden_units', [4, 8, 16, 32, 64, 128, 256]),
            ])

    @property
    def search_space(self):
        return self.cell_space

    @property
    def action_list(self):
        return list(self.cell_space.keys()) * self.num_of_layers

    @property
    def num_tokens(self):
        return [len(v) for v in self.cell_space.values()]
