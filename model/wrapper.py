from model.mygin import G_GIN
from model.molecule_encoder import MoleculeEncoder


def get_model(name, args, nclass):
    if name == 'gin':
        model = G_GIN(input_dim=args.nfeat, hidden_dim=args.nhid, output_dim=nclass, nconvs=args.layers)
    elif name == 'gin_var':
        model = MoleculeEncoder(emb_dim=args.nhid, num_gc_layers=args.layers,
                                drop_ratio=args.drop_ratio, pooling_type=args.pooling_type)
    else:
        raise NotImplementedError
    return model
