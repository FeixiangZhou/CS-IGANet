

config = dict()

config['num_features'] = 32
# config['num_hidden'] = 120
config['num_hidden'] = 64*40
config['num_classes'] = 12
config['pooling_ratio'] = 0.375    #0.25-2  0.375-3
# config['num_joints'] = 3
config['dropout'] = 0.5
config['input_dim'] = 40*64     #T C
config['mab_conv'] = 'GCN'
# config['conv'] = 'GCN'
config['conv'] = 'liner'
config['multi_head'] = 1
# config['model_string'] = 'GMPool_G-SelfAtt-GMPool_I'
config['model_string'] = 'GMPool_G-GMPool_I'
config['model_string_s2'] = 'GMPool_G'
config['ln'] = True
config['cluster'] = False