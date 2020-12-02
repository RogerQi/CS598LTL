'''
Subclass for the composition 'composition'
'''

from __future__ import print_function
import torch
torch.set_default_tensor_type('torch.cuda.FloatTensor')
nn_device='cuda:0'
torch.device(nn_device)
from composition import Composer
from structure import Structure
from variable_dimension_structure import VariableDimensionStructure
from IPython import embed
from pdb import set_trace

class MultiSumComposer(Composer):
  def __init__(self, composer, module_list, loss_fn=None, structure={},
      instructions={}):
    super().__init__(composer=composer,
        module_list=module_list, loss_fn=loss_fn,
        structure=structure, instructions=instructions)

  def forward_no_weights(self, x):
    res = []
    # self.structure['modules'] contains a list of length n
    # where n is the number of output dimension
    mod_idx = 0
    chunk_size = self.structure["chunk_size"]
    for i in range(self.structure["out_mods"]):
        res_val = 0
        for j in range(self.structure["in_mods"]):
            mod = self.structure["modules"][mod_idx]
            res_val += self.module_list[mod](x[:, chunk_size*j:chunk_size*(j+1)])
            mod_idx += 1
        res.append(res_val)
    x = torch.cat(res, dim = -1)
    return x
  
  def forward_with_weights(self, x, weights):
    res = []
    # self.structure['modules'] contains a list of length n
    # where n is the number of output dimension
    mod_idx = 0
    chunk_size = self.structure["chunk_size"]
    for i in range(self.structure["out_mods"]):
        res_val = 0
        for j in range(self.structure["in_mods"]):
            mod = self.structure["modules"][mod_idx]
            res_val += self.module_list[mod](x[:, chunk_size*j:chunk_size*(j+1)],
                weights=weights, prefix='module_list.'+str(mod)+'.features.')
            mod_idx += 1
        res.append(res_val)
    x = torch.cat(res, dim = -1)
    return x

class MultiSumStructure(VariableDimensionStructure):
  def __init__(self, args):
    self.composer = 'MultiSumComposer'
    self.composer_class = MultiSumComposer
    self.composer_abbreviation = 'M'
    super().__init__(args=args)

  def propose_new_structure(self, new_structure):
    return self.default_propose_new_structure(new_structure)

  def initialize_structure(self, in_dim, out_dim):
    return self.default_initialize_structure(in_dim, out_dim)

  def update_Usage_counters(self, METRICS, T):
    return self.default_update_Usage_counters(METRICS, T)

  def modules_given_structure(self, structure):
    return structure['modules']
