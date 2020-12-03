'''
Subclass for the composition 'composition'
'''

from __future__ import print_function
import torch
import numpy as np
torch.set_default_tensor_type('torch.cuda.FloatTensor')
nn_device='cuda:0'
torch.device(nn_device)
from composition import Composer
from structure import Structure
from variable_dimension_structure import VariableDimensionStructure
from IPython import embed
from pdb import set_trace

class GatedSumComposer(Composer):
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
    gate_offset = len(self.structure["modules"]) // 2
    for i in range(self.structure["out_mods"]):
        res_val = 0
        for j in range(self.structure["in_mods"]):
            mod = self.structure["modules"][mod_idx]
            gate_mod = self.structure["modules"][mod_idx+gate_offset]
            gate_val = self.module_list[gate_mod](x[:, chunk_size*j:chunk_size*(j+1)])
            assert gate_val.shape[1] == 1
            res_val += gate_val * self.module_list[mod](x[:, 
                chunk_size*j:chunk_size*(j+1)])
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
    gate_offset = len(self.structure["modules"]) // 2
    for i in range(self.structure["out_mods"]):
        res_val = 0
        for j in range(self.structure["in_mods"]):
            mod = self.structure["modules"][mod_idx]
            gate_mod = self.structure["modules"][mod_idx+gate_offset]
            gate_val = self.module_list[gate_mod](x[:, chunk_size*j:chunk_size*(j+1)],
                    weights=weights, prefix='module_list.'+str(gate_mod)
                            +'.features.')
            assert gate_val.shape[1] == 1
            res_val += gate_val * self.module_list[mod](x[:, 
                chunk_size*j:chunk_size*(j+1)],
                weights=weights, prefix='module_list.'+str(mod)+'.features.')
            mod_idx += 1
        res.append(res_val)
    x = torch.cat(res, dim = -1)
    return x

class GatedSumStructure(VariableDimensionStructure):
  def __init__(self, args):
    self.composer = 'GatedSumComposer'
    self.composer_class = GatedSumComposer
    self.composer_abbreviation = 'M'
    super().__init__(args=args)

  def propose_new_structure(self, new_structure):
    pos = np.random.randint(len(new_structure['modules'])//2)
    gate_offset = len(new_structure["modules"]) // 2
    act_type = np.random.randint(self.num_types-1)
    act_mod = np.random.randint(self.num_modules[act_type])
    gate_mod = np.random.randint(self.num_modules[-1])
    new_structure['modules'][pos] = self.Modules[act_type][act_mod]
    new_structure['modules'][gate_offset+pos] = self.Modules[-1][gate_mod]

  def initialize_structure(self, in_dim, out_dim):
    structure = {"modules" : []}
    if out_dim % self.module_out_size != 0:
      raise ValueError("Problem dim is not a multiple of module_out_size")
    if in_dim % self.module_in_size != 0:
      raise ValueError("Problem input dim is not a multiple of module_in_size")
    in_mods = in_dim // self.module_in_size
    out_mods = out_dim // self.module_out_size
    structure_size = in_mods * out_mods
    gate_mods = []
    for _ in range(structure_size):
      act_type = np.random.randint(len(self.Modules)-1)
      act_mod = np.random.randint(self.num_modules[act_type])
      gate_mod = np.random.randint(self.num_modules[-1])
      structure['modules'].append(self.Modules[act_type][act_mod])
      gate_mods.append(self.Modules[-1][gate_mod])
    structure["modules"] += gate_mods
    structure["out_mods"] = out_mods
    structure["in_mods"] = in_mods
    structure["chunk_size"] = self.module_in_size
    return structure

  def update_Usage_counters(self, METRICS, T):
    return self.default_update_Usage_counters(METRICS, T)

  def modules_given_structure(self, structure):
    return structure['modules']
