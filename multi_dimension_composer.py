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
from IPython import embed

class multi_dimension_Composer(Composer):
  def __init__(self, composer, module_list, loss_fn=None, structure={},
      instructions={}):
    super(multi_dimension_Composer, self).__init__(composer=composer,
        module_list=module_list, loss_fn=loss_fn,
        structure=structure, instructions=instructions)

  def forward_no_weights(self, x):
    res = []
    # self.structure['modules'] contains a list of length n
    # where n is the number of output dimension
    for mod in self.structure['modules']:
      res.append(self.module_list[mod](x))
    x = torch.cat(res, dim = -1)
    return x

  def forward_with_weights(self, x, weights):
    res = []
    for mod in self.structure['modules']:
      res.append(self.module_list[mod](x,
        weights=weights, prefix='module_list.'+str(mod)+'.features.'))
    x = torch.cat(res, dim = -1)
    return x

class multi_dimension_Structure(Structure):
  def __init__(self, args):
    self.composer = 'multiDimComposer'
    self.composer_class = multi_dimension_Composer
    self.composer_abbreviation = 'C'
    self.structure_size = args.structure_size
    super(multi_dimension_Structure, self).__init__(args=args)

  def propose_new_structure(self, new_structure):
    return self.default_propose_new_structure(new_structure)

  def initialize_structure(self):
    return self.default_initialize_structure()

  def update_Usage_counters(self, METRICS, T):
    return self.default_update_Usage_counters(METRICS, T)

  def modules_given_structure(self, structure):
    return structure['modules']
