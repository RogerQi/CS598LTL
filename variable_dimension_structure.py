import numpy as np
from structure import Structure 
from tqdm import tqdm as Tqdm

class VariableDimensionStructure(Structure):
  def default_initialize_structure(self, dimension):
    structure = {"modules" : []}
    if dimension % self.module_out_size != 0:
      raise ValueError("Problem dim is not a multiple of module_out_size")
    structure_size = dimension // self.module_out_size
    for _ in range(structure_size):
      act_type = np.random.randint(len(self.Modules))
      act_mod = np.random.randint(self.num_modules[act_type])
      structure['modules'].append(self.Modules[act_type][act_mod])
    return structure

  def initialize_all_structures(self, T, mtrain_copies=1):
    self.TrainStructures = [None for _ in range(mtrain_copies * T.mtrain)]
    self.ValStructures = [None for _ in T.MVAL]
    for i in Tqdm(range(len(self.TrainStructures))):
      output_shape = T.MTRAIN[i%T.mtrain].original_output_shape
      self.TrainStructures[i] = self.initialize_structure(dimension=output_shape[1])
      self.TrainStructures[i]['original_input_shape'] = (
              T.MTRAIN[i%T.mtrain].original_input_shape)
      self.TrainStructures[i]['original_output_shape'] = output_shape
    for i in range(len(self.ValStructures)):
      output_shape = T.MVAL[i].original_output_shape
      self.ValStructures[i] = self.initialize_structure(dimension=output_shape[1])
      self.ValStructures[i]['original_input_shape'] = (
              T.MVAL[i].original_input_shape)
      self.ValStructures[i]['original_output_shape'] = output_shape

  def default_propose_new_structure(self, new_structure):
    pos = np.random.randint(len(new_structure['modules']))
    act_type = np.random.randint(self.num_types)
    act_mod = np.random.randint(self.num_modules[act_type])
    new_structure['modules'][pos] = self.Modules[act_type][act_mod]
