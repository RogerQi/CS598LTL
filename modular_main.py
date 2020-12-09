import argparse
from tensorboardX import SummaryWriter
import pickle

from modular_metalearning import BounceGrad
from plot_ndim_sines import plot_ndim_sines
from plot_pushing_error import plot_pushing_error, save_predicted_trajs

from sum_composer import Sum_Structure
from functioncomposition_composer import FunctionComposition_Structure
from multi_dimension_composer import multi_dimension_Structure
from multi_sum_composer import MultiSumStructure
from concat_composer import ConcatStructure
from gated_sum_composer import GatedSumStructure

def main():
  #########
  # Flags #
  #########
  parser = argparse.ArgumentParser()
  # Compute flags
  parser.add_argument('--device', dest='nn_device', default='cuda:0',
      help='what device we want to run things in')
  # Data flags
  parser.add_argument('--data_desc', dest='data_desc',
      help='description of data source')
  parser.add_argument('--pickle_data', dest="pickle_data",
      help='Save the meta-dataset in a pickle', default="", type=str)
  parser.add_argument('--limit_data', dest='limit_data', type=int,
      help='maximum number of points per dataset', default=10000)
  parser.add_argument('--max_datasets', dest='max_datasets', type=int,
      default=256, help='maximum number of datasets')
  parser.add_argument('--data_split', dest='data_split', default='20,80,0',
      help='comma-separated distribution (in %) of train,val,test per dataset')
  parser.add_argument('--meta_split', dest='meta_split', default='90,10,0',
      help='comma-separated distribution (in %) of mtrain,mval,mtest')
  parser.add_argument('--split_by_file', dest='split_by_file',
      action='store_true')
  parser.add_argument('--smaller_MVals', dest='smaller_MVals', type=str,
      default='', help='List of extra smaller training sizes for MVal')
  parser.add_argument('--dont_normalize', dest='normalize_data',
      action='store_false')

  # BounceGrad flags
  parser.add_argument('--torch_seed', dest='torch_seed', type=int,
          default=0, help='random seed for pytorch')
  parser.add_argument('--mtrain_copies', dest='mtrain_copies', type=int,
          default=1, help='number of copies of meta-train, searching multiple \
                  structures in parallel')
  parser.add_argument('--dont_bounce', dest='do_bounce', action='store_false',
      help='Skip Simulated Annealing and proposing structures; only Grad step')
  parser.add_argument('--meta_lr', dest='meta_lr', type=float, default='1e-3',
      help='learning rate for module parameters')
  parser.add_argument('--num_modules', dest='num_modules',
      help='comma-separated list with size of each population of module type')
  parser.add_argument('--type_modules', dest='type_modules',
      help='comma-separated list describing the type of each module category')
  parser.add_argument('--composer', dest = 'composer', default='composition',
      help='Which type of composition to use; \
          for example "compositon,sum,concatenate,gnn"')
  parser.add_argument('--optimization_steps', dest='optimization_steps',
      type=int, default = 1000, help='number of BounceGrad steps')
  parser.add_argument('--meta_batch_size', dest='meta_batch_size', type=int,
      default=0, help='Number of metatrain cases between gradient steps;\
          0 if all MTRAIN')

  # MAML flags
  parser.add_argument('--MAML', dest='MAML', action='store_true',
      help='MAML loss instead of conventional loss')
  parser.add_argument('--MAML_separate', dest='MAML_separate', action='store_true',
      help='Apply MAML loss separately to modules which are in structure multiple times')
  parser.add_argument('--MAML_inner_updates', dest='MAML_inner_updates',
      type=int, default = 5, help='number of gradient steps in the inner loop')
  parser.add_argument('--MAML_step_size', dest='MAML_step_size', type=float,
      default = 1e-2, help='step size in MAML gradient steps')

  # Plotting flags
  parser.add_argument('--plot_name', dest='plot_name',
      default='default', help='Name for error plot')
  parser.add_argument('--store_video', dest='store_video', action='store_true',
       help='Store images every step into video folder')
  parser.add_argument('--plot_ymax', dest='plot_ymax', type=float,
      default = -1., help='maximum y in zoomed loss plot')
  parser.add_argument('--plot_freq', dest='plot_freq', type=int,
      default=5, help='Number of optimization steps between plots')

  # Flags mostly useful for restarting experiments
  parser.add_argument('--load_modules', dest='load_modules', type=str,
      default='', help='Filepath to load modules from self.L;\
          empty if dont want to load')
  parser.add_argument('--load_structures_and_metrics',
      dest='load_structures_and_metrics', type=str, default='',
      help='Filepath to load structures and metrics;\
          empty if dont want to load')
  parser.add_argument('--save_modules', dest='save_modules', type=str,
      default='', help='Filepath to save modules from self.L;\
          if empty use plot_name')
  parser.add_argument('--initial_temp', dest='initial_temp', type=float,
      default = 0, help='[log] initial temperature')
  parser.add_argument('--initial_acc', dest='initial_acc', type=float,
      default = 0, help='[log] initial acceptance ratio')

  # Flag for running tests
  parser.add_argument("--test", dest="test", action="store_true")
  parser.add_argument("--test_steps", dest="test_steps", type=int, default=-1)
  parser.add_argument("--save_test_structures", dest="save_test_structures", 
          type=str, default="")
  parser.add_argument("--load_test_structures", dest="load_test_structures", 
          type=str, default="")

  parser.add_argument("--plot_ndim_sines", dest="plot_ndim_sines", action="store_true")
  parser.add_argument("--plot_pushing_error", dest="plot_pushing_error", 
          action="store_true")
  parser.add_argument("--save_predicted_trajs", dest="save_predicted_trajs", 
          type=str, default="")

  # Parsing args
  args = parser.parse_args()

  # Tensorboard writer
  tensorboardX_writer = SummaryWriter(
      comment='composer='+args.composer+
      '_optsteps='+str(args.optimization_steps)+
      '_copies='+str(args.mtrain_copies)+
      '_data='+args.data_desc.split('/')[-1].split('.')[0]+'_'+
      str(args.meta_lr))

  # Finding composer
  composer = args.composer
  if composer.startswith('sum'):
    [composer, args.structure_size] = composer.split('-')
    args.structure_size=int(args.structure_size)
    S = Sum_Structure(args=args)
  elif composer.startswith('functionComposition'):
    [composer, args.structure_size] = composer.split('-')
    args.structure_size=int(args.structure_size)
    S = FunctionComposition_Structure(args=args)
  elif composer.startswith('two_dim_function_composition'):
    [composer, args.structure_size] = composer.split('-')
    args.structure_size=int(args.structure_size)
    S = multi_dimension_Structure(args=args)
  elif composer.startswith("concat_composition"):
    S = ConcatStructure(args=args)
  elif composer.startswith("multisum_composition"):
    S = MultiSumStructure(args=args)
  elif composer.startswith("gated_sum_composition"):
    S = GatedSumStructure(args=args)
  else:
    raise NotImplementedError

  bg = BounceGrad(S=S, args=args, tensorboardX_writer=tensorboardX_writer)
  if not args.test:
    bg.SAConfig_SGDModules(args.optimization_steps)
  else:
    if not args.load_test_structures:
      bg.run_test(args.test_steps)
    else:
      with open(args.load_test_structures, "rb") as f:
        bg.S.TestStructures = pickle.load(f)
    if args.save_test_structures:
      with open(args.save_test_structures, "wb") as f:
        pickle.dump(bg.S.TestStructures, f)

  if args.plot_ndim_sines:
    plot_ndim_sines(bg)

  if args.plot_pushing_error:
    plot_pushing_error(bg)

  if args.save_predicted_trajs:
    save_predicted_trajs(bg, out_dir=args.save_predicted_trajs)


if __name__ == '__main__':
  main()
