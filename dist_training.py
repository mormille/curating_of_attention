"""

Copy-paste from torch.nn.Transformer with modifications:
    * positional encodings are passed in MHattention
    * extra LN at the end of encoder is removed
    * decoder returns a stack of activations from all decoding layers
"""

from models.backbone import Backbone, EmbeddingNetwork
from models.encoder import EncoderModule
from models.utils.joiner3 import Joiner
from models.utils.losses import Attention_penalty_factor, Generator_loss
from models.unet import UNet

from fastai.vision.all import *
from fastai.distributed import *
from fastai.callback.tracker import SaveModelCallback
from fastprogress import fastprogress

import warnings

warnings.filterwarnings('ignore')
torch.backends.cudnn.benchmark = True
fastprogress.MAX_COLS = 120


def load_data(bs):
    path = rank0_first(lambda: untar_data(URLs.CIFAR)) 
    
    transform = [*aug_transforms(), Normalize.from_stats([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])]

    datablock = DataBlock(blocks=(ImageBlock, CategoryBlock), 
                     get_items=get_image_files, 
                     splitter=RandomSplitter(),
                     get_y=parent_label,
                     batch_tfms=transform)

    return datablock.dataloaders(path/'train', bs=bs, num_workers=num_cpus())


# path = untar_data(URLs.CIFAR)

# transform = ([*aug_transforms(),Normalize.from_stats([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])

# data = DataBlock(blocks=(ImageBlock, CategoryBlock), 
#                   get_items=get_image_files, 
#                   splitter=RandomSplitter(),
#                   get_y=parent_label,
#                   batch_tfms=transform)

if torch.cuda.is_available():
    print("CUDA Available")
    print(torch.cuda.device_count())
else:
    print("No CUDA")

@call_parse
def main(gpu: Param("GPU to run on", int) = 3,
         bs: Param("Batch size", int) = 25,
         runs: Param("Number of times to repeat training", int) = 1):
  
    gpu = setup_distrib(gpu)
    
    if gpu is not None: torch.cuda.set_device(gpu)
    n_gpu = torch.cuda.device_count()

    #data_loader = data.dataloaders(path,bs=bs) 
    data_loader = load_data(bs)
    model_dir = Path.home()/'Luiz/saved_models'
    
    
    net = load_learner(model_dir/'Untrained_model.pkl', cpu=False)
    
    #model = Joiner(backbone = False, bypass=False, hidden_dim=256, batch_size=25, image_h=32, image_w=32,grid_l=4,penalty_factor="2")
    
    save_model = SaveModelCallback(monitor='accuracy', fname='Trained_model')
    
    model_learner = Learner(data_loader, net.model, metrics=[accuracy,error_rate], cbs=[save_model], model_dir=model_dir)

    for run in range(runs):
        print(f'Run: {run}')

        # The old way to use DataParallel, or DistributedDataParallel training:
        # if gpu is None and n_gpu: learn.to_parallel()
        # if num_distrib()>1: learn.to_distributed(gpu) # Requires `-m fastai.launch`

        # the context manager way of dp/ddp, both can handle single GPU base case.
        ctx = model_learner.parallel_ctx if gpu is None and n_gpu else model_learner.distrib_ctx

        with partial(ctx, gpu)():  # distributed training requires "-m fastai.launch"
            print(f"Training in {ctx.__name__} context on GPU {gpu if gpu is not None else list(range(n_gpu))}")
            model_learner.fit_flat_cos(1, 3e-5)
        model_learner.export(model_dir/'Trained_model.pkl')
