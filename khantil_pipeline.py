from termcolor import colored
from parlai.core.teachers import register_teacher, DialogTeacher
from parlai.scripts.display_data import DisplayData
import torch
torch.cuda.empty_cache()


from config import *
from parlai.scripts.train_model import TrainModel


pre_path = ""

@register_teacher("my_teacher")
class MyTeacher(DialogTeacher):
    def __init__(self, opt, shared=None):
        opt['datafile'] = opt['datatype'].split(':')[0] + ".txt"
        super().__init__(opt, shared)
        
    
    def setup_data(self, datafile):
        print(f" ~~ Loading from {datafile} ~~ ")
        
        out = preprocessData(pre_path)
        for b, kk in enumerate(out):
            for a, i in enumerate(kk):
                if i[0] != "" and i[1] != "":
                    if a == 0:
                        yield (i[0], i[1]), True
                    else:
                        yield (i[0], i[1]), False
                            
class botPipeline:
    def __init__(self, data_path):
        global pre_path
        pre_path = data_path
        DisplayData.main(task="my_teacher")
        self.model_type = 'transformer/generator',
        
        
        
    def dataTrain(self, model_saving_path, pretrain_path = None):
        if pretrain_path != None:
            dict_path = pretrain_path + ".dict"
            print("pre_model_path=", pretrain_path)
            print("pre_model_dict_path=", dict_path)
            TrainModel.main(
                task='my_teacher', 
                model=self.model_type,
                model_file = model_saving_path,

                # init_model='zoo:tutorial_transformer_generator/model',
                init_model=pretrain_path,
                n_heads=16, n_layers=8, n_positions=512, text_truncate=512,
                label_truncate=128, ffn_size=2048, embedding_size=512,
                activation='gelu', variant='xlm',
                dict_lower=True, dict_tokenizer='bpe',
                # dict_file='zoo:tutorial_transformer_generator/model.dict',
                dict_file = dict_path,
                learn_positional_embeddings=True,
                lr=lr_, 
                optimizer=optimizer_,
                warmup_updates=100,
                validation_metric=validation_metric_,
                max_train_time=max_train_time_,
                validation_every_n_epochs=validation_every_n_epochs_,
                batchsize=batchsize_,
                fp16=True, fp16_impl='mem_efficient',
                skip_generation=True,
                dynamic_batching='full',
            )
        else:
            TrainModel.main(
                
                task='my_teacher', 
                model=self.model_type,
                model_file = model_saving_path,
                n_heads=16, n_layers=8, n_positions=512, text_truncate=512,
                label_truncate=128, ffn_size=2048, embedding_size=512,
                activation='gelu', variant='xlm',
                learn_positional_embeddings=True,
                lr=lr_, 
                optimizer=optimizer_,
                warmup_updates=100,
                validation_metric=validation_metric_,
                max_train_time=max_train_time_, 
                validation_every_n_epochs=validation_every_n_epochs_,
                batchsize=batchsize_, 
                fp16=True, fp16_impl='mem_efficient',
                skip_generation=True,
                dynamic_batching='full',
            )
            

        
        