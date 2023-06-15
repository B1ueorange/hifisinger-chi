import os
import pickle
import cProfile

import collections
import random

os.environ['FOR_DISABLE_CONSOLE_CTRL_HANDLER'] = 'T'    # This is ot prevent to be called Fortran Ctrl+C crash in Windows.

import torch
import numpy as np
import logging, yaml, sys, argparse, math
from tqdm import tqdm
from collections import defaultdict
import matplotlib
import torch.multiprocessing as mp
matplotlib.use('agg')
import matplotlib.pyplot as plt
from scipy.io import wavfile
import librosa

from Modules import HifiSinger, Discriminators, Gradient_Penalty
from Datasets import Dataset, Inference_Dataset, Collater, Inference_Collater, Dataset_Test, Collater1
from Radam import RAdam
from Noam_Scheduler import Modified_Noam_Scheduler
from Logger import Logger

from distributed import init_distributed, apply_gradient_allreduce, reduce_tensor
from Arg_Parser import Recursive_Parse

from torch.utils.data import DataLoader
from prefetch_generator import BackgroundGenerator
import Calc_Mel

class DataLoaderX(DataLoader):

    def __iter__(self):
        return BackgroundGenerator(super().__iter__())

def get_contributing_params(y, top_level=True):
    nf = y.grad_fn.next_functions if top_level else y.next_functions
    for f, _ in nf:
        try:
            yield f.variable
        except AttributeError:
            pass  # node has no tensor
        if f is not None:
            yield from get_contributing_params(f, top_level=False)

class Trainer:
    def __init__(self, hp_path, steps=0, gpu_id=0):  # 初始化
        self.hp_Path = hp_path
        # self.gpu_id = int(os.getenv('RANK', '0'))
        self.gpu_id = gpu_id
        self.num_gpus = int(os.getenv("WORLD_SIZE", '1'))

        self.hp = Recursive_Parse(yaml.load(
            open(self.hp_Path, encoding='utf-8'),
            Loader=yaml.Loader
        ))

        if not torch.cuda.is_available():
            self.device = torch.device('cpu')
        else:
            self.device = torch.device('cuda:{}'.format(self.gpu_id))
            torch.backends.cudnn.enabled = True
            torch.backends.cudnn.benchmark = True
            #torch.cuda.set_device(0)

        self.steps = steps

        self.Datset_Generate()
        self.Model_Generate()
        try:
            self.Load_Checkpoint()
        except Exception as e:
            print('skip')
            print(e)
        self._Set_Distribution()

        self.scalar_Dict = {
            'Train': defaultdict(float),
            'Evaluation': defaultdict(float),
        }

        self.writer_Dict = {
            'Train': Logger(os.path.join(self.hp.Log_Path, 'Train')),
            'Evaluation': Logger(os.path.join(self.hp.Log_Path, 'Evaluation')),
        }

    def Datset_Generate(self):  # 好像是将数据输入进去
        # token_Dict = yaml.load(open(self.hp.Token_Path), Loader=yaml.Loader)  # 这玩意是啥？好奇
        token_Dict = {}
        with open('./token3.txt', 'r', encoding='utf-8') as f:
            for line in f:
                tmp1 = line.split()
                print(tmp1)
                token_Dict[tmp1[0]] = int(tmp1[2])
        token_Dict['AP'] = 70
        token_Dict['SP'] = 70
        token_Dict['<X>'] = 70

        train_Dataset = Dataset(
            pattern_path=self.hp.Train.Train_Pattern.Path,
            Metadata_file=self.hp.Train.Train_Pattern.Metadata_File,
            token_dict=token_Dict,
            accumulated_dataset_epoch=self.hp.Train.Train_Pattern.Accumulated_Dataset_Epoch,
            use_cache=self.hp.Train.Use_Pattern_Cache
        )
        eval_Dataset = Dataset(  # 这个应该是测试集
            pattern_path=self.hp.Train.Eval_Pattern.Path,
            Metadata_file=self.hp.Train.Eval_Pattern.Metadata_File,
            token_dict=token_Dict,
            use_cache=self.hp.Train.Use_Pattern_Cache
        )
        eval_Dataset1 = Dataset_Test(  # 这个应该是测试集
            pattern_path=self.hp.Train.Eval_Pattern.Path,
            Metadata_file=self.hp.Train.Eval_Pattern.Metadata_File,
            token_dict=token_Dict,
            use_cache=self.hp.Train.Use_Pattern_Cache
        )
        inference_Dataset = Inference_Dataset(
            token_dict=token_Dict,
            pattern_path=self.hp.Train.Eval_Pattern.Path,
            Metadata_file=self.hp.Train.Eval_Pattern.Metadata_File,
            use_cache=False
        )

        if self.gpu_id == 0:
            logging.info('The number of train patterns = {}.'.format(train_Dataset.base_Length))
            # logging.info('The number of development patterns = {}.'.format(eval_Dataset.base_Length))
            # logging.info('The number of inference patterns = {}.'.format(len(inference_Dataset)))

        collater = Collater(  # 校对
            token_dict=token_Dict,
            max_abs_mel=self.hp.Sound.Max_Abs_Mel
        )

        collater1 = Collater1(  # 校对
            token_dict=token_Dict,
            max_abs_mel=self.hp.Sound.Max_Abs_Mel
        )
        inference_Collater = Inference_Collater(  # 这个参数更少不知道为啥
            token_dict=token_Dict,
            max_abs_mel=self.hp.Sound.Max_Abs_Mel
        )

        self.dataLoader_Dict = {}
        self.dataLoader_Dict['Train'] = DataLoaderX(
            dataset=train_Dataset,
            sampler=torch.utils.data.DistributedSampler(train_Dataset, shuffle=True) \
                if self.hp.Use_Multi_GPU else \
                torch.utils.data.RandomSampler(train_Dataset),
            collate_fn=collater,  # 将一个batch的数据和标签进行合并操作
            batch_size=self.hp.Train.Batch_Size,
            num_workers=self.hp.Train.Num_Workers,
            pin_memory=True
        )

        # self.dataLoader_Dict['Train'] = torch.utils.data.DataLoader(
        #     dataset=train_Dataset,
        #     sampler=torch.utils.data.DistributedSampler(train_Dataset, shuffle=True) \
        #         if self.hp.Use_Multi_GPU else \
        #         torch.utils.data.RandomSampler(train_Dataset),
        #     collate_fn=collater,  # 将一个batch的数据和标签进行合并操作
        #     batch_size=self.hp.Train.Batch_Size,
        #     num_workers=self.hp.Train.Num_Workers,
        #     pin_memory=True
        # )
        self.dataLoader_Dict['Eval'] = DataLoaderX(
            dataset=eval_Dataset,
            sampler=torch.utils.data.RandomSampler(eval_Dataset),
            collate_fn=collater,
            batch_size=self.hp.Train.Batch_Size,
            num_workers=self.hp.Train.Num_Workers,
            pin_memory=True
        )
        self.dataLoader_Dict['Inference'] = DataLoaderX(
            dataset=inference_Dataset,
            sampler=torch.utils.data.SequentialSampler(inference_Dataset),
            collate_fn=inference_Collater,
            batch_size=self.hp.Inference_Batch_Size or self.hp.Train.Batch_Size,
            num_workers=self.hp.Train.Num_Workers,
            pin_memory=True
        )
        self.dataLoader_Dict['Eval1'] = DataLoaderX(
            dataset=eval_Dataset1,
            sampler=torch.utils.data.RandomSampler(eval_Dataset1),
            collate_fn=collater1,
            batch_size=self.hp.Train.Batch_Size,
            num_workers=self.hp.Train.Num_Workers,
            pin_memory=True
        )

    def Model_Generate(self):
        # self.model_Dict = {
        #     'Generator': HifiSinger(self.hp).to(self.device),
        #     'Discriminator': Discriminators(self.hp).to(self.device)
        # }

        if self.hp.Use_Multi_GPU:
            self.model_Dict = {
                'Generator': torch.nn.parallel.DistributedDataParallel(
                    HifiSinger(self.hp).to(self.device),
                    device_ids=[self.gpu_id],
                    # find_unused_parameters=True
                    ),
                'Discriminator': torch.nn.parallel.DistributedDataParallel(
                    Discriminators(self.hp).to(self.device),
                    device_ids=[self.gpu_id],
                    # find_unused_parameters=True
                    )
                }
        else:
            self.model_Dict = {
                'Generator': HifiSinger(self.hp).to(self.device),
                'Discriminator': Discriminators(self.hp).to(self.device)
                }

        self.model_Dict['Generator'].requires_grad_(False)
        self.model_Dict['Discriminator'].requires_grad_(False)

        self.criterion_Dict = {
            'Mean_Absolute_Error': torch.nn.L1Loss(reduction='none').to(self.device),
            'Gradient_Penalty': Gradient_Penalty(
                gamma=self.hp.Train.Discriminator_Gradient_Panelty_Gamma
            ).to(self.device),
        }

        self.optimizer_Dict = {
            'Generator': RAdam(
                params=self.model_Dict['Generator'].parameters(),
                lr=self.hp.Train.Learning_Rate.Generator.Initial,
                betas=(self.hp.Train.ADAM.Beta1, self.hp.Train.ADAM.Beta2),
                eps=self.hp.Train.ADAM.Epsilon,
                weight_decay=self.hp.Train.Weight_Decay
            ),
            'Discriminator': RAdam(
                params=self.model_Dict['Discriminator'].parameters(),
                lr=self.hp.Train.Learning_Rate.Discriminator.Initial,
                betas=(self.hp.Train.ADAM.Beta1, self.hp.Train.ADAM.Beta2),
                eps=self.hp.Train.ADAM.Epsilon,
                weight_decay=self.hp.Train.Weight_Decay
            )
        }

        self.scheduler_Dict = {
            'Generator': Modified_Noam_Scheduler(
                optimizer=self.optimizer_Dict['Generator'],
                base=self.hp.Train.Learning_Rate.Generator.Base
            ),
            'Discriminator': Modified_Noam_Scheduler(
                optimizer=self.optimizer_Dict['Discriminator'],
                base=self.hp.Train.Learning_Rate.Discriminator.Base
            )
        }

        self.scaler = torch.cuda.amp.GradScaler(enabled=self.hp.Use_Mixed_Precision)  # pytorch的自动混合精度 减少计算压力

        self.vocoder = None
        # if not self.hp.Vocoder_Path is None:
        #     self.vocoder = torch.jit.load(self.hp.Vocoder_Path).to(self.device)

        if self.gpu_id == 0:
            logging.info('#' * 100)
            logging.info('Generator structure')
            logging.info(self.model_Dict['Generator'])
            logging.info('#' * 100)
            logging.info('Discriminator structure')
            logging.info(self.model_Dict['Discriminator'])

    def Train_Step(self, durations, tokens, notes, token_lengths, mels, silences, pitches, mel_lengths, speaker_embeddings=None):
        loss_Dict = {}

        durations = durations.to(self.device, non_blocking=True)  # 拷贝数据到gpu/cpu
        tokens = tokens.to(self.device, non_blocking=True)
        notes = notes.to(self.device, non_blocking=True)
        token_lengths = token_lengths.to(self.device, non_blocking=True)
        mels = mels.to(self.device, non_blocking=True)
        silences = silences.to(self.device, non_blocking=True)
        pitches = pitches.to(self.device, non_blocking=True)
        mel_lengths = mel_lengths.to(self.device, non_blocking=True)
        if speaker_embeddings is not None:
            speaker_embeddings = speaker_embeddings.to(self.device, non_blocking=True)

        with torch.cuda.amp.autocast(enabled=self.hp.Use_Mixed_Precision):  # 混合精度的训练机制？
            # Generator loss
            self.optimizer_Dict['Generator'].zero_grad()
            self.optimizer_Dict['Discriminator'].zero_grad()
            self.model_Dict['Generator'].requires_grad_(True)
            # self.model_Dict['Discriminator'].requires_grad_(False)
            predicted_Mels, predicted_Silences, predicted_Pitches, predicted_Durations = self.model_Dict['Generator'](
                durations=durations,
                tokens=tokens,
                notes=notes,
                token_lengths=token_lengths,
                speaker_embeddings=speaker_embeddings
            )  # 这里好像就是丢进去训练？？？
            # print('\n')
            # print(self.steps)
            # if(self.steps == 1):
            #     print('biaoji')
            # discriminations = []
            # try:
            #     discriminations = self.model_Dict['Discriminator'](predicted_Mels, mel_lengths)
            # except:
            # contributing_parameters = set(get_contributing_params(self.model_Dict['Discriminator'](predicted_Mels, mel_lengths)))
            # all_parameters = set(self.model_Dict['Discriminator'].parameters())
            # non_contributing = all_parameters - contributing_parameters
            # print(non_contributing)
            # for param in self.model_Dict['Discriminator'].parameters():
            #     if param.grad is None:
            #         print(param)
            # self.model_Dict['Discriminator'].requires_grad_(True)
            discriminations = self.model_Dict['Discriminator'](predicted_Mels, mel_lengths)
            # self.model_Dict['Discriminator'].requires_grad_(False)
            loss_Dict['Mel'] = self.criterion_Dict['Mean_Absolute_Error'](predicted_Mels, mels)
            loss_Dict['Mel'] = loss_Dict['Mel'].sum(dim=2).mean(dim=1) / mel_lengths.float()
            loss_Dict['Mel'] = loss_Dict['Mel'].mean()
            loss_Dict['Silence'] = self.criterion_Dict['Mean_Absolute_Error'](predicted_Silences,
                                                                              silences)  # BCE is faster, but loss increase infinity because the silence cannot tracking perfectly.
            loss_Dict['Silence'] = loss_Dict['Silence'].sum(dim=1) / mel_lengths.float()
            loss_Dict['Silence'] = loss_Dict['Silence'].mean()
            loss_Dict['Pitch'] = self.criterion_Dict['Mean_Absolute_Error'](predicted_Pitches, pitches)
            loss_Dict['Pitch'] = loss_Dict['Pitch'].sum(dim=1) / mel_lengths.float()
            loss_Dict['Pitch'] = loss_Dict['Pitch'].mean()
            loss_Dict['Predicted_Duration'] = self.criterion_Dict['Mean_Absolute_Error'](predicted_Durations,
                                                                                         durations.float()).mean()
            loss_Dict['Adversarial'] = torch.stack(
                [torch.nn.functional.softplus(-x).mean() for x in discriminations]).sum()  # 这个是啥
            loss_Dict['Generator'] = loss_Dict['Mel'] + loss_Dict['Silence'] + loss_Dict['Pitch'] + loss_Dict[
                'Predicted_Duration'] + loss_Dict['Adversarial']
            # loss_Dict['Generator'] = loss_Dict['Mel'] + loss_Dict['Silence'] + loss_Dict['Pitch'] + loss_Dict[
            #    'Predicted_Duration']

        self.scaler.scale(loss_Dict['Generator']).backward()
        if self.hp.Train.Gradient_Norm > 0.0:
            self.scaler.unscale_(self.optimizer_Dict['Generator'])
            torch.nn.utils.clip_grad_norm_(  # 梯度剪裁 防止下降过快，应该是出自wgan的思想
                parameters=self.model_Dict['Generator'].parameters(),
                max_norm=self.hp.Train.Gradient_Norm
            )

        self.model_Dict['Generator'].requires_grad_(False)
        self.scaler.step(self.optimizer_Dict['Generator'])
        self.scaler.update()
        self.scheduler_Dict['Generator'].step()

        with torch.cuda.amp.autocast(enabled=self.hp.Use_Mixed_Precision):
            # Fake discrimination
            self.optimizer_Dict['Discriminator'].zero_grad()
            self.model_Dict['Discriminator'].requires_grad_(True)
            fakes, *_ = self.model_Dict['Generator'](
                durations=durations,
                tokens=tokens,
                notes=notes,
                token_lengths=token_lengths,
                speaker_embeddings=speaker_embeddings
            )
            discriminations = self.model_Dict['Discriminator'](fakes, mel_lengths)
            loss_Dict['Fake'] = torch.stack([torch.nn.functional.softplus(x).mean() for x in discriminations]).sum()

        self.scaler.scale(loss_Dict['Fake']).backward()
        if self.hp.Train.Gradient_Norm > 0.0:  # 反向传播discriminator，这边传的是生成的样本吧
            self.scaler.unscale_(self.optimizer_Dict['Discriminator'])
            torch.nn.utils.clip_grad_norm_(
                parameters=self.model_Dict['Discriminator'].parameters(),
                max_norm=self.hp.Train.Gradient_Norm
            )

        self.model_Dict['Discriminator'].requires_grad_(False)
        self.scaler.step(self.optimizer_Dict['Discriminator'])
        self.scaler.update()
        self.scheduler_Dict['Discriminator'].step()

        with torch.cuda.amp.autocast(enabled=self.hp.Use_Mixed_Precision):  # 这边传的就是正确的答案做判别
            # Real discrimination
            self.optimizer_Dict['Discriminator'].zero_grad()
            self.model_Dict['Discriminator'].requires_grad_(True)
            discriminations = self.model_Dict['Discriminator'](mels, mel_lengths)
            loss_Dict['Real'] = torch.stack([torch.nn.functional.softplus(-x).mean() for x in discriminations]).sum()

        self.scaler.scale(loss_Dict['Real']).backward()
        if self.hp.Train.Gradient_Norm > 0.0:
            self.scaler.unscale_(self.optimizer_Dict['Discriminator'])
            torch.nn.utils.clip_grad_norm_(
                parameters=self.model_Dict['Discriminator'].parameters(),
                max_norm=self.hp.Train.Gradient_Norm
            )

        self.model_Dict['Discriminator'].requires_grad_(False)
        self.scaler.step(self.optimizer_Dict['Discriminator'])
        self.scaler.update()
        self.scheduler_Dict['Discriminator'].step()

        # with torch.cuda.amp.autocast(enabled=self.hp.Use_Mixed_Precision):  # 怎么个梯度惩罚法
        #     # Gradient penalty
        #     reals_for_GP = mels.detach().requires_grad_(True)  # This is required to calculate the gradient penalties.
        #     self.optimizer_Dict['Discriminator'].zero_grad()
        #     self.model_Dict['Discriminator'].requires_grad_(True)
        #     discriminations = self.model_Dict['Discriminator'](reals_for_GP, mel_lengths)
        #     loss_Dict['Gradient_Penalty'] = self.criterion_Dict['Gradient_Penalty'](
        #         reals=reals_for_GP,
        #         discriminations=torch.stack(discriminations, dim=-1).sum(dim=(1, 2, 3))
        #     )
        #
        # self.scaler.scale(loss_Dict['Gradient_Penalty']).backward()
        # if self.hp.Train.Gradient_Norm > 0.0:
        #     self.scaler.unscale_(self.optimizer_Dict['Discriminator'])
        #     torch.nn.utils.clip_grad_norm_(
        #         parameters=self.model_Dict['Discriminator'].parameters(),
        #         max_norm=self.hp.Train.Gradient_Norm
        #     )
        #
        # self.model_Dict['Discriminator'].requires_grad_(False)
        # self.scaler.step(self.optimizer_Dict['Discriminator'])
        # self.scaler.update()
        # self.scheduler_Dict['Discriminator'].step()

        # for param in self.model_Dict['Discriminator'].parameters():
        #     if param.grad is None:
        #         print(param)

        self.steps += 1
        self.tqdm.update(1)

        # print("The step has finished.")

        for tag, loss in loss_Dict.items():
            loss = reduce_tensor(loss.data, self.num_gpus).item() if self.num_gpus > 1 else loss.item()
            self.scalar_Dict['Train']['Loss/{}'.format(tag)] += loss

    def Train_Epoch(self):
        for durations, tokens, notes, token_lengths, mels, silences, pitches, mel_lengths, speaker_embeddings in self.dataLoader_Dict[
            'Train']:
            self.Train_Step(durations, tokens, notes, token_lengths, mels, silences, pitches, mel_lengths, speaker_embeddings)

            if self.steps % self.hp.Train.Checkpoint_Save_Interval == 0:
                self.Save_Checkpoint()  # 存档

            if self.steps % self.hp.Train.Logging_Interval == 0:  #
                self.scalar_Dict['Train'] = {
                    tag: loss / self.hp.Train.Logging_Interval
                    for tag, loss in self.scalar_Dict['Train'].items()
                }
                self.scalar_Dict['Train']['Learning_Rate/Generator'] = self.scheduler_Dict['Generator'].get_last_lr()
                self.scalar_Dict['Train']['Learning_Rate/Discriminator'] = self.scheduler_Dict[
                    'Discriminator'].get_last_lr()
                self.writer_Dict['Train'].add_scalar_dict(self.scalar_Dict['Train'], self.steps)
                self.scalar_Dict['Train'] = defaultdict(float)

            # if self.steps % self.hp.Train.Evaluation_Interval == 0:
            #     self.Evaluation_Epoch()
            #
            # if self.steps % self.hp.Train.Inference_Interval == 0:
            #     self.Inference_Epoch()

            if self.steps >= self.hp.Train.Max_Step:
                return

    # @torch.no_grad()  Gradient needs to calculate gradient penalty losses.
    def Evaluation_Step(self, durations, tokens, notes, token_lengths, mels, silences, pitches, mel_lengths, speaker_embeddings=None):
        loss_Dict = {}

        durations = durations.to(self.device, non_blocking=True)
        tokens = tokens.to(self.device, non_blocking=True)
        notes = notes.to(self.device, non_blocking=True)
        token_lengths = token_lengths.to(self.device, non_blocking=True)
        mels = mels.to(self.device, non_blocking=True)
        silences = silences.to(self.device, non_blocking=True)
        pitches = pitches.to(self.device, non_blocking=True)
        mel_lengths = mel_lengths.to(self.device, non_blocking=True)
        if speaker_embeddings is not None:
            speaker_embeddings = speaker_embeddings.to(self.device, non_blocking=True)

        # Generator loss
        predicted_Mels, predicted_Silences, predicted_Pitches, predicted_Durations = self.model_Dict['Generator'](
            durations=durations,
            tokens=tokens,
            notes=notes,
            token_lengths=token_lengths,
            speaker_embeddings=speaker_embeddings
        )  # 我认为后头这段真的是无所谓了
        discriminations = self.model_Dict['Discriminator'](predicted_Mels, mel_lengths)
        loss_Dict['Mel'] = self.criterion_Dict['Mean_Absolute_Error'](predicted_Mels, mels)
        loss_Dict['Mel'] = loss_Dict['Mel'].sum(dim=2).mean(dim=1) / mel_lengths.float()
        loss_Dict['Mel'] = loss_Dict['Mel'].mean()
        loss_Dict['Silence'] = self.criterion_Dict['Mean_Absolute_Error'](predicted_Silences,
                                                                          silences)  # BCE is faster, but loss increase infinity because the silence cannot tracking perfectly.
        loss_Dict['Silence'] = loss_Dict['Silence'].sum(dim=1) / mel_lengths.float()
        loss_Dict['Silence'] = loss_Dict['Silence'].mean()
        loss_Dict['Pitch'] = self.criterion_Dict['Mean_Absolute_Error'](predicted_Pitches, pitches)
        loss_Dict['Pitch'] = loss_Dict['Pitch'].sum(dim=1) / mel_lengths.float()
        loss_Dict['Pitch'] = loss_Dict['Pitch'].mean()
        loss_Dict['Predicted_Duration'] = self.criterion_Dict['Mean_Absolute_Error'](predicted_Durations,
                                                                                     durations.float()).mean()
        loss_Dict['Adversarial'] = torch.stack([torch.nn.functional.softplus(-x).mean() for x in discriminations]).sum()
        loss_Dict['Generator'] = loss_Dict['Mel'] + loss_Dict['Silence'] + loss_Dict['Pitch'] + loss_Dict[
            'Predicted_Duration'] + loss_Dict['Adversarial']

        # Fake discrimination
        fakes, *_ = self.model_Dict['Generator'](
            durations=durations,
            tokens=tokens,
            notes=notes,
            token_lengths=token_lengths,
            speaker_embeddings=speaker_embeddings
        )
        discriminations = self.model_Dict['Discriminator'](fakes, mel_lengths)
        loss_Dict['Fake'] = torch.stack([torch.nn.functional.softplus(x).mean() for x in discriminations]).sum()

        # Real discrimination
        discriminations = self.model_Dict['Discriminator'](mels, mel_lengths)
        loss_Dict['Real'] = torch.stack([torch.nn.functional.softplus(-x).mean() for x in discriminations]).sum()

        # Gradient penalty
        reals_for_GP = mels.detach().requires_grad_(True)  # This is required to calculate the gradient penalties.
        self.optimizer_Dict['Discriminator'].zero_grad()
        # self.model_Dict['Discriminator'].requires_grad_(True)
        # discriminations = self.model_Dict['Discriminator'](reals_for_GP, mel_lengths)
        # loss_Dict['Gradient_Penalty'] = self.criterion_Dict['Gradient_Penalty'](
        #     reals=reals_for_GP,
        #     discriminations=torch.stack(discriminations, dim=-1).sum(dim=(1, 2, 3))
        # )
        # self.model_Dict['Discriminator'].requires_grad_(False)
        # self.optimizer_Dict['Discriminator'].zero_grad()

        for tag, loss in loss_Dict.items():
            loss = reduce_tensor(loss.data, self.num_gpus).item() if self.num_gpus > 1 else loss.item()
            self.scalar_Dict['Evaluation']['Loss/{}'.format(tag)] += loss
        return predicted_Mels, predicted_Silences, predicted_Pitches, predicted_Durations

    def Evaluation_Epoch(self):
        if self.gpu_id != 0:
            return

        logging.info('(Steps: {}) Start evaluation in GPU {}.'.format(self.steps, self.gpu_id))

        self.model_Dict['Generator'].eval()
        self.model_Dict['Discriminator'].eval()

        for step, (durations, tokens, notes, token_lengths, mels, silences, pitches, mel_lengths, speaker_embeddings) in tqdm(
                enumerate(self.dataLoader_Dict['Eval'], 1),
                desc='[Evaluation]',
                total=math.ceil(len(self.dataLoader_Dict['Eval'].dataset) / self.hp.Train.Batch_Size)
        ):
            predicted_Mels, predicted_Silences, predicted_Pitches, predicted_Durations = self.Evaluation_Step(durations,
                                                                                                              tokens,
                                                                                                              notes,
                                                                                                              token_lengths,
                                                                                                              mels,
                                                                                                              silences,
                                                                                                              pitches,
                                                                                                              mel_lengths,
                                                                                                              speaker_embeddings)

        self.scalar_Dict['Evaluation'] = {
            tag: loss / step
            for tag, loss in self.scalar_Dict['Evaluation'].items()
        }
        self.writer_Dict['Evaluation'].add_scalar_dict(self.scalar_Dict['Evaluation'], self.steps)
        self.writer_Dict['Evaluation'].add_histogram_model(self.model_Dict['Generator'], 'Generator', self.steps,
                                                           delete_keywords=['layer_Dict', 'layer'])
        self.writer_Dict['Evaluation'].add_histogram_model(self.model_Dict['Discriminator'], 'Discriminator',
                                                           self.steps, delete_keywords=['layer_Dict', 'layer'])
        self.scalar_Dict['Evaluation'] = defaultdict(float)

        duration = durations[-1]
        duration = torch.arange(duration.size(0)).repeat_interleave(duration.cpu()).numpy()
        predicted_Duration = predicted_Durations[-1].ceil().long().clamp(0, self.hp.Max_Duration)
        predicted_Duration = torch.arange(predicted_Duration.size(0)).repeat_interleave(
            predicted_Duration.cpu()).numpy()
        image_Dict = {
            'Mel/Target': (mels[-1, :, :mel_lengths[-1]].cpu().numpy(), None),
            'Mel/Prediction': (predicted_Mels[-1, :, :mel_lengths[-1]].cpu().numpy(), None),
            'Silence/Target': (silences[-1, :mel_lengths[-1]].cpu().numpy(), None),
            'Silence/Prediction': (predicted_Silences[-1, :mel_lengths[-1]].cpu().numpy(), None),
            'Pitch/Target': (pitches[-1, :mel_lengths[-1]].cpu().numpy(), None),
            'Pitch/Prediction': (predicted_Pitches[-1, :mel_lengths[-1]].cpu().numpy(), None),
            'Duration/Target': (duration, None),
            'Duration/Prediction': (predicted_Duration, None),
        }
        self.writer_Dict['Evaluation'].add_image_dict(image_Dict, self.steps)

        self.model_Dict['Generator'].train()
        self.model_Dict['Discriminator'].train()

    def Evaluation_Step_Test(self, durations, tokens, notes, token_lengths, mels, silences, pitches, mel_lengths, speaker_embeddings, labels):
        loss_Dict = {}

        durations = durations.to(self.device, non_blocking=True)
        tokens = tokens.to(self.device, non_blocking=True)
        notes = notes.to(self.device, non_blocking=True)
        token_lengths = token_lengths.to(self.device, non_blocking=True)
        mels = mels.to(self.device, non_blocking=True)
        silences = silences.to(self.device, non_blocking=True)
        pitches = pitches.to(self.device, non_blocking=True)
        mel_lengths = mel_lengths.to(self.device, non_blocking=True)
        if speaker_embeddings is not None:
            speaker_embeddings = speaker_embeddings.to(self.device, non_blocking=True)

        # Generator loss
        predicted_Mels, predicted_Silences, predicted_Pitches, predicted_Durations = self.model_Dict['Generator'](
            durations=durations,
            tokens=tokens,
            notes=notes,
            token_lengths=token_lengths,
            speaker_embeddings=speaker_embeddings
        )  # 我认为后头这段真的是无所谓了
        discriminations = self.model_Dict['Discriminator'](predicted_Mels, mel_lengths)
        loss_Dict['Mel'] = self.criterion_Dict['Mean_Absolute_Error'](predicted_Mels, mels)
        loss_Dict['Mel'] = loss_Dict['Mel'].sum(dim=2).mean(dim=1) / mel_lengths.float()
        loss_Dict['Mel'] = loss_Dict['Mel'].mean()
        loss_Dict['Silence'] = self.criterion_Dict['Mean_Absolute_Error'](predicted_Silences,
                                                                          silences)  # BCE is faster, but loss increase infinity because the silence cannot tracking perfectly.
        loss_Dict['Silence'] = loss_Dict['Silence'].sum(dim=1) / mel_lengths.float()
        loss_Dict['Silence'] = loss_Dict['Silence'].mean()
        loss_Dict['Pitch'] = self.criterion_Dict['Mean_Absolute_Error'](predicted_Pitches, pitches)
        loss_Dict['Pitch'] = loss_Dict['Pitch'].sum(dim=1) / mel_lengths.float()
        loss_Dict['Pitch'] = loss_Dict['Pitch'].mean()
        loss_Dict['Predicted_Duration'] = self.criterion_Dict['Mean_Absolute_Error'](predicted_Durations,
                                                                                     durations.float()).mean()
        loss_Dict['Adversarial'] = torch.stack([torch.nn.functional.softplus(-x).mean() for x in discriminations]).sum()
        loss_Dict['Generator'] = loss_Dict['Mel'] + loss_Dict['Silence'] + loss_Dict['Pitch'] + loss_Dict[
            'Predicted_Duration'] + loss_Dict['Adversarial']

        # Fake discrimination
        fakes, *_ = self.model_Dict['Generator'](
            durations=durations,
            tokens=tokens,
            notes=notes,
            token_lengths=token_lengths,
            speaker_embeddings=speaker_embeddings
        )
        discriminations = self.model_Dict['Discriminator'](fakes, mel_lengths)
        loss_Dict['Fake'] = torch.stack([torch.nn.functional.softplus(x).mean() for x in discriminations]).sum()

        # Real discrimination
        discriminations = self.model_Dict['Discriminator'](mels, mel_lengths)
        loss_Dict['Real'] = torch.stack([torch.nn.functional.softplus(-x).mean() for x in discriminations]).sum()

        # Gradient penalty
        reals_for_GP = mels.detach().requires_grad_(True)  # This is required to calculate the gradient penalties.
        self.optimizer_Dict['Discriminator'].zero_grad()
        self.model_Dict['Discriminator'].requires_grad_(True)
        discriminations = self.model_Dict['Discriminator'](reals_for_GP, mel_lengths)
        loss_Dict['Gradient_Penalty'] = self.criterion_Dict['Gradient_Penalty'](
            reals=reals_for_GP,
            discriminations=torch.stack(discriminations, dim=-1).sum(dim=(1, 2, 3))
        )
        self.model_Dict['Discriminator'].requires_grad_(False)
        self.optimizer_Dict['Discriminator'].zero_grad()

        for tag, loss in loss_Dict.items():
            loss = reduce_tensor(loss.data, self.num_gpus).item() if self.num_gpus > 1 else loss.item()
            self.scalar_Dict['Evaluation']['Loss/{}'.format(tag)] += loss
        cnm = 0
        for mel, true_mel, silence, pitch, duration in zip(
                predicted_Mels.cpu(),
                mels.cpu(),
                predicted_Silences.cpu(),
                predicted_Pitches.cpu(),
                predicted_Durations.cpu(),
        ):
            cnm += 1
            # if(cnm >= 11):
            #     continue
            title = 'Note infomation: {}_{}'.format(labels, cnm)
            new_Figure = plt.figure(figsize=(25, 5 * 5), dpi=100)
            plt.subplot2grid((4, 1), (0, 0))
            plt.imshow(mel, aspect='auto', origin='lower')
            plt.title('Mel    {}'.format(title))
            plt.colorbar()
            plt.subplot2grid((4, 1), (1, 0))
            plt.imshow(true_mel, aspect='auto', origin='lower')
            plt.title('Ground_Truth_Mel    {}'.format(title))
            plt.colorbar()
            # plt.subplot2grid((4, 1), (2, 0))
            # plt.plot(silence)
            # plt.margins(x=0)
            # plt.title('Silence    {}'.format(title))
            # plt.colorbar()
            plt.subplot2grid((4, 1), (2, 0))
            plt.plot(pitch)
            plt.margins(x=0)
            plt.title('Pitch    {}'.format(title))
            plt.colorbar()
            duration = duration.ceil().long().clamp(0, self.hp.Max_Duration)
            duration = torch.arange(duration.size(0)).repeat_interleave(duration)
            plt.subplot2grid((4, 1), (3, 0))
            plt.plot(duration)
            plt.margins(x=0)
            plt.title('Duration    {}'.format(title))
            plt.colorbar()
            plt.tight_layout()
            plt.savefig(os.path.join(self.hp.Inference_Path, 'Step-{}'.format(self.steps), 'PNG',
                                     '{}_{}.png'.format(labels, cnm)).replace('\\', '/'))
            plt.close(new_Figure)

            # wav = librosa.feature.inverse.mel_to_audio(true_mel.numpy(),
            #                                            sr=self.hp.Sound.Sample_Rate,
            #                                            n_fft=2048,
            #                                            hop_length=240,
            #                                            win_length=960,
            #                                            window='hann',
            #                                            center=True,
            #                                            pad_mode='reflect',
            #                                            power=2.0
            #                                            )
            # # wav = Calc_Mel.melspectrogram2wav(mel.numpy())
            # wavfile.write(
            #     filename=os.path.join(self.hp.Inference_Path, 'Step-{}'.format(self.steps), 'Wav',
            #                           'groundtruth_{}.wav'.format(labels)).replace('\\', '/'),
            #     data=(np.clip(wav, -1.0 + 1e-7, 1.0 - 1e-7) * 32767.5).astype(np.int16),
            #     rate=self.hp.Sound.Sample_Rate
            # )
            np.save(
                os.path.join(self.hp.Inference_Path, 'Step-{}'.format(self.steps), 'NPY', 'Mel', str(labels) + '_' + str(cnm)).replace('\\',
                                                                                                               '/'),
                mel.T,
                allow_pickle=False
            )
            np.save(
                os.path.join(self.hp.Inference_Path, 'Step-{}'.format(self.steps), 'NPY', 'Mel', str(labels) + '_' + str(cnm) + '_groundtruth').replace(
                    '\\',
                    '/'),
                true_mel.T,
                allow_pickle=False
            )
            # wav = librosa.feature.inverse.mel_to_audio(mel.numpy(),
            #                                            sr=self.hp.Sound.Sample_Rate,
            #                                            n_fft=2048,
            #                                            hop_length=240,
            #                                            win_length=960,
            #                                            window='hann',
            #                                            center=True,
            #                                            pad_mode='reflect',
            #                                            power=2.0
            #                                            )
            # # wav = Calc_Mel.melspectrogram2wav(mel.numpy())
            # wavfile.write(
            #             filename=os.path.join(self.hp.Inference_Path, 'Step-{}'.format(self.steps), 'Wav',
            #                                   'predict_{}.wav'.format(labels)).replace('\\', '/'),
            #             data=(np.clip(wav, -1.0 + 1e-7, 1.0 - 1e-7) * 32767.5).astype(np.int16),
            #             rate=self.hp.Sound.Sample_Rate
            #         )
        # if not self.vocoder is None:
        #     os.makedirs(os.path.join(self.hp.Inference_Path, 'Step-{}'.format(self.steps), 'Wav').replace('\\', '/'),
        #                 exist_ok=True)
        #     for mel, silence, pitch in zip(predicted_Mels, predicted_Silences, predicted_Pitches):
        #         mel = mel.unsqueeze(0)
        #         silence = silence.unsqueeze(0)
        #         pitch = pitch.unsqueeze(0)
        #         x = torch.randn(size=(mel.size(0), self.hp.Sound.Frame_Shift * mel.size(2))).to(mel.device)
        #         mel = torch.nn.functional.pad(mel, (2, 2), 'reflect')
        #         silence = torch.nn.functional.pad(silence.unsqueeze(dim=1), (2, 2), 'reflect').squeeze(dim=1)
        #         pitch = torch.nn.functional.pad(pitch.unsqueeze(dim=1), (2, 2), 'reflect').squeeze(dim=1)
        #
        #         wav = self.vocoder(x, mel, silence, pitch).cpu().numpy()[0]
        #         wavfile.write(
        #             filename=os.path.join(self.hp.Inference_Path, 'Step-{}'.format(self.steps), 'Wav',
        #                                   'predict_{}.wav'.format(labels)).replace('\\', '/'),
        #             data=(np.clip(wav, -1.0 + 1e-7, 1.0 - 1e-7) * 32767.5).astype(np.int16),
        #             rate=self.hp.Sound.Sample_Rate
        #         )
        #     for mel, silence, pitch in zip(mels, silences, pitches):
        #         mel = mel.unsqueeze(0)
        #         silence = silence.unsqueeze(0)
        #         pitch = pitch.unsqueeze(0)
        #         x = torch.randn(size=(mel.size(0), self.hp.Sound.Frame_Shift * mel.size(2))).to(mel.device)
        #         mel = torch.nn.functional.pad(mel, (2, 2), 'reflect')
        #         silence = torch.nn.functional.pad(silence.unsqueeze(dim=1), (2, 2), 'reflect').squeeze(dim=1)
        #         pitch = torch.nn.functional.pad(pitch.unsqueeze(dim=1), (2, 2), 'reflect').squeeze(dim=1)
        #
        #         wav = self.vocoder(x, mel, silence, pitch).cpu().numpy()[0]
        #         wavfile.write(
        #             filename=os.path.join(self.hp.Inference_Path, 'Step-{}'.format(self.steps), 'Wav',
        #                                   'groundtruth_{}.wav'.format(labels)).replace('\\', '/'),
        #             data=(np.clip(wav, -1.0 + 1e-7, 1.0 - 1e-7) * 32767.5).astype(np.int16),
        #             rate=self.hp.Sound.Sample_Rate
        #         )
        return predicted_Mels, predicted_Silences, predicted_Pitches, predicted_Durations

    def Evaluation_Epoch_Test(self):
        if self.gpu_id != 0:
            return

        logging.info('(Steps: {}) Start evaluation in GPU {}.'.format(self.steps, self.gpu_id))

        self.model_Dict['Generator'].eval()
        self.model_Dict['Discriminator'].eval()

        cnt = -1
        for step, (durations, tokens, notes, token_lengths, mels, silences, pitches, mel_lengths, speaker_embeddings) in tqdm(
                enumerate(self.dataLoader_Dict['Eval'], 1),
                desc='[Evaluation]',
                total=math.ceil(len(self.dataLoader_Dict['Eval'].dataset) / self.hp.Train.Batch_Size)
        ):
            cnt += 1
            predicted_Mels, predicted_Silences, predicted_Pitches, predicted_Durations = self.Evaluation_Step_Test(
                durations,
                tokens,
                notes,
                token_lengths,
                mels,
                silences,
                pitches,
                mel_lengths,
                speaker_embeddings,
                cnt)
            # with torch.autograd.profiler.profile(enabled=True, use_cuda=False, record_shapes=False,
            #                                      profile_memory=False) as prof:

             #   print(prof.table())

        self.scalar_Dict['Evaluation'] = {
            tag: loss / step
            for tag, loss in self.scalar_Dict['Evaluation'].items()
        }
        self.writer_Dict['Evaluation'].add_scalar_dict(self.scalar_Dict['Evaluation'], self.steps)
        self.writer_Dict['Evaluation'].add_histogram_model(self.model_Dict['Generator'], 'Generator', self.steps,
                                                           delete_keywords=['layer_Dict', 'layer'])
        self.writer_Dict['Evaluation'].add_histogram_model(self.model_Dict['Discriminator'], 'Discriminator',
                                                           self.steps, delete_keywords=['layer_Dict', 'layer'])
        self.scalar_Dict['Evaluation'] = defaultdict(float)

        duration = durations[-1]
        duration = torch.arange(duration.size(0)).repeat_interleave(duration.cpu()).numpy()
        predicted_Duration = predicted_Durations[-1].ceil().long().clamp(0, self.hp.Max_Duration)
        predicted_Duration = torch.arange(predicted_Duration.size(0)).repeat_interleave(
            predicted_Duration.cpu()).numpy()
        image_Dict = {
            'Mel/Target': (mels[-1, :, :mel_lengths[-1]].cpu().numpy(), None),
            'Mel/Prediction': (predicted_Mels[-1, :, :mel_lengths[-1]].cpu().numpy(), None),
            'Silence/Target': (silences[-1, :mel_lengths[-1]].cpu().numpy(), None),
            'Silence/Prediction': (predicted_Silences[-1, :mel_lengths[-1]].cpu().numpy(), None),
            'Pitch/Target': (pitches[-1, :mel_lengths[-1]].cpu().numpy(), None),
            'Pitch/Prediction': (predicted_Pitches[-1, :mel_lengths[-1]].cpu().numpy(), None),
            'Duration/Target': (duration, None),
            'Duration/Prediction': (predicted_Duration, None),
        }
        self.writer_Dict['Evaluation'].add_image_dict(image_Dict, self.steps)

        self.model_Dict['Generator'].train()
        self.model_Dict['Discriminator'].train()

    def Evaluation_Epoch_Test1(self):
        if self.gpu_id != 0:
            return

        logging.info('(Steps: {}) Start evaluation in GPU {}.'.format(self.steps, self.gpu_id))

        self.model_Dict['Generator'].eval()
        self.model_Dict['Discriminator'].eval()

        cnt = -1
        for step, (durations, tokens, notes, token_lengths, mels, silences, pitches, mel_lengths, speaker_embeddings, paths) in tqdm(
                enumerate(self.dataLoader_Dict['Eval1'], 1),
                desc='[Evaluation]',
                total=math.ceil(len(self.dataLoader_Dict['Eval1'].dataset) / self.hp.Train.Batch_Size)
        ):
            for true_mel, silence, pitch, duration, path in zip(
                    mels.cpu(),
                    silences.cpu(),
                    pitches.cpu(),
                    durations.cpu(),
                    paths
            ):
                cnt += 1
                title = 'Note infomation: {}'.format(path)
                print(path)
                pattern_content = pickle.load(open(path, 'rb'))
                mellenl = pattern_content['Mel'].shape
                mellen = mellenl[0]
                new_Figure = plt.figure(figsize=(25, 5 * 5), dpi=100)
                plt.subplot2grid((4, 1), (0, 0))
                plt.imshow(true_mel[:, :mellen], aspect='auto', origin='lower')
                plt.title('Ground_Truth_Mel    {}'.format(title))
                plt.colorbar()
                # plt.subplot2grid((4, 1), (1, 0))
                # plt.plot(silence)
                # plt.margins(x=0)
                # plt.title('Silence    {}'.format(title))
                # plt.colorbar()
                plt.subplot2grid((4, 1), (1, 0))
                plt.imshow(pattern_content['Mel'].T, aspect='auto', origin='lower')
                plt.title('Raw_Ground_Truth_Mel    {}'.format(title))
                plt.colorbar()
                plt.subplot2grid((4, 1), (2, 0))
                plt.plot(pitch)
                plt.margins(x=0)
                plt.title('Pitch    {}'.format(title))
                plt.colorbar()
                # duration = duration.ceil().long().clamp(0, self.hp.Max_Duration)
                # duration = torch.arange(duration.size(0)).repeat_interleave(duration)
                duration = torch.tensor(duration)
                duration = duration.long().clamp(0, self.hp.Max_Duration)
                duration = torch.arange(duration.size(0)).repeat_interleave(duration)
                plt.subplot2grid((4, 1), (3, 0))
                plt.plot(duration)
                plt.margins(x=0)
                plt.title('Duration    {}'.format(title))
                plt.colorbar()
                plt.tight_layout()
                plt.savefig(os.path.join(self.hp.Inference_Path, 'Step-{}'.format(self.steps), 'PNG',
                                         '{}.png'.format(cnt)).replace('\\', '/'))
                plt.close(new_Figure)
                np.save(
                    os.path.join(self.hp.Inference_Path, 'Step-{}'.format(self.steps), 'NPY', 'Mel',
                                 str(cnt) + '_groundtruth').replace(
                        '\\',
                        '/'),
                    true_mel.T,
                    allow_pickle=False
                )
                np.save(
                    os.path.join(self.hp.Inference_Path, 'Step-{}'.format(self.steps), 'NPY', 'Mel',
                                 str(cnt) + '_rawgroundtruth').replace(
                        '\\',
                        '/'),
                    pattern_content['Mel'],
                    allow_pickle=False
                )

    def Evaluation_Epoch_Test2(self):
        if self.gpu_id != 0:
            return

        logging.info('(Steps: {}) Start evaluation in GPU {}.'.format(self.steps, self.gpu_id))

        self.model_Dict['Generator'].eval()
        self.model_Dict['Discriminator'].eval()
        aim_cnt = random.randint(0, math.ceil(len(self.dataLoader_Dict['Eval'].dataset) / self.hp.Train.Batch_Size))

        cnt = -1
        for step, (durations, tokens, notes, token_lengths, mels, silences, pitches, mel_lengths, speaker_embeddings) in tqdm(
                enumerate(self.dataLoader_Dict['Eval'], 1),
                desc='[Evaluation]',
                total=math.ceil(len(self.dataLoader_Dict['Eval'].dataset) / self.hp.Train.Batch_Size)
        ):
            cnt += 1
            if(cnt != aim_cnt):
                continue
            predicted_Mels, predicted_Silences, predicted_Pitches, predicted_Durations = self.Evaluation_Step_Test(
                durations,
                tokens,
                notes,
                token_lengths,
                mels,
                silences,
                pitches,
                mel_lengths,
                speaker_embeddings,
                cnt)
            # with torch.autograd.profiler.profile(enabled=True, use_cuda=False, record_shapes=False,
            #                                      profile_memory=False) as prof:

             #   print(prof.table())

        self.scalar_Dict['Evaluation'] = {
            tag: loss / step
            for tag, loss in self.scalar_Dict['Evaluation'].items()
        }
        self.writer_Dict['Evaluation'].add_scalar_dict(self.scalar_Dict['Evaluation'], self.steps)
        self.writer_Dict['Evaluation'].add_histogram_model(self.model_Dict['Generator'], 'Generator', self.steps,
                                                           delete_keywords=['layer_Dict', 'layer'])
        self.writer_Dict['Evaluation'].add_histogram_model(self.model_Dict['Discriminator'], 'Discriminator',
                                                           self.steps, delete_keywords=['layer_Dict', 'layer'])
        self.scalar_Dict['Evaluation'] = defaultdict(float)

        duration = durations[-1]
        duration = torch.arange(duration.size(0)).repeat_interleave(duration.cpu()).numpy()
        predicted_Duration = predicted_Durations[-1].ceil().long().clamp(0, self.hp.Max_Duration)
        predicted_Duration = torch.arange(predicted_Duration.size(0)).repeat_interleave(
            predicted_Duration.cpu()).numpy()
        image_Dict = {
            'Mel/Target': (mels[-1, :, :mel_lengths[-1]].cpu().numpy(), None),
            'Mel/Prediction': (predicted_Mels[-1, :, :mel_lengths[-1]].cpu().numpy(), None),
            'Silence/Target': (silences[-1, :mel_lengths[-1]].cpu().numpy(), None),
            'Silence/Prediction': (predicted_Silences[-1, :mel_lengths[-1]].cpu().numpy(), None),
            'Pitch/Target': (pitches[-1, :mel_lengths[-1]].cpu().numpy(), None),
            'Pitch/Prediction': (predicted_Pitches[-1, :mel_lengths[-1]].cpu().numpy(), None),
            'Duration/Target': (duration, None),
            'Duration/Prediction': (predicted_Duration, None),
        }
        self.writer_Dict['Evaluation'].add_image_dict(image_Dict, self.steps)

        self.model_Dict['Generator'].train()
        self.model_Dict['Discriminator'].train()

    @torch.no_grad()
    def Inference_Step(self, durations, tokens, notes, token_lengths, speaker_embeddings, labels, start_index=0, tag_step=False):
        durations = durations.to(self.device, non_blocking=True)
        tokens = tokens.to(self.device, non_blocking=True)
        notes = notes.to(self.device, non_blocking=True)
        token_lengths = token_lengths.to(self.device, non_blocking=True)
        if speaker_embeddings is not None:
            speaker_embeddings = speaker_embeddings.to(self.device, non_blocking=True)

        predicted_Mels, predicted_Silences, predicted_Pitches, predicted_Durations = self.model_Dict['Generator'](
            durations=durations,
            tokens=tokens,
            notes=notes,
            token_lengths=token_lengths,
            speaker_embeddings=speaker_embeddings
        )

        files = []
        for index, label in enumerate(labels):
            tags = []
            if tag_step: tags.append('Step-{}'.format(self.steps))
            tags.append(label)
            tags.append('IDX_{}'.format(index + start_index))
            files.append('.'.join(tags))

        os.makedirs(os.path.join(self.hp.Inference_Path, 'Step-{}'.format(self.steps), 'PNG').replace('\\', '/'),
                    exist_ok=True)
        os.makedirs(os.path.join(self.hp.Inference_Path, 'Step-{}'.format(self.steps), 'NPY', 'Mel').replace('\\', '/'),
                    exist_ok=True)
        for mel, silence, pitch, duration, label, file in zip(
                predicted_Mels.cpu(),
                predicted_Silences.cpu(),
                predicted_Pitches.cpu(),
                predicted_Durations.cpu(),
                labels,
                files
        ):
            title = 'Note infomation: {}'.format(label)
            new_Figure = plt.figure(figsize=(20, 5 * 4), dpi=100)
            plt.subplot2grid((4, 1), (0, 0))
            plt.imshow(mel, aspect='auto', origin='lower')
            plt.title('Mel    {}'.format(title))
            plt.colorbar()
            plt.subplot2grid((4, 1), (1, 0))
            plt.plot(silence)
            plt.margins(x=0)
            plt.title('Silence    {}'.format(title))
            plt.colorbar()
            plt.subplot2grid((4, 1), (2, 0))
            plt.plot(pitch)
            plt.margins(x=0)
            plt.title('Pitch    {}'.format(title))
            plt.colorbar()
            duration = duration.ceil().long().clamp(0, self.hp.Max_Duration)
            duration = torch.arange(duration.size(0)).repeat_interleave(duration)
            plt.subplot2grid((4, 1), (3, 0))
            plt.plot(duration)
            plt.margins(x=0)
            plt.title('Duration    {}'.format(title))
            plt.colorbar()
            plt.tight_layout()
            plt.savefig(os.path.join(self.hp.Inference_Path, 'Step-{}'.format(self.steps), 'PNG',
                                     '{}.png'.format(file)).replace('\\', '/'))
            plt.close(new_Figure)

            np.save(
                os.path.join(self.hp.Inference_Path, 'Step-{}'.format(self.steps), 'NPY', 'Mel', file).replace('\\',
                                                                                                               '/'),
                mel.T,
                allow_pickle=False
            )

        # This part may be changed depending on the vocoder used.
        if not self.vocoder is None:
            os.makedirs(os.path.join(self.hp.Inference_Path, 'Step-{}'.format(self.steps), 'Wav').replace('\\', '/'),
                        exist_ok=True)
            for mel, silence, pitch, file in zip(predicted_Mels, predicted_Silences, predicted_Pitches, files):
                mel = mel.unsqueeze(0)
                silence = silence.unsqueeze(0)
                pitch = pitch.unsqueeze(0)
                x = torch.randn(size=(mel.size(0), self.hp.Sound.Frame_Shift * mel.size(2))).to(mel.device)
                mel = torch.nn.functional.pad(mel, (2, 2), 'reflect')
                silence = torch.nn.functional.pad(silence.unsqueeze(dim=1), (2, 2), 'reflect').squeeze(dim=1)
                pitch = torch.nn.functional.pad(pitch.unsqueeze(dim=1), (2, 2), 'reflect').squeeze(dim=1)

                wav = self.vocoder(x, mel, silence, pitch).cpu().numpy()[0]
                wavfile.write(
                    filename=os.path.join(self.hp.Inference_Path, 'Step-{}'.format(self.steps), 'Wav',
                                          '{}.wav'.format(file)).replace('\\', '/'),
                    data=(np.clip(wav, -1.0 + 1e-7, 1.0 - 1e-7) * 32767.5).astype(np.int16),
                    rate=self.hp.Sound.Sample_Rate
                )

    def Inference_Epoch(self):
        if self.gpu_id != 0:
            return

        logging.info('(Steps: {}) Start inference in GPU {}.'.format(self.steps, self.gpu_id))

        self.model_Dict['Generator'].eval()

        for step, (durations, tokens, notes, token_lengths, labels) in tqdm(
                enumerate(self.dataLoader_Dict['Inference']),
                desc='[Inference]',
                total=math.ceil(len(self.dataLoader_Dict['Inference'].dataset) / (
                        self.hp.Inference_Batch_Size or self.hp.Train.Batch_Size))
        ):
            self.Inference_Step(durations, tokens, notes, token_lengths, labels,
                                start_index=step * (self.hp.Inference_Batch_Size or self.hp.Train.Batch_Size))

        self.model_Dict['Generator'].train()

    def Inference_Test(self, pos):
        if self.gpu_id != 0:
            return
        cnt = -1

        logging.info('(Steps: {}) Start inference in GPU {}.'.format(self.steps, self.gpu_id))

        self.model_Dict['Generator'].eval()

        for step, (durations, tokens, notes, token_lengths, speaker_embeddings, labels) in tqdm(
                enumerate(self.dataLoader_Dict['Inference']),
                desc='[Inference]',
                total=math.ceil(len(self.dataLoader_Dict['Inference'].dataset) / (
                        self.hp.Inference_Batch_Size or self.hp.Train.Batch_Size))
        ):
            cnt += 1
            if(cnt != pos):
                continue
            self.Inference_Step(durations, tokens, notes, token_lengths, speaker_embeddings, labels,
                                start_index=step * (self.hp.Inference_Batch_Size or self.hp.Train.Batch_Size))

        self.model_Dict['Generator'].train()

    def Load_Checkpoint(self):
        if self.steps == 0:
            # paths = [
            #     os.path.join(root, file).replace('\\', '/')
            #     for root, _, files in os.walk(self.hp.Checkpoint_Path)
            #     for file in files
            #     if os.path.splitext(file)[1] == '.pt'
            # ]
            # if len(paths) > 0:
            #     path = max(paths, key=os.path.getctime)
            # else:
            #     return  # Initial training
            return
        else:
            path = os.path.join(self.hp.Checkpoint_Path, 'S_{}.pt'.format(self.steps).replace('\\', '/'))
        # self.model_Dict['Generator'] = HifiSinger(self.hp).to(self.device)
        # self.model_Dict['Discriminator'] = Discriminators(self.hp).to(self.device)
        # Error(s) in loading
        # state_dict
        # for DistributedDataParallel:
        #     Missing
        #     key(s) in state_dict: "module.layer_Dict.Encoder.layer_Dict.Phoneme_Embedding.weight", "module.layer_Dict.Encoder.layer_Dict.Duration_Embedding.weight", "module.layer_Dict.Encoder.layer_Dict.Note_Embedding.weight", "module.layer_Dict.Encoder.layer_Dict.Positional_Embedding.pe", "module.layer_Dict.Encoder.layer_Dict.FFT_Block_0.slf_attn.w_qs.weight", "module.layer_Dict.Encoder.layer_Dict.FFT_Block_0.slf_attn.w_qs.bias", "module.layer_Dict.Encoder.layer_Dict.FFT_Block_0.slf_attn.w_ks.weight", "module.layer_Dict.Encoder.layer_Dict.FFT_Block_0.slf_attn.w_ks.bias", "module.layer_Dict.Encoder.layer_Dict.FFT_Block_0.slf_attn.w_vs.weight", "module.layer_Dict.Encoder.layer_Dict.FFT_Block_0.slf_attn.w_vs.bias", "module.layer_Dict.Encoder.layer_Dict.FFT_Block_0.slf_attn.layer_norm.weight", "module.layer_Dict.Encoder.layer_Dict.FFT_Block_0.slf_attn.layer_norm.bias", "module.layer_Dict.Encoder.layer_Dict.FFT_Block_0.slf_attn.fc.weight", "module.layer_Dict.Encoder.layer_Dict.FFT_Block_0.slf_attn.fc.bias", "module.layer_Dict.Encoder.layer_Dict.FFT_Block_0.pos_ffn.w_1.weight", "module.layer_Dict.Encoder.layer_Dict.FFT_Block_0.pos_ffn.w_1.bias", "module.layer_Dict.Encoder.layer_Dict.FFT_Block_0.pos_ffn.w_2.weight", "module.layer_Dict.Encoder.layer_Dict.FFT_Block_0.pos_ffn.w_2.bias", "module.layer_Dict.Encoder.layer_Dict.FFT_Block_0.pos_ffn.layer_norm.weight", "module.layer_Dict.Encoder.layer_Dict.FFT_Block_0.pos_ffn.layer_norm.bias", "module.layer_Dict.Encoder.layer_Dict.FFT_Block_1.slf_attn.w_qs.weight", "module.layer_Dict.Encoder.layer_Dict.FFT_Block_1.slf_attn.w_qs.bias", "module.layer_Dict.Encoder.layer_Dict.FFT_Block_1.slf_attn.w_ks.weight", "module.layer_Dict.Encoder.layer_Dict.FFT_Block_1.slf_attn.w_ks.bias", "module.layer_Dict.Encoder.layer_Dict.FFT_Block_1.slf_attn.w_vs.weight", "module.layer_Dict.Encoder.layer_Dict.FFT_Block_1.slf_attn.w_vs.bias", "module.layer_Dict.Encoder.layer_Dict.FFT_Block_1.slf_attn.layer_norm.weight", "module.layer_Dict.Encoder.layer_Dict.FFT_Block_1.slf_attn.layer_norm.bias", "module.layer_Dict.Encoder.layer_Dict.FFT_Block_1.slf_attn.fc.weight", "module.layer_Dict.Encoder.layer_Dict.FFT_Block_1.slf_attn.fc.bias", "module.layer_Dict.Encoder.layer_Dict.FFT_Block_1.pos_ffn.w_1.weight", "module.layer_Dict.Encoder.layer_Dict.FFT_Block_1.pos_ffn.w_1.bias", "module.layer_Dict.Encoder.layer_Dict.FFT_Block_1.pos_ffn.w_2.weight", "module.layer_Dict.Encoder.layer_Dict.FFT_Block_1.pos_ffn.w_2.bias", "module.layer_Dict.Encoder.layer_Dict.FFT_Block_1.pos_ffn.layer_norm.weight", "module.layer_Dict.Encoder.layer_Dict.FFT_Block_1.pos_ffn.layer_norm.bias", "module.layer_Dict.Encoder.layer_Dict.FFT_Block_2.slf_attn.w_qs.weight", "module.layer_Dict.Encoder.layer_Dict.FFT_Block_2.slf_attn.w_qs.bias", "module.layer_Dict.Encoder.layer_Dict.FFT_Block_2.slf_attn.w_ks.weight", "module.layer_Dict.Encoder.layer_Dict.FFT_Block_2.slf_attn.w_ks.bias", "module.layer_Dict.Encoder.layer_Dict.FFT_Block_2.slf_attn.w_vs.weight", "module.layer_Dict.Encoder.layer_Dict.FFT_Block_2.slf_attn.w_vs.bias", "module.layer_Dict.Encoder.layer_Dict.FFT_Block_2.slf_attn.layer_norm.weight", "module.layer_Dict.Encoder.layer_Dict.FFT_Block_2.slf_attn.layer_norm.bias", "module.layer_Dict.Encoder.layer_Dict.FFT_Block_2.slf_attn.fc.weight", "module.layer_Dict.Encoder.layer_Dict.FFT_Block_2.slf_attn.fc.bias", "module.layer_Dict.Encoder.layer_Dict.FFT_Block_2.pos_ffn.w_1.weight", "module.layer_Dict.Encoder.layer_Dict.FFT_Block_2.pos_ffn.w_1.bias", "module.layer_Dict.Encoder.layer_Dict.FFT_Block_2.pos_ffn.w_2.weight", "module.layer_Dict.Encoder.layer_Dict.FFT_Block_2.pos_ffn.w_2.bias", "module.layer_Dict.Encoder.layer_Dict.FFT_Block_2.pos_ffn.layer_norm.weight", "module.layer_Dict.Encoder.layer_Dict.FFT_Block_2.pos_ffn.layer_norm.bias", "module.layer_Dict.Encoder.layer_Dict.FFT_Block_3.slf_attn.w_qs.weight", "module.layer_Dict.Encoder.layer_Dict.FFT_Block_3.slf_attn.w_qs.bias", "module.layer_Dict.Encoder.layer_Dict.FFT_Block_3.slf_attn.w_ks.weight", "module.layer_Dict.Encoder.layer_Dict.FFT_Block_3.slf_attn.w_ks.bias", "module.layer_Dict.Encoder.layer_Dict.FFT_Block_3.slf_attn.w_vs.weight", "module.layer_Dict.Encoder.layer_Dict.FFT_Block_3.slf_attn.w_vs.bias", "module.layer_Dict.Encoder.layer_Dict.FFT_Block_3.slf_attn.layer_norm.weight", "module.layer_Dict.Encoder.layer_Dict.FFT_Block_3.slf_attn.layer_norm.bias", "module.layer_Dict.Encoder.layer_Dict.FFT_Block_3.slf_attn.fc.weight", "module.layer_Dict.Encoder.layer_Dict.FFT_Block_3.slf_attn.fc.bias", "module.layer_Dict.Encoder.layer_Dict.FFT_Block_3.pos_ffn.w_1.weight", "module.layer_Dict.Encoder.layer_Dict.FFT_Block_3.pos_ffn.w_1.bias", "module.layer_Dict.Encoder.layer_Dict.FFT_Block_3.pos_ffn.w_2.weight", "module.layer_Dict.Encoder.layer_Dict.FFT_Block_3.pos_ffn.w_2.bias", "module.layer_Dict.Encoder.layer_Dict.FFT_Block_3.pos_ffn.layer_norm.weight", "module.layer_Dict.Encoder.layer_Dict.FFT_Block_3.pos_ffn.layer_norm.bias", "module.layer_Dict.Encoder.layer_Dict.FFT_Block_4.slf_attn.w_qs.weight", "module.layer_Dict.Encoder.layer_Dict.FFT_Block_4.slf_attn.w_qs.bias", "module.layer_Dict.Encoder.layer_Dict.FFT_Block_4.slf_attn.w_ks.weight", "module.layer_Dict.Encoder.layer_Dict.FFT_Block_4.slf_attn.w_ks.bias", "module.layer_Dict.Encoder.layer_Dict.FFT_Block_4.slf_attn.w_vs.weight", "module.layer_Dict.Encoder.layer_Dict.FFT_Block_4.slf_attn.w_vs.bias", "module.layer_Dict.Encoder.layer_Dict.FFT_Block_4.slf_attn.layer_norm.weight", "module.layer_Dict.Encoder.layer_Dict.FFT_Block_4.slf_attn.layer_norm.bias", "module.layer_Dict.Encoder.layer_Dict.FFT_Block_4.slf_attn.fc.weight", "module.layer_Dict.Encoder.layer_Dict.FFT_Block_4.slf_attn.fc.bias", "module.layer_Dict.Encoder.layer_Dict.FFT_Block_4.pos_ffn.w_1.weight", "module.layer_Dict.Encoder.layer_Dict.FFT_Block_4.pos_ffn.w_1.bias", "module.layer_Dict.Encoder.layer_Dict.FFT_Block_4.pos_ffn.w_2.weight", "module.layer_Dict.Encoder.layer_Dict.FFT_Block_4.pos_ffn.w_2.bias", "module.layer_Dict.Encoder.layer_Dict.FFT_Block_4.pos_ffn.layer_norm.weight", "module.layer_Dict.Encoder.layer_Dict.FFT_Block_4.pos_ffn.layer_norm.bias", "module.layer_Dict.Encoder.layer_Dict.FFT_Block_5.slf_attn.w_qs.weight", "module.layer_Dict.Encoder.layer_Dict.FFT_Block_5.slf_attn.w_qs.bias", "module.layer_Dict.Encoder.layer_Dict.FFT_Block_5.slf_attn.w_ks.weight", "module.layer_Dict.Encoder.layer_Dict.FFT_Block_5.slf_attn.w_ks.bias", "module.layer_Dict.Encoder.layer_Dict.FFT_Block_5.slf_attn.w_vs.weight", "module.layer_Dict.Encoder.layer_Dict.FFT_Block_5.slf_attn.w_vs.bias", "module.layer_Dict.Encoder.layer_Dict.FFT_Block_5.slf_attn.layer_norm.weight", "module.layer_Dict.Encoder.layer_Dict.FFT_Block_5.slf_attn.layer_norm.bias", "module.layer_Dict.Encoder.layer_Dict.FFT_Block_5.slf_attn.fc.weight", "module.layer_Dict.Encoder.layer_Dict.FFT_Block_5.slf_attn.fc.bias", "module.layer_Dict.Encoder.layer_Dict.FFT_Block_5.pos_ffn.w_1.weight", "module.layer_Dict.Encoder.layer_Dict.FFT_Block_5.pos_ffn.w_1.bias", "module.layer_Dict.Encoder.layer_Dict.FFT_Block_5.pos_ffn.w_2.weight", "module.layer_Dict.Encoder.layer_Dict.FFT_Block_5.pos_ffn.w_2.bias", "module.layer_Dict.Encoder.layer_Dict.FFT_Block_5.pos_ffn.layer_norm.weight", "module.layer_Dict.Encoder.layer_Dict.FFT_Block_5.pos_ffn.layer_norm.bias", "module.layer_Dict.Duration_Predictor.layer_Dict.Conv_0.weight", "module.layer_Dict.Duration_Predictor.layer_Dict.Conv_0.bias", "module.layer_Dict.Duration_Predictor.layer_Dict.LayerNorm_0.weight", "module.layer_Dict.Duration_Predictor.layer_Dict.LayerNorm_0.bias", "module.layer_Dict.Duration_Predictor.layer_Dict.Conv_1.weight", "module.layer_Dict.Duration_Predictor.layer_Dict.Conv_1.bias", "module.layer_Dict.Duration_Predictor.layer_Dict.LayerNorm_1.weight", "module.layer_Dict.Duration_Predictor.layer_Dict.LayerNorm_1.bias", "module.layer_Dict.Duration_Predictor.layer_Dict.Projection.Conv.weight", "module.layer_Dict.Duration_Predictor.layer_Dict.Projection.Conv.bias", "module.layer_Dict.Decoder.layer_Dict.Positional_Embedding.pe", "module.layer_Dict.Decoder.layer_Dict.FFT_Block_0.slf_attn.w_qs.weight", "module.layer_Dict.Decoder.layer_Dict.FFT_Block_0.slf_attn.w_qs.bias", "module.layer_Dict.Decoder.layer_Dict.FFT_Block_0.slf_attn.w_ks.weight", "module.layer_Dict.Decoder.layer_Dict.FFT_Block_0.slf_attn.w_ks.bias", "module.layer_Dict.Decoder.layer_Dict.FFT_Block_0.slf_attn.w_vs.weight", "module.layer_Dict.Decoder.layer_Dict.FFT_Block_0.slf_attn.w_vs.bias", "module.layer_Dict.Decoder.layer_Dict.FFT_Block_0.slf_attn.layer_norm.weight", "module.layer_Dict.Decoder.layer_Dict.FFT_Block_0.slf_attn.layer_norm.bias", "module.layer_Dict.Decoder.layer_Dict.FFT_Block_0.slf_attn.fc.weight", "module.layer_Dict.Decoder.layer_Dict.FFT_Block_0.slf_attn.fc.bias", "module.layer_Dict.Decoder.layer_Dict.FFT_Block_0.pos_ffn.w_1.weight", "module.layer_Dict.Decoder.layer_Dict.FFT_Block_0.pos_ffn.w_1.bias", "module.layer_Dict.Decoder.layer_Dict.FFT_Block_0.pos_ffn.w_2.weight", "module.layer_Dict.Decoder.layer_Dict.FFT_Block_0.pos_ffn.w_2.bias", "module.layer_Dict.Decoder.layer_Dict.FFT_Block_0.pos_ffn.layer_norm.weight", "module.layer_Dict.Decoder.layer_Dict.FFT_Block_0.pos_ffn.layer_norm.bias", "module.layer_Dict.Decoder.layer_Dict.FFT_Block_1.slf_attn.w_qs.weight", "module.layer_Dict.Decoder.layer_Dict.FFT_Block_1.slf_attn.w_qs.bias", "module.layer_Dict.Decoder.layer_Dict.FFT_Block_1.slf_attn.w_ks.weight", "module.layer_Dict.Decoder.layer_Dict.FFT_Block_1.slf_attn.w_ks.bias", "module.layer_Dict.Decoder.layer_Dict.FFT_Block_1.slf_attn.w_vs.weight", "module.layer_Dict.Decoder.layer_Dict.FFT_Block_1.slf_attn.w_vs.bias", "module.layer_Dict.Decoder.layer_Dict.FFT_Block_1.slf_attn.layer_norm.weight", "module.layer_Dict.Decoder.layer_Dict.FFT_Block_1.slf_attn.layer_norm.bias", "module.layer_Dict.Decoder.layer_Dict.FFT_Block_1.slf_attn.fc.weight", "module.layer_Dict.Decoder.layer_Dict.FFT_Block_1.slf_attn.fc.bias", "module.layer_Dict.Decoder.layer_Dict.FFT_Block_1.pos_ffn.w_1.weight", "module.layer_Dict.Decoder.layer_Dict.FFT_Block_1.pos_ffn.w_1.bias", "module.layer_Dict.Decoder.layer_Dict.FFT_Block_1.pos_ffn.w_2.weight", "module.layer_Dict.Decoder.layer_Dict.FFT_Block_1.pos_ffn.w_2.bias", "module.layer_Dict.Decoder.layer_Dict.FFT_Block_1.pos_ffn.layer_norm.weight", "module.layer_Dict.Decoder.layer_Dict.FFT_Block_1.pos_ffn.layer_norm.bias", "module.layer_Dict.Decoder.layer_Dict.FFT_Block_2.slf_attn.w_qs.weight", "module.layer_Dict.Decoder.layer_Dict.FFT_Block_2.slf_attn.w_qs.bias", "module.layer_Dict.Decoder.layer_Dict.FFT_Block_2.slf_attn.w_ks.weight", "module.layer_Dict.Decoder.layer_Dict.FFT_Block_2.slf_attn.w_ks.bias", "module.layer_Dict.Decoder.layer_Dict.FFT_Block_2.slf_attn.w_vs.weight", "module.layer_Dict.Decoder.layer_Dict.FFT_Block_2.slf_attn.w_vs.bias", "module.layer_Dict.Decoder.layer_Dict.FFT_Block_2.slf_attn.layer_norm.weight", "module.layer_Dict.Decoder.layer_Dict.FFT_Block_2.slf_attn.layer_norm.bias", "module.layer_Dict.Decoder.layer_Dict.FFT_Block_2.slf_attn.fc.weight", "module.layer_Dict.Decoder.layer_Dict.FFT_Block_2.slf_attn.fc.bias", "module.layer_Dict.Decoder.layer_Dict.FFT_Block_2.pos_ffn.w_1.weight", "module.layer_Dict.Decoder.layer_Dict.FFT_Block_2.pos_ffn.w_1.bias", "module.layer_Dict.Decoder.layer_Dict.FFT_Block_2.pos_ffn.w_2.weight", "module.layer_Dict.Decoder.layer_Dict.FFT_Block_2.pos_ffn.w_2.bias", "module.layer_Dict.Decoder.layer_Dict.FFT_Block_2.pos_ffn.layer_norm.weight", "module.layer_Dict.Decoder.layer_Dict.FFT_Block_2.pos_ffn.layer_norm.bias", "module.layer_Dict.Decoder.layer_Dict.FFT_Block_3.slf_attn.w_qs.weight", "module.layer_Dict.Decoder.layer_Dict.FFT_Block_3.slf_attn.w_qs.bias", "module.layer_Dict.Decoder.layer_Dict.FFT_Block_3.slf_attn.w_ks.weight", "module.layer_Dict.Decoder.layer_Dict.FFT_Block_3.slf_attn.w_ks.bias", "module.layer_Dict.Decoder.layer_Dict.FFT_Block_3.slf_attn.w_vs.weight", "module.layer_Dict.Decoder.layer_Dict.FFT_Block_3.slf_attn.w_vs.bias", "module.layer_Dict.Decoder.layer_Dict.FFT_Block_3.slf_attn.layer_norm.weight", "module.layer_Dict.Decoder.layer_Dict.FFT_Block_3.slf_attn.layer_norm.bias", "module.layer_Dict.Decoder.layer_Dict.FFT_Block_3.slf_attn.fc.weight", "module.layer_Dict.Decoder.layer_Dict.FFT_Block_3.slf_attn.fc.bias", "module.layer_Dict.Decoder.layer_Dict.FFT_Block_3.pos_ffn.w_1.weight", "module.layer_Dict.Decoder.layer_Dict.FFT_Block_3.pos_ffn.w_1.bias", "module.layer_Dict.Decoder.layer_Dict.FFT_Block_3.pos_ffn.w_2.weight", "module.layer_Dict.Decoder.layer_Dict.FFT_Block_3.pos_ffn.w_2.bias", "module.layer_Dict.Decoder.layer_Dict.FFT_Block_3.pos_ffn.layer_norm.weight", "module.layer_Dict.Decoder.layer_Dict.FFT_Block_3.pos_ffn.layer_norm.bias", "module.layer_Dict.Decoder.layer_Dict.FFT_Block_4.slf_attn.w_qs.weight", "module.layer_Dict.Decoder.layer_Dict.FFT_Block_4.slf_attn.w_qs.bias", "module.layer_Dict.Decoder.layer_Dict.FFT_Block_4.slf_attn.w_ks.weight", "module.layer_Dict.Decoder.layer_Dict.FFT_Block_4.slf_attn.w_ks.bias", "module.layer_Dict.Decoder.layer_Dict.FFT_Block_4.slf_attn.w_vs.weight", "module.layer_Dict.Decoder.layer_Dict.FFT_Block_4.slf_attn.w_vs.bias", "module.layer_Dict.Decoder.layer_Dict.FFT_Block_4.slf_attn.layer_norm.weight", "module.layer_Dict.Decoder.layer_Dict.FFT_Block_4.slf_attn.layer_norm.bias", "module.layer_Dict.Decoder.layer_Dict.FFT_Block_4.slf_attn.fc.weight", "module.layer_Dict.Decoder.layer_Dict.FFT_Block_4.slf_attn.fc.bias", "module.layer_Dict.Decoder.layer_Dict.FFT_Block_4.pos_ffn.w_1.weight", "module.layer_Dict.Decoder.layer_Dict.FFT_Block_4.pos_ffn.w_1.bias", "module.layer_Dict.Decoder.layer_Dict.FFT_Block_4.pos_ffn.w_2.weight", "module.layer_Dict.Decoder.layer_Dict.FFT_Block_4.pos_ffn.w_2.bias", "module.layer_Dict.Decoder.layer_Dict.FFT_Block_4.pos_ffn.layer_norm.weight", "module.layer_Dict.Decoder.layer_Dict.FFT_Block_4.pos_ffn.layer_norm.bias", "module.layer_Dict.Decoder.layer_Dict.FFT_Block_5.slf_attn.w_qs.weight", "module.layer_Dict.Decoder.layer_Dict.FFT_Block_5.slf_attn.w_qs.bias", "module.layer_Dict.Decoder.layer_Dict.FFT_Block_5.slf_attn.w_ks.weight", "module.layer_Dict.Decoder.layer_Dict.FFT_Block_5.slf_attn.w_ks.bias", "module.layer_Dict.Decoder.layer_Dict.FFT_Block_5.slf_attn.w_vs.weight", "module.layer_Dict.Decoder.layer_Dict.FFT_Block_5.slf_attn.w_vs.bias", "module.layer_Dict.Decoder.layer_Dict.FFT_Block_5.slf_attn.layer_norm.weight", "module.layer_Dict.Decoder.layer_Dict.FFT_Block_5.slf_attn.layer_norm.bias", "module.layer_Dict.Decoder.layer_Dict.FFT_Block_5.slf_attn.fc.weight", "module.layer_Dict.Decoder.layer_Dict.FFT_Block_5.slf_attn.fc.bias", "module.layer_Dict.Decoder.layer_Dict.FFT_Block_5.pos_ffn.w_1.weight", "module.layer_Dict.Decoder.layer_Dict.FFT_Block_5.pos_ffn.w_1.bias", "module.layer_Dict.Decoder.layer_Dict.FFT_Block_5.pos_ffn.w_2.weight", "module.layer_Dict.Decoder.layer_Dict.FFT_Block_5.pos_ffn.w_2.bias", "module.layer_Dict.Decoder.layer_Dict.FFT_Block_5.pos_ffn.layer_norm.weight", "module.layer_Dict.Decoder.layer_Dict.FFT_Block_5.pos_ffn.layer_norm.bias", "module.layer_Dict.Decoder.layer_Dict.Projection.weight", "module.layer_Dict.Decoder.layer_Dict.Projection.bias".
        #     Unexpected
        #     key(s) in state_dict: "layer_Dict.Encoder.layer_Dict.Phoneme_Embedding.weight", "layer_Dict.Encoder.layer_Dict.Duration_Embedding.weight", "layer_Dict.Encoder.layer_Dict.Note_Embedding.weight", "layer_Dict.Encoder.layer_Dict.Positional_Embedding.pe", "layer_Dict.Encoder.layer_Dict.FFT_Block_0.slf_attn.w_qs.weight", "layer_Dict.Encoder.layer_Dict.FFT_Block_0.slf_attn.w_qs.bias", "layer_Dict.Encoder.layer_Dict.FFT_Block_0.slf_attn.w_ks.weight", "layer_Dict.Encoder.layer_Dict.FFT_Block_0.slf_attn.w_ks.bias", "layer_Dict.Encoder.layer_Dict.FFT_Block_0.slf_attn.w_vs.weight", "layer_Dict.Encoder.layer_Dict.FFT_Block_0.slf_attn.w_vs.bias", "layer_Dict.Encoder.layer_Dict.FFT_Block_0.slf_attn.layer_norm.weight", "layer_Dict.Encoder.layer_Dict.FFT_Block_0.slf_attn.layer_norm.bias", "layer_Dict.Encoder.layer_Dict.FFT_Block_0.slf_attn.fc.weight", "layer_Dict.Encoder.layer_Dict.FFT_Block_0.slf_attn.fc.bias", "layer_Dict.Encoder.layer_Dict.FFT_Block_0.pos_ffn.w_1.weight", "layer_Dict.Encoder.layer_Dict.FFT_Block_0.pos_ffn.w_1.bias", "layer_Dict.Encoder.layer_Dict.FFT_Block_0.pos_ffn.w_2.weight", "layer_Dict.Encoder.layer_Dict.FFT_Block_0.pos_ffn.w_2.bias", "layer_Dict.Encoder.layer_Dict.FFT_Block_0.pos_ffn.layer_norm.weight", "layer_Dict.Encoder.layer_Dict.FFT_Block_0.pos_ffn.layer_norm.bias", "layer_Dict.Encoder.layer_Dict.FFT_Block_1.slf_attn.w_qs.weight", "layer_Dict.Encoder.layer_Dict.FFT_Block_1.slf_attn.w_qs.bias", "layer_Dict.Encoder.layer_Dict.FFT_Block_1.slf_attn.w_ks.weight", "layer_Dict.Encoder.layer_Dict.FFT_Block_1.slf_attn.w_ks.bias", "layer_Dict.Encoder.layer_Dict.FFT_Block_1.slf_attn.w_vs.weight", "layer_Dict.Encoder.layer_Dict.FFT_Block_1.slf_attn.w_vs.bias", "layer_Dict.Encoder.layer_Dict.FFT_Block_1.slf_attn.layer_norm.weight", "layer_Dict.Encoder.layer_Dict.FFT_Block_1.slf_attn.layer_norm.bias", "layer_Dict.Encoder.layer_Dict.FFT_Block_1.slf_attn.fc.weight", "layer_Dict.Encoder.layer_Dict.FFT_Block_1.slf_attn.fc.bias", "layer_Dict.Encoder.layer_Dict.FFT_Block_1.pos_ffn.w_1.weight", "layer_Dict.Encoder.layer_Dict.FFT_Block_1.pos_ffn.w_1.bias", "layer_Dict.Encoder.layer_Dict.FFT_Block_1.pos_ffn.w_2.weight", "layer_Dict.Encoder.layer_Dict.FFT_Block_1.pos_ffn.w_2.bias", "layer_Dict.Encoder.layer_Dict.FFT_Block_1.pos_ffn.layer_norm.weight", "layer_Dict.Encoder.layer_Dict.FFT_Block_1.pos_ffn.layer_norm.bias", "layer_Dict.Encoder.layer_Dict.FFT_Block_2.slf_attn.w_qs.weight", "layer_Dict.Encoder.layer_Dict.FFT_Block_2.slf_attn.w_qs.bias", "layer_Dict.Encoder.layer_Dict.FFT_Block_2.slf_attn.w_ks.weight", "layer_Dict.Encoder.layer_Dict.FFT_Block_2.slf_attn.w_ks.bias", "layer_Dict.Encoder.layer_Dict.FFT_Block_2.slf_attn.w_vs.weight", "layer_Dict.Encoder.layer_Dict.FFT_Block_2.slf_attn.w_vs.bias", "layer_Dict.Encoder.layer_Dict.FFT_Block_2.slf_attn.layer_norm.weight", "layer_Dict.Encoder.layer_Dict.FFT_Block_2.slf_attn.layer_norm.bias", "layer_Dict.Encoder.layer_Dict.FFT_Block_2.slf_attn.fc.weight", "layer_Dict.Encoder.layer_Dict.FFT_Block_2.slf_attn.fc.bias", "layer_Dict.Encoder.layer_Dict.FFT_Block_2.pos_ffn.w_1.weight", "layer_Dict.Encoder.layer_Dict.FFT_Block_2.pos_ffn.w_1.bias", "layer_Dict.Encoder.layer_Dict.FFT_Block_2.pos_ffn.w_2.weight", "layer_Dict.Encoder.layer_Dict.FFT_Block_2.pos_ffn.w_2.bias", "layer_Dict.Encoder.layer_Dict.FFT_Block_2.pos_ffn.layer_norm.weight", "layer_Dict.Encoder.layer_Dict.FFT_Block_2.pos_ffn.layer_norm.bias", "layer_Dict.Encoder.layer_Dict.FFT_Block_3.slf_attn.w_qs.weight", "layer_Dict.Encoder.layer_Dict.FFT_Block_3.slf_attn.w_qs.bias", "layer_Dict.Encoder.layer_Dict.FFT_Block_3.slf_attn.w_ks.weight", "layer_Dict.Encoder.layer_Dict.FFT_Block_3.slf_attn.w_ks.bias", "layer_Dict.Encoder.layer_Dict.FFT_Block_3.slf_attn.w_vs.weight", "layer_Dict.Encoder.layer_Dict.FFT_Block_3.slf_attn.w_vs.bias", "layer_Dict.Encoder.layer_Dict.FFT_Block_3.slf_attn.layer_norm.weight", "layer_Dict.Encoder.layer_Dict.FFT_Block_3.slf_attn.layer_norm.bias", "layer_Dict.Encoder.layer_Dict.FFT_Block_3.slf_attn.fc.weight", "layer_Dict.Encoder.layer_Dict.FFT_Block_3.slf_attn.fc.bias", "layer_Dict.Encoder.layer_Dict.FFT_Block_3.pos_ffn.w_1.weight", "layer_Dict.Encoder.layer_Dict.FFT_Block_3.pos_ffn.w_1.bias", "layer_Dict.Encoder.layer_Dict.FFT_Block_3.pos_ffn.w_2.weight", "layer_Dict.Encoder.layer_Dict.FFT_Block_3.pos_ffn.w_2.bias", "layer_Dict.Encoder.layer_Dict.FFT_Block_3.pos_ffn.layer_norm.weight", "layer_Dict.Encoder.layer_Dict.FFT_Block_3.pos_ffn.layer_norm.bias", "layer_Dict.Encoder.layer_Dict.FFT_Block_4.slf_attn.w_qs.weight", "layer_Dict.Encoder.layer_Dict.FFT_Block_4.slf_attn.w_qs.bias", "layer_Dict.Encoder.layer_Dict.FFT_Block_4.slf_attn.w_ks.weight", "layer_Dict.Encoder.layer_Dict.FFT_Block_4.slf_attn.w_ks.bias", "layer_Dict.Encoder.layer_Dict.FFT_Block_4.slf_attn.w_vs.weight", "layer_Dict.Encoder.layer_Dict.FFT_Block_4.slf_attn.w_vs.bias", "layer_Dict.Encoder.layer_Dict.FFT_Block_4.slf_attn.layer_norm.weight", "layer_Dict.Encoder.layer_Dict.FFT_Block_4.slf_attn.layer_norm.bias", "layer_Dict.Encoder.layer_Dict.FFT_Block_4.slf_attn.fc.weight", "layer_Dict.Encoder.layer_Dict.FFT_Block_4.slf_attn.fc.bias", "layer_Dict.Encoder.layer_Dict.FFT_Block_4.pos_ffn.w_1.weight", "layer_Dict.Encoder.layer_Dict.FFT_Block_4.pos_ffn.w_1.bias", "layer_Dict.Encoder.layer_Dict.FFT_Block_4.pos_ffn.w_2.weight", "layer_Dict.Encoder.layer_Dict.FFT_Block_4.pos_ffn.w_2.bias", "layer_Dict.Encoder.layer_Dict.FFT_Block_4.pos_ffn.layer_norm.weight", "layer_Dict.Encoder.layer_Dict.FFT_Block_4.pos_ffn.layer_norm.bias", "layer_Dict.Encoder.layer_Dict.FFT_Block_5.slf_attn.w_qs.weight", "layer_Dict.Encoder.layer_Dict.FFT_Block_5.slf_attn.w_qs.bias", "layer_Dict.Encoder.layer_Dict.FFT_Block_5.slf_attn.w_ks.weight", "layer_Dict.Encoder.layer_Dict.FFT_Block_5.slf_attn.w_ks.bias", "layer_Dict.Encoder.layer_Dict.FFT_Block_5.slf_attn.w_vs.weight", "layer_Dict.Encoder.layer_Dict.FFT_Block_5.slf_attn.w_vs.bias", "layer_Dict.Encoder.layer_Dict.FFT_Block_5.slf_attn.layer_norm.weight", "layer_Dict.Encoder.layer_Dict.FFT_Block_5.slf_attn.layer_norm.bias", "layer_Dict.Encoder.layer_Dict.FFT_Block_5.slf_attn.fc.weight", "layer_Dict.Encoder.layer_Dict.FFT_Block_5.slf_attn.fc.bias", "layer_Dict.Encoder.layer_Dict.FFT_Block_5.pos_ffn.w_1.weight", "layer_Dict.Encoder.layer_Dict.FFT_Block_5.pos_ffn.w_1.bias", "layer_Dict.Encoder.layer_Dict.FFT_Block_5.pos_ffn.w_2.weight", "layer_Dict.Encoder.layer_Dict.FFT_Block_5.pos_ffn.w_2.bias", "layer_Dict.Encoder.layer_Dict.FFT_Block_5.pos_ffn.layer_norm.weight", "layer_Dict.Encoder.layer_Dict.FFT_Block_5.pos_ffn.layer_norm.bias", "layer_Dict.Duration_Predictor.layer_Dict.Conv_0.weight", "layer_Dict.Duration_Predictor.layer_Dict.Conv_0.bias", "layer_Dict.Duration_Predictor.layer_Dict.LayerNorm_0.weight", "layer_Dict.Duration_Predictor.layer_Dict.LayerNorm_0.bias", "layer_Dict.Duration_Predictor.layer_Dict.Conv_1.weight", "layer_Dict.Duration_Predictor.layer_Dict.Conv_1.bias", "layer_Dict.Duration_Predictor.layer_Dict.LayerNorm_1.weight", "layer_Dict.Duration_Predictor.layer_Dict.LayerNorm_1.bias", "layer_Dict.Duration_Predictor.layer_Dict.Projection.Conv.weight", "layer_Dict.Duration_Predictor.layer_Dict.Projection.Conv.bias", "layer_Dict.Decoder.layer_Dict.Positional_Embedding.pe", "layer_Dict.Decoder.layer_Dict.FFT_Block_0.slf_attn.w_qs.weight", "layer_Dict.Decoder.layer_Dict.FFT_Block_0.slf_attn.w_qs.bias", "layer_Dict.Decoder.layer_Dict.FFT_Block_0.slf_attn.w_ks.weight", "layer_Dict.Decoder.layer_Dict.FFT_Block_0.slf_attn.w_ks.bias", "layer_Dict.Decoder.layer_Dict.FFT_Block_0.slf_attn.w_vs.weight", "layer_Dict.Decoder.layer_Dict.FFT_Block_0.slf_attn.w_vs.bias", "layer_Dict.Decoder.layer_Dict.FFT_Block_0.slf_attn.layer_norm.weight", "layer_Dict.Decoder.layer_Dict.FFT_Block_0.slf_attn.layer_norm.bias", "layer_Dict.Decoder.layer_Dict.FFT_Block_0.slf_attn.fc.weight", "layer_Dict.Decoder.layer_Dict.FFT_Block_0.slf_attn.fc.bias", "layer_Dict.Decoder.layer_Dict.FFT_Block_0.pos_ffn.w_1.weight", "layer_Dict.Decoder.layer_Dict.FFT_Block_0.pos_ffn.w_1.bias", "layer_Dict.Decoder.layer_Dict.FFT_Block_0.pos_ffn.w_2.weight", "layer_Dict.Decoder.layer_Dict.FFT_Block_0.pos_ffn.w_2.bias", "layer_Dict.Decoder.layer_Dict.FFT_Block_0.pos_ffn.layer_norm.weight", "layer_Dict.Decoder.layer_Dict.FFT_Block_0.pos_ffn.layer_norm.bias", "layer_Dict.Decoder.layer_Dict.FFT_Block_1.slf_attn.w_qs.weight", "layer_Dict.Decoder.layer_Dict.FFT_Block_1.slf_attn.w_qs.bias", "layer_Dict.Decoder.layer_Dict.FFT_Block_1.slf_attn.w_ks.weight", "layer_Dict.Decoder.layer_Dict.FFT_Block_1.slf_attn.w_ks.bias", "layer_Dict.Decoder.layer_Dict.FFT_Block_1.slf_attn.w_vs.weight", "layer_Dict.Decoder.layer_Dict.FFT_Block_1.slf_attn.w_vs.bias", "layer_Dict.Decoder.layer_Dict.FFT_Block_1.slf_attn.layer_norm.weight", "layer_Dict.Decoder.layer_Dict.FFT_Block_1.slf_attn.layer_norm.bias", "layer_Dict.Decoder.layer_Dict.FFT_Block_1.slf_attn.fc.weight", "layer_Dict.Decoder.layer_Dict.FFT_Block_1.slf_attn.fc.bias", "layer_Dict.Decoder.layer_Dict.FFT_Block_1.pos_ffn.w_1.weight", "layer_Dict.Decoder.layer_Dict.FFT_Block_1.pos_ffn.w_1.bias", "layer_Dict.Decoder.layer_Dict.FFT_Block_1.pos_ffn.w_2.weight", "layer_Dict.Decoder.layer_Dict.FFT_Block_1.pos_ffn.w_2.bias", "layer_Dict.Decoder.layer_Dict.FFT_Block_1.pos_ffn.layer_norm.weight", "layer_Dict.Decoder.layer_Dict.FFT_Block_1.pos_ffn.layer_norm.bias", "layer_Dict.Decoder.layer_Dict.FFT_Block_2.slf_attn.w_qs.weight", "layer_Dict.Decoder.layer_Dict.FFT_Block_2.slf_attn.w_qs.bias", "layer_Dict.Decoder.layer_Dict.FFT_Block_2.slf_attn.w_ks.weight", "layer_Dict.Decoder.layer_Dict.FFT_Block_2.slf_attn.w_ks.bias", "layer_Dict.Decoder.layer_Dict.FFT_Block_2.slf_attn.w_vs.weight", "layer_Dict.Decoder.layer_Dict.FFT_Block_2.slf_attn.w_vs.bias", "layer_Dict.Decoder.layer_Dict.FFT_Block_2.slf_attn.layer_norm.weight", "layer_Dict.Decoder.layer_Dict.FFT_Block_2.slf_attn.layer_norm.bias", "layer_Dict.Decoder.layer_Dict.FFT_Block_2.slf_attn.fc.weight", "layer_Dict.Decoder.layer_Dict.FFT_Block_2.slf_attn.fc.bias", "layer_Dict.Decoder.layer_Dict.FFT_Block_2.pos_ffn.w_1.weight", "layer_Dict.Decoder.layer_Dict.FFT_Block_2.pos_ffn.w_1.bias", "layer_Dict.Decoder.layer_Dict.FFT_Block_2.pos_ffn.w_2.weight", "layer_Dict.Decoder.layer_Dict.FFT_Block_2.pos_ffn.w_2.bias", "layer_Dict.Decoder.layer_Dict.FFT_Block_2.pos_ffn.layer_norm.weight", "layer_Dict.Decoder.layer_Dict.FFT_Block_2.pos_ffn.layer_norm.bias", "layer_Dict.Decoder.layer_Dict.FFT_Block_3.slf_attn.w_qs.weight", "layer_Dict.Decoder.layer_Dict.FFT_Block_3.slf_attn.w_qs.bias", "layer_Dict.Decoder.layer_Dict.FFT_Block_3.slf_attn.w_ks.weight", "layer_Dict.Decoder.layer_Dict.FFT_Block_3.slf_attn.w_ks.bias", "layer_Dict.Decoder.layer_Dict.FFT_Block_3.slf_attn.w_vs.weight", "layer_Dict.Decoder.layer_Dict.FFT_Block_3.slf_attn.w_vs.bias", "layer_Dict.Decoder.layer_Dict.FFT_Block_3.slf_attn.layer_norm.weight", "layer_Dict.Decoder.layer_Dict.FFT_Block_3.slf_attn.layer_norm.bias", "layer_Dict.Decoder.layer_Dict.FFT_Block_3.slf_attn.fc.weight", "layer_Dict.Decoder.layer_Dict.FFT_Block_3.slf_attn.fc.bias", "layer_Dict.Decoder.layer_Dict.FFT_Block_3.pos_ffn.w_1.weight", "layer_Dict.Decoder.layer_Dict.FFT_Block_3.pos_ffn.w_1.bias", "layer_Dict.Decoder.layer_Dict.FFT_Block_3.pos_ffn.w_2.weight", "layer_Dict.Decoder.layer_Dict.FFT_Block_3.pos_ffn.w_2.bias", "layer_Dict.Decoder.layer_Dict.FFT_Block_3.pos_ffn.layer_norm.weight", "layer_Dict.Decoder.layer_Dict.FFT_Block_3.pos_ffn.layer_norm.bias", "layer_Dict.Decoder.layer_Dict.FFT_Block_4.slf_attn.w_qs.weight", "layer_Dict.Decoder.layer_Dict.FFT_Block_4.slf_attn.w_qs.bias", "layer_Dict.Decoder.layer_Dict.FFT_Block_4.slf_attn.w_ks.weight", "layer_Dict.Decoder.layer_Dict.FFT_Block_4.slf_attn.w_ks.bias", "layer_Dict.Decoder.layer_Dict.FFT_Block_4.slf_attn.w_vs.weight", "layer_Dict.Decoder.layer_Dict.FFT_Block_4.slf_attn.w_vs.bias", "layer_Dict.Decoder.layer_Dict.FFT_Block_4.slf_attn.layer_norm.weight", "layer_Dict.Decoder.layer_Dict.FFT_Block_4.slf_attn.layer_norm.bias", "layer_Dict.Decoder.layer_Dict.FFT_Block_4.slf_attn.fc.weight", "layer_Dict.Decoder.layer_Dict.FFT_Block_4.slf_attn.fc.bias", "layer_Dict.Decoder.layer_Dict.FFT_Block_4.pos_ffn.w_1.weight", "layer_Dict.Decoder.layer_Dict.FFT_Block_4.pos_ffn.w_1.bias", "layer_Dict.Decoder.layer_Dict.FFT_Block_4.pos_ffn.w_2.weight", "layer_Dict.Decoder.layer_Dict.FFT_Block_4.pos_ffn.w_2.bias", "layer_Dict.Decoder.layer_Dict.FFT_Block_4.pos_ffn.layer_norm.weight", "layer_Dict.Decoder.layer_Dict.FFT_Block_4.pos_ffn.layer_norm.bias", "layer_Dict.Decoder.layer_Dict.FFT_Block_5.slf_attn.w_qs.weight", "layer_Dict.Decoder.layer_Dict.FFT_Block_5.slf_attn.w_qs.bias", "layer_Dict.Decoder.layer_Dict.FFT_Block_5.slf_attn.w_ks.weight", "layer_Dict.Decoder.layer_Dict.FFT_Block_5.slf_attn.w_ks.bias", "layer_Dict.Decoder.layer_Dict.FFT_Block_5.slf_attn.w_vs.weight", "layer_Dict.Decoder.layer_Dict.FFT_Block_5.slf_attn.w_vs.bias", "layer_Dict.Decoder.layer_Dict.FFT_Block_5.slf_attn.layer_norm.weight", "layer_Dict.Decoder.layer_Dict.FFT_Block_5.slf_attn.layer_norm.bias", "layer_Dict.Decoder.layer_Dict.FFT_Block_5.slf_attn.fc.weight", "layer_Dict.Decoder.layer_Dict.FFT_Block_5.slf_attn.fc.bias", "layer_Dict.Decoder.layer_Dict.FFT_Block_5.pos_ffn.w_1.weight", "layer_Dict.Decoder.layer_Dict.FFT_Block_5.pos_ffn.w_1.bias", "layer_Dict.Decoder.layer_Dict.FFT_Block_5.pos_ffn.w_2.weight", "layer_Dict.Decoder.layer_Dict.FFT_Block_5.pos_ffn.w_2.bias", "layer_Dict.Decoder.layer_Dict.FFT_Block_5.pos_ffn.layer_norm.weight", "layer_Dict.Decoder.layer_Dict.FFT_Block_5.pos_ffn.layer_norm.bias", "layer_Dict.Decoder.layer_Dict.Projection.weight", "layer_Dict.Decoder.layer_Dict.Projection.bias".

        state_Dict = torch.load(path, map_location='cpu')
        if self.hp.Use_Multi_GPU == True:
            new_state_dict1 = collections.OrderedDict()
            for k, v in state_Dict['Generator']['Model'].items():
                name = 'module.' + k  # module字段在最前面，从第7个字符开始就可以去掉module
                new_state_dict1[name] = v  # 新字典的key值对应的value一一对应
            state_Dict['Generator']['Model'] = new_state_dict1
            new_state_dict2 = collections.OrderedDict()
            for k, v in state_Dict['Discriminator']['Model'].items():
                name = 'module.' + k  # module字段在最前面，从第7个字符开始就可以去掉module
                new_state_dict2[name] = v  # 新字典的key值对应的value一一对应
            state_Dict['Discriminator']['Model'] =  new_state_dict2
        #     for k, v in state_Dict['Generator']['Optimizer'].items():
        #         name = 'module.' + k  # module字段在最前面，从第7个字符开始就可以去掉module
        #         new_state_dict['Generator']['Optimizer'][name] = v  # 新字典的key值对应的value一一对应
        #     for k, v in state_Dict['Discriminator']['Optimizer'].items():
        #         name = 'module.' + k  # module字段在最前面，从第7个字符开始就可以去掉module
        #         new_state_dict['Discriminator']['Optimizer'][name] = v  # 新字典的key值对应的value一一对应
        #     for k, v in state_Dict['Generator']['Scheduler'].items():
        #         name = 'module.' + k  # module字段在最前面，从第7个字符开始就可以去掉module
        #         new_state_dict['Generator']['Scheduler'][name] = v  # 新字典的key值对应的value一一对应
        #     for k, v in state_Dict['Discriminator']['Scheduler'].items():
        #         name = 'module.' + k  # module字段在最前面，从第7个字符开始就可以去掉module
        #         new_state_dict['Discriminator']['Scheduler'][name] = v  # 新字典的key值对应的value一一对应
        #     for k, v in state_Dict['Steps'].items():
        #         name = 'module.' + k  # module字段在最前面，从第7个字符开始就可以去掉module
        #         new_state_dict['Steps'][name] = v  # 新字典的key值对应的value一一对应
        #     for k, v in state_Dict['Generator']['Optimizer'].items():
        #         name = k  # module字段在最前面，从第7个字符开始就可以去掉module
        #         new_state_dict['Generator']['Optimizer'][name] = v  # 新字典的key值对应的value一一对应
        #     for k, v in state_Dict['Discriminator']['Optimizer'].items():
        #         name = k  # module字段在最前面，从第7个字符开始就可以去掉module
        #         new_state_dict['Discriminator']['Optimizer'][name] = v  # 新字典的key值对应的value一一对应
        #     for k, v in state_Dict['Generator']['Scheduler'].items():
        #         name = k  # module字段在最前面，从第7个字符开始就可以去掉module
        #         new_state_dict['Generator']['Scheduler'][name] = v  # 新字典的key值对应的value一一对应
        #     for k, v in state_Dict['Discriminator']['Scheduler'].items():
        #         name = k  # module字段在最前面，从第7个字符开始就可以去掉module
        #         new_state_dict['Discriminator']['Scheduler'][name] = v  # 新字典的key值对应的value一一对应
        #     for k, v in state_Dict['Steps'].items():
        #         name = k  # module字段在最前面，从第7个字符开始就可以去掉module
        #         new_state_dict['Steps'][name] = v  # 新字典的key值对应的value一一对应
        #     state_Dict = new_state_dict

        self.model_Dict['Generator'].load_state_dict(state_Dict['Generator']['Model'])
        self.model_Dict['Discriminator'].load_state_dict(state_Dict['Discriminator']['Model'])

        self.optimizer_Dict['Generator'].load_state_dict(state_Dict['Generator']['Optimizer'])
        self.optimizer_Dict['Discriminator'].load_state_dict(state_Dict['Discriminator']['Optimizer'])

        self.scheduler_Dict['Generator'].load_state_dict(state_Dict['Generator']['Scheduler'])
        self.scheduler_Dict['Discriminator'].load_state_dict(state_Dict['Discriminator']['Scheduler'])

        self.steps = state_Dict['Steps']
        # print(self.steps)

        # if self.hp.Use_Multi_GPU == True:
        #     self.model_Dict['Generator'] = torch.nn.parallel.DistributedDataParallel(self.model_Dict['Generator'](self.hp).to(self.device), device_ids=[self.gpu_id],
        #             # find_unused_parameters=True
        #             ).cuda()
        #     self.model_Dict['Discriminator'] = torch.nn.parallel.DistributedDataParallel(
        #         self.model_Dict['Discriminator'](self.hp).to(self.device), device_ids=[self.gpu_id],
        #         # find_unused_parameters=True
        #         ).cuda()

        logging.info('Checkpoint loaded at {} steps in GPU {}.'.format(self.steps, self.gpu_id))

    def Save_Checkpoint(self):
        if self.gpu_id != 0:
            return

        os.makedirs(self.hp.Checkpoint_Path, exist_ok=True)

        state_Dict = {
            'Generator': {
                'Model': self.model_Dict['Generator'].module.state_dict() if self.hp.Use_Multi_GPU else self.model_Dict[
                    'Generator'].state_dict(),
                'Optimizer': self.optimizer_Dict['Generator'].state_dict(),
                'Scheduler': self.scheduler_Dict['Generator'].state_dict(),
            },
            'Discriminator': {
                'Model': self.model_Dict['Discriminator'].module.state_dict() if self.hp.Use_Multi_GPU else
                self.model_Dict['Discriminator'].state_dict(),
                'Optimizer': self.optimizer_Dict['Discriminator'].state_dict(),
                'Scheduler': self.scheduler_Dict['Discriminator'].state_dict(),
            },
            'Steps': self.steps
        }

        torch.save(
            state_Dict,
            os.path.join(self.hp.Checkpoint_Path, 'S_{}.pt'.format(self.steps).replace('\\', '/'))
        )
        if self.hp.Use_Multi_GPU:
            state_Dict = {
                'Generator': {
                    'Model': self.model_Dict['Generator'].state_dict(),
                    'Optimizer': self.optimizer_Dict['Generator'].state_dict(),
                    'Scheduler': self.scheduler_Dict['Generator'].state_dict(),
                },
                'Discriminator': {
                    'Model': self.model_Dict['Discriminator'].state_dict(),
                    'Optimizer': self.optimizer_Dict['Discriminator'].state_dict(),
                    'Scheduler': self.scheduler_Dict['Discriminator'].state_dict(),
                },
                'Steps': self.steps
            }

            torch.save(
                state_Dict,
                os.path.join(self.hp.Checkpoint_Path, 'S_{}_multi.pt'.format(self.steps).replace('\\', '/'))
            )

        logging.info('Checkpoint saved at {} steps.'.format(self.steps))

    def _Set_Distribution(self):
        if self.num_gpus > 1:
            self.model = apply_gradient_allreduce(self.model)

    def Train(self):
        hp_Path = os.path.join(self.hp.Checkpoint_Path, 'Hyper_Parameters.yaml').replace('\\', '/')
        if not os.path.exists(hp_Path):
            from shutil import copyfile
            os.makedirs(self.hp.Checkpoint_Path, exist_ok=True)
            copyfile(self.hp_Path, hp_Path)

        # if self.steps == 0:
        #     self.Evaluation_Epoch()
        #
        # if self.hp.Train.Initial_Inference:
        #     self.Inference_Epoch()

        self.tqdm = tqdm(
            initial=self.steps,
            total=self.hp.Train.Max_Step,
            desc='[Training]'
        )

        # self.steps += 1

        while self.steps < self.hp.Train.Max_Step:
            try:
                self.Train_Epoch()
            except KeyboardInterrupt:
                self.Save_Checkpoint()
                exit(1)

        self.tqdm.close()
        logging.info('Finished training.')

def Worker(gpu, hp_path, steps):
    torch.distributed.init_process_group(
        backend= 'nccl',
        init_method='tcp://127.0.0.1:54321',
        world_size= torch.cuda.device_count(),
        rank= gpu
        )

    new_Trainer = Trainer(hp_path= hp_path, steps= steps, gpu_id= gpu)
    new_Trainer.Train()

# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    # parser.add_argument('-t', '--train', required=0, type=bool)
    # parser.add_argument('-i', '--inference', required=True, type=str)
    # parser.add_argument('-id', '--identify', required=True, type=str)
    # parser.add_argument('-hp', '--hyper_parameters', required=True, type=str)
    parser.add_argument('-s', '--steps', default=0, type=int)
    parser.add_argument('-p', '--port', default=54321, type=int)
    # parser.add_argument('-r', '--local_rank', default=0, type=int)
    args = parser.parse_args()

    hp = Recursive_Parse(yaml.load(
        open('./Hyper_Parameters.yaml', encoding='utf-8'),
        Loader=yaml.Loader
    ))  # 读取超参数
    os.environ['CUDA_VISIBLE_DEVICES'] = hp.Device

    calc_pattern = 5
    if calc_pattern == 0: #args.train:
        if hp.Use_Multi_GPU:
            # init_distributed(
            #     rank=int(os.getenv('RANK', '0')),
            #     num_gpus=int(os.getenv("WORLD_SIZE", '1')),
            #     dist_backend='nccl',
            #     dist_url='tcp://127.0.0.1:{}'.format(args.port)
            # )
            mp.spawn(
                Worker,
                nprocs=torch.cuda.device_count(),
                args=('Hyper_Parameters.yaml', 0)
            )
        else:
            new_Trainer = Trainer(hp_path='./Hyper_Parameters.yaml', steps=0)  # 开始训练
            new_Trainer.Train()
    elif calc_pattern == 1:
        new_Trainer = Trainer(hp_path='./Hyper_Parameters.yaml', steps=130000)
        new_Trainer.Inference_Test(-1)
    elif calc_pattern == 2:
        new_Trainer = Trainer(hp_path='./Hyper_Parameters.yaml', steps=130000)
        new_Trainer.Evaluation_Epoch_Test()
    elif calc_pattern == 3:
        new_Trainer = Trainer(hp_path='./Hyper_Parameters.yaml', steps=130000)
        new_Trainer.Evaluation_Epoch_Test1()
    elif calc_pattern == 4:
        new_Trainer = Trainer(hp_path='./Hyper_Parameters.yaml', steps=130000)  # 开始训练
        new_Trainer.Evaluation_Epoch()
    elif calc_pattern == 5:
        new_Trainer = Trainer(hp_path='./Hyper_Parameters.yaml', steps=400000)
        new_Trainer.Evaluation_Epoch_Test2()