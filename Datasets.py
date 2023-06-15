# If there is no duration in pattern dict, you must add the duration information
# Please use 'Get_Duration.py' in Pitchtron repository

import torch
import numpy as np
import pickle, os
from random import randint
from multiprocessing import Manager
import librosa
import torchaudio

def Text_to_Token(text: list, token_dict: dict): #将文本转化为音素，for走一遍所有的文本，token_dict是个map
    return [token_dict[x] for x in text]

note_dict = {}

def Init_Note():
    cnt = -1
    tmp_note_list = ['C', 'C#/Db', 'D', 'D#/Eb', 'E', 'F', 'F#/Gb', 'G', 'G#/Ab', 'A', 'A#/Bb', 'B']
    for i in range(-1, 10, 1):
        for j in range(len(tmp_note_list)):
            cnt += 1
            new_note_str = ''
            if(len(tmp_note_list[j]) == 1):
                new_note_str = tmp_note_list[j] + str(i)
            else:
                new_note_str = tmp_note_list[j][0:2] + str(i) + '/' + tmp_note_list[j][-2:] + str(i)
            note_dict[new_note_str] = cnt
            print(new_note_str)
            print(cnt)
    # note_dict['AP'] = 0
    # note_dict['SP'] = 0
    note_dict['rest'] = 0

def Note_to_Pitch(note: list, pitch_dict: dict):
    return [pitch_dict[x] for x in note]


def Mel_Stack(mels: list, max_abs_mel: float): #把mel图合并
    max_Mel_Length = max([mel.shape[0] for mel in mels])
    mels = np.stack(
        [np.pad(mel, [[0, max_Mel_Length - mel.shape[0]], [0, 0]], constant_values=-max_abs_mel) for mel in mels],
        axis=0
    )

    return mels


def Silence_Stack(silences: list):
    max_Silences_Length = max([silence.shape[0] for silence in silences])
    silences = np.stack(
        [np.pad(silence, [0, max_Silences_Length - silence.shape[0]], constant_values=0.0) for silence in silences],
        axis=0
    )

    return silences


def Pitch_Stack(pitches: list):
    max_Pitch_Length = max([pitch.shape[0] for pitch in pitches])
    pitches = np.stack(
        [np.pad(pitch, [0, max_Pitch_Length - pitch.shape[0]], constant_values=0.0) for pitch in pitches],
        axis=0
    )

    return pitches

def Speaker_Embedding_Stack(speaker_embeddings: list):
    speaker_embeddings = np.stack(
        [speaker_embedding for speaker_embedding in speaker_embeddings],
        axis=0
    )
    return speaker_embeddings

def Duration_Stack(durations: list):
    '''
    The length of durations becomes +1 for padding value of each duration.
    '''
    max_Duration = max([np.sum(duration) for duration in durations])
    max_Duration_Length = max([len(duration) for duration in durations]) + 1  # 1 is for padding duration(max - sum).

    durations = np.stack(
        [np.pad(duration, [0, max_Duration_Length - len(duration)], constant_values=0) for duration in durations],
        axis=0
    )
    durations[:, -1] = max_Duration - np.sum(durations, axis=1)  # To fit the time after sample

    return durations


def Token_Stack(tokens: list, token_dict: dict):
    '''
    The length of tokens becomes +1 for padding value of each duration.
    '''
    max_Token_Length = max([len(token) for token in tokens]) + 1  # 1 is for padding '<X>'

    tokens = np.stack(
        [np.pad(token, [0, max_Token_Length - len(token)], constant_values=token_dict['<X>']) for token in tokens],
        axis=0
    )

    return tokens


def Note_Stack(notes: list):
    '''
    The length of notes becomes +1 for padding value of each duration.
    '''
    max_Note_Length = max([len(note) for note in notes]) + 1  # 1 is for padding '<X>'

    notes = np.stack(
        [np.pad(note, [0, max_Note_Length - len(note)], constant_values=0) for note in notes],
        axis=0
    )

    return notes


class Node:
    def __init__(self,
                 number: str,
                 lysic: str,
                 tokens: list,
                 notes: list,
                 durations: list):
        self.number = number
        self.lysic = lysic
        self.texts = tokens
        self.notes = notes
        self.durations = durations


class Dataset(torch.utils.data.Dataset):
    def __init__(
            self,
            pattern_path: str,
            Metadata_file: str,
            token_dict: dict,
            accumulated_dataset_epoch: int = 1,
            use_cache: bool = False
    ):
        super(Dataset, self).__init__()
        self.pattern_Path = pattern_path
        self.token_Dict = token_dict
        self.use_cache = use_cache

        self.metadata_Path = os.path.join(pattern_path, Metadata_file).replace('\\', '/')
        metadata_Dict = pickle.load(open(self.metadata_Path, 'rb'))
        self.patterns = metadata_Dict['File_List']
        self.base_Length = len(self.patterns)
        self.patterns *= accumulated_dataset_epoch

        self.cache_Dict = Manager().dict()

    def __getitem__(self, idx: int):
        if (idx % self.base_Length) in self.cache_Dict.keys():  # 这个map存的啥
            return self.cache_Dict[self.metadata_Path, idx % self.base_Length]

        path = os.path.join(self.pattern_Path, self.patterns[idx]).replace('\\', '/')
        pattern_Dict = pickle.load(open(path, 'rb'))

        pattern = pattern_Dict['Duration'], Text_to_Token(pattern_Dict['Text'], self.token_Dict), pattern_Dict['Note'], \
                  pattern_Dict['Mel'], pattern_Dict['Silence'], pattern_Dict['Pitch'], pattern_Dict['Speaker_Embedding']
        if self.use_cache:
            self.cache_Dict[self.metadata_Path, idx % self.base_Length] = pattern

        return pattern

    def __len__(self):
        return len(self.patterns)

class Dataset_Test(torch.utils.data.Dataset):
    def __init__(
            self,
            pattern_path: str,
            Metadata_file: str,
            token_dict: dict,
            accumulated_dataset_epoch: int = 1,
            use_cache: bool = False
    ):
        super(Dataset_Test, self).__init__()
        self.pattern_Path = pattern_path
        self.token_Dict = token_dict
        self.use_cache = use_cache

        self.metadata_Path = os.path.join(pattern_path, Metadata_file).replace('\\', '/')
        metadata_Dict = pickle.load(open(self.metadata_Path, 'rb'))
        self.patterns = metadata_Dict['File_List']
        self.base_Length = len(self.patterns)
        self.patterns *= accumulated_dataset_epoch

        self.cache_Dict = Manager().dict()

    def __getitem__(self, idx: int):
        if (idx % self.base_Length) in self.cache_Dict.keys():  # 这个map存的啥
            return self.cache_Dict[self.metadata_Path, idx % self.base_Length]

        path = os.path.join(self.pattern_Path, self.patterns[idx]).replace('\\', '/')
        pattern_Dict = pickle.load(open(path, 'rb'))

        pattern = pattern_Dict['Duration'], Text_to_Token(pattern_Dict['Text'], self.token_Dict), pattern_Dict['Note'], \
                  pattern_Dict['Mel'], pattern_Dict['Silence'], pattern_Dict['Pitch'], path, pattern_Dict['Speaker_Embedding']
        if self.use_cache:
            self.cache_Dict[self.metadata_Path, idx % self.base_Length] = pattern

        return pattern

    def __len__(self):
        return len(self.patterns)

class Inference_Dataset(torch.utils.data.Dataset):
    def __init__(
            self,
            token_dict: dict,
            pattern_path: str,
            Metadata_file: str,
            use_cache: bool = False
    ):
        super(Inference_Dataset, self).__init__()
        self.token_Dict = token_dict
        self.use_cache = use_cache

        self.patterns = []

        self.pattern_Path = pattern_path
        self.token_Dict = token_dict
        self.use_cache = use_cache

        self.metadata_Path = os.path.join(pattern_path, Metadata_file).replace('\\', '/')
        metadata_Dict = pickle.load(open(self.metadata_Path, 'rb'))
        self.patterns = metadata_Dict['File_List']
        self.base_Length = len(self.patterns)

        self.cache_Dict = Manager().dict()

    def __getitem__(self, idx: int):
        if (idx % self.base_Length) in self.cache_Dict.keys():  # 这个map存的啥
            return self.cache_Dict[self.metadata_Path, idx % self.base_Length]

        path = os.path.join(self.pattern_Path, self.patterns[idx]).replace('\\', '/')
        pattern_Dict = pickle.load(open(path, 'rb'))

        pattern = pattern_Dict['Duration'], Text_to_Token(pattern_Dict['Text'], self.token_Dict), pattern_Dict['Note'], os.path.splitext(os.path.basename(path))[0], pattern_Dict['Speaker_Embedding']
        if self.use_cache:
            self.cache_Dict[self.metadata_Path, idx % self.base_Length] = pattern

        return pattern

    def __len__(self):
        return len(self.patterns)


# class Inference_Dataset(torch.utils.data.Dataset):
#     def __init__(
#             self,
#             token_dict: dict,
#             pattern_paths: list = ['./Inference_for_Training/Example.txt'],
#             use_cache: bool = False
#     ):
#         super(Inference_Dataset, self).__init__()
#         self.token_Dict = token_dict
#         self.use_cache = use_cache
#
#         self.patterns = []
#         for path in pattern_paths:
#             music = [
#                 (int(line.strip().split('\t')[0]), line.strip().split('\t')[1], int(line.strip().split('\t')[2]))
#                 for line in open(path, 'r', encoding='utf-8').readlines()[1:]
#             ]
#             duration, text, note = zip(*music)
#             self.patterns.append((duration, text, note, path))
#
#         self.cache_Dict = Manager().dict()
#
#     def __getitem__(self, idx: int):
#         if idx in self.cache_Dict.keys():
#             return self.cache_Dict['Inference', idx]
#
#         duration, text, note, path = self.patterns[idx]
#         pattern = duration, Text_to_Token(text, self.token_Dict), note, os.path.splitext(os.path.basename(path))[0]
#
#         if self.use_cache:
#             self.cache_Dict['Inference', idx] = pattern
#
#         return pattern
#
#     def __len__(self):
#         return len(self.patterns)

# class Collater: # 校验者
#     def __init__(
#             self,
#             token_dict: dict,
#             max_abs_mel: float
#     ):
#         self.token_Dict = token_dict
#         self.max_ABS_Mel = max_abs_mel
#
#     def __call__(self, batch: list):
#         durations, tokens, notes, mels, silences, pitches = zip(*batch)  # 这里是解压
#
#         token_Lengths = [len(token) + 1 for token in tokens]
#         mel_Lengths = [mel.shape[0] for mel in mels]
#
#         durations = Duration_Stack(durations)
#         tokens = Token_Stack(tokens, self.token_Dict)
#         notes = Note_Stack(notes)
#         mels = Mel_Stack(mels, self.max_ABS_Mel)
#         # silences = Silence_Stack(silences)
#         pitches = Pitch_Stack(pitches)
#
#         durations = torch.LongTensor(durations)  # [Batch, Time]
#         tokens = torch.LongTensor(tokens)  # [Batch, Time]
#         token_Lengths = torch.LongTensor(token_Lengths)  # [Batch]
#         notes = torch.LongTensor(notes)  # [Batch, Time]
#         mels = torch.FloatTensor(mels).transpose(2, 1)  # [Batch, Mel_dim, Time]
#         mel_Lengths = torch.LongTensor(mel_Lengths)  # [Batch]
#         silences = torch.FloatTensor(silences)  # [Batch, Time]
#         pitches = torch.FloatTensor(pitches)  # [Batch, Time]
#
#         return durations, tokens, notes, token_Lengths, mels, pitches, mel_Lengths #, sliences

class Collater:  # 校对啥啊
    def __init__(
            self,
            token_dict: dict,
            max_abs_mel: float
    ):
        self.token_Dict = token_dict
        self.max_ABS_Mel = max_abs_mel

    def __call__(self, batch: list):
        durations, tokens, notes, mels, silences, pitches, speaker_embeddings = zip(*batch)  # 这里是解压

        token_Lengths = [len(token) + 1 for token in tokens]
        mel_Lengths = [mel.shape[0] for mel in mels]

        durations = Duration_Stack(durations)
        tokens = Token_Stack(tokens, self.token_Dict)
        notes = Note_Stack(notes)
        mels = Mel_Stack(mels, self.max_ABS_Mel)
        silences = Silence_Stack(silences)
        pitches = Pitch_Stack(pitches)
        speaker_embeddings = Speaker_Embedding_Stack(speaker_embeddings)

        durations = torch.LongTensor(durations)  # [Batch, Time]
        tokens = torch.LongTensor(tokens)  # [Batch, Time]
        token_Lengths = torch.LongTensor(token_Lengths)  # [Batch]
        notes = torch.LongTensor(notes)  # [Batch, Time]
        mels = torch.FloatTensor(mels).transpose(2, 1)  # [Batch, Mel_dim, Time]
        mel_Lengths = torch.LongTensor(mel_Lengths)  # [Batch]
        silences = torch.FloatTensor(silences)  # [Batch, Time]
        pitches = torch.FloatTensor(pitches)  # [Batch, Time]
        speaker_embeddings = torch.FloatTensor(speaker_embeddings) # [Batch, 256]

        return durations, tokens, notes, token_Lengths, mels, silences, pitches, mel_Lengths, speaker_embeddings

class Collater1:  # 校对啥啊
    def __init__(
            self,
            token_dict: dict,
            max_abs_mel: float
    ):
        self.token_Dict = token_dict
        self.max_ABS_Mel = max_abs_mel

    def __call__(self, batch: list):
        durations, tokens, notes, mels, silences, pitches, paths, speaker_embeddings = zip(*batch)  # 这里是解压

        token_Lengths = [len(token) + 1 for token in tokens]
        mel_Lengths = [mel.shape[0] for mel in mels]

        durations = Duration_Stack(durations)
        tokens = Token_Stack(tokens, self.token_Dict)
        notes = Note_Stack(notes)
        mels = Mel_Stack(mels, self.max_ABS_Mel)
        silences = Silence_Stack(silences)
        pitches = Pitch_Stack(pitches)
        speaker_embeddings = Speaker_Embedding_Stack(speaker_embeddings)

        durations = torch.LongTensor(durations)  # [Batch, Time]
        tokens = torch.LongTensor(tokens)  # [Batch, Time]
        token_Lengths = torch.LongTensor(token_Lengths)  # [Batch]
        notes = torch.LongTensor(notes)  # [Batch, Time]
        mels = torch.FloatTensor(mels).transpose(2, 1)  # [Batch, Mel_dim, Time]
        mel_Lengths = torch.LongTensor(mel_Lengths)  # [Batch]
        silences = torch.FloatTensor(silences)  # [Batch, Time]
        pitches = torch.FloatTensor(pitches)  # [Batch, Time]
        speaker_embeddings = torch.FloatTensor(speaker_embeddings)

        return durations, tokens, notes, token_Lengths, mels, silences, pitches, mel_Lengths, paths, speaker_embeddings

class Inference_Collater:
    def __init__(
            self,
            token_dict: dict,
            max_abs_mel: float
    ):
        self.token_Dict = token_dict
        self.max_ABS_Mel = max_abs_mel

    def __call__(self, batch: list):
        durations, tokens, notes, labels, speaker_embeddings = zip(*batch)

        token_Lengths = [len(token) + 1 for token in tokens]

        durations = Duration_Stack(durations)
        tokens = Token_Stack(tokens, self.token_Dict)
        notes = Note_Stack(notes)
        speaker_embeddings = Speaker_Embedding_Stack(speaker_embeddings)

        durations = torch.LongTensor(durations)  # [Batch, Time]
        tokens = torch.LongTensor(tokens)  # [Batch, Time]
        token_Lengths = torch.LongTensor(token_Lengths)  # [Batch]
        notes = torch.LongTensor(notes)  # [Batch, Time]
        speaker_embeddings = torch.FloatTensor(speaker_embeddings)

        return durations, tokens, notes, token_Lengths, labels, speaker_embeddings


def test1():
    import yaml
    from Arg_Parser import Recursive_Parse

    hp = Recursive_Parse(yaml.load(
        open('Hyper_Parameters.yaml', encoding='utf-8'),
        Loader=yaml.Loader
    ))
    print(233)
    # token_Dict = yaml.load(open(hp.Token_Path), Loader=yaml.Loader)
    token_Dict = {}
    with open('./token1.txt', 'r', encoding='utf-8') as f:
        for line in f:
            tmp1 = line.split()
            print(tmp1)
            token_Dict[tmp1[0]] = int(tmp1[2])
    token_Dict['AP'] = 59
    token_Dict['SP'] = 59
    token_Dict['<X>'] = 59

    dataset = Dataset(
        pattern_path=hp.Train.Train_Pattern.Path,
        Metadata_file=hp.Train.Train_Pattern.Metadata_File,
        token_dict=token_Dict,
        accumulated_dataset_epoch=hp.Train.Train_Pattern.Accumulated_Dataset_Epoch,
    )
    collater = Collater(
        token_dict=token_Dict,
        max_abs_mel=hp.Sound.Max_Abs_Mel
    )
    dataLoader = torch.utils.data.DataLoader(
        dataset=dataset,
        collate_fn=collater,  # torch.utils.data.DataLoader
        sampler=torch.utils.data.RandomSampler(dataset),
        batch_size=hp.Train.Batch_Size,
        num_workers=hp.Train.Num_Workers,
        pin_memory=True
    )

    print(next(iter(dataLoader))[0])

    # inference_Dataset = Inference_Dataset(
    #     token_dict=token_Dict,
    #     pattern_paths=['./Inference_for_Training/Example.txt'],
    #     use_cache=False
    # )
    # inference_Collater = Inference_Collater(
    #     token_dict=token_Dict,
    #     max_abs_mel=hp.Sound.Max_Abs_Mel
    # )
    # inference_DataLoader = torch.utils.data.DataLoader(
    #     dataset=inference_Dataset,
    #     collate_fn=inference_Collater,
    #     sampler=torch.utils.data.SequentialSampler(inference_Dataset),
    #     batch_size=hp.Train.Batch_Size,
    #     num_workers=hp.Inference_Batch_Size or hp.Train.Num_Workers,
    #     pin_memory=True
    # )
    # print(next(iter(inference_DataLoader)))
    assert False

if __name__ == "__main__":
    test1()