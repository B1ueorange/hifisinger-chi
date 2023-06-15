import numpy as np
import os, pickle, yaml, argparse, math

import textgrid
from tqdm import tqdm
import librosa
import torch
from argparse import Namespace  # for type

from Audio import Audio_Prep, Mel_Generate, Audio_Prep1
from yin import pitch_calc
from Arg_Parser import Recursive_Parse
import matplotlib
# from resemblyzer import VoiceEncoder, preprocess_wav
import parselmouth
# import speechbrain
import torchaudio
# from resemblyzer import VoiceEncoder, preprocess_wav
# from speechbrain.pretrained import EncoderClassifier
matplotlib.use('agg')
import matplotlib.pyplot as plt

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
            # print(new_note_str)
            # print(cnt)
    note_dict['rest'] = 0
    note_dict['G♯-7'] = 0
    # note_dict['AP'] = 0
    # note_dict['SP'] = 0

def Note_to_Pitch(note: list, pitch_dict: dict):
    return [pitch_dict[x] for x in note]

def normalize(S):
    return np.clip((S + 100) / 100, -2, 2)

def logmelfilterbank(audio, eps=1e-10):

    x_stft = librosa.stft(audio, n_fft=512, hop_length=128,  # stft变换
                          win_length=512, window="hann", pad_mode="reflect")
    spc = np.abs(x_stft).T  # (#frames, #bins)

    # get mel basis  得到mel偏移量
    mel_basis = librosa.filters.mel(sr=24000, n_fft=512,
                                    n_mels=80, fmin=30, fmax=12000)
                                    # norm=None if config['use_same_high_mel'] else 1)

    return 20 * np.log10(np.maximum(eps, np.dot(spc, mel_basis.T)))

def extract_mels(wav):
    wav = wav / np.abs(wav).max() * 0.5
    mel = logmelfilterbank(wav)  #
    frames = len(mel)
    mel = normalize(mel) * 2
    return mel.astype(np.float32)


def Pattern_Test(
    hyper_paramters: Namespace,
    aim: str
    ):
    min_Duration, max_Duration = math.inf, -math.inf
    min_Note, max_Note = math.inf, -math.inf

    paths = []
    Init_Note()

    # aim_pos = 'train'
    aim_pos = aim
    content_dir = (hyper_paramters.Train.Train_Pattern.Path if aim_pos == 'train' else hyper_paramters.Train.Eval_Pattern.Path) + '/' + ('train' if aim_pos == 'train' else 'test') + '.txt'
    cnt = -1
    print(content_dir)
    # encoder = VoiceEncoder()
    with open(content_dir, 'r', encoding='utf-8') as f:
        for line in f:
            tmp1 = line.split('|')
            cnt += 1
            music = []
            tmp_phoneme = tmp1[2].split()
            tmp_note = tmp1[3].split()
            tmp_dur1 = tmp1[4].split()
            tmp_dur2 = tmp1[5].split()
            # last_dur_note = -0.1
            # now_sum_lyrics = 0.0
            for i in range(len(tmp_phoneme)):
                if((tmp_note[i] in ['rest', None]) or (tmp_phoneme[i] in ['AP', 'SP', None])):
                    music.append((float(tmp_dur2[i]), '<X>', 0))
                else:
                    music.append((float(tmp_dur2[i]), tmp_phoneme[i], Note_to_Pitch([tmp_note[i]], note_dict)[0]))
            wav_Path = (hyper_paramters.Train.Train_Pattern.Path if aim_pos == 'train' else hyper_paramters.Train.Eval_Pattern.Path) + '/wavs/' + tmp1[0] + '.wav'
            audio = Audio_Prep(wav_Path, hyper_paramters.Sound.Sample_Rate)
            # audio1 = Audio_Prep1(wav_Path, hyper_paramters.Sound.Sample_Rate)
            # speaker_embedding = encoder.embed_utterance(preprocess_wav(wav_Path))
            if music[0][1] == '<X>':
                audio = audio[int(music[0][0] * hyper_paramters.Sound.Sample_Rate):]
                music = music[1:]
            if music[-1][1] == '<X>':
                audio = audio[:-int(music[-1][0] * hyper_paramters.Sound.Sample_Rate)]
                music = music[:-1]

            previous_Used = 0
            absolute_Position = 0
            mel_Based = []
            for x in music:
                duration = int(x[0] * hyper_paramters.Sound.Sample_Rate) + previous_Used
                previous_Used = duration % hyper_paramters.Sound.Frame_Shift
                duration = duration // hyper_paramters.Sound.Frame_Shift
                #if x[1] == '<X>':
                mel_Based.append((absolute_Position, duration, x[1], x[2]))
                absolute_Position += duration
                    # lyrics = Decompose(x[1])
                    # mel_Based.append((absolute_Position, 2, lyrics[0], x[2]))  # Onset
                    # absolute_Position += 2
                    # mel_Based.append((absolute_Position, duration - 4, lyrics[1], x[2]))  # Onset
                    # absolute_Position += duration - 4
                    # mel_Based.append((absolute_Position, 2, lyrics[2], x[2]))  # Onset
                    # absolute_Position += 2
            music = mel_Based

            # mel = Mel_Generate(
            #     audio,
            #     sample_rate=hyper_paramters.Sound.Sample_Rate,
            #     num_mel=hyper_paramters.Sound.Mel_Dim,
            #     num_frequency=hyper_paramters.Sound.Spectrogram_Dim,
            #     window_length=hyper_paramters.Sound.Frame_Length,
            #     hop_length=hyper_paramters.Sound.Frame_Shift,
            #     mel_fmin=hyper_paramters.Sound.Mel_F_Min,
            #     mel_fmax=hyper_paramters.Sound.Mel_F_Max,
            #     max_abs_value=hyper_paramters.Sound.Max_Abs_Mel
            # )[:absolute_Position]
            mel1 = extract_mels(audio)[:absolute_Position]
            mel = mel1

            pitch = pitch_calc(
                sig=audio,
                sr=hyper_paramters.Sound.Sample_Rate,
                w_len=hyper_paramters.Sound.Frame_Length,
                w_step=hyper_paramters.Sound.Frame_Shift,
                f0_min=hyper_paramters.Sound.F0_Min,
                f0_max=hyper_paramters.Sound.F0_Max,
                confidence_threshold=hyper_paramters.Sound.Confidence_Threshold,
                gaussian_smoothing_sigma=hyper_paramters.Sound.Gaussian_Smoothing_Sigma
            )[:absolute_Position] / hyper_paramters.Sound.F0_Max

            silence = np.where(np.mean(mel, axis=1) < -3.5, np.zeros_like(np.mean(mel, axis=1)),
                               np.ones_like(np.mean(mel, axis=1)))
            cnt1 = -1
            pattern_Index = 0
            for start_Index in tqdm(range(len(music)), desc=os.path.basename(wav_Path)):
                cnt1 += 1
                for end_Index in range(start_Index + 1, len(music), 5):
                    music_Sample = music[start_Index:end_Index]
                    sample_Length = music_Sample[-1][0] + music_Sample[-1][1] - music_Sample[0][0]
                    if sample_Length < hyper_paramters.Min_Duration:
                        continue
                    elif sample_Length > hyper_paramters.Max_Duration:
                        break

                    audio_Sample = audio[music_Sample[0][0] * hyper_paramters.Sound.Frame_Shift:(music_Sample[-1][0] +
                                                                                                 music_Sample[-1][
                                                                                                     1]) * hyper_paramters.Sound.Frame_Shift]
                    mel_Sample = mel[music_Sample[0][0]:music_Sample[-1][0] + music_Sample[-1][1]]
                    silence_Sample = silence[music_Sample[0][0]:music_Sample[-1][0] + music_Sample[-1][1]]
                    pitch_Sample = pitch[music_Sample[0][0]:music_Sample[-1][0] + music_Sample[-1][1]]

                    _, duration_Sample, text_Sample, Note_Sample = zip(*music_Sample)

                    pattern = {
                        'Audio': audio_Sample.astype(np.float32),
                        # 'Speaker_Embedding': speaker_embedding.astype(np.float32),
                        'Mel': mel_Sample.astype(np.float32),
                        'Silence': silence_Sample.astype(np.uint8),
                        'Pitch': pitch_Sample.astype(np.float32),
                        'Duration': duration_Sample,
                        'Text': text_Sample,
                        'Note': Note_Sample,
                        'Singer': 'Female_0',
                        'Dataset': 'NAMS',
                    }
                    pattern_Path = os.path.join(
                        hyper_paramters.Train.Train_Pattern.Path if aim_pos == 'train' else hyper_paramters.Train.Eval_Pattern.Path,
                        'NAMS' + ('1' if aim_pos == 'eval' else ''),
                        '{:03d}'.format(cnt),
                        'NAMS' + ('1' if aim_pos == 'eval' else '') + '.S_{:03d}.P_{:05d}.pickle'.format(cnt,
                                                                                                         pattern_Index)
                    ).replace('\\', '/')

                    # pattern_Path = os.path.join(
                    #     hyper_paramters.Train.Train_Pattern.Path,
                    #     'NAMS_' + aim,
                    #     '{:03d}'.format(cnt),
                    #     'NAMS.S_{:03d}.P_{:05d}.pickle'.format(cnt, pattern_Index)
                    # ).replace('\\', '/')
                    print(pattern_Path)
                    print(np.shape(mel_Sample.T))
                    np.save(os.path.join('tmpnpy', 'NAMS.S_{:03d}.P_{:05d}'.format(cnt, pattern_Index) + '.npy'), pattern['Mel'], allow_pickle=False)
                    tmp_pattern = pickle.load(open(pattern_Path, 'rb'))
                    np.save(os.path.join('tmpnpy', 'NAMS.S_{:03d}.P_{:05d}_after_pickle'.format(cnt, pattern_Index) + '.npy'), tmp_pattern['Mel'], allow_pickle=False)
                    # os.makedirs(os.path.dirname(pattern_Path), exist_ok=True)
                    # pickle.dump(
                    #     pattern,
                    #     open(pattern_Path, 'wb'),
                    #     protocol=4
                    # )
                    pattern_Index += 1

                    min_Duration, max_Duration = min(sample_Length, min_Duration), max(sample_Length, max_Duration)
                # for j in range(len(music)):
                #     print(music[j])
                #     min_Note = min(min_Note, music[j][3])
                #     max_Note = max(max_Note, music[j][3])
            min_Note, max_Note = min(list(zip(*music))[3] + (min_Note,)), max(list(zip(*music))[3] + (max_Note,))
            print('Duration range: {} - {}'.format(min_Duration, max_Duration))
            print('Note range: {} - {}'.format(min_Note, max_Note))
            # for mel_Sample, mel1_Sample, isilence, ipitch in zip(
            #     mel,
            #     mel1,
            #     silence,
            #     pitch
            # ):
            #     cnt += 1
            #     title = 'Note infomation: {}'.format(cnt)
            #     new_Figure = plt.figure(figsize=(25, 5 * 5), dpi=100)
            #     plt.subplot2grid((4, 1), (0, 0))
            #     plt.imshow(mel_Sample, aspect='auto', origin='lower')
            #     plt.title('Mel    {}'.format(title))
            #     plt.colorbar()
            #     plt.subplot2grid((4, 1), (1, 0))
            #     plt.imshow(mel1_Sample, aspect='auto', origin='lower')
            #     plt.title('Ground_Truth_Mel    {}'.format(title))
            #     plt.colorbar()
            #     plt.subplot2grid((4, 1), (2, 0))
            #     plt.plot(isilence)
            #     plt.margins(x=0)
            #     plt.title('Silence    {}'.format(title))
            #     plt.colorbar()
            #     plt.subplot2grid((4, 1), (3, 0))
            #     plt.plot(ipitch)
            #     plt.margins(x=0)
            #     plt.title('Pitch    {}'.format(title))
            #     plt.colorbar()
            #     # duration = duration.ceil().long().clamp(0, self.hp.Max_Duration)
            #     # duration = torch.arange(duration.size(0)).repeat_interleave(duration)
            #     # plt.subplot2grid((4, 1), (3, 0))
            #     # plt.plot(duration)
            #     # plt.margins(x=0)
            #     # plt.title('Duration    {}'.format(title))
            #     # plt.colorbar()
            #     plt.tight_layout()
            #     plt.savefig(os.path.join(hyper_paramters.Inference_Path, 'Step-{}'.format(400000), 'PNG',
            #                              '{}.png'.format(cnt)).replace('\\', '/'))
            #     plt.close(new_Figure)
            #
            # return

            # pattern_Index = 0
            # for start_Index in tqdm(range(len(music)), desc=os.path.basename(wav_Path)):
            #     for end_Index in range(start_Index + 1, len(music), 5):
            #         print(start_Index, end_Index, pattern_Index)
            #         music_Sample = music[start_Index:end_Index]
            #         sample_Length = music_Sample[-1][0] + music_Sample[-1][1] - music_Sample[0][0]
            #         if sample_Length < hyper_paramters.Min_Duration:
            #             continue
            #         elif sample_Length > hyper_paramters.Max_Duration:
            #             break
            #
            #         audio_Sample = audio[music_Sample[0][0] * hyper_paramters.Sound.Frame_Shift:(music_Sample[-1][0] +
            #                                                                                      music_Sample[-1][
            #                                                                                          1]) * hyper_paramters.Sound.Frame_Shift]
            #         mel_Sample = mel[music_Sample[0][0]:music_Sample[-1][0] + music_Sample[-1][1]]
            #         silence_Sample = silence[music_Sample[0][0]:music_Sample[-1][0] + music_Sample[-1][1]]
            #         pitch_Sample = pitch[music_Sample[0][0]:music_Sample[-1][0] + music_Sample[-1][1]]
            #         mel1_Sample = mel1[music_Sample[0][0]:music_Sample[-1][0] + music_Sample[-1][1]]
            #         _, duration_Sample, text_Sample, Note_Sample = zip(*music_Sample)
            #
            #         pattern = {
            #             'Audio': audio_Sample.astype(np.float32),
            #             'Mel': mel_Sample.astype(np.float32),
            #             'Silence': silence_Sample.astype(np.uint8),
            #             'Pitch': pitch_Sample.astype(np.float32),
            #             'Duration': duration_Sample,
            #             'Text': text_Sample,
            #             'Note': Note_Sample,
            #             'Singer': 'Female_0',
            #             'Dataset': 'NAMS',
            #         }
            #
            #         pattern_Path = os.path.join(
            #             hyper_paramters.Train.Train_Pattern.Path if aim_pos == 'train' else hyper_paramters.Train.Eval_Pattern.Path,
            #             'NAMS' + ('1' if aim_pos == 'eval' else ''),
            #             '{:03d}'.format(cnt),
            #             'NAMS' + ('1' if aim_pos == 'eval' else '') + '.S_{:03d}.P_{:05d}.pickle'.format(cnt,
            #                                                                                              pattern_Index)
            #         ).replace('\\', '/')
            #
            #         title = 'Note infomation: {}'.format(pattern_Path)
            #         new_Figure = plt.figure(figsize=(25, 5 * 5), dpi=100)
            #         plt.subplot2grid((4, 1), (0, 0))
            #         plt.imshow(mel_Sample.T, aspect='auto', origin='lower')
            #         plt.title('Mel    {}'.format(title))
            #         plt.colorbar()
            #         plt.subplot2grid((4, 1), (1, 0))
            #         plt.imshow(mel1_Sample.T, aspect='auto', origin='lower')
            #         plt.title('Ground_Truth_Mel    {}'.format(title))
            #         plt.colorbar()
            #         # plt.subplot2grid((4, 1), (2, 0))
            #         # plt.plot(silence_Sample)
            #         # plt.margins(x=0)
            #         # plt.title('Silence    {}'.format(title))
            #         # plt.colorbar()
            #         plt.subplot2grid((4, 1), (3, 0))
            #         plt.plot(pitch_Sample)
            #         plt.margins(x=0)
            #         plt.title('Pitch    {}'.format(title))
            #         plt.colorbar()
            #         duration_Sample = torch.tensor(duration_Sample)
            #         duration_Sample = duration_Sample.long().clamp(0, hyper_paramters.Max_Duration)
            #         duration_Sample = torch.arange(duration_Sample.size(0)).repeat_interleave(duration_Sample)
            #         # duration_Sample = np.array(duration_Sample, dtype=int)
            #         # duration_Sample = np.clip(duration_Sample, 0, hyper_paramters.Max_Duration)
            #         # duration_Sample = torch.arange(duration_Sample.size(0)).repeat_interleave(duration_Sample)
            #         plt.subplot2grid((4, 1), (2, 0))
            #         plt.plot(duration_Sample)
            #         plt.margins(x=0)
            #         plt.title('Duration    {}'.format(title))
            #         plt.colorbar()
            #         plt.tight_layout()
            #         plt.savefig(os.path.join(hyper_paramters.Inference_Path, 'Step-{}'.format(400000), 'PNG',
            #                                  '{}.png'.format(os.path.basename(pattern_Path))).replace('\\', '/'))
            #         plt.close(new_Figure)
            #
            #
            #         print(pattern_Path)
            #         os.makedirs(os.path.dirname(pattern_Path), exist_ok=True)
            #         pickle.dump(
            #             pattern,
            #             open(pattern_Path, 'wb'),
            #             protocol=4
            #         )
            #         pattern_Index += 1
            #         if(pattern_Index >= 10):
            #             return
            #
            #         min_Duration, max_Duration = min(sample_Length, min_Duration), max(sample_Length, max_Duration)


def Pattern_Generate(
    hyper_paramters: Namespace,
    aim: str
    ):
    min_Duration, max_Duration = math.inf, -math.inf
    min_Note, max_Note = math.inf, -math.inf

    paths = []
    Init_Note()

    # aim_pos = 'train'
    aim_pos = aim
    content_dir = (hyper_paramters.Train.Train_Pattern.Path if aim_pos == 'train' else hyper_paramters.Train.Eval_Pattern.Path) + '/' + ('train' if aim_pos == 'train' else 'test') + '.txt'
    cnt = -1
    print(content_dir)
    with open(content_dir, 'r', encoding='utf-8') as f:
        for line in f:
            tmp1 = line.split('|')
            cnt += 1
            music = []
            tmp_phoneme = tmp1[2].split()
            tmp_note = tmp1[3].split()
            tmp_dur1 = tmp1[4].split()
            tmp_dur2 = tmp1[5].split()
            # last_dur_note = -0.1
            # now_sum_lyrics = 0.0
            for i in range(len(tmp_phoneme)):
                if((tmp_note[i] in ['rest', None]) or (tmp_phoneme[i] in ['AP', 'SP', None])):
                    music.append((float(tmp_dur2[i]), '<X>', 0))
                else:
                    music.append((float(tmp_dur2[i]), tmp_phoneme[i], Note_to_Pitch([tmp_note[i]], note_dict)[0]))
            wav_Path = (hyper_paramters.Train.Train_Pattern.Path if aim_pos == 'train' else hyper_paramters.Train.Eval_Pattern.Path) + '/wavs/' + tmp1[0] + '.wav'
            audio = Audio_Prep(wav_Path, hyper_paramters.Sound.Sample_Rate)
            audio1 = Audio_Prep1(wav_Path, hyper_paramters.Sound.Sample_Rate)
            if music[0][1] == '<X>':
                audio = audio[int(music[0][0] * hyper_paramters.Sound.Sample_Rate):]
                music = music[1:]
            if music[-1][1] == '<X>':
                audio = audio[:-int(music[-1][0] * hyper_paramters.Sound.Sample_Rate)]
                music = music[:-1]

            previous_Used = 0
            absolute_Position = 0
            mel_Based = []
            for x in music:
                duration = int(x[0] * hyper_paramters.Sound.Sample_Rate) + previous_Used
                previous_Used = duration % hyper_paramters.Sound.Frame_Shift
                duration = duration // hyper_paramters.Sound.Frame_Shift
                #if x[1] == '<X>':
                mel_Based.append((absolute_Position, duration, x[1], x[2]))
                absolute_Position += duration
                    # lyrics = Decompose(x[1])
                    # mel_Based.append((absolute_Position, 2, lyrics[0], x[2]))  # Onset
                    # absolute_Position += 2
                    # mel_Based.append((absolute_Position, duration - 4, lyrics[1], x[2]))  # Onset
                    # absolute_Position += duration - 4
                    # mel_Based.append((absolute_Position, 2, lyrics[2], x[2]))  # Onset
                    # absolute_Position += 2
            music = mel_Based

            # mel = Mel_Generate(
            #     audio,
            #     sample_rate=hyper_paramters.Sound.Sample_Rate,
            #     num_mel=hyper_paramters.Sound.Mel_Dim,
            #     num_frequency=hyper_paramters.Sound.Spectrogram_Dim,
            #     window_length=hyper_paramters.Sound.Frame_Length,
            #     hop_length=hyper_paramters.Sound.Frame_Shift,
            #     mel_fmin=hyper_paramters.Sound.Mel_F_Min,
            #     mel_fmax=hyper_paramters.Sound.Mel_F_Max,
            #     max_abs_value=hyper_paramters.Sound.Max_Abs_Mel
            # )[:absolute_Position]

            mel1 = extract_mels(audio)[:absolute_Position]
            mel = mel1

            pitch = pitch_calc(
                sig=audio,
                sr=hyper_paramters.Sound.Sample_Rate,
                w_len=hyper_paramters.Sound.Frame_Length,
                w_step=hyper_paramters.Sound.Frame_Shift,
                f0_min=hyper_paramters.Sound.F0_Min,
                f0_max=hyper_paramters.Sound.F0_Max,
                confidence_threshold=hyper_paramters.Sound.Confidence_Threshold,
                gaussian_smoothing_sigma=hyper_paramters.Sound.Gaussian_Smoothing_Sigma
            )[:absolute_Position] / hyper_paramters.Sound.F0_Max

            silence = np.where(np.mean(mel, axis=1) < -3.5, np.zeros_like(np.mean(mel, axis=1)),
                               np.ones_like(np.mean(mel, axis=1)))

            pattern_Index = 0
            for start_Index in tqdm(range(len(music)), desc=os.path.basename(wav_Path)):
                for end_Index in range(start_Index + 1, len(music), 5):
                    music_Sample = music[start_Index:end_Index]
                    sample_Length = music_Sample[-1][0] + music_Sample[-1][1] - music_Sample[0][0]
                    if sample_Length < hyper_paramters.Min_Duration:
                        continue
                    elif sample_Length > hyper_paramters.Max_Duration:
                        break

                    audio_Sample = audio[music_Sample[0][0] * hyper_paramters.Sound.Frame_Shift:(music_Sample[-1][0] +
                                                                                                 music_Sample[-1][
                                                                                                     1]) * hyper_paramters.Sound.Frame_Shift]
                    mel_Sample = mel[music_Sample[0][0]:music_Sample[-1][0] + music_Sample[-1][1]]
                    silence_Sample = silence[music_Sample[0][0]:music_Sample[-1][0] + music_Sample[-1][1]]
                    pitch_Sample = pitch[music_Sample[0][0]:music_Sample[-1][0] + music_Sample[-1][1]]

                    _, duration_Sample, text_Sample, Note_Sample = zip(*music_Sample)

                    pattern = {
                        'Audio': audio_Sample.astype(np.float32),
                        'Mel': mel_Sample.astype(np.float32),
                        'Silence': silence_Sample.astype(np.uint8),
                        'Pitch': pitch_Sample.astype(np.float32),
                        'Duration': duration_Sample,
                        'Text': text_Sample,
                        'Note': Note_Sample,
                        'Singer': 'Female_0',
                        'Dataset': 'NAMS',
                    }

                    pattern_Path = os.path.join(
                        hyper_paramters.Train.Train_Pattern.Path if aim_pos == 'train' else hyper_paramters.Train.Eval_Pattern.Path,
                        'NAMS' + ('1' if aim_pos == 'eval' else ''),
                        '{:03d}'.format(cnt),
                        'NAMS' + ('1' if aim_pos == 'eval' else '') + '.S_{:03d}.P_{:05d}.pickle'.format(cnt, pattern_Index)
                    ).replace('\\', '/')
                    print(pattern_Path)
                    os.makedirs(os.path.dirname(pattern_Path), exist_ok=True)
                    pickle.dump(
                        pattern,
                        open(pattern_Path, 'wb'),
                        protocol=4
                    )
                    pattern_Index += 1

                    min_Duration, max_Duration = min(sample_Length, min_Duration), max(sample_Length, max_Duration)
            # for j in range(len(music)):
            #     print(music[j])
            #     min_Note = min(min_Note, music[j][3])
            #     max_Note = max(max_Note, music[j][3])
            min_Note, max_Note = min(list(zip(*music))[3] + (min_Note,)), max(list(zip(*music))[3] + (max_Note,))
    print('Duration range: {} - {}'.format(min_Duration, max_Duration))
    print('Note range: {} - {}'.format(min_Note, max_Note))

def Calc_Note(content_dir, hyper_paramters):
    cnt = -1
    fenzi = 0
    fenmu = 0
    fenzi_1 = 0
    print(content_dir)
    shengmu_str = 'b p m f d t n l g k h j q x zh ch sh r z c s y w'
    shengmu_list = shengmu_str.split()
    shengmu_dic = {}
    for i in shengmu_list:
        shengmu_dic[i] = 1
    with open(content_dir, 'r', encoding='utf-8') as f:
        for line in f:
            tmp1 = line.split('|')
            cnt += 1
            music = []
            tmp_phoneme = tmp1[2].split()
            tmp_note = tmp1[3].split()
            tmp_dur1 = tmp1[4].split()
            tmp_dur2 = tmp1[5].split()
            # last_dur_note = -0.1
            # now_sum_lyrics = 0.0
            # wav_Path = (hyper_paramters.Train.Train_Pattern.Path if aim_pos == 'train' else hyper_paramters.Train.Eval_Pattern.Path) + '/wavs/' + \
            #            tmp1[0] + '.wav'
            wav_Path = ''
            audio1 = parselmouth.Sound(wav_Path)
            audio = audio1.to_pitch()
            pitch_values = audio.selected_array['frequency']
            pitch_values[pitch_values == 0] = 0.1


            notes = librosa.hz_to_note(pitch_values)
            last_time = 0.0
            new_time = 0.0
            tmp_fenzi = 0
            tmp_fenmu = 0

            tmp_ans = ''
            tmp_ans_list = []

            smax = audio1.xmax
            smin = audio1.xmin
            for i in range(len(tmp_phoneme)):
                if (tmp_phoneme[i] == 'AP' or tmp_phoneme[i] == 'SP'):
                    tmp_ans_list.append('G♯-7')
                    continue
                tmp_note_dict = {}
                tmp_cnt = 0
                fenmu += 1
                tmp_fenmu += 1
                new_time = last_time + float(tmp_dur2[i])
                l = int(math.ceil(last_time * 100))
                l = int(last_time * 100)
                r = min(int(new_time * 100), len(notes) - 1)
                if (r != len(notes) - 1):
                    r = max(0, r - 4)
                last_time = new_time
                for j in range(l, r + 1):
                    if (tmp_note_dict.__contains__(notes[j])):
                        tmp_note_dict[notes[j]] = tmp_note_dict[notes[j]] + 1
                        if (tmp_note_dict[notes[j]] > tmp_cnt):
                            tmp_ans = notes[j]
                            tmp_cnt = tmp_note_dict[notes[j]]
                    else:
                        tmp_note_dict[notes[j]] = 1
                        if (tmp_note_dict[notes[j]] > tmp_cnt):
                            tmp_ans = notes[j]
                            tmp_cnt = tmp_note_dict[notes[j]]
                tmp_ans_list.append(tmp_ans)
                if (tmp_ans == 'G♯-7'):
                    tmp_debug = notes[l:r + 1]
                    # draw_pitch(audio, l, r, smin, smax)
                    fenzi_1 += 1
                    # os.system("pause")
                    continue
                if (tmp_ans[1] == '♯'):
                    if (tmp_ans[0] == 'C'):
                        tmp_ans = tmp_ans[0] + '#' + tmp_ans[2:] + '/' + 'Db' + tmp_ans[2:]
                    if (tmp_ans[0] == 'D'):
                        tmp_ans = tmp_ans[0] + '#' + tmp_ans[2:] + '/' + 'Eb' + tmp_ans[2:]
                    if (tmp_ans[0] == 'F'):
                        tmp_ans = tmp_ans[0] + '#' + tmp_ans[2:] + '/' + 'Gb' + tmp_ans[2:]
                    if (tmp_ans[0] == 'G'):
                        tmp_ans = tmp_ans[0] + '#' + tmp_ans[2:] + '/' + 'Ab' + tmp_ans[2:]
                    if (tmp_ans[0] == 'A'):
                        tmp_ans = tmp_ans[0] + '#' + tmp_ans[2:] + '/' + 'Bb' + tmp_ans[2:]
                if (tmp_ans == tmp_note[i] or abs(note_dict[tmp_ans] - note_dict[tmp_note[i]]) <= 0):
                    tmp_fenzi += 1
                    fenzi += 1
            # tmp_jishu = -1
            # print(len(tmp_phoneme))
            # print(len(tmp_ans_list))
            for i in range(len(tmp_phoneme)):
                if (tmp_phoneme[i] == 'AP' or tmp_phoneme[i] == 'SP'):
                    continue
                if (not tmp_ans_list[i] == 'G♯-7'):
                    continue
                if (shengmu_dic.__contains__(tmp_phoneme[i])):
                    if ((i + 1) < len(tmp_phoneme)):
                        if (not tmp_ans_list[i + 1] == 'G♯-7'):
                            tmp_ans_list[i] = tmp_ans_list[i + 1]
                    else:
                        tmp_ans_list[i] = tmp_ans_list[i - 1]
                else:
                    if (i - 1 >= 0):
                        if (not tmp_ans_list[i - 1] == 'G♯-7'):
                            tmp_ans_list[i] = tmp_ans_list[i - 1]
                    else:
                        tmp_ans_list[i] = tmp_ans_list[i + 1]
                if (tmp_ans_list[i] == 'G♯-7' or tmp_ans_list[i] == '-1'):
                    continue
                if (tmp_ans_list[i][1] == '♯'):
                    if (tmp_ans_list[i][0] == 'C'):
                        tmp_ans_list[i] = tmp_ans_list[i][0] + '#' + tmp_ans_list[i][2:] + '/' + 'Db' + tmp_ans_list[i][
                                                                                                        2:]
                    if (tmp_ans_list[i][0] == 'D'):
                        tmp_ans_list[i] = tmp_ans_list[i][0] + '#' + tmp_ans_list[i][2:] + '/' + 'Eb' + tmp_ans_list[i][
                                                                                                        2:]
                    if (tmp_ans_list[i][0] == 'F'):
                        tmp_ans_list[i] = tmp_ans_list[i][0] + '#' + tmp_ans_list[i][2:] + '/' + 'Gb' + tmp_ans_list[i][
                                                                                                        2:]
                    if (tmp_ans_list[i][0] == 'G'):
                        tmp_ans_list[i] = tmp_ans_list[i][0] + '#' + tmp_ans_list[i][2:] + '/' + 'Ab' + tmp_ans_list[i][
                                                                                                        2:]
                    if (tmp_ans_list[i][0] == 'A'):
                        tmp_ans_list[i] = tmp_ans_list[i][0] + '#' + tmp_ans_list[i][2:] + '/' + 'Bb' + tmp_ans_list[i][
                                                                                                        2:]
                if (tmp_ans_list[i] == tmp_note[i] or abs(note_dict[tmp_ans_list[i]] - note_dict[tmp_note[i]]) <= 0):
                    tmp_fenzi += 1
                    fenzi += 1
                    fenzi_1 -= 1

def Pattern_Generate_OpenSinger(
    hyper_paramters: Namespace,
    aim: str
    ):
    min_Duration, max_Duration = math.inf, -math.inf
    min_Note, max_Note = math.inf, -math.inf

    paths = []
    Init_Note()
    aim_pos = aim
    # content_dir = (hyper_paramters.Train.Train_Pattern.Path if aim_pos == 'train' else hyper_paramters.Train.Eval_Pattern.Path) + '/' + ('train' if aim_pos == 'train' else 'test') + '.txt'
    # content_dir = 'D:/Download/OpenSinger1/' + aim + '/'
    content_dir = './OpenSinger1/' + aim + '/'
    cnt = -1
    # print(content_dir)
    shengmu_str = 'b p m f d t n l g k h j q x zh ch sh r z c s y w'
    shengmu_list = shengmu_str.split()
    shengmu_dic = {}
    # encoder = VoiceEncoder()
    for i in shengmu_list:
        shengmu_dic[i] = 1
    file_list = os.listdir(content_dir)
    # grid_dir = 'D:/Download/output1/'
    grid_dir = './output1/'
    for f in file_list:
        # print(f)
        filename = f.split('.')
        # print(filename[1])
        if(filename[1] != 'wav'):
            # print(filename[1])
            continue
        cnt += 1
        # if(cnt <= 31581):
        #     continue
        # last_dur_note = -0.1
        # now_sum_lyrics = 0.0
        # for i in range(len(tmp_phoneme)):
        #     if ((tmp_note[i] in ['rest', None]) or (tmp_phoneme[i] in ['AP', 'SP', None])):
        #         music.append((float(tmp_dur2[i]), '<X>', 0))
        #     else:
        #         music.append((float(tmp_dur2[i]), tmp_phoneme[i], Note_to_Pitch([tmp_note[i]], note_dict)[0]))
        # wav_Path = (hyper_paramters.Train.Train_Pattern.Path if aim_pos == 'train' else hyper_paramters.Train.Eval_Pattern.Path) + '/wavs/' + tmp1[0] + '.wav'
        wav_Path = content_dir + '/' + f

        # audio1 = Audio_Prep1(wav_Path, hyper_paramters.Sound.Sample_Rate)
        grid_file_dir = grid_dir + filename[0] + '.TextGrid'
        if (not os.path.exists(grid_file_dir)):
            continue
        # print(grid_file_dir)
        music = []
        # print(grid_file_dir)
        tg = textgrid.TextGrid()
        tg.read(grid_file_dir)
        tmp = tg.tiers[0]
        for j in tmp:
            if(j.mark == ''):
                music.append([float(j.maxTime) - float(j.minTime), 'SP', 0])
            else:
                music.append([float(j.maxTime) - float(j.minTime), j.mark, 0])

        for j in range(0, len(music)):
            if(music[j][1] == ''):
                las = j - 1
                nex = j + 1
                if(las >= 0 and shengmu_dic.__contains__(music[las][1]) and music[nex][1] != 'SP'):
                    if(music[las][0] > music[nex][0]):
                        music[nex][0] += music[j][0]
                    else:
                        music[las][0] += music[j][0] / 2
                        music[nex][0] += music[j][0] / 2
                    music[j][0] = 0
                else:
                    if(las >= 0 and music[las][0] < 0.17):
                        d = min(music[j][0], 0.17 - music[las][0])
                        music[las][0] += d
                        music[j][0] -= d
                    if(nex < len(music) and music[nex][0] < 0.17):
                        d = min(music[j][0], 0.17 - music[nex][0])
                        music[nex][0] += d
                        music[j][0] -= d
        music1 = []
        for j in range(0, len(music)):
            if(abs(music[j][0] - 0.0) > 1e-2):
                music1.append(music[j])
        music = music1

        audio1 = parselmouth.Sound(wav_Path)
        audio = audio1.to_pitch()
        pitch_values = audio.selected_array['frequency']
        pitch_values[pitch_values == 0] = 0.1

        notes = librosa.hz_to_note(pitch_values)
        last_time = 0.0
        new_time = 0.0
        tmp_fenzi = 0
        tmp_fenmu = 0

        tmp_ans = ''
        tmp_ans_list = []

        smax = audio1.xmax
        smin = audio1.xmin
        for i in range(len(music)):
            if (music[i][1] == 'AP' or music[i][1] == 'SP'):
                tmp_ans_list.append('G♯-7')
                continue
            tmp_note_dict = {}
            tmp_cnt = 0
            new_time = last_time + music[i][0]
            l = int(math.ceil(last_time * 100))
            l = int(last_time * 100)
            r = min(int(new_time * 100), len(notes) - 1)
            if (r != len(notes) - 1):
                r = max(0, r - 4)
            last_time = new_time
            tmp_ans = 'G♯-7'
            for j in range(l, r + 1):
                if (notes[j] == 'G♯-7'):
                    continue
                if (tmp_note_dict.__contains__(notes[j])):
                    tmp_note_dict[notes[j]] = tmp_note_dict[notes[j]] + 1
                    if (tmp_note_dict[notes[j]] > tmp_cnt):
                        tmp_ans = notes[j]
                        tmp_cnt = tmp_note_dict[notes[j]]
                else:
                    tmp_note_dict[notes[j]] = 1
                    if (tmp_note_dict[notes[j]] > tmp_cnt):
                        tmp_ans = notes[j]
                        tmp_cnt = tmp_note_dict[notes[j]]
            if (tmp_ans == 'G♯-7'):
                tmp_ans_list.append(tmp_ans)
                tmp_debug = notes[l:r + 1]
                # draw_pitch(audio, l, r, smin, smax)

                # os.system("pause")
                continue
            if (tmp_ans[1] == '♯'):
                if (tmp_ans[0] == 'C'):
                    tmp_ans = tmp_ans[0] + '#' + tmp_ans[2:] + '/' + 'Db' + tmp_ans[2:]
                if (tmp_ans[0] == 'D'):
                    tmp_ans = tmp_ans[0] + '#' + tmp_ans[2:] + '/' + 'Eb' + tmp_ans[2:]
                if (tmp_ans[0] == 'F'):
                    tmp_ans = tmp_ans[0] + '#' + tmp_ans[2:] + '/' + 'Gb' + tmp_ans[2:]
                if (tmp_ans[0] == 'G'):
                    tmp_ans = tmp_ans[0] + '#' + tmp_ans[2:] + '/' + 'Ab' + tmp_ans[2:]
                if (tmp_ans[0] == 'A'):
                    tmp_ans = tmp_ans[0] + '#' + tmp_ans[2:] + '/' + 'Bb' + tmp_ans[2:]
            tmp_ans_list.append(tmp_ans)

        tmp_jishu = -1
        # print(len(tmp_phoneme))
        # print(len(tmp_ans_list))
        for i in range(len(music)):
            if (music[i][1] == 'AP' or music[i][1] == 'SP'):
                continue
            if (not tmp_ans_list[i] == 'G♯-7'):
                continue
            if (shengmu_dic.__contains__(music[i][1])):
                if ((i + 1) < len(music[i][1])):
                    if (not tmp_ans_list[i + 1] == 'G♯-7'):
                        tmp_ans_list[i] = tmp_ans_list[i + 1]
                else:
                    tmp_ans_list[i] = tmp_ans_list[i - 1]
            else:
                if (i - 1 >= 0):
                    if (not tmp_ans_list[i - 1] == 'G♯-7'):
                        tmp_ans_list[i] = tmp_ans_list[i - 1]
                else:
                    tmp_ans_list[i] = tmp_ans_list[i + 1]
            if (tmp_ans_list[i] == 'G♯-7' or tmp_ans_list[i] == '-1'):
                continue
            if (tmp_ans_list[i][1] == '♯'):
                if (tmp_ans_list[i][0] == 'C'):
                    tmp_ans_list[i] = tmp_ans_list[i][0] + '#' + tmp_ans_list[i][2:] + '/' + 'Db' + tmp_ans_list[i][2:]
                if (tmp_ans_list[i][0] == 'D'):
                    tmp_ans_list[i] = tmp_ans_list[i][0] + '#' + tmp_ans_list[i][2:] + '/' + 'Eb' + tmp_ans_list[i][2:]
                if (tmp_ans_list[i][0] == 'F'):
                    tmp_ans_list[i] = tmp_ans_list[i][0] + '#' + tmp_ans_list[i][2:] + '/' + 'Gb' + tmp_ans_list[i][2:]
                if (tmp_ans_list[i][0] == 'G'):
                    tmp_ans_list[i] = tmp_ans_list[i][0] + '#' + tmp_ans_list[i][2:] + '/' + 'Ab' + tmp_ans_list[i][2:]
                if (tmp_ans_list[i][0] == 'A'):
                    tmp_ans_list[i] = tmp_ans_list[i][0] + '#' + tmp_ans_list[i][2:] + '/' + 'Bb' + tmp_ans_list[i][2:]

        for i in range(len(music)):
            if(music[i][1] == 'SP' or music[i][1] == 'AP'):
                music[i][2] = 0
            else:
                music[i][2] = Note_to_Pitch([tmp_ans_list[i]], note_dict)[0]
        # tmp_note = Calc_Note(wav_Path)
        # wav = preprocess_wav(wav_Path)
        # encoder = VoiceEncoder()
        # embed = encoder.embed_utterance(wav)
        audio = Audio_Prep(wav_Path, hyper_paramters.Sound.Sample_Rate)

        if music[0][1] == 'SP' or music[0][1] == 'AP':
            audio = audio[int(music[0][0] * hyper_paramters.Sound.Sample_Rate):]
            music = music[1:]
        if music[0][1] == 'SP' or music[0][1] == 'AP':
            audio = audio[:-int(music[-1][0] * hyper_paramters.Sound.Sample_Rate)]
            music = music[:-1]
        print(f)
        previous_Used = 0
        absolute_Position = 0
        mel_Based = []
        for x in music:
            duration = int(x[0] * hyper_paramters.Sound.Sample_Rate) + previous_Used
            previous_Used = duration % hyper_paramters.Sound.Frame_Shift
            duration = duration // hyper_paramters.Sound.Frame_Shift
            # if x[1] == '<X>':
            mel_Based.append((absolute_Position, duration, x[1], x[2]))
            absolute_Position += duration
        music = mel_Based
        mel1 = extract_mels(audio)[:absolute_Position]
        mel = mel1

        pitch = pitch_calc(
            sig=audio,
            sr=hyper_paramters.Sound.Sample_Rate,
            w_len=hyper_paramters.Sound.Frame_Length,
            w_step=hyper_paramters.Sound.Frame_Shift,
            f0_min=hyper_paramters.Sound.F0_Min,
            f0_max=hyper_paramters.Sound.F0_Max,
            confidence_threshold=hyper_paramters.Sound.Confidence_Threshold,
            gaussian_smoothing_sigma=hyper_paramters.Sound.Gaussian_Smoothing_Sigma
        )[:absolute_Position] / hyper_paramters.Sound.F0_Max

        silence = np.where(np.mean(mel, axis=1) < -3.5, np.zeros_like(np.mean(mel, axis=1)),
                           np.ones_like(np.mean(mel, axis=1)))

        pattern_Index = 0

        # wav = preprocess_wav(wav_Path)
        # embed = encoder.embed_utterance(wav)

        for start_Index in tqdm(range(len(music)), desc=os.path.basename(wav_Path)):
            for end_Index in range(start_Index + 1, len(music), 5):
                music_Sample = music[start_Index:end_Index]
                sample_Length = music_Sample[-1][0] + music_Sample[-1][1] - music_Sample[0][0]
                if sample_Length < hyper_paramters.Min_Duration:
                    continue
                elif sample_Length > hyper_paramters.Max_Duration:
                    break

                audio_Sample = audio[music_Sample[0][0] * hyper_paramters.Sound.Frame_Shift:(music_Sample[-1][0] +
                                                                                             music_Sample[-1][
                                                                                                 1]) * hyper_paramters.Sound.Frame_Shift]
                mel_Sample = mel[music_Sample[0][0]:music_Sample[-1][0] + music_Sample[-1][1]]
                silence_Sample = silence[music_Sample[0][0]:music_Sample[-1][0] + music_Sample[-1][1]]
                pitch_Sample = pitch[music_Sample[0][0]:music_Sample[-1][0] + music_Sample[-1][1]]

                _, duration_Sample, text_Sample, Note_Sample = zip(*music_Sample)
                # sss = filename.split('_')[0] + '_' + filename.split('_')[1] + '_' + filename.split('_')[2]
                pattern = {
                    'Audio': audio_Sample.astype(np.float32),
                    'Mel': mel_Sample.astype(np.float32),
                    'Silence': silence_Sample.astype(np.uint8),
                    'Pitch': pitch_Sample.astype(np.float32),
                    'Duration': duration_Sample,
                    'Text': text_Sample,
                    'Note': Note_Sample,
                    # 'Speaker_Embedding': embed,
                    # 'Singer': 'Female_0',
                    'Singer': filename[0],
                    'Dataset': aim,
                }

                # pattern_Path = os.path.join(
                #     hyper_paramters.Train.Train_Pattern.Path if aim_pos == 'train' else hyper_paramters.Train.Eval_Pattern.Path,
                #     'NAMS' + ('1' if aim_pos == 'eval' else ''),
                #     '{:03d}'.format(cnt),
                #     'NAMS' + ('1' if aim_pos == 'eval' else '') + '.S_{:03d}.P_{:05d}.pickle'.format(cnt, pattern_Index)
                # ).replace('\\', '/')
                # pattern_Path = os.path.join(
                #     'D:/Download/OpenSingerPickle',
                #     aim,
                #     '{:03d}'.format(cnt),
                #     aim + '.S_{:03d}.P_{:05d}.pickle'.format(cnt, pattern_Index)
                # ).replace('\\', '/')
                pattern_Path = os.path.join(
                    './OpenSingerPickle',
                    aim,
                    '{:03d}'.format(cnt),
                    aim + '.S_{:03d}.P_{:05d}.pickle'.format(cnt, pattern_Index)
                ).replace('\\', '/')
                print(pattern_Path)
                os.makedirs(os.path.dirname(pattern_Path), exist_ok=True)
                pickle.dump(
                    pattern,
                    open(pattern_Path, 'wb'),
                    protocol=4
                )
                pattern_Index += 1

                min_Duration, max_Duration = min(sample_Length, min_Duration), max(sample_Length, max_Duration)
        # for j in range(len(music)):
        #     print(music[j])
        #     min_Note = min(min_Note, music[j][3])
        #     max_Note = max(max_Note, music[j][3])
        # min_Note, max_Note = min(list(zip(*music))[3] + (min_Note,)), max(list(zip(*music))[3] + (max_Note,))

    # print('Duration range: {} - {}'.format(min_Duration, max_Duration))
    # print('Note range: {} - {}'.format(min_Note, max_Note))

def Pattern_Generate_OpenSinger_20230301(
    hyper_paramters: Namespace,
    aim: str
    ):
    min_Duration, max_Duration = math.inf, -math.inf
    min_Note, max_Note = math.inf, -math.inf

    paths = []
    Init_Note()
    aim_pos = aim
    # content_dir = (hyper_paramters.Train.Train_Pattern.Path if aim_pos == 'train' else hyper_paramters.Train.Eval_Pattern.Path) + '/' + ('train' if aim_pos == 'train' else 'test') + '.txt'
    # content_dir = 'D:/Download/OpenSinger1/' + aim + '/'
    content_dir = './OpenSinger1/' + aim + '/'
    cnt = -1
    # print(content_dir)
    shengmu_str = 'b p m f d t n l g k h j q x zh ch sh r z c s y w'
    shengmu_list = shengmu_str.split()
    shengmu_dic = {}
    # encoder = VoiceEncoder()
    for i in shengmu_list:
        shengmu_dic[i] = 1
    file_list = os.listdir(content_dir)
    # grid_dir = 'D:/Download/output1/'
    grid_dir = './output1/'
    for f in file_list:
        # print(f)
        filename = f.split('.')
        # print(filename[1])
        if(filename[1] != 'wav'):
            # print(filename[1])
            continue

        cnt += 1
        wav_Path = content_dir + '/' + f

        # audio1 = Audio_Prep1(wav_Path, hyper_paramters.Sound.Sample_Rate)
        grid_file_dir = grid_dir + filename[0] + '.TextGrid'
        if (not os.path.exists(grid_file_dir)):
            continue
        # print(grid_file_dir)
        music = []
        # print(grid_file_dir)
        tg = textgrid.TextGrid()
        tg.read(grid_file_dir)
        tmp = tg.tiers[0]
        for j in tmp:
            if(j.mark == ''):
                music.append([float(j.maxTime) - float(j.minTime), 'SP', 0])
            else:
                music.append([float(j.maxTime) - float(j.minTime), j.mark, 0])

        for j in range(0, len(music)):
            if(music[j][1] == ''):
                las = j - 1
                nex = j + 1
                if(las >= 0 and shengmu_dic.__contains__(music[las][1]) and music[nex][1] != 'SP'):
                    if(music[las][0] > music[nex][0]):
                        music[nex][0] += music[j][0]
                    else:
                        music[las][0] += music[j][0] / 2
                        music[nex][0] += music[j][0] / 2
                    music[j][0] = 0
                else:
                    if(las >= 0 and music[las][0] < 0.17):
                        d = min(music[j][0], 0.17 - music[las][0])
                        music[las][0] += d
                        music[j][0] -= d
                    if(nex < len(music) and music[nex][0] < 0.17):
                        d = min(music[j][0], 0.17 - music[nex][0])
                        music[nex][0] += d
                        music[j][0] -= d
        music1 = []
        for j in range(0, len(music)):
            if(abs(music[j][0] - 0.0) > 1e-2):
                music1.append(music[j])
        music = music1

        audio1 = parselmouth.Sound(wav_Path)
        audio = audio1.to_pitch()
        pitch_values = audio.selected_array['frequency']
        pitch_values[pitch_values == 0] = 0.1

        notes = librosa.hz_to_note(pitch_values)
        last_time = 0.0
        new_time = 0.0
        tmp_fenzi = 0
        tmp_fenmu = 0

        tmp_ans = ''
        tmp_ans_list = []

        smax = audio1.xmax
        smin = audio1.xmin
        for i in range(len(music)):
            if (music[i][1] == 'AP' or music[i][1] == 'SP'):
                tmp_ans_list.append('G♯-7')
                continue
            tmp_note_dict = {}
            tmp_cnt = 0
            new_time = last_time + music[i][0]
            l = int(math.ceil(last_time * 100))
            l = int(last_time * 100)
            r = min(int(new_time * 100), len(notes) - 1)
            if (r != len(notes) - 1):
                r = max(0, r - 4)
            last_time = new_time
            tmp_ans = 'G♯-7'
            for j in range(l, r + 1):
                if (notes[j] == 'G♯-7'):
                    continue
                if (tmp_note_dict.__contains__(notes[j])):
                    tmp_note_dict[notes[j]] = tmp_note_dict[notes[j]] + 1
                    if (tmp_note_dict[notes[j]] > tmp_cnt):
                        tmp_ans = notes[j]
                        tmp_cnt = tmp_note_dict[notes[j]]
                else:
                    tmp_note_dict[notes[j]] = 1
                    if (tmp_note_dict[notes[j]] > tmp_cnt):
                        tmp_ans = notes[j]
                        tmp_cnt = tmp_note_dict[notes[j]]
            if (tmp_ans == 'G♯-7'):
                tmp_ans_list.append(tmp_ans)
                tmp_debug = notes[l:r + 1]
                # draw_pitch(audio, l, r, smin, smax)

                # os.system("pause")
                continue
            if (tmp_ans[1] == '♯'):
                if (tmp_ans[0] == 'C'):
                    tmp_ans = tmp_ans[0] + '#' + tmp_ans[2:] + '/' + 'Db' + tmp_ans[2:]
                if (tmp_ans[0] == 'D'):
                    tmp_ans = tmp_ans[0] + '#' + tmp_ans[2:] + '/' + 'Eb' + tmp_ans[2:]
                if (tmp_ans[0] == 'F'):
                    tmp_ans = tmp_ans[0] + '#' + tmp_ans[2:] + '/' + 'Gb' + tmp_ans[2:]
                if (tmp_ans[0] == 'G'):
                    tmp_ans = tmp_ans[0] + '#' + tmp_ans[2:] + '/' + 'Ab' + tmp_ans[2:]
                if (tmp_ans[0] == 'A'):
                    tmp_ans = tmp_ans[0] + '#' + tmp_ans[2:] + '/' + 'Bb' + tmp_ans[2:]
            tmp_ans_list.append(tmp_ans)

        tmp_jishu = -1
        for i in range(len(music)):
            if (music[i][1] == 'AP' or music[i][1] == 'SP'):
                continue
            if (not tmp_ans_list[i] == 'G♯-7'):
                continue
            if (shengmu_dic.__contains__(music[i][1])):
                if ((i + 1) < len(music[i][1])):
                    if (not tmp_ans_list[i + 1] == 'G♯-7'):
                        tmp_ans_list[i] = tmp_ans_list[i + 1]
                else:
                    tmp_ans_list[i] = tmp_ans_list[i - 1]
            else:
                if (i - 1 >= 0):
                    if (not tmp_ans_list[i - 1] == 'G♯-7'):
                        tmp_ans_list[i] = tmp_ans_list[i - 1]
                else:
                    tmp_ans_list[i] = tmp_ans_list[i + 1]
            if (tmp_ans_list[i] == 'G♯-7' or tmp_ans_list[i] == '-1'):
                continue
            if (tmp_ans_list[i][1] == '♯'):
                if (tmp_ans_list[i][0] == 'C'):
                    tmp_ans_list[i] = tmp_ans_list[i][0] + '#' + tmp_ans_list[i][2:] + '/' + 'Db' + tmp_ans_list[i][2:]
                if (tmp_ans_list[i][0] == 'D'):
                    tmp_ans_list[i] = tmp_ans_list[i][0] + '#' + tmp_ans_list[i][2:] + '/' + 'Eb' + tmp_ans_list[i][2:]
                if (tmp_ans_list[i][0] == 'F'):
                    tmp_ans_list[i] = tmp_ans_list[i][0] + '#' + tmp_ans_list[i][2:] + '/' + 'Gb' + tmp_ans_list[i][2:]
                if (tmp_ans_list[i][0] == 'G'):
                    tmp_ans_list[i] = tmp_ans_list[i][0] + '#' + tmp_ans_list[i][2:] + '/' + 'Ab' + tmp_ans_list[i][2:]
                if (tmp_ans_list[i][0] == 'A'):
                    tmp_ans_list[i] = tmp_ans_list[i][0] + '#' + tmp_ans_list[i][2:] + '/' + 'Bb' + tmp_ans_list[i][2:]

        for i in range(len(music)):
            if(music[i][1] == 'SP' or music[i][1] == 'AP'):
                music[i][2] = 0
            else:
                music[i][2] = Note_to_Pitch([tmp_ans_list[i]], note_dict)[0]

        audio = Audio_Prep(wav_Path, hyper_paramters.Sound.Sample_Rate)

        if music[0][1] == 'SP' or music[0][1] == 'AP':
            audio = audio[int(music[0][0] * hyper_paramters.Sound.Sample_Rate):]
            music = music[1:]
        if music[0][1] == 'SP' or music[0][1] == 'AP':
            audio = audio[:-int(music[-1][0] * hyper_paramters.Sound.Sample_Rate)]
            music = music[:-1]
        print(f)
        previous_Used = 0
        absolute_Position = 0
        mel_Based = []
        for x in music:
            duration = int(x[0] * hyper_paramters.Sound.Sample_Rate) + previous_Used
            previous_Used = duration % hyper_paramters.Sound.Frame_Shift
            duration = duration // hyper_paramters.Sound.Frame_Shift
            # if x[1] == '<X>':
            mel_Based.append((absolute_Position, duration, x[1], x[2]))
            absolute_Position += duration
        music = mel_Based
        mel1 = extract_mels(audio)[:absolute_Position]
        mel = mel1

        pitch = pitch_calc(
            sig=audio,
            sr=hyper_paramters.Sound.Sample_Rate,
            w_len=hyper_paramters.Sound.Frame_Length,
            w_step=hyper_paramters.Sound.Frame_Shift,
            f0_min=hyper_paramters.Sound.F0_Min,
            f0_max=hyper_paramters.Sound.F0_Max,
            confidence_threshold=hyper_paramters.Sound.Confidence_Threshold,
            gaussian_smoothing_sigma=hyper_paramters.Sound.Gaussian_Smoothing_Sigma
        )[:absolute_Position] / hyper_paramters.Sound.F0_Max

        silence = np.where(np.mean(mel, axis=1) < -3.5, np.zeros_like(np.mean(mel, axis=1)),
                           np.ones_like(np.mean(mel, axis=1)))

        pattern_Index = 0

        # wav = preprocess_wav(wav_Path)
        # embed = encoder.embed_utterance(wav)

        pattern_contain = []
        pattern_address = []

        for start_Index in tqdm(range(len(music)), desc=os.path.basename(wav_Path)):
            for end_Index in range(start_Index + 1, len(music), 5):
                music_Sample = music[start_Index:end_Index]
                sample_Length = music_Sample[-1][0] + music_Sample[-1][1] - music_Sample[0][0]
                if sample_Length < hyper_paramters.Min_Duration:
                    continue
                elif sample_Length > hyper_paramters.Max_Duration:
                    break

                audio_Sample = audio[music_Sample[0][0] * hyper_paramters.Sound.Frame_Shift:(music_Sample[-1][0] +
                                                                                             music_Sample[-1][
                                                                                                 1]) * hyper_paramters.Sound.Frame_Shift]
                mel_Sample = mel[music_Sample[0][0]:music_Sample[-1][0] + music_Sample[-1][1]]
                silence_Sample = silence[music_Sample[0][0]:music_Sample[-1][0] + music_Sample[-1][1]]
                pitch_Sample = pitch[music_Sample[0][0]:music_Sample[-1][0] + music_Sample[-1][1]]

                _, duration_Sample, text_Sample, Note_Sample = zip(*music_Sample)
                # sss = filename.split('_')[0] + '_' + filename.split('_')[1] + '_' + filename.split('_')[2]
                pattern = {
                    'Audio': audio_Sample.astype(np.float32),
                    'Mel': mel_Sample.astype(np.float32),
                    'Silence': silence_Sample.astype(np.uint8),
                    'Pitch': pitch_Sample.astype(np.float32),
                    'Duration': duration_Sample,
                    'Text': text_Sample,
                    'Note': Note_Sample,

                    'Ref_Audio': None,
                    'Ref_Mel': None,
                    'Ref_Silence': None,
                    'Ref_Pitch': None,
                    'Ref_Duration': None,
                    'Ref_Text': None,
                    'Ref_Note': None,

                    'Singer': filename[0],
                    'Dataset': aim,
                }

                pattern_Path = os.path.join(
                    './OpenSingerPickle_20230301',
                    aim,
                    '{:03d}'.format(cnt),
                    aim + '.S_{:03d}.P_{:05d}.pickle'.format(cnt, pattern_Index)
                ).replace('\\', '/')
                pattern_contain.append(pattern)
                pattern_address.append(pattern_Path)
                # print(pattern_Path)
                # os.makedirs(os.path.dirname(pattern_Path), exist_ok=True)
                # pickle.dump(
                #     pattern,
                #     open(pattern_Path, 'wb'),
                #     protocol=4
                # )
                pattern_Index += 1

                min_Duration, max_Duration = min(sample_Length, min_Duration), max(sample_Length, max_Duration)

        for i in range(pattern_Index):
            pos = (i + (int)(pattern_Index / 2)) % pattern_Index
            pattern_contain[i]['Ref_Audio'] = pattern_contain[pos]['Audio']
            pattern_contain[i]['Ref_Mel'] = pattern_contain[pos]['Mel']
            pattern_contain[i]['Ref_Silence'] = pattern_contain[pos]['Silence']
            pattern_contain[i]['Ref_Pitch'] = pattern_contain[pos]['Silence']
            pattern_contain[i]['Ref_Duration'] = pattern_contain[pos]['Duration']
            pattern_contain[i]['Ref_Text'] = pattern_contain[pos]['Text']
            pattern_contain[i]['Ref_Note'] = pattern_contain[pos]['Note']
            pattern_Path = pattern_address[i]
            print(pattern_Path)
            os.makedirs(os.path.dirname(pattern_Path), exist_ok=True)
            pickle.dump(
                pattern_contain[i],
                open(pattern_Path, 'wb'),
                protocol=4
            )

def Metadata_Generate(
    hyper_parameters: Namespace,
    eval: bool= False
    ):
    pattern_Path = hyper_parameters.Train.Eval_Pattern.Path if eval else hyper_parameters.Train.Train_Pattern.Path
    metadata_File = hyper_parameters.Train.Eval_Pattern.Metadata_File if eval else hyper_parameters.Train.Train_Pattern.Metadata_File

    # pattern_Path = '/home/share/chengxuanang/hifisinger-chinese/segment/NAMS' + ('1' if eval else '')
    pattern_Path = 'D:/Download/segment/NAMS' + ('1' if eval else '')
    # pattern_Path = '/data/gongjunhao/hifisinger-chinese/segment/NAMS' + ('1' if eval else '')
    new_Metadata_Dict = {
        'Spectrogram_Dim': hyper_parameters.Sound.Spectrogram_Dim,
        'Mel_Dim': hyper_parameters.Sound.Mel_Dim,
        'Frame_Shift': hyper_parameters.Sound.Frame_Shift,
        'Frame_Length': hyper_parameters.Sound.Frame_Length,
        'Sample_Rate': hyper_parameters.Sound.Sample_Rate,
        'Max_Abs_Mel': hyper_parameters.Sound.Max_Abs_Mel,
        'Mel_F_Min': hyper_parameters.Sound.Mel_F_Min,
        'Mel_F_Max': hyper_parameters.Sound.Mel_F_Max,
        'File_List': [],
        'Audio_Length_Dict': {},
        'Mel_Length_Dict': {},
        'Music_Length_Dict': {},
        }

    files_TQDM = tqdm(
        total= sum([len(files) for root, _, files in os.walk(pattern_Path)]),
        desc= 'Eval_Pattern' if eval else 'Train_Pattern'
        )

    for root, _, files in os.walk(pattern_Path):
        for file in files:
            print(file)
            with open(os.path.join(root, file).replace("\\", "/"), "rb") as f:
                pattern_Dict = pickle.load(f)
            file = os.path.join(root, file).replace("\\", "/").replace(pattern_Path, '').lstrip('/')
            try:
                if not all([
                    key in pattern_Dict.keys()
                    for key in ('Audio', 'Mel', 'Silence', 'Pitch', 'Duration', 'Text', 'Note', 'Singer', 'Dataset')
                    ]):
                    continue
                new_Metadata_Dict['Audio_Length_Dict'][file] = pattern_Dict['Audio'].shape[0]
                new_Metadata_Dict['Mel_Length_Dict'][file] = pattern_Dict['Mel'].shape[0]
                new_Metadata_Dict['Music_Length_Dict'][file] = len(pattern_Dict['Duration'])
                new_Metadata_Dict['File_List'].append(file)
            except:
                print('File \'{}\' is not correct pattern file. This file is ignored.'.format(file))
            files_TQDM.update(1)

    with open(os.path.join(pattern_Path, metadata_File.upper()).replace("\\", "/"), 'wb') as f:
        pickle.dump(new_Metadata_Dict, f, protocol= 4)

    print('Metadata generate done.')

def Metadata_Generate_OpenSinger(
    hyper_parameters: Namespace,
    aim: str,
    eval: bool= False
    ):
    # pattern_Path = hyper_parameters.Train.Eval_Pattern.Path if eval else hyper_parameters.Train.Train_Pattern.Path
    # pattern_Path = '/home/share/chengxuanang/hifisinger-chinese/segment/NAMS' + ('1' if eval else '')
    # pattern_Path = 'D:/Download/OpenSingerPickle/NAMS' + ('1' if eval else '')
    # pattern_Path = '/data/gongjunhao/hifisinger-chinese/segment/NAMS' + ('1' if eval else '')
    # pattern_Path = 'D:/Download/OpenSingerPickle/' + aim
    # pattern_Path = './OpenSingerPickle/' + aim
    pattern_Path = './OpenSingerPickle_20230301/' + aim
    metadata_File = hyper_parameters.Train.Eval_Pattern.Metadata_File if eval else hyper_parameters.Train.Train_Pattern.Metadata_File

    cc = 0
    new_Metadata_Dict = {
        'Spectrogram_Dim': hyper_parameters.Sound.Spectrogram_Dim,
        'Mel_Dim': hyper_parameters.Sound.Mel_Dim,
        'Frame_Shift': hyper_parameters.Sound.Frame_Shift,
        'Frame_Length': hyper_parameters.Sound.Frame_Length,
        'Sample_Rate': hyper_parameters.Sound.Sample_Rate,
        'Max_Abs_Mel': hyper_parameters.Sound.Max_Abs_Mel,
        'Mel_F_Min': hyper_parameters.Sound.Mel_F_Min,
        'Mel_F_Max': hyper_parameters.Sound.Mel_F_Max,
        'File_List': [],
        'Audio_Length_Dict': {},
        'Mel_Length_Dict': {},
        'Music_Length_Dict': {},
        }

    files_TQDM = tqdm(
        total= sum([len(files) for root, _, files in os.walk(pattern_Path)]),
        desc= 'Eval_Pattern' if eval else 'Train_Pattern'
        )

    for root, _, files in os.walk(pattern_Path):
        # cc += 1
        # if (cc >= 11):
        #     break
        for file in files:
            # tmp_str_list = file.split('_')
            # if(len(tmp_str_list) > 1 and tmp_str_list[0] == 'NAMS.S' and int((tmp_str_list[1].split('.'))[0]) >= 10000):
            #     print(file)
            print(file)
            with open(os.path.join(root, file).replace("\\", "/"), "rb") as f:
                pattern_Dict = pickle.load(f)
            file = os.path.join(root, file).replace("\\", "/").replace(pattern_Path, '').lstrip('/')
            try:
                if not all([
                    key in pattern_Dict.keys()
                    for key in ('Audio', 'Mel', 'Silence', 'Pitch', 'Duration', 'Text', 'Note', 'Singer', 'Dataset')
                    ]):
                    continue
                new_Metadata_Dict['Audio_Length_Dict'][file] = pattern_Dict['Audio'].shape[0]
                new_Metadata_Dict['Mel_Length_Dict'][file] = pattern_Dict['Mel'].shape[0]
                new_Metadata_Dict['Music_Length_Dict'][file] = len(pattern_Dict['Duration'])
                new_Metadata_Dict['File_List'].append(file)
            except:
                print('File \'{}\' is not correct pattern file. This file is ignored.'.format(file))
            files_TQDM.update(1)

    with open(os.path.join(pattern_Path, metadata_File.upper()).replace("\\", "/"), 'wb') as f:
        pickle.dump(new_Metadata_Dict, f, protocol= 4)

    print('Metadata generate done.')

def Metadata_Generate_OpenSinger_Man(
    hyper_parameters: Namespace,
    aim: str,
    eval: bool= False
    ):
    # pattern_Path = hyper_parameters.Train.Eval_Pattern.Path if eval else hyper_parameters.Train.Train_Pattern.Path
    # pattern_Path = '/home/share/chengxuanang/hifisinger-chinese/segment/NAMS' + ('1' if eval else '')
    # pattern_Path = 'D:/Download/OpenSingerPickle/NAMS' + ('1' if eval else '')
    # pattern_Path = '/data/gongjunhao/hifisinger-chinese/segment/NAMS' + ('1' if eval else '')
    # pattern_Path = 'D:/Download/OpenSingerPickle/' + aim
    pattern_Path = './OpenSingerPickle/' + aim
    pattern_Path = './OpenSingerPickle_20230301/' + aim
    metadata_File = hyper_parameters.Train.Eval_Pattern.Metadata_File if eval else hyper_parameters.Train.Train_Pattern.Metadata_File

    cc = 0
    new_Metadata_Dict = {
        'Spectrogram_Dim': hyper_parameters.Sound.Spectrogram_Dim,
        'Mel_Dim': hyper_parameters.Sound.Mel_Dim,
        'Frame_Shift': hyper_parameters.Sound.Frame_Shift,
        'Frame_Length': hyper_parameters.Sound.Frame_Length,
        'Sample_Rate': hyper_parameters.Sound.Sample_Rate,
        'Max_Abs_Mel': hyper_parameters.Sound.Max_Abs_Mel,
        'Mel_F_Min': hyper_parameters.Sound.Mel_F_Min,
        'Mel_F_Max': hyper_parameters.Sound.Mel_F_Max,
        'File_List': [],
        'Audio_Length_Dict': {},
        'Mel_Length_Dict': {},
        'Music_Length_Dict': {},
        }

    files_TQDM = tqdm(
        total= sum([len(files) for root, _, files in os.walk(pattern_Path)]),
        desc= 'Eval_Pattern' if eval else 'Train_Pattern'
        )

    for root, _, files in os.walk(pattern_Path):
        # cc += 1
        # if (cc >= 11):
        #     break
        for file in files:
            # tmp_str_list = file.split('_')
            # if(len(tmp_str_list) > 1 and tmp_str_list[0] == 'NAMS.S' and int((tmp_str_list[1].split('.'))[0]) >= 10000):
            #     print(file)
            # print(file)
            with open(os.path.join(root, file).replace("\\", "/"), "rb") as f:
                pattern_Dict = pickle.load(f)
            file = os.path.join(root, file).replace("\\", "/").replace(pattern_Path, '').lstrip('/')
            try:
                if not all([
                    key in pattern_Dict.keys()
                    for key in ('Audio', 'Mel', 'Silence', 'Pitch', 'Duration', 'Text', 'Note', 'Singer', 'Dataset')
                    ]):
                    continue
                p_str = pattern_Dict['Singer'].split('_')[0]
                if (p_str == 'WomanRaw'):
                    continue
                new_Metadata_Dict['Audio_Length_Dict'][file] = pattern_Dict['Audio'].shape[0]
                new_Metadata_Dict['Mel_Length_Dict'][file] = pattern_Dict['Mel'].shape[0]
                new_Metadata_Dict['Music_Length_Dict'][file] = len(pattern_Dict['Duration'])
                new_Metadata_Dict['File_List'].append(file)
            except Exception as e:
                # print(e)
                print('File \'{}\' is not correct pattern file. This file is ignored.'.format(file))
            files_TQDM.update(1)

    with open(os.path.join(pattern_Path, metadata_File.upper()).replace("\\", "/"), 'wb') as f:
        pickle.dump(new_Metadata_Dict, f, protocol= 4)

    print('Metadata generate done.')


if __name__ == "__main__":

    hp = Recursive_Parse(yaml.load(
        open('Hyper_Parameters.yaml', encoding='utf-8'),
        Loader=yaml.Loader
        ))

    # Token_Dict_Generate(hyper_parameters= hp)
    # Pattern_Generate(hyper_paramters= hp, aim= 'train')
    # Pattern_Generate(hyper_paramters=hp, aim='eval')
    # Metadata_Generate(hp, False)
    # Metadata_Generate(hp, True)
    # Pattern_Test(hyper_paramters= hp, aim= 'eval')
    # Pattern_Generate_OpenSinger(hyper_paramters= hp, aim= 'train')
    # Metadata_Generate_OpenSinger(hp, False)
    # Pattern_Generate_OpenSinger(hyper_paramters=hp, aim='NAMS')
    # Metadata_Generate_OpenSinger(hp, aim='NAMS')
    # Pattern_Generate_OpenSinger(hyper_paramters=hp, aim='NAMS1')
    # Metadata_Generate_OpenSinger(hp, aim='NAMS1')
    # Pattern_Generate_OpenSinger(hyper_paramters=hp, aim='NAMS2')
    # Metadata_Generate_OpenSinger_Man(hp, aim='NAMS2')

    # Pattern_Generate_OpenSinger(hyper_paramters=hp, aim='NAMS')
    # Metadata_Generate_OpenSinger_Man(hp, aim='NAMS')
    # Pattern_Generate_OpenSinger(hyper_paramters=hp, aim='NAMS1')
    # Metadata_Generate_OpenSinger_Man(hp, aim='NAMS1')
    # Pattern_Generate_OpenSinger(hyper_paramters=hp, aim='NAMS2')
    # Metadata_Generate_OpenSinger_Man(hp, aim='NAMS2')
    # Pattern_Generate_OpenSinger_20230301(hyper_paramters=hp, aim='NAMS')
    # Metadata_Generate_OpenSinger(hp, aim='NAMS')
    # Pattern_Generate_OpenSinger_20230301(hyper_paramters=hp, aim='NAMS1')
    # Metadata_Generate_OpenSinger(hp, aim='NAMS1')
    # Pattern_Generate_OpenSinger_20230301(hyper_paramters=hp, aim='NAMS_WOMAN')
    Metadata_Generate_OpenSinger(hp, aim='NAMS_WOMAN')
    # Pattern_Generate_OpenSinger_20230301(hyper_paramters=hp, aim='NAMS_WOMAN_EVAL')
    # Metadata_Generate_OpenSinger(hp, aim='NAMS_WOMAN_EVAL')
    # Pattern_Generate_OpenSinger_20230301(hyper_paramters=hp, aim='NAMS_MAN')
    # Metadata_Generate_OpenSinger(hp, aim='NAMS_MAN')
    # Pattern_Generate_OpenSinger_20230301(hyper_paramters=hp, aim='NAMS_MAN_EVAL')
    # Metadata_Generate_OpenSinger(hp, aim='NAMS_MAN_EVAL')
    # python Pattern_Generator.py -hp Hyper_Parameters.yaml -d E:/Kor_Music_Confidential

    # nohup /home/chengxuanang/miniconda3/envs/ActionRA/bin/python3 Pattern_Generator.py &