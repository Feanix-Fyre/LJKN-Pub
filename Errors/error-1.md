
Person: 
- What's happening here

(STT) PS C:\AI\STT\User\whisper-diarization> python diarize.py -a .\PTT-20220420-WA0000.opus
[NeMo W 2023-05-12 11:44:25 optimizers:54] Apex was not found. Using the lamb or fused_adam optimizer will error out.
[NeMo W 2023-05-12 11:44:26 experimental:27] Module <class 'nemo.collections.asr.modules.audio_modules.SpectrogramToMultichannelFeatures'> is experimental, not ready for production and is not fully supported. Use at your own risk.
Selected model is a bag of 1 models. You will see that many progress bars per track.
Separated tracks will be stored in C:\AI\STT\User\whisper-diarization\temp_outputs\htdemucs
Separating track PTT-20220420-WA0000.opus
100%|██████████████████████████████████████████████| 17.549999999999997/17.549999999999997 [00:04<00:00,  3.80seconds/s]
[NeMo I 2023-05-12 11:44:57 msdd_models:1092] Loading pretrained diar_msdd_telephonic model from NGC
[NeMo I 2023-05-12 11:44:57 cloud:58] Found existing object C:\Users\kijan\.cache\torch\NeMo\NeMo_1.17.0\diar_msdd_telephonic\3c3697a0a46f945574fa407149975a13\diar_msdd_telephonic.nemo.
[NeMo I 2023-05-12 11:44:57 cloud:64] Re-using file from: C:\Users\kijan\.cache\torch\NeMo\NeMo_1.17.0\diar_msdd_telephonic\3c3697a0a46f945574fa407149975a13\diar_msdd_telephonic.nemo 
[NeMo I 2023-05-12 11:44:57 common:913] Instantiating model from pre-trained checkpoint
[NeMo W 2023-05-12 11:44:58 modelPT:161] If you intend to do training or fine-tuning, please call the ModelPT.setup_training_data() method and provide a valid configuration file to setup the train data loader.
    Train config :
    manifest_filepath: null
    emb_dir: null
    sample_rate: 16000
    num_spks: 2
    soft_label_thres: 0.5
    labels: null
    batch_size: 15
    emb_batch_size: 0
    shuffle: true

[NeMo W 2023-05-12 11:44:58 modelPT:168] If you intend to do validation, please call the ModelPT.setup_validation_data() or ModelPT.setup_multiple_validation_data() method and provide a valid configuration file to setup the validation data loader(s).
    Validation config :
    manifest_filepath: null
    emb_dir: null
    sample_rate: 16000
    num_spks: 2
    soft_label_thres: 0.5
    labels: null
    batch_size: 15
    emb_batch_size: 0
    shuffle: false

[NeMo W 2023-05-12 11:44:58 modelPT:174] Please call the ModelPT.setup_test_data() or ModelPT.setup_multiple_test_data() method and provide a valid configuration file to setup the test data loader(s).
    Test config :
    manifest_filepath: null
    emb_dir: null
    sample_rate: 16000
    num_spks: 2
    soft_label_thres: 0.5
    labels: null
    batch_size: 15
    emb_batch_size: 0
    shuffle: false
    seq_eval_mode: false

[NeMo I 2023-05-12 11:44:58 features:287] PADDING: 16
[NeMo I 2023-05-12 11:44:58 features:287] PADDING: 16
[NeMo I 2023-05-12 11:44:59 save_restore_connector:247] Model EncDecDiarLabelModel was successfully restored from C:\Users\kijan\.cache\torch\NeMo\NeMo_1.17.0\diar_msdd_telephonic\3c3697a0a46f945574fa407149975a13\diar_msdd_telephonic.nemo.
[NeMo I 2023-05-12 11:44:59 features:287] PADDING: 16
[NeMo I 2023-05-12 11:44:59 clustering_diarizer:127] Loading pretrained vad_multilingual_marblenet model from NGC
[NeMo I 2023-05-12 11:44:59 cloud:58] Found existing object C:\Users\kijan\.cache\torch\NeMo\NeMo_1.17.0\vad_multilingual_marblenet\670f425c7f186060b7a7268ba6dfacb2\vad_multilingual_marblenet.nemo.
[NeMo I 2023-05-12 11:44:59 cloud:64] Re-using file from: C:\Users\kijan\.cache\torch\NeMo\NeMo_1.17.0\vad_multilingual_marblenet\670f425c7f186060b7a7268ba6dfacb2\vad_multilingual_marblenet.nemo
[NeMo I 2023-05-12 11:44:59 common:913] Instantiating model from pre-trained checkpoint
[NeMo W 2023-05-12 11:44:59 modelPT:161] If you intend to do training or fine-tuning, please call the ModelPT.setup_training_data() method and provide a valid configuration file to setup the train data loader.
    Train config :
    manifest_filepath: /manifests/ami_train_0.63.json,/manifests/freesound_background_train.json,/manifests/freesound_laughter_train.json,/manifests/fisher_2004_background.json,/manifests/fisher_2004_speech_sampled.json,/manifests/google_train_manifest.json,/manifests/icsi_all_0.63.json,/manifests/musan_freesound_train.json,/manifests/musan_music_train.json,/manifests/musan_soundbible_train.json,/manifests/mandarin_train_sample.json,/manifests/german_train_sample.json,/manifests/spanish_train_sample.json,/manifests/french_train_sample.json,/manifests/russian_train_sample.json
    sample_rate: 16000
    labels:
    - background
    - speech
    batch_size: 256
    shuffle: true
    is_tarred: false
    tarred_audio_filepaths: null
    tarred_shard_strategy: scatter
    augmentor:
      shift:
        prob: 0.5
        min_shift_ms: -10.0
        max_shift_ms: 10.0
      white_noise:
        prob: 0.5
        min_level: -90
        max_level: -46
        norm: true
      noise:
        prob: 0.5
        manifest_path: /manifests/noise_0_1_musan_fs.json
        min_snr_db: 0
        max_snr_db: 30
        max_gain_db: 300.0
        norm: true
      gain:
        prob: 0.5
        min_gain_dbfs: -10.0
        max_gain_dbfs: 10.0
        norm: true
    num_workers: 16
    pin_memory: true

[NeMo W 2023-05-12 11:44:59 modelPT:168] If you intend to do validation, please call the ModelPT.setup_validation_data() or ModelPT.setup_multiple_validation_data() method and provide a valid configuration file to setup the validation data loader(s).
    Validation config :
    manifest_filepath: /manifests/ami_dev_0.63.json,/manifests/freesound_background_dev.json,/manifests/freesound_laughter_dev.json,/manifests/ch120_moved_0.63.json,/manifests/fisher_2005_500_speech_sampled.json,/manifests/google_dev_manifest.json,/manifests/musan_music_dev.json,/manifests/mandarin_dev.json,/manifests/german_dev.json,/manifests/spanish_dev.json,/manifests/french_dev.json,/manifests/russian_dev.json
    sample_rate: 16000
    labels:
    - background
    - speech
    batch_size: 256
    shuffle: false
    val_loss_idx: 0
    num_workers: 16
    pin_memory: true

[NeMo W 2023-05-12 11:44:59 modelPT:174] Please call the ModelPT.setup_test_data() or ModelPT.setup_multiple_test_data() method and provide a valid configuration file to setup the test data loader(s).
    Test config :
    manifest_filepath: null
    sample_rate: 16000
    labels:
    - background
    - speech
    batch_size: 128
    shuffle: false
    test_loss_idx: 0

[NeMo I 2023-05-12 11:44:59 features:287] PADDING: 16
[NeMo I 2023-05-12 11:44:59 save_restore_connector:247] Model EncDecClassificationModel was successfully restored from C:\Users\kijan\.cache\torch\NeMo\NeMo_1.17.0\vad_multilingual_marblenet\670f425c7f186060b7a7268ba6dfacb2\vad_multilingual_marblenet.nemo.
[NeMo I 2023-05-12 11:44:59 msdd_models:864] Multiscale Weights: [1, 1, 1, 1, 1]
[NeMo I 2023-05-12 11:44:59 msdd_models:865] Clustering Parameters: {
        "oracle_num_speakers": false,
        "max_num_speakers": 8,
        "enhanced_count_thres": 80,
        "max_rp_threshold": 0.25,
        "sparse_search_volume": 30,
        "maj_vote_spk_count": false
    }
[NeMo W 2023-05-12 11:44:59 clustering_diarizer:411] Deleting previous clustering diarizer outputs.
[NeMo I 2023-05-12 11:44:59 speaker_utils:93] Number of files to diarize: 1
[NeMo I 2023-05-12 11:44:59 clustering_diarizer:309] Split long audio file to avoid CUDA memory issue
splitting manifest: 100%|██████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 1/1 [00:00<00:00, 110.64it/s] 
[NeMo I 2023-05-12 11:44:59 vad_utils:101] The prepared manifest file exists. Overwriting!
[NeMo I 2023-05-12 11:44:59 classification_models:263] Perform streaming frame-level VAD
[NeMo I 2023-05-12 11:44:59 collections:298] Filtered duration for loading collection is 0.000000.
[NeMo I 2023-05-12 11:44:59 collections:301] Dataset loaded with 1 items, total duration of  0.00 hours.
[NeMo I 2023-05-12 11:44:59 collections:303] # 1 files loaded accounting to # 1 labels
vad:   0%|                                                                                                                                                      | 0/1 [00:00<?, ?it/s] 
Traceback (most recent call last):
  File "C:\AI\STT\User\whisper-diarization\diarize.py", line 105, in <module>
    msdd_model.diarize()
  File "C:\AI\STT\lib\site-packages\torch\utils\_contextlib.py", line 115, in decorate_context
    return func(*args, **kwargs)
  File "C:\AI\STT\lib\site-packages\nemo\collections\asr\models\msdd_models.py", line 1180, in diarize
    self.clustering_embedding.prepare_cluster_embs_infer()
  File "C:\AI\STT\lib\site-packages\nemo\collections\asr\models\msdd_models.py", line 699, in prepare_cluster_embs_infer
    self.emb_sess_test_dict, self.emb_seq_test, self.clus_test_label_dict, _ = self.run_clustering_diarizer(
  File "C:\AI\STT\lib\site-packages\nemo\collections\asr\models\msdd_models.py", line 866, in run_clustering_diarizer
    scores = self.clus_diar_model.diarize(batch_size=self.cfg_diar_infer.batch_size)
  File "C:\AI\STT\lib\site-packages\nemo\collections\asr\models\clustering_diarizer.py", line 437, in diarize
    self._perform_speech_activity_detection()
  File "C:\AI\STT\lib\site-packages\nemo\collections\asr\models\clustering_diarizer.py", line 325, in _perform_speech_activity_detection
    self._run_vad(manifest_vad_input)
  File "C:\AI\STT\lib\site-packages\nemo\collections\asr\models\clustering_diarizer.py", line 218, in _run_vad
    for i, test_batch in enumerate(
  File "C:\AI\STT\lib\site-packages\tqdm\std.py", line 1178, in __iter__
    for obj in iterable:
  File "C:\AI\STT\lib\site-packages\torch\utils\data\dataloader.py", line 441, in __iter__
    return self._get_iterator()
  File "C:\AI\STT\lib\site-packages\torch\utils\data\dataloader.py", line 388, in _get_iterator
    return _MultiProcessingDataLoaderIter(self)
  File "C:\AI\STT\lib\site-packages\torch\utils\data\dataloader.py", line 1042, in __init__
    w.start()
  File "C:\Users\kijan\AppData\Local\Programs\Python\Python310\lib\multiprocessing\process.py", line 121, in start
    self._popen = self._Popen(self)
  File "C:\Users\kijan\AppData\Local\Programs\Python\Python310\lib\multiprocessing\context.py", line 224, in _Popen
    return _default_context.get_context().Process._Popen(process_obj)
  File "C:\Users\kijan\AppData\Local\Programs\Python\Python310\lib\multiprocessing\context.py", line 336, in _Popen
    return Popen(process_obj)
  File "C:\Users\kijan\AppData\Local\Programs\Python\Python310\lib\multiprocessing\popen_spawn_win32.py", line 93, in __init__
    reduction.dump(process_obj, to_child)
  File "C:\Users\kijan\AppData\Local\Programs\Python\Python310\lib\multiprocessing\reduction.py", line 60, in dump
    ForkingPickler(file, protocol).dump(obj)
_pickle.PicklingError: Can't pickle <class 'nemo.collections.common.parts.preprocessing.collections.SpeechLabelEntity'>: attribute lookup SpeechLabelEntity on nemo.collections.common.parts.preprocessing.collections failed
(STT) PS C:\AI\STT\User\whisper-diarization> [NeMo W 2023-05-12 11:45:04 optimizers:54] Apex was not found. Using the lamb or fused_adam optimizer will error out.
[NeMo W 2023-05-12 11:45:05 experimental:27] Module <class 'nemo.collections.asr.modules.audio_modules.SpectrogramToMultichannelFeatures'> is experimental, not ready for production and is not fully supported. Use at your own risk.
Selected model is a bag of 1 models. You will see that many progress bars per track.
Separated tracks will be stored in C:\AI\STT\User\whisper-diarization\temp_outputs\htdemucs
Separating track PTT-20220420-WA0000.opus
100%|██████████████████████████████████████████████| 17.549999999999997/17.549999999999997 [00:00<00:00, 17.83seconds/s]
[NeMo I 2023-05-12 11:45:26 msdd_models:1092] Loading pretrained diar_msdd_telephonic model from NGC
[NeMo I 2023-05-12 11:45:26 cloud:58] Found existing object C:\Users\kijan\.cache\torch\NeMo\NeMo_1.17.0\diar_msdd_telephonic\3c3697a0a46f945574fa407149975a13\diar_msdd_telephonic.nemo.
[NeMo I 2023-05-12 11:45:26 cloud:64] Re-using file from: C:\Users\kijan\.cache\torch\NeMo\NeMo_1.17.0\diar_msdd_telephonic\3c3697a0a46f945574fa407149975a13\diar_msdd_telephonic.nemo 
[NeMo I 2023-05-12 11:45:26 common:913] Instantiating model from pre-trained checkpoint
[NeMo W 2023-05-12 11:45:27 modelPT:161] If you intend to do training or fine-tuning, please call the ModelPT.setup_training_data() method and provide a valid configuration file to setup the train data loader.
    Train config :
    manifest_filepath: null
    emb_dir: null
    sample_rate: 16000
    num_spks: 2
    soft_label_thres: 0.5
    labels: null
    batch_size: 15
    emb_batch_size: 0
    shuffle: true

[NeMo W 2023-05-12 11:45:27 modelPT:168] If you intend to do validation, please call the ModelPT.setup_validation_data() or ModelPT.setup_multiple_validation_data() method and provide a valid configuration file to setup the validation data loader(s).
    Validation config :
    manifest_filepath: null
    emb_dir: null
    sample_rate: 16000
    num_spks: 2
    soft_label_thres: 0.5
    labels: null
    batch_size: 15
    emb_batch_size: 0
    shuffle: false

[NeMo W 2023-05-12 11:45:27 modelPT:174] Please call the ModelPT.setup_test_data() or ModelPT.setup_multiple_test_data() method and provide a valid configuration file to setup the test data loader(s).
    Test config :
    manifest_filepath: null
    emb_dir: null
    sample_rate: 16000
    num_spks: 2
    soft_label_thres: 0.5
    labels: null
    batch_size: 15
    emb_batch_size: 0
    shuffle: false
    seq_eval_mode: false

[NeMo I 2023-05-12 11:45:27 features:287] PADDING: 16
[NeMo I 2023-05-12 11:45:27 features:287] PADDING: 16
[NeMo I 2023-05-12 11:45:28 save_restore_connector:247] Model EncDecDiarLabelModel was successfully restored from C:\Users\kijan\.cache\torch\NeMo\NeMo_1.17.0\diar_msdd_telephonic\3c3697a0a46f945574fa407149975a13\diar_msdd_telephonic.nemo.
[NeMo I 2023-05-12 11:45:28 features:287] PADDING: 16
[NeMo I 2023-05-12 11:45:28 clustering_diarizer:127] Loading pretrained vad_multilingual_marblenet model from NGC
[NeMo I 2023-05-12 11:45:28 cloud:58] Found existing object C:\Users\kijan\.cache\torch\NeMo\NeMo_1.17.0\vad_multilingual_marblenet\670f425c7f186060b7a7268ba6dfacb2\vad_multilingual_marblenet.nemo.
[NeMo I 2023-05-12 11:45:28 cloud:64] Re-using file from: C:\Users\kijan\.cache\torch\NeMo\NeMo_1.17.0\vad_multilingual_marblenet\670f425c7f186060b7a7268ba6dfacb2\vad_multilingual_marblenet.nemo
[NeMo I 2023-05-12 11:45:28 common:913] Instantiating model from pre-trained checkpoint
[NeMo W 2023-05-12 11:45:28 modelPT:161] If you intend to do training or fine-tuning, please call the ModelPT.setup_training_data() method and provide a valid configuration file to setup the train data loader.
    Train config :
    manifest_filepath: /manifests/ami_train_0.63.json,/manifests/freesound_background_train.json,/manifests/freesound_laughter_train.json,/manifests/fisher_2004_background.json,/manifests/fisher_2004_speech_sampled.json,/manifests/google_train_manifest.json,/manifests/icsi_all_0.63.json,/manifests/musan_freesound_train.json,/manifests/musan_music_train.json,/manifests/musan_soundbible_train.json,/manifests/mandarin_train_sample.json,/manifests/german_train_sample.json,/manifests/spanish_train_sample.json,/manifests/french_train_sample.json,/manifests/russian_train_sample.json
    sample_rate: 16000
    labels:
    - background
    - speech
    batch_size: 256
    shuffle: true
    is_tarred: false
    tarred_audio_filepaths: null
    tarred_shard_strategy: scatter
    augmentor:
      shift:
        prob: 0.5
        min_shift_ms: -10.0
        max_shift_ms: 10.0
      white_noise:
        prob: 0.5
        min_level: -90
        max_level: -46
        norm: true
      noise:
        prob: 0.5
        manifest_path: /manifests/noise_0_1_musan_fs.json
        min_snr_db: 0
        max_snr_db: 30
        max_gain_db: 300.0
        norm: true
      gain:
        prob: 0.5
        min_gain_dbfs: -10.0
        max_gain_dbfs: 10.0
        norm: true
    num_workers: 16
    pin_memory: true

[NeMo W 2023-05-12 11:45:28 modelPT:168] If you intend to do validation, please call the ModelPT.setup_validation_data() or ModelPT.setup_multiple_validation_data() method and provide a valid configuration file to setup the validation data loader(s).
    Validation config :
    manifest_filepath: /manifests/ami_dev_0.63.json,/manifests/freesound_background_dev.json,/manifests/freesound_laughter_dev.json,/manifests/ch120_moved_0.63.json,/manifests/fisher_2005_500_speech_sampled.json,/manifests/google_dev_manifest.json,/manifests/musan_music_dev.json,/manifests/mandarin_dev.json,/manifests/german_dev.json,/manifests/spanish_dev.json,/manifests/french_dev.json,/manifests/russian_dev.json
    sample_rate: 16000
    labels:
    - background
    - speech
    batch_size: 256
    shuffle: false
    val_loss_idx: 0
    num_workers: 16
    pin_memory: true

[NeMo W 2023-05-12 11:45:28 modelPT:174] Please call the ModelPT.setup_test_data() or ModelPT.setup_multiple_test_data() method and provide a valid configuration file to setup the test data loader(s).
    Test config :
    manifest_filepath: null
    sample_rate: 16000
    labels:
    - background
    - speech
    batch_size: 128
    shuffle: false
    test_loss_idx: 0

[NeMo I 2023-05-12 11:45:28 features:287] PADDING: 16
[NeMo I 2023-05-12 11:45:28 save_restore_connector:247] Model EncDecClassificationModel was successfully restored from C:\Users\kijan\.cache\torch\NeMo\NeMo_1.17.0\vad_multilingual_marblenet\670f425c7f186060b7a7268ba6dfacb2\vad_multilingual_marblenet.nemo.
[NeMo I 2023-05-12 11:45:28 msdd_models:864] Multiscale Weights: [1, 1, 1, 1, 1]
[NeMo I 2023-05-12 11:45:28 msdd_models:865] Clustering Parameters: {
        "oracle_num_speakers": false,
        "max_num_speakers": 8,
        "enhanced_count_thres": 80,
        "max_rp_threshold": 0.25,
        "sparse_search_volume": 30,
        "maj_vote_spk_count": false
    }
[NeMo W 2023-05-12 11:45:28 clustering_diarizer:411] Deleting previous clustering diarizer outputs.
[NeMo I 2023-05-12 11:45:28 speaker_utils:93] Number of files to diarize: 1
[NeMo I 2023-05-12 11:45:28 clustering_diarizer:309] Split long audio file to avoid CUDA memory issue
splitting manifest: 100%|██████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 1/1 [00:00<00:00, 120.58it/s] 
[NeMo I 2023-05-12 11:45:28 vad_utils:101] The prepared manifest file exists. Overwriting!
[NeMo I 2023-05-12 11:45:28 classification_models:263] Perform streaming frame-level VAD
[NeMo I 2023-05-12 11:45:28 collections:298] Filtered duration for loading collection is 0.000000.
[NeMo I 2023-05-12 11:45:28 collections:301] Dataset loaded with 1 items, total duration of  0.00 hours.
[NeMo I 2023-05-12 11:45:28 collections:303] # 1 files loaded accounting to # 1 labels
vad:   0%|                                                                                                                                                      | 0/1 [00:00<?, ?it/s]
Traceback (most recent call last):
  File "<string>", line 1, in <module>
  File "C:\Users\kijan\AppData\Local\Programs\Python\Python310\lib\multiprocessing\spawn.py", line 116, in spawn_main
    exitcode = _main(fd, parent_sentinel)
  File "C:\Users\kijan\AppData\Local\Programs\Python\Python310\lib\multiprocessing\spawn.py", line 125, in _main
    prepare(preparation_data)
  File "C:\Users\kijan\AppData\Local\Programs\Python\Python310\lib\multiprocessing\spawn.py", line 236, in prepare
    _fixup_main_from_path(data['init_main_from_path'])
  File "C:\Users\kijan\AppData\Local\Programs\Python\Python310\lib\multiprocessing\spawn.py", line 287, in _fixup_main_from_path
    main_content = runpy.run_path(main_path,
  File "C:\Users\kijan\AppData\Local\Programs\Python\Python310\lib\runpy.py", line 289, in run_path
    return _run_module_code(code, init_globals, run_name,
  File "C:\Users\kijan\AppData\Local\Programs\Python\Python310\lib\runpy.py", line 96, in _run_module_code
    _run_code(code, mod_globals, init_globals,
  File "C:\Users\kijan\AppData\Local\Programs\Python\Python310\lib\runpy.py", line 86, in _run_code
    exec(code, run_globals)
  File "C:\AI\STT\User\whisper-diarization\diarize.py", line 105, in <module>
    msdd_model.diarize()
  File "C:\AI\STT\lib\site-packages\torch\utils\_contextlib.py", line 115, in decorate_context
    return func(*args, **kwargs)
  File "C:\AI\STT\lib\site-packages\nemo\collections\asr\models\msdd_models.py", line 1180, in diarize
    self.clustering_embedding.prepare_cluster_embs_infer()
  File "C:\AI\STT\lib\site-packages\nemo\collections\asr\models\msdd_models.py", line 699, in prepare_cluster_embs_infer
    self.emb_sess_test_dict, self.emb_seq_test, self.clus_test_label_dict, _ = self.run_clustering_diarizer(
  File "C:\AI\STT\lib\site-packages\nemo\collections\asr\models\msdd_models.py", line 866, in run_clustering_diarizer
    scores = self.clus_diar_model.diarize(batch_size=self.cfg_diar_infer.batch_size)
  File "C:\AI\STT\lib\site-packages\nemo\collections\asr\models\clustering_diarizer.py", line 437, in diarize
    self._perform_speech_activity_detection()
  File "C:\AI\STT\lib\site-packages\nemo\collections\asr\models\clustering_diarizer.py", line 325, in _perform_speech_activity_detection
    self._run_vad(manifest_vad_input)
  File "C:\AI\STT\lib\site-packages\nemo\collections\asr\models\clustering_diarizer.py", line 218, in _run_vad
    for i, test_batch in enumerate(
  File "C:\AI\STT\lib\site-packages\tqdm\std.py", line 1178, in __iter__
    for obj in iterable:
  File "C:\AI\STT\lib\site-packages\torch\utils\data\dataloader.py", line 441, in __iter__
    return self._get_iterator()
  File "C:\AI\STT\lib\site-packages\torch\utils\data\dataloader.py", line 388, in _get_iterator
    return _MultiProcessingDataLoaderIter(self)
  File "C:\AI\STT\lib\site-packages\torch\utils\data\dataloader.py", line 1042, in __init__
    w.start()
  File "C:\Users\kijan\AppData\Local\Programs\Python\Python310\lib\multiprocessing\process.py", line 121, in start
    self._popen = self._Popen(self)
  File "C:\Users\kijan\AppData\Local\Programs\Python\Python310\lib\multiprocessing\context.py", line 224, in _Popen
    return _default_context.get_context().Process._Popen(process_obj)
  File "C:\Users\kijan\AppData\Local\Programs\Python\Python310\lib\multiprocessing\context.py", line 336, in _Popen
    return Popen(process_obj)
  File "C:\Users\kijan\AppData\Local\Programs\Python\Python310\lib\multiprocessing\popen_spawn_win32.py", line 45, in __init__
    prep_data = spawn.get_preparation_data(process_obj._name)
  File "C:\Users\kijan\AppData\Local\Programs\Python\Python310\lib\multiprocessing\spawn.py", line 154, in get_preparation_data
    _check_not_importing_main()
  File "C:\Users\kijan\AppData\Local\Programs\Python\Python310\lib\multiprocessing\spawn.py", line 134, in _check_not_importing_main
    raise RuntimeError('''
RuntimeError:
        An attempt has been made to start a new process before the
        current process has finished its bootstrapping phase.

        This probably means that you are not using fork to start your
        child processes and you have forgotten to use the proper idiom
        in the main module:

            if __name__ == '__main__':
                freeze_support()
                ...

        The "freeze_support()" line can be omitted if the program
        is not going to be frozen to produce an executable.

ChatGPT: 
- The given output appears to be the console output of a command being run in a Python script. The script seems to be performing speaker diarization on an audio file using the NeMo library, which is a framework for building and training speech and natural language processing models.

Here are the main steps and information indicated in the output: 
1. The script is running the command `python diarize.py -a .\PTT-20220420-WA0000.opus` to execute the `diarize.py` script and perform diarization on the audio file `PTT-20220420-WA0000.opus`. 
2. The script initializes and loads the required models for diarization. It mentions that the selected model is a bag of 1 models. 
3. The script separates the tracks from the audio file and stores them in a temporary directory. 
4. It loads a pretrained diarization model and a voice activity detection (VAD) model. 
5. There are several warnings related to the model configuration for training, validation, and testing data. These warnings suggest that certain methods or configuration files are missing, which might be relevant for training or evaluating the models but not necessarily for the diarization task. 
6. The script proceeds with the diarization process, splitting the audio file into smaller segments to avoid CUDA memory issues. 
7. The VAD (speech activity detection) is performed using the loaded VAD model. 
8. Finally, an error occurs during the VAD step, resulting in a traceback. It seems that there is an issue with the input data or the VAD model.

To understand the exact cause of the error and resolve it, further investigation into the `diarize.py` script and the specific configurations and data being used would be required.

Person: 
- Can you talk a bit more about everything after the 5th point?

ChatGPT: 
- Certainly! After the fifth point, the code is providing information about the setup and configuration of the models being used for speech-to-text (STT) tasks. 
1. The code displays the padding value used for the models: `PADDING: 16`. 
2. The code loads a pre-trained model called `vad_multilingual_marblenet` for Voice Activity Detection (VAD). It mentions that the model is being loaded from the NGC (NVIDIA GPU Cloud) repository. 
3. It mentions that the model checkpoint file for `vad_multilingual_marblenet` already exists in the local cache and will be reused. 
4. The code instantiates the `vad_multilingual_marblenet` model.
5. There are warnings indicating that the model is not set up for training, validation, or testing. It suggests calling the appropriate setup methods for each of these tasks.

From the 6th point onwards, the code provides information about another model called `diar_msdd_telephonic`, which is used for diarization (the process of separating and labeling individual speakers in an audio recording). 
1. It mentions that the `diar_msdd_telephonic` model is being loaded from the NGC repository. 
2. The model checkpoint file for `diar_msdd_telephonic` already exists in the local cache and will be reused. 
3. The code instantiates the `diar_msdd_telephonic` model.
4. It specifies some configuration parameters for the diarization process, such as the number of speakers, clustering parameters, and more.
5. The code mentions that the previous diarization outputs are being deleted.
6. It indicates that there is one audio file to be diarized.
7. The code prepares the audio file for diarization, splitting it to avoid CUDA memory issues.
8. It mentions that the manifest file used for VAD (voice activity detection) is being prepared and overwritten. 
9. The code performs streaming frame-level VAD using the `vad_multilingual_marblenet` model. 
10. Finally, an error occurs during the VAD process. The traceback shows that the error occurred in the method `_run_vad` of the `clustering_diarizer.py` file. The specific cause of the error is not visible in the provided code snippet.

Person: 
- What information do you need to find the specific cause of the error?

ChatGPT: 
- To find the specific cause of the error, I would need the complete error message or traceback that occurred during the execution of the code. The provided code snippet only shows that an error occurred in the `_run_vad` method of the `clustering_diarizer.py` file. However, without the full error message or traceback, it is difficult to determine the exact cause of the error.

If you can provide the complete error message or traceback, I will be able to analyze it and provide more specific information about the cause of the error.

Person: 
- Is this enough information?

Traceback (most recent call last):
  File "C:\AI\STT\User\whisper-diarization\diarize.py", line 105, in <module>
    msdd_model.diarize()
  File "C:\AI\STT\lib\site-packages\torch\utils\_contextlib.py", line 115, in decorate_context
    return func(*args, **kwargs)
  File "C:\AI\STT\lib\site-packages\nemo\collections\asr\models\msdd_models.py", line 1180, in diarize
    self.clustering_embedding.prepare_cluster_embs_infer()
  File "C:\AI\STT\lib\site-packages\nemo\collections\asr\models\msdd_models.py", line 699, in prepare_cluster_embs_infer
    self.emb_sess_test_dict, self.emb_seq_test, self.clus_test_label_dict, _ = self.run_clustering_diarizer(
  File "C:\AI\STT\lib\site-packages\nemo\collections\asr\models\msdd_models.py", line 866, in run_clustering_diarizer
    scores = self.clus_diar_model.diarize(batch_size=self.cfg_diar_infer.batch_size)
  File "C:\AI\STT\lib\site-packages\nemo\collections\asr\models\clustering_diarizer.py", line 437, in diarize
    self._perform_speech_activity_detection()
  File "C:\AI\STT\lib\site-packages\nemo\collections\asr\models\clustering_diarizer.py", line 325, in _perform_speech_activity_detection
    self._run_vad(manifest_vad_input)
  File "C:\AI\STT\lib\site-packages\nemo\collections\asr\models\clustering_diarizer.py", line 218, in _run_vad
    for i, test_batch in enumerate(
  File "C:\AI\STT\lib\site-packages\tqdm\std.py", line 1178, in __iter__
    for obj in iterable:
  File "C:\AI\STT\lib\site-packages\torch\utils\data\dataloader.py", line 441, in __iter__
    return self._get_iterator()
  File "C:\AI\STT\lib\site-packages\torch\utils\data\dataloader.py", line 388, in _get_iterator
    return _MultiProcessingDataLoaderIter(self)
  File "C:\AI\STT\lib\site-packages\torch\utils\data\dataloader.py", line 1042, in __init__
    w.start()
  File "C:\Users\kijan\AppData\Local\Programs\Python\Python310\lib\multiprocessing\process.py", line 121, in start
    self._popen = self._Popen(self)
  File "C:\Users\kijan\AppData\Local\Programs\Python\Python310\lib\multiprocessing\context.py", line 224, in _Popen
    return _default_context.get_context().Process._Popen(process_obj)
  File "C:\Users\kijan\AppData\Local\Programs\Python\Python310\lib\multiprocessing\context.py", line 336, in _Popen
    return Popen(process_obj)
  File "C:\Users\kijan\AppData\Local\Programs\Python\Python310\lib\multiprocessing\popen_spawn_win32.py", line 93, in __init__
    reduction.dump(process_obj, to_child)
  File "C:\Users\kijan\AppData\Local\Programs\Python\Python310\lib\multiprocessing\reduction.py", line 60, in dump
    ForkingPickler(file, protocol).dump(obj)
_pickle.PicklingError: Can't pickle <class 'nemo.collections.common.parts.preprocessing.collections.SpeechLabelEntity'>: attribute lookup SpeechLabelEntity on nemo.collections.common.parts.preprocessing.collections failed

ChatGPT: 
- Thank you for providing the traceback. From the provided information, it appears that the error occurred during the pickling process, specifically when trying to pickle the `SpeechLabelEntity` class from the `nemo.collections.common.parts.preprocessing.collections` module.

Pickling is the process of serializing Python objects into a binary format, and it seems that the `SpeechLabelEntity` class is causing an issue during this serialization process. The error message indicates that the attribute lookup for `SpeechLabelEntity` failed, suggesting that the class may not be importable or accessible at the location where the pickling is being performed.

To further diagnose the issue, you can try the following steps: 
1. Ensure that the `nemo.collections.common.parts.preprocessing.collections` module is correctly installed and accessible in your environment. 
2. Check if there are any import errors or issues with the `nemo` library or its dependencies. 
3. Verify that the `SpeechLabelEntity` class is defined and accessible within the `nemo.collections.common.parts.preprocessing.collections` module. 
4. Make sure that the versions of the `nemo` library and its dependencies are compatible with each other.

By investigating these factors, you may be able to identify the specific cause of the pickling error.

Person: 
- How would I go about investigating these factors?

ChatGPT: 
- To investigate the factors that may be causing the pickling error, you can follow these steps: 
1. Verify module installation: Check if the `nemo` library and its dependencies are correctly installed in your environment. You can use the package manager or the command-line tool you typically use to manage Python packages. For example, if you're using pip, you can run `pip list` to see the installed packages.

If `nemo` or any of its dependencies are missing or outdated, you can update or reinstall them using the package manager. For example, `pip install --upgrade nemo` will upgrade the `nemo` library. 
2. Check for import errors: Ensure that there are no import errors when importing the `nemo` library or any of its dependencies. Run the script again and pay attention to any import-related error messages that may appear in the traceback.

If there are import errors, it's possible that the library or a required module is not properly installed or accessible. Reinstalling the library or resolving any missing dependencies should help. 
3. Verify class definition: Confirm that the `SpeechLabelEntity` class is defined within the `nemo.collections.common.parts.preprocessing.collections` module. You can open the relevant Python file in a text editor or IDE and search for the class definition.

If the class is not found in the expected module or if the module itself is missing, you might need to update or reinstall the `nemo` library. It's also possible that there may be different versions of the library installed in your environment, causing conflicts. In such cases, removing the conflicting installations and reinstalling the library can resolve the issue. 
4. Check compatibility: Ensure that the versions of the `nemo` library and its dependencies are compatible with each other. Incompatibilities between different library versions can sometimes lead to errors.

You can refer to the documentation or the release notes of the `nemo` library to find information about the compatible versions of its dependencies. If necessary, you may need to update or downgrade specific dependencies to achieve compatibility.

By going through

Person: 
- Sorry you got cut off there

ChatGPT: 
- Apologies for the cutoff. Here's the continuation:

By going through these steps, you can identify and resolve potential issues that could be causing the pickling error. It's important to ensure that the required modules and dependencies are correctly installed, that there are no import errors, and that the class `SpeechLabelEntity` is defined in the expected module. Additionally, verifying the compatibility of different library versions can help resolve conflicts.

If the issue persists after following these steps, you can consider reaching out to the `nemo` library's support channels, such as their GitHub repository or community forums. They may be able to provide further assistance in diagnosing and resolving the specific error you're encountering.

Person: 
- I did the pip upgrade and then ran it again. I got the same error. I anticipate that the issue could be a compatibility issue. How would I investigate this most effectively?

ChatGPT: 
- If you suspect that the error is caused by a compatibility issue, there are several steps you can take to investigate and resolve it: 
1. Check library compatibility: Ensure that all the libraries and dependencies you are using, including `nemo`, are compatible with each other. Review the documentation and release notes of each library to verify their compatibility requirements. Pay special attention to the versions of Python, PyTorch, and other major dependencies. 
2. Review the `nemo` documentation: Check the `nemo` documentation for any specific compatibility requirements or known issues. Look for any recommended library versions or configurations that might help resolve the problem. 
3. Verify Python version: Make sure you are using the correct version of Python recommended by `nemo` or the library documentation. It's possible that the code or dependencies require a specific Python version, and using a different version could lead to compatibility issues. 
4. Examine the error message: Carefully analyze the error message you received. Look for any specific module or class names mentioned in the error, such as `SpeechLabelEntity`. This can help you identify which library or module might be causing the issue. 
5. Search for similar issues: Search online forums, GitHub repositories, or other community platforms for similar issues or error messages related to the libraries you're using. It's possible that someone else has encountered the same problem and has found a solution or workaround. Pay attention to any discussions about compatibility issues or specific configurations. 
6. Create a minimal reproducible example: If you haven't already, try to isolate the error by creating a minimal code example that reproduces the issue. Strip down your code to the essential parts required to reproduce the error and remove any unrelated code. This will make it easier to share and discuss the issue with others. 
7. Reach out for support: If you've followed the steps above and haven't found a solution, consider reaching out for support. You can post your issue on relevant forums, GitHub repositories, or community platforms, providing the error message, relevant code snippets, and details about your environment (Python version, library versions, etc.). The `nemo` community or the developers of the libraries you're using may be able to provide further guidance or suggest possible solutions.

Remember to be patient and provide as much detail as possible when seeking support. This will help others understand your problem and provide more accurate assistance.

Person: 
- when did pytorch 2.o release?

ChatGPT: 
- As of my knowledge cutoff in September 2021, PyTorch 2.0 has not been released yet. At that time, PyTorch 1.9 was the latest stable release. However, please note that my information might be outdated, and there could have been new releases since then. I recommend checking the official PyTorch website or other reliable sources for the most up-to-date information on PyTorch releases.
