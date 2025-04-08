# Logging parameters
RUN_NAME = "GPT_XTTS_v2.0_LJSpeech_FT_LoRA"
PROJECT_NAME = "XTTS_trainer_LoRA"
DASHBOARD_LOGGER = "tensorboard"
LOGGER_URI = None

# 출력 경로 설정
OUT_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "run", "training")

# 훈련 파라미터
OPTIMIZER_WD_ONLY_ON_WEIGHTS = True
START_WITH_EVAL = True
BATCH_SIZE = 4  # LoRA는 더 적은 메모리를 사용하므로 배치 크기를 늘릴 수 있습니다
GRAD_ACUMM_STEPS = 64  # 배치 크기와 비례하여 조정

# 데이터셋 설정
config_dataset = BaseDatasetConfig(
    formatter="ljspeech",
    dataset_name="ljspeech",
    path="/Users/changhyeoncheon/dev/Vreeze-AI/wavs_processed",
    meta_file_train="/Users/changhyeoncheon/dev/Vreeze-AI/metadata.txt",
    language="ko",
)

DATASETS_CONFIG_LIST = [config_dataset]

# 체크포인트 경로 설정
CHECKPOINTS_OUT_PATH = os.path.join(OUT_PATH, "XTTS_v2.0_original_model_files/")
os.makedirs(CHECKPOINTS_OUT_PATH, exist_ok=True)

# DVAE 파일
DVAE_CHECKPOINT_LINK = "https://coqui.gateway.scarf.sh/hf-coqui/XTTS-v2/main/dvae.pth"
MEL_NORM_LINK = "https://coqui.gateway.scarf.sh/hf-coqui/XTTS-v2/main/mel_stats.pth"

DVAE_CHECKPOINT = os.path.join(CHECKPOINTS_OUT_PATH, os.path.basename(DVAE_CHECKPOINT_LINK))
MEL_NORM_FILE = os.path.join(CHECKPOINTS_OUT_PATH, os.path.basename(MEL_NORM_LINK))

# 필요한 파일 다운로드
if not os.path.isfile(DVAE_CHECKPOINT) or not os.path.isfile(MEL_NORM_FILE):
    print(" > Downloading DVAE files!")
    ModelManager._download_model_files([MEL_NORM_LINK, DVAE_CHECKPOINT_LINK], CHECKPOINTS_OUT_PATH, progress_bar=True)

# XTTS 체크포인트 다운로드
TOKENIZER_FILE_LINK = "https://coqui.gateway.scarf.sh/hf-coqui/XTTS-v2/main/vocab.json"
XTTS_CHECKPOINT_LINK = "https://coqui.gateway.scarf.sh/hf-coqui/XTTS-v2/main/model.pth"

TOKENIZER_FILE = os.path.join(CHECKPOINTS_OUT_PATH, os.path.basename(TOKENIZER_FILE_LINK))
XTTS_CHECKPOINT = os.path.join(CHECKPOINTS_OUT_PATH, os.path.basename(XTTS_CHECKPOINT_LINK))

if not os.path.isfile(TOKENIZER_FILE) or not os.path.isfile(XTTS_CHECKPOINT):
    print(" > Downloading XTTS v2.0 files!")
    ModelManager._download_model_files(
        [TOKENIZER_FILE_LINK, XTTS_CHECKPOINT_LINK], CHECKPOINTS_OUT_PATH, progress_bar=True
    )

# 테스트용 화자 참조
SPEAKER_REFERENCE = [
    "./tests/data/ljspeech/wavs/LJ001-0002.wav"
]
LANGUAGE = config_dataset.language

# LoRA 설정
LORA_RANK = 8  # LoRA 랭크
LORA_ALPHA = 16  # LoRA 알파
LORA_DROPOUT = 0.05  # LoRA 드롭아웃 비율
TARGET_MODULES = ["q_proj", "v_proj", "k_proj", "o_proj"]  # LoRA를 적용할 모듈

def apply_lora_to_model(model):
    """모델에 LoRA를 적용하는 함수"""
    # LoRA 설정 생성
    lora_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,  # 캐주얼 LM 태스크 유형
        r=LORA_RANK,
        lora_alpha=LORA_ALPHA,
        lora_dropout=LORA_DROPOUT,
        target_modules=TARGET_MODULES,
        bias="none",
        inference_mode=False,
    )
    
    # 모델에 LoRA 적용
    lora_model = get_peft_model(model.gpt, lora_config)
    model.gpt = lora_model
    
    # LoRA 파라미터만 훈련 가능하도록 설정
    for param in model.gpt.parameters():
        param.requires_grad = False
    
    for name, param in model.gpt.named_parameters():
        if "lora" in name:
            param.requires_grad = True
    
    return model

def main():
    # 모델 인자 설정
    model_args = GPTArgs(
        max_conditioning_length=132300,
        min_conditioning_length=66150,
        debug_loading_failures=False,
        max_wav_length=255995,
        max_text_length=200,
        mel_norm_file=MEL_NORM_FILE,
        dvae_checkpoint=DVAE_CHECKPOINT,
        xtts_checkpoint=XTTS_CHECKPOINT,
        tokenizer_file=TOKENIZER_FILE,
        gpt_num_audio_tokens=1026,
        gpt_start_audio_token=1024,
        gpt_stop_audio_token=1025,
        gpt_use_masking_gt_prompt_approach=True,
        gpt_use_perceiver_resampler=True,
    )
    
    # 오디오 설정
    audio_config = XttsAudioConfig(sample_rate=24000, dvae_sample_rate=24000, output_sample_rate=24000)
    
    # 훈련 설정
    config = GPTTrainerConfig(
        output_path=OUT_PATH,
        model_args=model_args,
        run_name=RUN_NAME,
        project_name=PROJECT_NAME,
        run_description="""
            GPT XTTS training with LoRA fine-tuning
            """,
        dashboard_logger=DASHBOARD_LOGGER,
        logger_uri=LOGGER_URI,
        audio=audio_config,
        batch_size=BATCH_SIZE,
        batch_group_size=48,
        eval_batch_size=BATCH_SIZE,
        num_loader_workers=8,
        eval_split_max_size=256,
        print_step=50,
        plot_step=100,
        log_model_step=1000,
        save_step=5000,  # LoRA는 더 자주 저장 가능
        save_n_checkpoints=2,
        save_checkpoints=True,
        optimizer="AdamW",
        optimizer_wd_only_on_weights=OPTIMIZER_WD_ONLY_ON_WEIGHTS,
        optimizer_params={"betas": [0.9, 0.96], "eps": 1e-8, "weight_decay": 1e-2},
        lr=1e-04,  # LoRA에 대해 학습률 조정
        lr_scheduler="MultiStepLR",
        lr_scheduler_params={"milestones": [25000, 75000, 150000], "gamma": 0.5, "last_epoch": -1},
        test_sentences=[
            {
                "text": "It took me quite a long time to develop a voice, and now that I have it I'm not going to be silent.",
                "speaker_wav": SPEAKER_REFERENCE,
                "language": LANGUAGE,
            },
            {
                "text": "This cake is great. It's so delicious and moist.",
                "speaker_wav": SPEAKER_REFERENCE,
                "language": LANGUAGE,
            },
        ],
    )

    # 모델 초기화
    model = GPTTrainer.init_from_config(config)
    
    # LoRA 적용
    model = apply_lora_to_model(model)
    
    # 훈련/평가 샘플 로드
    train_samples, eval_samples = load_tts_samples(
        DATASETS_CONFIG_LIST,
        eval_split=True,
        eval_split_max_size=config.eval_split_max_size,
        eval_split_size=config.eval_split_size,
    )

    # 훈련 시작
    trainer = Trainer(
        TrainerArgs(
            restore_path=None,
            skip_train_epoch=False,
            start_with_eval=START_WITH_EVAL,
            grad_accum_steps=GRAD_ACUMM_STEPS,
        ),
        config,
        output_path=OUT_PATH,
        model=model,
        train_samples=train_samples,
        eval_samples=eval_samples,
    )
    trainer.fit()
    
    # LoRA 가중치만 저장
    lora_weights_path = os.path.join(OUT_PATH, "lora_weights")
    os.makedirs(lora_weights_path, exist_ok=True)
    model.gpt.save_pretrained(lora_weights_path)
    print(f"LoRA 가중치가 {lora_weights_path}에 저장되었습니다.")

if __name__ == "__main__":
    main()