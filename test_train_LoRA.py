import os

from trainer import Trainer, TrainerArgs
import torch
from peft import LoraConfig, get_peft_model, TaskType
from tqdm import tqdm

from TTS.TTS.config.shared_configs import BaseDatasetConfig
from TTS.TTS.tts.datasets import load_tts_samples
from TTS.TTS.tts.layers.xtts.trainer.gpt_trainer import GPTArgs, GPTTrainer, GPTTrainerConfig, XttsAudioConfig
from TTS.TTS.utils.manage import ModelManager

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
    dataset_name="audio",
    path="/Users/changhyeoncheon/dev/Vreeze-AI",
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
    "/Users/changhyeoncheon/dev/Vreeze-AI/wavs/audio138.wav"
]
LANGUAGE = config_dataset.language

# LoRA 설정
LORA_RANK = 8  # LoRA 랭크
LORA_ALPHA = 16  # LoRA 알파
LORA_DROPOUT = 0.05  # LoRA 드롭아웃 비율
TARGET_MODULES = ["q_proj", "v_proj", "k_proj", "o_proj"]  # LoRA를 적용할 모듈

def apply_lora_to_model(model):
    """GPTTrainer 모델에 LoRA를 적용하는 함수"""
    # GPTTrainer 클래스에서는 xtts.gpt.gpt를 통해 GPT 모델에 접근
    target_module = model.xtts.gpt.gpt
    
    # LoRA 설정 생성
    lora_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,  # 이 부분이 중요합니다!
        r=8,                      # LoRA 랭크
        lora_alpha=16,            # LoRA 알파
        lora_dropout=0.05,        # 드롭아웃 비율
        # GPT2 모델의 주요 어텐션 모듈
        target_modules=["c_attn", "c_proj"],
        bias="none",
        inference_mode=False,
    )
    
    try:
        # 모델에 LoRA 적용
        print("LoRA를 GPT 모델에 적용합니다...")
        
        # 원본 forward 메서드 보존
        original_forward = target_module.forward
        
        # LoRA 모델 생성
        lora_model = get_peft_model(target_module, lora_config)
        model.xtts.gpt.gpt = lora_model
        
        # original_forward 메서드를 lora_model에 연결
        def custom_forward(*args, **kwargs):
            # 'labels' 인수 제거 (필요한 경우)
            if 'labels' in kwargs:
                del kwargs['labels']
            return original_forward(*args, **kwargs)
        
        # 커스텀 forward 메서드 설정
        model.xtts.gpt.gpt.model.forward = custom_forward
        
        # LoRA 파라미터만 훈련 가능하도록 설정
        print("모든 파라미터를 freeze 합니다...")
        for param in model.parameters():
            param.requires_grad = False
        
        # LoRA 파라미터 훈련 가능하게 설정
        print("LoRA 파라미터를 훈련 가능하게 설정합니다...")
        trainable_params = 0
        all_params = 0
        
        for name, param in model.named_parameters():
            all_params += param.numel()
            if "lora" in name:
                param.requires_grad = True
                trainable_params += param.numel()
                print(f"훈련 가능한 LoRA 파라미터: {name}")
        
        print(f"전체 파라미터: {all_params:,} 개")
        print(f"훈련 가능한 파라미터: {trainable_params:,} 개 ({trainable_params / all_params:.2%})")
        print("LoRA가 성공적으로 적용되었습니다.")
    except Exception as e:
        print(f"LoRA 적용 중 오류 발생: {str(e)}")
    
    return model

def main():
    # 모델 인자 설정
    model_args = GPTArgs(
        max_conditioning_length=132300,
        min_conditioning_length=20000, #66150
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
                "text": "브라우저로 다음 주소에 접속하면, 텐서보드 대시보드를 볼 수 있습니다. 저는 TTS입니다.",
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

    if not eval_samples:
        print("경고: 평가 데이터셋이 비어 있습니다. 훈련 데이터셋에서 일부를 가져옵니다.")
        num_eval_samples = min(10, len(train_samples))  # 최소 10개 또는 전체 훈련 샘플 수
        eval_samples = train_samples[:num_eval_samples]
        train_samples = train_samples[num_eval_samples:]
        print(f"평가 데이터셋 크기: {len(eval_samples)}")
        print(f"훈련 데이터셋 크기: {len(train_samples)}")
        # 평가 샘플 확인 및 처리
    print(f"초기 평가 샘플 수: {len(eval_samples)}")
    
    # 평가 샘플이 없거나 너무 적은 경우
    if len(eval_samples) < 5:  # 최소 5개 이상의 샘플이 필요하다고 가정
        print("평가 데이터셋이 부족합니다. 훈련 데이터셋에서 추가 샘플을 가져옵니다.")
        
        # 훈련 샘플의 유효성 미리 확인 (필터링 테스트)
        valid_train_samples = []
        for sample in train_samples[:50]:  # 처음 50개 샘플만 확인
            # 필터링 로직 추가 - 오디오 파일이 존재하고 접근 가능한지 확인
            audio_file = sample.get("audio_file", None) or sample.get("wav_file", None)
            if audio_file and os.path.exists(audio_file) and os.path.getsize(audio_file) > 0:
                try:
                    # 간단한 오디오 메타데이터 확인 (선택적)
                    import wave
                    with wave.open(audio_file, 'rb') as wf:
                        if wf.getnframes() > 0:
                            valid_train_samples.append(sample)
                except Exception as e:
                    print(f"오디오 파일 확인 중 오류: {audio_file} - {str(e)}")
                    continue
        
        print(f"유효한 훈련 샘플 수: {len(valid_train_samples)}")
        
        # 유효한 샘플에서 평가 데이터 추가
        num_eval_samples = min(10, len(valid_train_samples))
        new_eval_samples = valid_train_samples[:num_eval_samples]
        
        # 기존 평가 샘플에 추가
        if eval_samples:
            eval_samples.extend(new_eval_samples)
        else:
            eval_samples = new_eval_samples
            
        # 사용한 샘플은 훈련 데이터에서 제외 (선택적)
        train_sample_ids = set(s.get("audio_file", "") for s in train_samples)
        eval_sample_ids = set(s.get("audio_file", "") for s in eval_samples)
        train_samples = [s for s in train_samples if s.get("audio_file", "") not in eval_sample_ids]
        
        print(f"최종 평가 데이터셋 크기: {len(eval_samples)}")
        print(f"최종 훈련 데이터셋 크기: {len(train_samples)}")
    
    # 최종 확인 - 평가 데이터가 여전히 비어 있으면 오류 발생
    if not eval_samples:
        raise ValueError("유효한 평가 샘플을 생성할 수 없습니다. 데이터셋을 확인하세요.")

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