import torch
from PIL import Image
from transformers import LlavaForConditionalGeneration, AutoProcessor, BitsAndBytesConfig
from typing import Optional
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ModelManager:
    """싱글톤 패턴으로 모델 관리"""
    
    _instance = None
    _model = None
    _processor = None
    _model_id = "llava-hf/llava-1.5-7b-hf"
    # _model_id = "bczhou/tiny-llava-v1-hf"  # TinyLLaVA는 processor 호환성 이슈 있음

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    def load_model(self, use_quantization: bool = True ,use_mps: bool = True):
        """모델 로딩 (맥북 최적화)"""
        if self._model is not None:
            logger.info("Model already loaded")
            return self._model, self._processor
        
        logger.info(f"Loading model: {self._model_id}")

       

        # 프로세서 로드
        self._processor = AutoProcessor.from_pretrained(self._model_id)

        # patch_size가 None인 경우 명시적으로 설정
        if hasattr(self._processor, 'image_processor'):
            if hasattr(self._processor.image_processor, 'patch_size'):
                if self._processor.image_processor.patch_size is None:
                    self._processor.image_processor.patch_size = 14  # LLaVA 기본값
                    logger.info("Set patch_size to 14 (default)")
            # size 속성도 확인
            if hasattr(self._processor.image_processor, 'size'):
                if self._processor.image_processor.size is None:
                    self._processor.image_processor.size = {"height": 336, "width": 336}
                    logger.info("Set image size to 336x336 (default)")

        if use_quantization:
            logger.info("Loading with 4-bit quantization...")

            # 4-bit 양자화 설정
            quantization_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=torch.float16,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_use_double_quant=True
            )
            self._model = LlavaForConditionalGeneration.from_pretrained(
                self._model_id,
                quantization_config=quantization_config,
                torch_dtype=torch.float16,
                low_cpu_mem_usage=True
            )

            logger.info("4-bit quantized model loaded!")
            logger.info(f" Memory saved")

        else:
            device = "mps" if use_mps and torch.backends.mps.is_available() else "cpu"
            logger.info(f"Using device: {device}")

            # 모델 로드 (device_map 사용하지 않음 - MPS에서는 지원 안됨)
            self._model = LlavaForConditionalGeneration.from_pretrained(
                self._model_id,
                torch_dtype=torch.float16,
                low_cpu_mem_usage=True
            )

            if device == "mps":
                logger.info("Moving model to MPS device...")
                self._model = self._model.to(device)
                logger.info(f"Model device after move: {self._model.device}")

        # Warm up
        logger.info("Warming up model...")
        self._warmup()

        logger.info("Model loaded successfully!")
        return self._model, self._processor
    
    def _warmup(self):
        """더미 입력으로 모델 준비"""
        try:
            dummy_image = Image.new('RGB', (336, 336), color='white')  # LLaVA 기본 사이즈
            dummy_prompt = "USER: <image>\nDescribe this image.\nASSISTANT:"

            inputs = self._processor(
                text=dummy_prompt,
                images=dummy_image,
                return_tensors="pt"
            )

            # MPS 디바이스로 이동
            device = torch.device("mps") if torch.backends.mps.is_available() else torch.device("cpu")
            inputs = {k: v.to(device) for k, v in inputs.items()}

            with torch.no_grad():
                self._model.generate(**inputs, max_new_tokens=10)

            logger.info("Warmup completed successfully!")
        except Exception as e:
            logger.warning(f"Warmup failed (non-critical): {e}")

    def analyze_image(
            self,
            image: Image.Image,
            question: str,
            max_tokens: int = 100,
            temperature: float = 0.7
    ) -> str:
        """이미지 분석 메인 함수"""
        if self._model is None:
            raise RuntimeError("Model not loaded. Call load_model() first")

        # 이미지를 RGB로 변환하고 적절한 크기로 조정
        if image.mode != 'RGB':
            image = image.convert('RGB')

        prompt = f"USER: <image>\n{question}\nASSISTANT:"

        # padding과 truncation 명시적으로 지정
        inputs = self._processor(
            text=prompt,
            images=image,
            return_tensors="pt",
            padding=True
        )

        # MPS 디바이스로 이동 (무조건 mps로)
        device = torch.device("mps") if torch.backends.mps.is_available() else torch.device("cpu")
        inputs = {k: v.to(device) for k, v in inputs.items()}

        with torch.no_grad():
            output = self._model.generate(
                **inputs,
                max_new_tokens=max_tokens,
                temperature=temperature,
                do_sample=True if temperature > 0 else False,
                num_beams=1,  # beam search 끄기 (속도 향상)
                use_cache=True  # KV 캐시 사용 (속도 향상)
            )

        # decoding
        full_response = self._processor.decode(output[0], skip_special_tokens=True)

        answer = full_response.split("ASSISTANT:")[-1].strip()

        return answer
    
    def batch_analyze(
        self,
        images: list[Image.Image],
        questions: list[str],
        max_tokens: int = 100
    ) -> list[str]:
        """배치 처리"""
        if self._model is None:
            raise RuntimeError("Model not loaded.")
        
        # 프롬프트 리스트
        prompts = [f"USER: <image>\n{q}\nASSISTANT:" for q in questions]

        inputs = self._processor(
            text=prompts,
            images=images,
            return_tensors="pt",
            padding=True
        )

        # MPS 디바이스로 이동 (무조건 mps로)
        device = torch.device("mps") if torch.backends.mps.is_available() else torch.device("cpu")
        inputs = {k: v.to(device) for k, v in inputs.items()}
        
        with torch.no_grad():
            outputs = self._model.generate(
                **inputs,
                max_new_tokens=max_tokens,
                do_sample=False
            )

        answers = []

        for output in outputs:
            full_response = self._processor.decode(output, skip_special_tokens=True)
            answer = full_response.split("ASSISTANT:")[-1].strip()
            answers.append(answer)

        return answers
        
    @property
    def is_loaded(self) -> bool:
        return self._model is not None
