import torch
from PIL import Image
from transformers import LlavaForConditionalGeneration, AutoProcessor
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

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    def load_model(self, use_mps: bool = True):
        """모델 로딩 (맥북 최적화)"""
        if self._model is not None:
            logger.info("Model already loaded")
            return self._model, self._processor
        
        logger.info(f"Loading model: {self._model_id}")

        device = "mps" if use_mps and torch.backends.mps.is_available() else "cpu"
        logger.info(f"Using device: {device}")

        # 프로세서 로드
        self._processor = AutoProcessor.from_pretrained(self._model_id)

        # 모델 로드
        self._model = LlavaForConditionalGeneration.from_pretrained(
            self._model_id,
            torch_dtype=torch.float16,
            low_cpu_mem_usage=True
        )

        if device == "mps":
            self._model = self._model.to(device)

        # Warm up
        logger.info("Warming up model...")
        self._warmup()

        logger.info("Model loaded successfully!")
        return self._model, self._processor
    
    def _warmup(self):
        """더미 입력으로 모델 준비"""
        dummy_image = Image.new('RGB', (224,224), color='white')
        dummy_prompt = "USER: <image>\nDescribe this image.\nASSISTANT:"

        inputs = self._processor(
            text=dummy_prompt,
            images=dummy_image,
            return_tensors="pt"
        )

        if self._model.device.type == "mps":
            inputs = {k: v.to("mps") for k,v in inputs.items()}

        with torch.no_grad():
            self._model.generate(**inputs, max_new_tokens=10)

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

        prompt = f"USER: <image>\n{question}\nASSISTANT:"

        inputs = self._processor(
            text= prompt,
            images=image,
            return_tensors="pt"
        )

        # MPS 디바이스로 이동
        if self._model.device.type == "mps":
            inputs = {k: v.to("mps") for k,v in inputs.items()}

        with torch.no_grad():
            output = self._model.generate(
                **inputs,
                max_new_tokens=max_tokens,
                temperature=temperature,
                do_sample=True if temperature > 0 else False
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

        if self._model.device.type == "mps":
            inputs = {k: v.to("mps") for k, v in inputs.items()}
        
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
