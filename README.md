# Virtual Try-On Workflow (Qwen Image Edit)

AI 가상 피팅 워크플로우입니다. 사람 이미지와 의류 이미지를 입력하면 해당 사람이 옷을 입은 모습을 생성합니다.

## 흐름

```
[의류 1 + 의류 2] → [사람 이미지 아래에 배치] → [Qwen AI 처리] → [결과]
```

---

## 노드 구성

### 1. Garment Selection (의류 선택)

| 노드 | 역할 |
|------|------|
| LoadImage (Garment 1) | 첫 번째 의류 이미지 |
| LoadImage (Garment 2) | 두 번째 의류 이미지 |
| LoadImage (Garment 3) | 세 번째 의류 (현재 미연결) |
| ImageStitch | 의류들을 가로로 합침 (`direction: right`) |

### 2. Person Selection (사람 선택)

| 노드 | 역할 |
|------|------|
| LoadImage (Person) | 옷을 입힐 대상 사람 이미지 |
| ImageStitch | 사람(위) + 의류(아래) 세로 합성 (`direction: down`) |

### 3. Final Input Image for Qwen (입력 이미지 준비)

| 노드 | 역할 | 설정값 |
|------|------|--------|
| ImageResizeKJv2 | Qwen 권장 해상도로 조정 | `832 × 1248` (세로) |
| PreviewImage | 합성 결과 미리보기 | - |

**Qwen 권장 해상도:**
- 세로: 832×1248, 672×1568
- 가로: 1248×832
- 정사각형: 1024×1024

### 4. Generation (이미지 생성)

| 노드 | 설정값 |
|------|--------|
| UNETLoader | `qwen_image_edit_2509_fp8_e4m3fn.safetensors` |
| CLIPLoader | `qwen_2.5_vl_7b_fp8_scaled.safetensors` |
| VAELoader | `qwen_image_vae.safetensors` |
| ModelSamplingAuraFlow | `shift: 3` |
| CFGNorm | `strength: 1` |
| TextEncodeQwenImageEdit (Prompt) | `Style the model in the top of the image, with every article of clothing on the bottom` |
| TextEncodeQwenImageEdit (Negative) | 비어있음 |

**KSampler 설정:**

| 파라미터 | 값 | 조정 가이드 |
|----------|-----|-------------|
| seed | 1088 (fixed) | 같은 결과 재현시 고정, 다양한 결과는 randomize |
| steps | 30 | 20~50 (높을수록 품질↑, 속도↓) |
| cfg | 2.5 | 2.0~3.0 권장 (높을수록 프롬프트 준수↑) |
| sampler_name | euler | 빠르고 안정적 |
| scheduler | simple | - |
| denoise | 1 | 1.0 = 완전 새로 생성 |

### 5. Results (결과)

| 노드 | 역할 |
|------|------|
| VAEDecode | latent → 이미지 변환 |
| SaveImage | 결과 이미지 저장 |
| ImageStitch + SaveImage | 입력/결과 비교 이미지 저장 |

---

## 핵심 노드 설명

### UNETLoader
이미지 생성의 핵심 AI 모델(diffusion model)을 로드합니다. UNET은 노이즈가 있는 이미지에서 점진적으로 노이즈를 제거하며 최종 이미지를 만들어내는 신경망 구조입니다.

**모델 파일:** `qwen_image_edit_2509_fp8_e4m3fn.safetensors`
- `fp8_e4m3fn`: 8비트 부동소수점 형식. 메모리 사용량을 줄이면서 품질 유지 (VRAM이 적은 GPU에 유용)

### CLIPLoader
텍스트 프롬프트를 AI가 이해할 수 있는 벡터(embedding)로 변환하는 모델입니다. 사용자가 입력한 "Style the model..."같은 문장을 숫자 배열로 인코딩하여 UNET에 전달합니다.

**모델 파일:** `qwen_2.5_vl_7b_fp8_scaled.safetensors`
- Qwen 2.5 Vision-Language 7B 모델 기반

### VAELoader
VAE(Variational Auto-Encoder)는 이미지와 latent space 사이의 변환을 담당합니다.
- **인코딩**: 이미지 → latent (압축된 표현, AI가 처리하기 쉬운 형태)
- **디코딩**: latent → 이미지 (사람이 볼 수 있는 형태로 복원)

AI는 픽셀을 직접 다루지 않고 latent 공간에서 작업하기 때문에 VAE가 필수입니다.

### ModelSamplingAuraFlow
Qwen 모델의 노이즈 스케줄링 방식을 설정합니다.

| 파라미터 | 값 | 설명 |
|----------|-----|------|
| shift | 3 | 노이즈 스케줄 시프트 값. Qwen 모델은 일반 SD와 다른 노이즈 처리 방식을 사용하므로 이 노드가 없으면 결과가 노이즈로 가득 참 |

### CFGNorm
CFG(Classifier-Free Guidance) 값을 정규화하여 안정적인 생성을 돕습니다.

| 파라미터 | 값 | 설명 |
|----------|-----|------|
| strength | 1 | 정규화 강도. 이 노드가 없으면 색상 이상이나 아티팩트 발생 가능 |

### KSampler
실제 이미지 생성(denoising)을 수행하는 핵심 노드입니다.

| 파라미터 | 현재값 | 설명 |
|----------|--------|------|
| **seed** | 1088 (fixed) | 랜덤 시드. 같은 값 = 같은 결과 재현. `randomize`로 변경하면 매번 다른 결과 |
| **steps** | 30 | 노이즈 제거 반복 횟수. 높을수록 디테일↑, 시간↑. 권장: 20~50 |
| **cfg** | 2.5 | 프롬프트 준수 강도. 낮으면(1~2) 창의적이지만 불안정, 높으면(4+) 프롬프트에 과집착. Qwen 권장: 2.0~3.0 |
| **sampler_name** | euler | 샘플링 알고리즘. `euler`는 빠르고 안정적. 대안: `euler_ancestral`(다양한 결과), `dpmpp_2m`(고품질, 느림) |
| **scheduler** | simple | 노이즈 감소 스케줄. `simple`은 선형 감소 |
| **denoise** | 1 | 노이즈 제거 강도. 1.0 = 완전히 새로 생성, 0.5 = 원본 50% 유지 |

### VAEEncode / VAEDecode
- **VAEEncode**: 입력 이미지를 latent로 변환 (KSampler 입력용)
- **VAEDecode**: KSampler 출력(latent)을 이미지로 변환

### TextEncodeQwenImageEdit
프롬프트를 Qwen 모델용 conditioning으로 변환합니다.
- **Prompt**: AI에게 수행할 작업 지시
- **Negative Prompt**: 피해야 할 요소 (예: `blurry, distorted face, extra limbs`)

---
