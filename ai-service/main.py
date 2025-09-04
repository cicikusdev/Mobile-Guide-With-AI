from fastapi import FastAPI, File, UploadFile, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
import base64
import io
import cv2
import numpy as np
from PIL import Image
from ultralytics import YOLO
import logging
from typing import Optional
import time

# Logging konfigürasyonu
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(
    title="🧠 Mobil Rehber AI Servisi",
    description="YOLOv5 ile görsel analizi yapan AI servisi",
    version="1.0.0"
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# YOLOv5 modeli yükle
try:
    model = YOLO('yolov5s.pt')  # Küçük model, hızlı çalışır
    logger.info("✅ YOLOv5 modeli başarıyla yüklendi")
except Exception as e:
    logger.error(f"❌ YOLOv5 modeli yüklenemedi: {e}")
    model = None

# COCO sınıf isimleri (YOLOv5'in varsayılan sınıfları)
COCO_CLASSES = [
    'person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus', 'train', 'truck', 'boat',
    'traffic light', 'fire hydrant', 'stop sign', 'parking meter', 'bench', 'bird', 'cat',
    'dog', 'horse', 'sheep', 'cow', 'elephant', 'bear', 'zebra', 'giraffe', 'backpack',
    'umbrella', 'handbag', 'tie', 'suitcase', 'frisbee', 'skis', 'snowboard', 'sports ball',
    'kite', 'baseball bat', 'baseball glove', 'skateboard', 'surfboard', 'tennis racket',
    'bottle', 'wine glass', 'cup', 'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple',
    'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza', 'donut', 'cake',
    'chair', 'couch', 'potted plant', 'bed', 'dining table', 'toilet', 'tv', 'laptop',
    'mouse', 'remote', 'keyboard', 'cell phone', 'microwave', 'oven', 'toaster', 'sink',
    'refrigerator', 'book', 'clock', 'vase', 'scissors', 'teddy bear', 'hair drier', 'toothbrush'
]

@app.get("/")
async def root():
    """Ana endpoint"""
    return {
        "message": "🧠 Mobil Rehber AI Servisi",
        "version": "1.0.0",
        "model_loaded": model is not None,
        "endpoints": {
            "/analyze": "POST - Görsel analizi",
            "/health": "GET - Sistem durumu"
        }
    }

@app.get("/health")
async def health_check():
    """Sistem durumu kontrolü"""
    return {
        "status": "healthy",
        "model_loaded": model is not None,
        "timestamp": time.time()
    }

@app.post("/analyze")
async def analyze_image(request: Request):
    """Görsel analizi endpoint'i"""

    if not model:
        raise HTTPException(status_code=500, detail="AI modeli yüklenemedi")

    try:
        # JSON body'den veriyi oku
        body = await request.json()
        image_base64 = body.get("image_base64")

        if not image_base64:
            raise HTTPException(status_code=400, detail="Görsel bulunamadı")

        # Görseli işle
        image_data = base64.b64decode(image_base64)
        logger.info(f"📱 Base64 formatında görsel alındı: {len(image_data)} bytes")

        pil_image = Image.open(io.BytesIO(image_data))
        cv_image = cv2.cvtColor(np.array(pil_image), cv2.COLOR_RGB2BGR)

        # YOLOv5 ile analiz
        logger.info("🔍 YOLOv5 analizi başlatılıyor...")
        results = model(cv_image)

        # Sonuçları işle
        objects, confidence, processing_time = process_yolo_results_optimized(results[0], cv_image)
        analysis_result = process_yolo_results(results[0], cv_image)

        logger.info(f"✅ Analiz tamamlandı: {analysis_result}")

        return {
            "success": True,
            "objects": objects,
            "confidence": confidence,
            "processing_time": processing_time,
            "description": analysis_result
        }

    except HTTPException as e:
        raise e  # HTTP hatalarını doğrudan ilet
    except Exception as e:
        logger.error(f"❌ Analiz hatası: {e}")
        raise HTTPException(status_code=500, detail=f"Analiz hatası: {str(e)}")

def process_yolo_results_optimized(result, image):
    """YOLOv5 sonuçlarını işle ve mobil uygulama için optimize et"""
    
    start_time = time.time()
    
    if result.boxes is None or len(result.boxes) == 0:
        return [], 0.0, 0.0
    
    # Tespit edilen nesneleri grupla
    detected_objects = {}
    total_confidence = 0.0
    object_count = 0
    
    for box in result.boxes:
        if box.conf > 0.5:  # Güven eşiği
            class_id = int(box.cls[0])
            class_name = COCO_CLASSES[class_id]
            confidence = float(box.conf[0])
            
            if class_name not in detected_objects:
                detected_objects[class_name] = []
            detected_objects[class_name].append(confidence)
            total_confidence += confidence
            object_count += 1
    
    if not detected_objects:
        return [], 0.0, 0.0
    
    # Ortalama güven oranı
    avg_confidence = total_confidence / object_count if object_count > 0 else 0.0
    
    # Nesne listesini oluştur (tekrar edenleri kaldır)
    unique_objects = list(detected_objects.keys())
    
    # İşlem süresi
    processing_time = time.time() - start_time
    
    return unique_objects, avg_confidence, processing_time

def process_yolo_results(result, image):
    """YOLOv5 sonuçlarını işle ve Türkçe açıklama oluştur"""
    
    if result.boxes is None or len(result.boxes) == 0:
        return "Görselde herhangi bir nesne tespit edilemedi."
    
    # Tespit edilen nesneleri grupla
    detected_objects = {}
    for box in result.boxes:
        if box.conf > 0.5:  # Güven eşiği
            class_id = int(box.cls[0])
            class_name = COCO_CLASSES[class_id]
            confidence = float(box.conf[0])
            
            if class_name not in detected_objects:
                detected_objects[class_name] = []
            detected_objects[class_name].append(confidence)
    
    if not detected_objects:
        return "Görselde güvenilir nesne tespit edilemedi."
    
    # Türkçe açıklama oluştur
    descriptions = []
    
    # Ana nesneleri belirle
    main_objects = []
    for obj_name, confidences in detected_objects.items():
        avg_conf = sum(confidences) / len(confidences)
        if avg_conf > 0.7:  # Yüksek güvenilirlik
            main_objects.append((obj_name, avg_conf))
    
    # Ana nesneleri güvenilirlik sırasına göre sırala
    main_objects.sort(key=lambda x: x[1], reverse=True)
    
    # Açıklama oluştur
    if len(main_objects) == 1:
        obj_name, conf = main_objects[0]
        descriptions.append(f"Görselde {translate_to_turkish(obj_name)} tespit edildi.")
    else:
        obj_names = [translate_to_turkish(obj[0]) for obj in main_objects[:3]]  # En fazla 3 nesne
        descriptions.append(f"Görselde {', '.join(obj_names)} tespit edildi.")
    
    # Güvenlik uyarıları
    safety_warnings = []
    for obj_name in detected_objects.keys():
        if obj_name in ['person', 'car', 'bicycle', 'motorcycle']:
            safety_warnings.append(f"Dikkat: {translate_to_turkish(obj_name)} var, güvenli mesafeyi koruyun.")
        elif obj_name in ['stairs', 'step', 'chair', 'table']:
            safety_warnings.append(f"Engel: {translate_to_turkish(obj_name)} tespit edildi.")
    
    if safety_warnings:
        descriptions.extend(safety_warnings)
    
    # Yön önerisi
    if len(main_objects) > 1:
        descriptions.append("Çevrenizde birden fazla nesne var, dikkatli hareket edin.")
    
    return " ".join(descriptions)

def translate_to_turkish(english_name):
    """İngilizce nesne isimlerini Türkçe'ye çevir"""
    translations = {
        'person': 'kişi',
        'car': 'araba',
        'bicycle': 'bisiklet',
        'motorcycle': 'motosiklet',
        'chair': 'sandalye',
        'table': 'masa',
        'bed': 'yatak',
        'couch': 'koltuk',
        'tv': 'televizyon',
        'laptop': 'dizüstü bilgisayar',
        'phone': 'telefon',
        'book': 'kitap',
        'cup': 'bardak',
        'bottle': 'şişe',
        'dog': 'köpek',
        'cat': 'kedi',
        'bird': 'kuş',
        'plant': 'bitki',
        'tree': 'ağaç',
        'flower': 'çiçek'
    }
    
    return translations.get(english_name, english_name)

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)


