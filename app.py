import base64
import csv
import os
from datetime import datetime
from io import BytesIO
from urllib.parse import quote
import logging

import cv2

try:
    cv2.setLogLevel(0)
except Exception:
    pass

import torch
import torch.nn as nn
from flask import Flask, jsonify, render_template, request, send_file, Response
from PIL import Image
from torchvision import models, transforms
from ultralytics import YOLO

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
SAVE_DIR = os.path.join(BASE_DIR, "skin_images")
os.makedirs(SAVE_DIR, exist_ok=True)

PRODUCT_DIR = os.path.join(BASE_DIR, "product")
CSV_PATH = os.path.join(BASE_DIR, "skin_diagnosis_results.csv")
QR_PATH = os.path.join(BASE_DIR, "qr.png")
MODEL_PATH = os.path.join(BASE_DIR, "skin_od_resnet18.pth")
ACNE_MODEL_PATH = os.path.join(BASE_DIR, "best.pt")
DEVICE = torch.device("cpu")

app = Flask(__name__)

camera = None
selected_camera_index = 0

od_model = None
od_classes = []
acne_model = None

od_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])

TEXTS = {
    "kr": {
        "window_title": "피부 진단 테스트",
        "app_title": "피부 진단 테스트",
        "start_title": "AI 피부 큐레이션",
        "start_subtitle": "스마트 문진과 AI 안면 정밀 스캔을 통해\n피부 타입에 맞는 맞춤 루틴을 추천드립니다.",
        "start_btn": "시작",
        "gender_title": "성별 선택",
        "gender_subtitle": "보다 정확한 피부 분석을 위해 선택해 주세요.",
        "male": "남자",
        "female": "여자",
        "camera_title": "피부 촬영 준비",
        "camera_wait": "3초 후 피부 촬영이 시작됩니다.",
        "camera_capture": "선명한 분석을 위해 피부 이미지를 총 3회 촬영합니다.",
        "capture_btn": "촬영",
        "diagnosis_title": "피부 분석 중",
        "diagnosis_guide": "촬영된 이미지와 문진 결과를 바탕으로\n피부 상태를 분석하고 있습니다.",
        "result_title": "피부 분석 결과",
        "gender": "성별",
        "skin_type": "피부 타입",
        "additional": "보조판정",
        "cosmetic_title": "추천 화장품",
        "purchase": "구매하기",
        "home": "처음으로",
        "camera_error": "카메라를 열 수 없습니다.",
        "questions": [
            {"q": "연령대를 선택해 주세요.", "a": ["10~20대", "30대", "40대 이상"]},
            {"q": "평소 피부가 가렵거나 따가운 느낌을\n얼마나 자주 경험하나요?", "a": ["거의 없음", "가끔 있음", "자주 있음"]},
            {"q": "세안 후 또는 일상 중 피부 속당김을\n어느 정도 느끼나요?", "a": ["거의 없음", "약간 있음", "강하게 있음"]},
            {"q": "평소 유분, 번들거림 또는\n트러블에 대한 고민이 있나요?", "a": ["거의 없음", "가끔 있음", "자주 있음"]},
            {"q": "미세먼지, 자외선, 건조한 환경 등에\n자주 노출되나요?", "a": ["거의 아님", "보통", "자주 노출됨"]}
        ],
    },
    "en": {
        "window_title": "Skin Diagnosis Test",
        "app_title": "SKIN DIAGNOSIS TEST",
        "start_title": "AI Skin Curation",
        "start_subtitle": "Personalized skincare recommendations\nthrough smart questionnaire and AI facial scan.",
        "start_btn": "Start",
        "gender_title": "Select Gender",
        "gender_subtitle": "Select this for a more accurate skin analysis.",
        "male": "Male",
        "female": "Female",
        "camera_title": "Skin Capture Ready",
        "camera_wait": "Skin capture will start in 3 seconds.",
        "camera_capture": "Skin images are taken a total of 3 times.",
        "capture_btn": "Capture",
        "diagnosis_title": "Analyzing Skin",
        "diagnosis_guide": "Analyzing your skin condition based on\ncaptured images and questionnaire results.",
        "result_title": "Skin Analysis Result",
        "gender": "Gender",
        "skin_type": "Skin Type",
        "additional": "Additional",
        "cosmetic_title": "Recommended Cosmetics",
        "purchase": "Purchase",
        "home": "Home",
        "camera_error": "Cannot open camera.",
        "questions": [
            {"q": "Select your age group.", "a": ["10s~20s", "30s", "40s+"]},
            {"q": "How often do you experience itching\nor stinging on your skin?", "a": ["Rarely", "Sometimes", "Often"]},
            {"q": "How much inner dryness do you feel\nafter cleansing or during the day?", "a": ["Rarely", "Mildly", "Strongly"]},
            {"q": "How often are oiliness, shine,\nor breakouts a concern for you?", "a": ["Rarely", "Sometimes", "Often"]},
            {"q": "How often is your skin exposed to\nfine dust, UV rays, or dry environments?", "a": ["Rarely", "Moderately", "Often"]}
        ],
    }
}


def open_camera():
    global camera, selected_camera_index

    if camera is not None and camera.isOpened():
        return camera

    camera_candidates = []

    if hasattr(cv2, "CAP_V4L2"):
        camera_candidates.append((selected_camera_index, cv2.CAP_V4L2))

    camera_candidates.append((selected_camera_index, None))

    for cam_index, backend in camera_candidates:
        if backend is None:
            temp_camera = cv2.VideoCapture(cam_index)
        else:
            temp_camera = cv2.VideoCapture(cam_index, backend)

        if temp_camera.isOpened():
            temp_camera.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
            temp_camera.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

            ret, frame = temp_camera.read()

            if ret:
                camera = temp_camera
                print("카메라 연결 성공:", cam_index, backend)
                return camera

        temp_camera.release()

    camera = None
    return None


def close_camera():
    global camera

    if camera is not None:
        camera.release()
        camera = None

def is_any_camera_connected():
    global camera

    if camera is not None and camera.isOpened():
        return True

    camera_candidates = []

    if hasattr(cv2, "CAP_V4L2"):
        camera_candidates.extend([
            (0, cv2.CAP_V4L2),
            (1, cv2.CAP_V4L2),
            (2, cv2.CAP_V4L2)
        ])

    camera_candidates.extend([
        (0, None),
        (1, None),
        (2, None)
    ])

    for cam_index, backend in camera_candidates:
        if backend is None:
            temp_camera = cv2.VideoCapture(cam_index)
        else:
            temp_camera = cv2.VideoCapture(cam_index, backend)

        if temp_camera.isOpened():
            temp_camera.release()
            return True

        temp_camera.release()

    return False


def load_od_model():
    global od_model, od_classes

    if not os.path.exists(MODEL_PATH):
        print("건성/지성 모델 파일 없음:", MODEL_PATH)
        return

    checkpoint = torch.load(MODEL_PATH, map_location=DEVICE)
    od_classes = checkpoint["classes"]

    model = models.resnet18(weights=None)
    num_features = model.fc.in_features
    model.fc = nn.Linear(num_features, len(od_classes))

    model.load_state_dict(checkpoint["model_state_dict"])
    model = model.to(DEVICE)
    model.eval()

    od_model = model
    print("건성/지성 모델 로드 완료:", MODEL_PATH)
    print("클래스:", od_classes)


def load_acne_model():
    global acne_model

    if not os.path.exists(ACNE_MODEL_PATH):
        print("여드름 모델 파일 없음:", ACNE_MODEL_PATH)
        return

    acne_model = YOLO(ACNE_MODEL_PATH)
    acne_model.to("cpu")
    print("여드름 모델 로드 완료:", ACNE_MODEL_PATH)


def decode_and_save_image(data_url, index):
    if "," in data_url:
        data_url = data_url.split(",", 1)[1]

    image_bytes = base64.b64decode(data_url)
    image = Image.open(BytesIO(image_bytes)).convert("RGB")

    file_name = datetime.now().strftime(f"skin_%Y%m%d_%H%M%S_%f_{index}") + ".jpg"
    image_path = os.path.join(SAVE_DIR, file_name)
    image.save(image_path, "JPEG", quality=95)

    return file_name, image_path


def predict_od_from_images(image_paths):
    if od_model is None:
        return None, 0.0

    if not image_paths:
        return None, 0.0

    prob_sum = torch.zeros(len(od_classes)).to(DEVICE)
    valid_count = 0

    with torch.no_grad():
        for image_path in image_paths:
            if not os.path.exists(image_path):
                continue

            image = Image.open(image_path).convert("RGB")
            image = od_transform(image).unsqueeze(0).to(DEVICE)

            output = od_model(image)
            probs = torch.softmax(output, dim=1)[0]

            prob_sum += probs
            valid_count += 1

    if valid_count == 0:
        return None, 0.0

    avg_probs = prob_sum / valid_count
    confidence, predicted = torch.max(avg_probs, 0)

    pred_class = od_classes[predicted.item()]
    confidence = confidence.item() * 100

    return pred_class, confidence


def predict_acne_from_images(image_paths, lang):
    if acne_model is None:
        return ""

    if not image_paths:
        return ""

    acne_found = False

    for image_path in image_paths:
        if not os.path.exists(image_path):
            continue

        results = acne_model(image_path, verbose=False, device="cpu")

        for result in results:
            if result.boxes is not None and len(result.boxes) > 0:
                acne_found = True

                boxed_image = result.plot()
                base_name, ext = os.path.splitext(image_path)
                boxed_image_path = base_name + "_1" + ext
                success, encoded_image = cv2.imencode(ext, boxed_image)

                if success:
                    encoded_image.tofile(boxed_image_path)

    if acne_found:
        return "있음" if lang == "kr" else "Present"

    return "없음" if lang == "kr" else "Absent"


def analyze_question_scores(question_answers):
    scores = {
        "Sensitivity": 0,
        "Oil": 0,
        "Dry": 0,
        "Early_Aging": 0,
        "Fixed_Aging": 0,
        "Healthy": 0
    }

    if len(question_answers) >= 1:
        age_index = question_answers[0]["answer_index"]
        scores["Early_Aging"] += [0, 2, 3][age_index]
        scores["Fixed_Aging"] += [0, 0, 2][age_index]

    if len(question_answers) >= 2:
        scores["Sensitivity"] += [0, 4, 8][question_answers[1]["answer_index"]]

    if len(question_answers) >= 3:
        inner_dryness_score = [0, 4, 8][question_answers[2]["answer_index"]]
        scores["Dry"] += inner_dryness_score

        if question_answers[2]["answer_index"] == 2:
            scores["Sensitivity"] += 2

    if len(question_answers) >= 4:
        scores["Oil"] += [0, 4, 8][question_answers[3]["answer_index"]]

    if len(question_answers) >= 5:
        urban_score = [0, 3, 5][question_answers[4]["answer_index"]]
        scores["Dry"] += urban_score
        scores["Early_Aging"] += urban_score

    return scores


def get_skin_type_desc(skin_type, lang):
    desc_map_kr = {
        "Sensitivity": "극민감 · 아토피성 장벽 붕괴 피부",
        "Oil": "지성 · 과피지 · 트러블성 피부",
        "Dry": "건성 · 속당김 · 도심형 생기 부족 피부",
        "Early_Aging": "초기 노화 · 탄력 저하 피부",
        "Fixed_Aging": "고정 노화 · 진피 치밀도 저하 피부",
        "Healthy": "건강 · 유수분 균형 피부"
    }

    desc_map_en = {
        "Sensitivity": "Extreme Sensitive & Atopic",
        "Oil": "Oily & Acne-Prone",
        "Dry": "Dehydrated & Dull Urban Skin",
        "Early_Aging": "Early Aging & Fatigue",
        "Fixed_Aging": "Advanced Aging & Dermal Loss",
        "Healthy": "Healthy & Perfectly Balanced"
    }

    desc_map = desc_map_kr if lang == "kr" else desc_map_en
    return desc_map.get(skin_type, skin_type)


def get_extra_condition(question_answers, lang):
    conditions = []

    if len(question_answers) >= 2 and question_answers[1]["answer_index"] == 2:
        conditions.append("가려움·따가움 높음" if lang == "kr" else "High itching/stinging")

    if len(question_answers) >= 3 and question_answers[2]["answer_index"] == 2:
        conditions.append("속당김 강함" if lang == "kr" else "Strong inner dryness")

    if len(question_answers) >= 4 and question_answers[3]["answer_index"] == 2:
        conditions.append("유분·트러블 높음" if lang == "kr" else "High oil/trouble")

    if len(question_answers) >= 5 and question_answers[4]["answer_index"] == 2:
        conditions.append("도심 유해환경 노출 높음" if lang == "kr" else "High urban exposure")

    return conditions


def get_recommended_cosmetics(skin_type, question_answers, lang):
    recommended_cosmetics = []

    if lang == "kr":
        if skin_type == "Sensitivity":
            recommended_cosmetics = [
                "테라피토 퓨어 pH 밸런싱 폼 클렌저",
                "테라피토 큐어 크림",
                "테라피토 인텐시브 큐어 오인트먼트",
                "테라피토 큐어 립케어"
            ]
        elif skin_type == "Oil":
            recommended_cosmetics = [
                "테라피토 퓨어 pH 밸런싱 폼 클렌저",
                "테라피토 큐어 로션",
                "테라피토 마일드 선크림"
            ]
        elif skin_type == "Dry":
            recommended_cosmetics = [
                "테라피토 크림 미스트",
                "릴렉사 리페어 앰플",
                "릴렉사 리페어 모이스트 크림",
                "에이벨 센 바이탈 마스크"
            ]
        elif skin_type == "Early_Aging":
            recommended_cosmetics = [
                "릴렉사 리페어 부스팅 토너",
                "릴렉사 리페어 크림",
                "에이벨 센 바이탈 마스크"
            ]
        elif skin_type == "Fixed_Aging":
            recommended_cosmetics = [
                "릴렉사 리페어 부스팅 토너",
                "릴렉사 리페어 앰플",
                "릴렉사 리페어 크림"
            ]
        elif skin_type == "Healthy":
            recommended_cosmetics = [
                "테라피토 퓨어 pH 밸런싱 폼 클렌저",
                "릴렉사 리페어 모이스트 크림",
                "테라피토 마일드 선크림"
            ]
    else:
        if skin_type == "Sensitivity":
            recommended_cosmetics = [
                "Theraphyto Pure pH Balancing Foam Cleanser",
                "Theraphyto Cure Cream",
                "Theraphyto Intensive-Cure Ointment",
                "Theraphyto Cure Lip Care"
            ]
        elif skin_type == "Oil":
            recommended_cosmetics = [
                "Theraphyto Pure pH Balancing Foam Cleanser",
                "Theraphyto Cure Lotion",
                "Theraphyto Mild Sun Cream"
            ]
        elif skin_type == "Dry":
            recommended_cosmetics = [
                "Theraphyto Cream Mist",
                "Relaxa Repair Ampoule",
                "Relaxa Repair Moist Cream",
                "Abel S-en Vital Mask"
            ]
        elif skin_type == "Early_Aging":
            recommended_cosmetics = [
                "Relaxa Repair Boosting Toner",
                "Relaxa Repair Cream",
                "Abel S-en Vital Mask"
            ]
        elif skin_type == "Fixed_Aging":
            recommended_cosmetics = [
                "Relaxa Repair Boosting Toner",
                "Relaxa Repair Ampoule",
                "Relaxa Repair Cream"
            ]
        elif skin_type == "Healthy":
            recommended_cosmetics = [
                "Theraphyto Pure pH Balancing Foam Cleanser",
                "Relaxa Repair Moist Cream",
                "Theraphyto Mild Sun Cream"
            ]

    if len(question_answers) >= 5:
        urban_exposure_answer = question_answers[4]["answer_index"]

        if urban_exposure_answer >= 1:
            suncream = "테라피토 마일드 선크림" if lang == "kr" else "Theraphyto Mild Sun Cream"

            if suncream not in recommended_cosmetics:
                recommended_cosmetics.append(suncream)

    return list(dict.fromkeys(recommended_cosmetics))[:4]


PRODUCT_IMAGE_MAP = {
    "Theraphyto Cure Lotion": "테라피토 큐어 로션",
    "Theraphyto Cure Cream": "테라피토 큐어 크림",
    "Theraphyto Pure pH Balancing Foam Cleanser": "테라피토 퓨어 pH 밸런싱 폼 클렌저",
    "Theraphyto Mild Sun Cream": "테라피토 마일드 선크림",
    "Theraphyto Intensive-Cure Ointment": "테라피토 인텐시브 큐어 오인트먼트",
    "Theraphyto Cream Mist": "테라피토 크림 미스트",
    "Relaxa Repair Boosting Toner": "릴렉사 리페어 부스팅 토너",
    "Relaxa Repair Ampoule": "릴렉사 리페어 앰플",
    "Relaxa Repair Moist Cream": "릴렉사 리페어 모이스트 크림",
    "Relaxa Repair Cream": "릴렉사 리페어 크림",
    "Abel S-en Vital Serum": "에이벨 센 바이탈 세럼",
    "Abel S-en Vital Cream": "에이벨 센 바이탈 크림",
    "Abel S-en Vital Mask": "에이벨 센 바이탈 마스크",
    "Abel S-en Vital Serum & Vital Cream": "에이벨 센 바이탈 세럼 & 바이탈 크림",

    "테라피토 큐어 로션": "테라피토 큐어 로션",
    "테라피토 큐어 크림": "테라피토 큐어 크림",
    "테라피토 퓨어 pH 밸런싱 폼 클렌저": "테라피토 퓨어 pH 밸런싱 폼 클렌저",
    "테라피토 마일드 선크림": "테라피토 마일드 선크림",
    "테라피토 인텐시브 큐어 오인트먼트": "테라피토 인텐시브 큐어 오인트먼트",
    "테라피토 크림 미스트": "테라피토 크림 미스트",
    "릴렉사 리페어 부스팅 토너": "릴렉사 리페어 부스팅 토너",
    "릴렉사 리페어 앰플": "릴렉사 리페어 앰플",
    "릴렉사 리페어 모이스트 크림": "릴렉사 리페어 모이스트 크림",
    "릴렉사 리페어 크림": "릴렉사 리페어 크림",
    "에이벨 센 바이탈 세럼": "에이벨 센 바이탈 세럼",
    "에이벨 센 바이탈 크림": "에이벨 센 바이탈 크림",
    "에이벨 센 바이탈 마스크": "에이벨 센 바이탈 마스크",
    "에이벨 센 바이탈 세럼 & 바이탈 크림": "에이벨 센 바이탈 세럼 & 바이탈 크림"
}


def find_product_image_path(image_key):
    for ext in [".png", ".jpg", ".jpeg", ".webp"]:
        image_path = os.path.join(PRODUCT_DIR, image_key + ext)

        if os.path.exists(image_path):
            return image_path

    return ""


def get_product_image_url(product_name):
    image_key = PRODUCT_IMAGE_MAP.get(product_name, "")

    if not image_key:
        return ""

    image_path = find_product_image_path(image_key)

    if image_path:
        return "/product-image/" + quote(image_key)

    return ""


def calculate_skin_type(gender, lang, question_answers, image_files, image_paths):
    pred_class, confidence = predict_od_from_images(image_paths)

    camera_scores = {
        "model_result": pred_class,
        "confidence": round(confidence, 2)
    }

    question_scores = analyze_question_scores(question_answers)

    if pred_class == "oil":
        question_scores["Oil"] += 5
        axis_od = "oil"
    elif pred_class == "dry":
        question_scores["Dry"] += 5
        axis_od = "dry"
    else:
        axis_od = ""

    acne_result = predict_acne_from_images(image_paths, lang)

    if acne_result == "있음" or acne_result == "Present":
        question_scores["Oil"] += 5

    issue_scores = {
        "Sensitivity": question_scores["Sensitivity"],
        "Oil": question_scores["Oil"],
        "Dry": question_scores["Dry"],
        "Early_Aging": question_scores["Early_Aging"],
        "Fixed_Aging": question_scores["Fixed_Aging"]
    }

    max_score = max(issue_scores.values())

    if max_score <= 3:
        skin_type = "Healthy"
    else:
        priority = [
            "Sensitivity",
            "Oil",
            "Dry",
            "Fixed_Aging",
            "Early_Aging"
        ]

        skin_type = max(
            priority,
            key=lambda key: (issue_scores[key], -priority.index(key))
        )

    skin_type_desc = get_skin_type_desc(skin_type, lang)
    extra_condition = get_extra_condition(question_answers, lang)

    if acne_result == "있음" or acne_result == "Present":
        acne_condition = "여드름성" if lang == "kr" else "Acne-prone"

        if acne_condition not in extra_condition:
            extra_condition.append(acne_condition)

    recommended_cosmetics = get_recommended_cosmetics(skin_type, question_answers, lang)

    save_result_to_csv(
        gender=gender,
        image_files=image_files,
        skin_type_desc=skin_type_desc,
        camera_scores=camera_scores,
        question_scores=question_scores,
        question_answers=question_answers,
        recommended_cosmetics=recommended_cosmetics
    )

    product_items = []

    for item in recommended_cosmetics:
        product_items.append({
            "name": item,
            "image_url": get_product_image_url(item)
        })

    print("건성/지성 모델 결과:", pred_class, round(confidence, 2), "%")
    print("문진 점수:", question_scores)
    print("피부 타입:", skin_type, skin_type_desc)
    print("추천 화장품 리스트:", recommended_cosmetics)

    return {
        "gender": gender,
        "skin_type": skin_type,
        "skin_type_desc": skin_type_desc,
        "extra_condition": extra_condition,
        "camera_scores": camera_scores,
        "question_scores": question_scores,
        "acne_result": acne_result,
        "recommended_cosmetics": recommended_cosmetics,
        "product_items": product_items
    }


def save_result_to_csv(
    gender,
    image_files,
    skin_type_desc,
    camera_scores,
    question_scores,
    question_answers,
    recommended_cosmetics
):
    answers = [""] * 5

    for i, item in enumerate(question_answers):
        if i < 5:
            answers[i] = item["answer"]

    row = {
        "datetime": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "gender": gender,
        "image_files": ",".join(image_files),
        "skin_type_desc": skin_type_desc,
        "camera_scores": str(camera_scores),
        "question_scores": str(question_scores),
        "q1_age_group": answers[0],
        "q2_itching_stinging": answers[1],
        "q3_inner_dryness": answers[2],
        "q4_oiliness_trouble": answers[3],
        "q5_urban_exposure": answers[4],
        "cosmetic_1": recommended_cosmetics[0] if len(recommended_cosmetics) > 0 else "",
        "cosmetic_2": recommended_cosmetics[1] if len(recommended_cosmetics) > 1 else "",
        "cosmetic_3": recommended_cosmetics[2] if len(recommended_cosmetics) > 2 else "",
        "cosmetic_4": recommended_cosmetics[3] if len(recommended_cosmetics) > 3 else ""
    }

    fieldnames = [
        "datetime",
        "gender",
        "image_files",
        "skin_type_desc",
        "camera_scores",
        "question_scores",
        "q1_age_group",
        "q2_itching_stinging",
        "q3_inner_dryness",
        "q4_oiliness_trouble",
        "q5_urban_exposure",
        "cosmetic_1",
        "cosmetic_2",
        "cosmetic_3",
        "cosmetic_4",
    ]

    file_exists = os.path.exists(CSV_PATH)

    with open(CSV_PATH, "a", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)

        if not file_exists:
            writer.writeheader()

        writer.writerow(row)

    print("CSV 저장 완료:", CSV_PATH)


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/texts")
def get_texts():
    return jsonify(TEXTS)


@app.route("/diagnose", methods=["POST"])
def diagnose():
    data = request.get_json()

    lang = data.get("lang", "kr")
    gender = data.get("gender", "")
    answers = data.get("answers", [])
    images = data.get("images", [])

    image_files = []
    image_paths = []

    for idx, image_data in enumerate(images):
        file_name, image_path = decode_and_save_image(image_data, idx + 1)
        image_files.append(file_name)
        image_paths.append(image_path)

    result = calculate_skin_type(
        gender=gender,
        lang=lang,
        question_answers=answers,
        image_files=image_files,
        image_paths=image_paths
    )

    return jsonify(result)


@app.route("/product-image/<path:image_key>")
def product_image(image_key):
    image_path = find_product_image_path(image_key)

    if not image_path:
        return "", 404

    return send_file(image_path)


@app.route("/qr")
def qr_image():
    if not os.path.exists(QR_PATH):
        return "", 404

    return send_file(QR_PATH)


@app.route("/camera-status")
def camera_status():
    connected = is_any_camera_connected()
    return jsonify({"connected": connected})


@app.route("/camera-start", methods=["POST"])
def camera_start():
    global selected_camera_index

    data = request.get_json(silent=True) or {}
    selected_camera_index = int(data.get("camera_index", 0))

    close_camera()
    cap = open_camera()

    if cap is None or not cap.isOpened():
        return jsonify({"ok": False, "message": "Cannot open camera"}), 500

    return jsonify({"ok": True})


@app.route("/camera-frame")
def camera_frame():
    cap = open_camera()

    if cap is None or not cap.isOpened():
        return jsonify({"ok": False, "message": "Cannot open camera"}), 500

    ret, frame = cap.read()

    if not ret:
        return jsonify({"ok": False, "message": "Cannot read camera frame"}), 500

    success, encoded_image = cv2.imencode(".jpg", frame)

    if not success:
        return jsonify({"ok": False, "message": "Cannot encode camera frame"}), 500

    return Response(
        encoded_image.tobytes(),
        mimetype="image/jpeg",
        headers={"Cache-Control": "no-store, no-cache, must-revalidate, max-age=0"}
    )


@app.route("/camera-capture", methods=["POST"])
def camera_capture():
    cap = open_camera()

    if cap is None or not cap.isOpened():
        return jsonify({"ok": False, "message": "Cannot open camera"}), 500

    ret, frame = cap.read()

    if not ret:
        return jsonify({"ok": False, "message": "Cannot read camera frame"}), 500

    success, encoded_image = cv2.imencode(".jpg", frame)

    if not success:
        return jsonify({"ok": False, "message": "Cannot encode camera frame"}), 500

    image_base64 = base64.b64encode(encoded_image.tobytes()).decode("utf-8")

    return jsonify({
        "ok": True,
        "image": "data:image/jpeg;base64," + image_base64
    })


@app.route("/camera-stop", methods=["POST"])
def camera_stop():
    close_camera()
    return jsonify({"ok": True})


if __name__ == "__main__":
    load_od_model()
    load_acne_model()
    app.run(host="0.0.0.0", port=5000, debug=False)
