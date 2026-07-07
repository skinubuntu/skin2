let texts = {};
let lang = "kr";
let gender = "";
let questionIndex = 0;
let answers = [];
let captureImages = [];
let stream = null;
let cameraTimer = null;
let lastResult = null;

const screen = document.getElementById("screen");
const headerTitle = document.getElementById("headerTitle");
const stepLabel = document.getElementById("stepLabel");
const settingBtn = document.getElementById("settingBtn");

let selectedCameraIndex = localStorage.getItem("selectedCameraIndex") || "0";
let settingCameraTimer = null;
let previousScreenFunction = null;
let cameraStatusTimer = null;
let cameraStatusChecking = false;

async function init() {
    const response = await fetch("/texts");
    texts = await response.json();
    showStartScreen();
    startCameraConnectionWatch();
}

function t(key) {
    return texts[lang][key];
}

function setHeader(step = null) {
    headerTitle.innerHTML = `
        <img src="/static/logo_1.png" alt="THERAPHYTO ABEL" class="brand-main">
    `;

    headerTitle.style.cursor = "pointer";
    headerTitle.onclick = () => {
        stopCamera();

        gender = "";
        questionIndex = 0;
        answers = [];
        captureImages = [];
        lastResult = null;

        showStartScreen();
    };

    if (step) {
        stepLabel.textContent = `STEP ${step}/5`;
        stepLabel.style.display = "inline-flex";
    } else {
        stepLabel.textContent = "";
        stepLabel.style.display = "none";
    }

    settingBtn.onclick = openSettingScreen;

    document.title = t("window_title");
}

function openSettingScreen() {
    previousScreenFunction = getCurrentScreenFunction();
    showSettingScreen();
}

function getCurrentScreenFunction() {
    if (screen.classList.contains("screen-start")) {
        return showStartScreen;
    }

    if (screen.classList.contains("screen-question")) {
        return showQuestionScreen;
    }

    if (screen.classList.contains("screen-camera-ready")) {
        return showCameraReadyScreen;
    }

    if (screen.classList.contains("screen-camera")) {
        return showCameraScreen;
    }

    if (screen.classList.contains("screen-diagnosis")) {
        return runSkinDiagnosis;
    }

    if (screen.classList.contains("screen-result")) {
        return showResultScreen;
    }

    if (screen.classList.contains("screen-product")) {
        return showProductScreen;
    }

    if (screen.classList.contains("screen-qr")) {
        return showQrScreen;
    }

    return showStartScreen;
}

function returnToPreviousScreen() {
    stopCamera();

    if (previousScreenFunction) {
        const targetScreen = previousScreenFunction;
        previousScreenFunction = null;
        targetScreen();
    } else {
        showStartScreen();
    }
}

async function showSettingScreen() {
    stopCamera();
    clearScreen();
    screen.classList.add("screen-setting");
    setHeader();

    const box = centerBox();

    const title = document.createElement("h1");
    title.className = "title";
    title.textContent = lang === "kr" ? "카메라 설정" : "Camera Settings";
    box.appendChild(title);

    const subtitle = document.createElement("p");
    subtitle.className = "subtitle";
    subtitle.textContent = lang === "kr"
        ? "사용할 카메라를 선택하고 화면을 확인해 주세요."
        : "Select a camera and check the preview.";
    box.appendChild(subtitle);

    const cameraBox = document.createElement("div");
    cameraBox.className = "camera-box setting-camera-box";

    const preview = document.createElement("img");
    preview.className = "camera-preview";
    preview.alt = "camera preview";
    cameraBox.appendChild(preview);

    box.appendChild(cameraBox);

    const list = document.createElement("div");
    list.className = "setting-camera-row";

    ["0", "1", "2"].forEach(index => {
        const label = selectedCameraIndex === index
            ? (lang === "kr" ? `카메라 ${index} ✓` : `Camera ${index} ✓`)
            : (lang === "kr" ? `카메라 ${index}` : `Camera ${index}`);

        list.appendChild(button(label, () => {
            selectedCameraIndex = index;
            localStorage.setItem("selectedCameraIndex", selectedCameraIndex);
            showSettingScreen();
        }, "small"));
    });

    box.appendChild(list);

    const settingButtonRow = document.createElement("div");
    settingButtonRow.className = "setting-bottom-row";

    settingButtonRow.appendChild(button(
        lang === "kr" ? "처음으로" : "Home",
        showStartScreen,
        "fit"
    ));

    settingButtonRow.appendChild(button(
        lang === "kr" ? "기존 화면으로" : "Back",
        returnToPreviousScreen,
        "fit"
    ));

    box.appendChild(settingButtonRow);

    await startSettingCameraPreview(preview);
}

async function startSettingCameraPreview(preview) {
    if (settingCameraTimer) {
        clearInterval(settingCameraTimer);
        settingCameraTimer = null;
    }

    try {
        const response = await fetch("/camera-start", {
            method: "POST",
            headers: {"Content-Type": "application/json"},
            body: JSON.stringify({
                camera_index: selectedCameraIndex
            })
        });

        if (!response.ok) {
            throw new Error("camera start failed");
        }

        preview.src = `/camera-frame?t=${Date.now()}`;

        settingCameraTimer = setInterval(() => {
            preview.src = `/camera-frame?t=${Date.now()}`;
        }, 120);
    } catch (error) {
        console.error(error);
        preview.alt = lang === "kr" ? "카메라 화면 없음" : "No camera preview";
    }
}

function startCameraConnectionWatch() {
    checkCameraConnection();

    if (cameraStatusTimer) {
        clearInterval(cameraStatusTimer);
        cameraStatusTimer = null;
    }

    cameraStatusTimer = setInterval(() => {
        checkCameraConnection();
    }, 300);
}

async function checkCameraConnection() {
    if (cameraStatusChecking) {
        return;
    }

    cameraStatusChecking = true;

    try {
        const response = await fetch(`/camera-status?t=${Date.now()}`, {
            cache: "no-store"
        });

        if (!response.ok) {
            throw new Error("camera status failed");
        }

        const data = await response.json();

        if (data.connected) {
            hideCameraWarning();
        } else {
            showCameraWarning();
        }
    } catch (error) {
        showCameraWarning();
    } finally {
        cameraStatusChecking = false;
    }
}

function showCameraWarning() {
    document.body.classList.add("camera-blocked");

    let warning = document.getElementById("cameraWarning");

    if (!warning) {
        warning = document.createElement("div");
        warning.id = "cameraWarning";
        warning.className = "camera-warning";
        document.body.appendChild(warning);
    }

    warning.textContent = lang === "kr"
        ? "⚠ 카메라 연결 필요"
        : "⚠ Camera connection required";

    warning.style.display = "flex";
}

function hideCameraWarning() {
    document.body.classList.remove("camera-blocked");

    const warning = document.getElementById("cameraWarning");

    if (warning) {
        warning.style.display = "none";
    }
}

function setLanguage(selectedLang) {
    lang = selectedLang;
    showStartScreen();
}

function button(text, onClick, className = "") {
    const btn = document.createElement("button");
    btn.className = `button ${className}`;
    btn.textContent = text;
    btn.onclick = onClick;
    return btn;
}

function clearScreen() {
    screen.innerHTML = "";
    screen.className = "screen";
}

function centerBox() {
    const box = document.createElement("div");
    box.className = "center";
    screen.appendChild(box);
    return box;
}

function showStartScreen() {
    stopCamera();
    clearScreen();
    screen.classList.add("screen-start");
    setHeader();

    const box = centerBox();

    const title = document.createElement("h1");
    title.className = "title";
    title.textContent = t("start_title");
    box.appendChild(title);

    const subtitle = document.createElement("p");
    subtitle.className = "subtitle";
    subtitle.textContent = t("start_subtitle");
    box.appendChild(subtitle);

    const startBtn = button(t("start_btn"), startDiagnosis);
    startBtn.classList.add("start-button");
    box.appendChild(startBtn);

    const langRow = document.createElement("div");
    langRow.className = "lang-row";
    langRow.style.marginTop = "18px";
    langRow.appendChild(button("한국어", () => setLanguage("kr"), "small"));
    langRow.appendChild(button("English", () => setLanguage("en"), "small"));
    box.appendChild(langRow);
}

function showGenderScreen() {
    clearScreen();
    screen.classList.add("screen-gender");
    setHeader(1);

    const box = centerBox();

    const title = document.createElement("h1");
    title.className = "title";
    title.textContent = t("gender_title");
    box.appendChild(title);

    const subtitle = document.createElement("p");
    subtitle.className = "subtitle";
    subtitle.textContent = t("gender_subtitle");
    box.appendChild(subtitle);

    const row = document.createElement("div");
    row.className = "gender-row";
    row.appendChild(button(t("male"), () => selectGender(t("male"))));
    row.appendChild(button(t("female"), () => selectGender(t("female"))));
    box.appendChild(row);
}

function startDiagnosis() {
    gender = "";
    questionIndex = -1;
    answers = [];
    captureImages = [];
    lastResult = null;
    showQuestionScreen();
}

function selectGender(selectedGender) {
    gender = selectedGender;
    questionIndex = 0;
    showQuestionScreen();
}

function showQuestionScreen() {
    clearScreen();
    screen.classList.add("screen-question");
    setHeader(1);

    const box = centerBox();
    const questions = texts[lang].questions;

    let currentQuestion = null;
    let progressText = "";

    if (questionIndex === -1) {
        currentQuestion = {
            q: lang === "kr" ? "성별을 선택해 주세요." : "Select your gender.",
            a: [t("male"), t("female")]
        };
        progressText = `1/${questions.length + 1}`;
    } else {
        currentQuestion = questions[questionIndex];
        progressText = `${questionIndex + 2}/${questions.length + 1}`;
    }

    const progress = document.createElement("div");
    progress.className = "progress";
    progress.textContent = progressText;
    box.appendChild(progress);

    const question = document.createElement("div");
    question.className = "question";
    question.textContent = currentQuestion.q;
    box.appendChild(question);

    const list = document.createElement("div");
    list.className = "answer-list";

    currentQuestion.a.forEach((answer, index) => {
        if (questionIndex === -1) {
            list.appendChild(button(answer, () => selectGender(answer)));
        } else {
            list.appendChild(button(answer, () => selectQuestionAnswer(answer, index)));
        }
    });

    if (questionIndex === -1) {
        const spacer = button("", () => {}, "spacer");
        spacer.disabled = true;
        list.appendChild(spacer);
    }

    box.appendChild(list);
}

function selectQuestionAnswer(answer, answerIndex) {
    const currentQuestion = texts[lang].questions[questionIndex];

    answers.push({
        question: currentQuestion.q,
        answer: answer,
        answer_index: answerIndex
    });

    questionIndex += 1;

    if (questionIndex < texts[lang].questions.length) {
        showQuestionScreen();
    } else {
        showCameraReadyScreen();
    }
}

function showCameraReadyScreen() {
    clearScreen();
    screen.classList.add("screen-camera-ready");
    setHeader(2);

    const box = centerBox();

    const title = document.createElement("h1");
    title.className = "title";
    title.textContent = t("camera_title");
    box.appendChild(title);

    const subtitle = document.createElement("p");
    subtitle.className = "subtitle";
    subtitle.textContent = t("camera_wait");
    box.appendChild(subtitle);

    const countdown = document.createElement("div");
    countdown.className = "countdown";
    box.appendChild(countdown);

    let count = 3;
    countdown.textContent = count;

    const timer = setInterval(() => {
        count -= 1;
        countdown.textContent = count;

        if (count <= 0) {
            clearInterval(timer);
            showCameraScreen();
        }
    }, 1000);
}

async function showCameraScreen() {
    clearScreen();
    screen.classList.add("screen-camera");
    setHeader(2);

    const box = centerBox();

    const guide = document.createElement("div");
    guide.className = "question";
    guide.style.height = "auto";
    guide.classList.add("camera-guide");
    guide.textContent = t("camera_capture");
    box.appendChild(guide);

    const cameraBox = document.createElement("div");
    cameraBox.className = "camera-box";

    const preview = document.createElement("img");
    preview.className = "camera-preview";
    preview.alt = "camera preview";
    cameraBox.appendChild(preview);

    const crossH = document.createElement("div");
    crossH.className = "cross-h";
    cameraBox.appendChild(crossH);

    const crossV = document.createElement("div");
    crossV.className = "cross-v";
    cameraBox.appendChild(crossV);

    const countLabel = document.createElement("div");
    countLabel.className = "capture-count";
    countLabel.textContent = "0/3";
    cameraBox.appendChild(countLabel);

    box.appendChild(cameraBox);

    const captureBtn = button(t("capture_btn"), () => captureImage(countLabel, captureBtn), "fit");
    box.appendChild(captureBtn);

    try {
        const response = await fetch("/camera-start", {
            method: "POST",
            headers: {"Content-Type": "application/json"},
            body: JSON.stringify({
                camera_index: selectedCameraIndex
            })
        });

        if (!response.ok) {
            throw new Error("camera start failed");
        }

        updateCameraPreview(preview);

        cameraTimer = setInterval(() => {
            updateCameraPreview(preview);
        }, 120);
    } catch (error) {
        console.error(error);

        guide.textContent = lang === "kr"
            ? "카메라를 열 수 없습니다.\n카메라 연결을 확인해 주세요."
            : "Cannot open camera.\nCheck camera connection.";

        captureBtn.classList.add("disabled");
    }
}

function updateCameraPreview(preview) {
    preview.src = `/camera-frame?t=${Date.now()}`;
}

async function captureImage(countLabel, captureBtn) {
    try {
        const response = await fetch("/camera-capture", {method: "POST"});

        if (!response.ok) {
            throw new Error("camera capture failed");
        }

        const data = await response.json();

        if (!data.ok || !data.image) {
            throw new Error("camera image empty");
        }

        captureImages.push(data.image);
        countLabel.textContent = `${captureImages.length}/3`;

        if (captureImages.length >= 3) {
            captureBtn.classList.add("disabled");
            stopCamera();

            setTimeout(() => {
                runSkinDiagnosis();
            }, 500);
        }
    } catch (error) {
        console.error(error);
    }
}

function stopCamera() {
    if (cameraTimer) {
        clearInterval(cameraTimer);
        cameraTimer = null;
    }

    if (settingCameraTimer) {
        clearInterval(settingCameraTimer);
        settingCameraTimer = null;
    }

    if (stream) {
        stream.getTracks().forEach(track => track.stop());
        stream = null;
    }

    fetch("/camera-stop", {method: "POST"}).catch(() => {});
}

async function runSkinDiagnosis() {
    clearScreen();
    screen.classList.add("screen-diagnosis");
    setHeader(3);

    const box = centerBox();

    const title = document.createElement("h1");
    title.className = "title";
    title.textContent = t("diagnosis_title");
    box.appendChild(title);

    const guide = document.createElement("p");
    guide.className = "subtitle";
    guide.textContent = t("diagnosis_guide");
    box.appendChild(guide);

    const loading = document.createElement("div");
    loading.className = "loading";
    loading.textContent = "●  ●  ●";
    box.appendChild(loading);

    try {
        const response = await fetch("/diagnose", {
            method: "POST",
            headers: {"Content-Type": "application/json"},
            body: JSON.stringify({
                lang: lang,
                gender: gender,
                answers: answers,
                images: captureImages
            })
        });

        lastResult = await response.json();

        setTimeout(() => {
            showResultScreen();
        }, 1000);
    } catch (error) {
        console.error(error);
        loading.textContent = "ERROR";
    }
}

function showResultScreen() {
    clearScreen();
    screen.classList.add("screen-result");
    setHeader(3);

    const box = centerBox();

    const title = document.createElement("h1");
    title.className = "title";
    title.textContent = t("result_title");
    box.appendChild(title);

    const resultBox = document.createElement("div");
    resultBox.className = "result-box";

    const table = document.createElement("div");
    table.className = "result-table";

    addResultRow(table, `${t("gender")}:`, lastResult.gender);
    addResultRow(table, `${t("skin_type")}:`, lastResult.skin_type_desc);

    if (lastResult.extra_condition && lastResult.extra_condition.length > 0) {
        addResultRow(table, `${t("additional")}:`, lastResult.extra_condition.join(", "));
    }

    resultBox.appendChild(table);
    box.appendChild(resultBox);

    box.appendChild(button(
        lang === "kr" ? "추천 화장품 보기" : "Recommended Products",
        showProductScreen,
        "fit"
    ));
}

function addResultRow(table, labelText, valueText) {
    const label = document.createElement("div");
    label.className = "label";
    label.textContent = labelText;

    const value = document.createElement("div");
    value.className = "value";
    value.textContent = valueText;

    table.appendChild(label);
    table.appendChild(value);
}

function showProductScreen() {
    clearScreen();
    screen.classList.add("screen-product");
    setHeader(4);

    const box = centerBox();

    const title = document.createElement("h1");
    title.className = "title";
    title.textContent = t("cosmetic_title");
    box.appendChild(title);

    const productBox = document.createElement("div");
    productBox.className = "product-box";

    const list = document.createElement("div");
    list.className = "product-list";

    if (lastResult.product_items.length === 3) {
        list.classList.add("product-list-3");
    }

    lastResult.product_items.forEach(item => {
        const itemBox = document.createElement("div");
        itemBox.className = "product-item";

        const imageBox = document.createElement("div");
        imageBox.className = "product-image";

        const placeholder = document.createElement("div");
        placeholder.className = "image-placeholder";
        placeholder.textContent = lang === "kr" ? "이미지 준비중" : "Image pending";

        if (item.image_url) {
            const img = document.createElement("img");
            img.src = item.image_url;
            img.alt = item.name;
            img.onerror = () => {
                img.remove();
                imageBox.appendChild(placeholder);
            };
            imageBox.appendChild(img);
        } else {
            imageBox.appendChild(placeholder);
        }

        const name = document.createElement("div");
        name.className = "product-name";
        name.textContent = item.name;

        itemBox.appendChild(imageBox);
        itemBox.appendChild(name);
        list.appendChild(itemBox);
    });

    productBox.appendChild(list);
    box.appendChild(productBox);

    box.appendChild(button(t("purchase"), showQrScreen, "fit"));
}

function showQrScreen() {
    clearScreen();
    screen.classList.add("screen-qr");
    setHeader(5);

    const box = centerBox();

    const title = document.createElement("h1");
    title.className = "title";
    title.textContent = t("purchase");
    box.appendChild(title);

    const qr = document.createElement("img");
    qr.className = "qr-image";
    qr.src = "/qr";
    qr.alt = "QR";
    box.appendChild(qr);

    box.appendChild(button(t("home"), showStartScreen, "fit"));
}

init();
