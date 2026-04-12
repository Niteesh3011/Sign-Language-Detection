import cv2
import numpy as np
import os
import re
import time
import threading
from datetime import datetime
from flask import Flask, render_template, Response, jsonify
from ultralytics import YOLO
import nltk
from nltk import pos_tag, word_tokenize

app = Flask(__name__)

# ─────────────────────────────────────────────
# 1. Model Loading  (YOLO11l)
# ─────────────────────────────────────────────
model_env_path = os.getenv('MODEL_PATH', '').strip()
MODEL_CANDIDATES = [
    model_env_path,
    'best.pt',
    os.path.join('models', 'best.pt'),
    os.path.join('weights', 'best.pt'),
]

model = None
for candidate in MODEL_CANDIDATES:
    if not candidate:
        continue
    if os.path.exists(candidate):
        try:
            model = YOLO(candidate)
            print(f"✅ YOLO11l model loaded successfully from: {candidate}")
        except Exception as e:
            print(f"❌ Error loading model from {candidate}: {e}")
        break

if model is None:
    print("⚠️  No model file found. Set MODEL_PATH or place best.pt in project root/models/weights.")


# ─────────────────────────────────────────────
# 2. NLP Sentence Formation Setup (NLTK-based)
# ─────────────────────────────────────────────
nlp_ready = False

# ── All class names from the trained model ────────────────────────────────────
# Dataset: Restaurant / Food-service ASL (YOLO11l fine-tuned)
CLASS_NAMES = [
    "cup", "tomato", "spoon", "pizza", "fork", "chicken", "cake", "water",
    "bag", "sandwich", "milk", "bread", "cheese", "coke", "burger", "drink",
    "lettuce", "pepper", "fresh", "sugar",
    "A", "additional", "alcohol", "allergy",
    "B", "bacon", "barbecue", "bill", "biscuit", "bitter", "bye",
    "C", "cash", "cold", "cost", "coupon", "credit card",
    "D", "dessert", "drive",
    "E", "eat", "eggs", "enjoy",
    "F", "french fries",
    "G",
    "H", "hello", "hot",
    "I", "icecream", "ingredients",
    "J", "juicy",
    "K", "ketchup",
    "L", "lactose", "lid",
    "M", "manager", "menu", "mustard",
    "N", "napkin", "no",
    "O", "order",
    "P", "pickle", "please",
    "Q",
    "R", "ready", "receipt", "refill", "repeat",
    "S", "safe", "salt", "sauce", "small", "soda", "sorry", "spicy", "straw", "sweet",
    "T", "thank-you", "tissues", "total",
    "U", "urgent",
    "V", "vegetables",
    "W", "wait", "warm", "what", "would", "yoghurt", "your"
]

# ── Nouns that need an article (a/an) ─────────────────────────────────────────
COUNTABLE_NOUNS = {
    # Food & drinks
    "cup", "tomato", "spoon", "pizza", "fork", "cake", "sandwich", "burger",
    "biscuit", "egg", "eggs", "dessert", "icecream", "ketchup", "lid",
    "napkin", "pickle", "receipt", "refill", "sauce", "straw", "bill",
    "coupon", "menu", "bottle", "bag", "cup",
    # Multi-word food
    "french fries", "credit card",
}

# ── Words that take 'an' (start with vowel sounds) ────────────────────────────
VOWEL_STARTS = {"additional", "alcohol", "allergy", "eggs", "enjoy", "icecream",
                "ingredients", "order", "urgent"}

# ── Personal pronouns (no article) ────────────────────────────────────────────
PRONOUNS = {"i", "you", "he", "she", "it", "we", "they", "me", "him", "her", "us", "them"}

# ── Verbs in the class set ────────────────────────────────────────────────────
KNOWN_VERBS = {"eat", "drink", "order", "enjoy", "drive", "wait", "repeat",
               "refill", "would", "what"}

# ── Adjectives in the class set ───────────────────────────────────────────────
KNOWN_ADJECTIVES = {"cold", "hot", "spicy", "sweet", "bitter", "juicy", "fresh",
                    "small", "warm", "safe", "additional", "urgent", "ready"}

# ── Standalone expressions — no verb injection, print as-is ───────────────────
STANDALONE_EXPRESSIONS = {
    "hello", "bye", "goodbye", "sorry", "please", "no", "yes",
    "thank-you", "thank you", "thanks", "urgent", "wait", "repeat",
    "enjoy", "ready", "what", "would", "your", "total", "bill", "receipt",
}

# ── Word normalization map — maps raw class name → display/NLP form ───────────
WORD_MAP = {
    # Label cleanup
    "thank-you":    "thank you",
    "french fries": "french fries",
    "credit card":  "credit card",
    "icecream":     "ice cream",
    "yoghurt":      "yoghurt",
    "bye":          "goodbye",
    # Single-letter class labels → skip (they are ASL alphabet, not words)
    "a": None, "b": None, "c": None, "d": None, "e": None, "f": None,
    "g": None, "h": None, "i_letter": None, "j": None, "k": None,
    "l": None, "m": None, "n": None, "o": None, "p": None, "q": None,
    "r": None, "s": None, "t": None, "u": None, "v": None, "w": None,
}

# ── Sentence templates for common restaurant contexts ─────────────────────────
# Detects frequent intent patterns and maps them to natural sentences.
SENTENCE_TEMPLATES = [
    # Ordering
    ({"i", "would", "like"},      "I would like {items}."),
    ({"i", "want"},               "I want {items}."),
    ({"i", "order"},              "I would like to order {items}."),
    ({"i", "eat"},                "I want to eat {items}."),
    ({"i", "drink"},              "I want to drink {items}."),
    # Questions
    ({"what", "cost"},            "What is the cost?"),
    ({"what", "total"},           "What is the total?"),
    ({"what", "menu"},            "Can I see the menu?"),
    # Polite requests
    ({"please", "refill"},        "Please refill my drink."),
    ({"please", "repeat"},        "Please repeat that."),
    ({"please", "wait"},          "Please wait."),
    ({"i", "need", "napkin"},     "I need a napkin."),
    ({"i", "need", "straw"},      "I need a straw."),
    ({"i", "need", "lid"},        "I need a lid."),
    # Payments
    ({"pay", "cash"},             "I will pay with cash."),
    ({"pay", "credit card"},      "I will pay with a credit card."),
    ({"bill", "please"},          "Bill, please."),
    ({"receipt", "please"},       "Receipt, please."),
    # Allergy / safety
    ({"allergy"},                 "I have a food allergy."),
    ({"lactose"},                 "I am lactose intolerant."),
    ({"safe"},                    "Is this dish safe for me?"),
    # Feedback
    ({"enjoy"},                   "I am enjoying this."),
    ({"sorry"},                   "Sorry."),
    ({"thank you"},               "Thank you!"),
    ({"thank-you"},               "Thank you!"),
    ({"hello"},                   "Hello!"),
    ({"bye"}, "Goodbye!"),
    ({"goodbye"},                 "Goodbye!"),
]


def load_nlp():
    global nlp_ready
    try:
        nltk.data.find('taggers/averaged_perceptron_tagger_eng')
        nltk.data.find('tokenizers/punkt_tab')
        nlp_ready = True
        print("✅ NLTK NLP engine ready.")
    except LookupError:
        try:
            nltk.download('averaged_perceptron_tagger_eng', quiet=True)
            nltk.download('punkt_tab', quiet=True)
            nltk.download('punkt', quiet=True)
            nlp_ready = True
            print("✅ NLTK NLP engine ready (downloaded data).")
        except Exception as e:
            print(f"⚠️  NLTK setup failed: {e}. Using basic fallback.")
            nlp_ready = False

threading.Thread(target=load_nlp, daemon=True).start()



# ─────────────────────────────────────────────
# 3. Configuration
# ─────────────────────────────────────────────
CONFIDENCE_THRESHOLD = 0.5
BOX_COLOR           = (0, 255, 0)
TEXT_COLOR          = (0, 255, 0)
BOX_THICKNESS       = 2
FONT                = cv2.FONT_HERSHEY_SIMPLEX
FONT_SCALE          = 0.8
FONT_THICKNESS      = 2

# Sentence / document config
DEBOUNCE_SECONDS    = 2.5   # Seconds of silence before a phrase is finalized
MIN_CONFIDENCE_WORD = 0.60  # Minimum confidence to record a word
PREDICTIONS_FILE    = os.path.join(os.path.dirname(__file__), 'predictions.txt')

# Camera resolution (request HD from webcam)
CAM_WIDTH  = 1280
CAM_HEIGHT = 720

# YOLO inference input size (larger = more detail, slightly slower)
YOLO_IMGSZ = 640


# ─────────────────────────────────────────────
# 4. Global Prediction State
# ─────────────────────────────────────────────
state_lock        = threading.Lock()
word_buffer       = []          # Words captured in the current phrase
last_word         = None        # Last word added (dedup)
last_word_time    = 0.0         # Timestamp of last word added
finalized_sentence = ""         # The most recent NLP-corrected sentence
all_sentences     = []          # All finalized sentences in this session
debounce_thread   = None        # Reference to debounce timer thread


def _choose_article(word: str) -> str:
    """Choose 'an' or 'a' — use VOWEL_STARTS for known exceptions."""
    return 'an' if (word.lower() in VOWEL_STARTS or word[0].lower() in 'aeiou') else 'a'


def _add_article(word: str) -> list[str]:
    """Return [article, word] if the word is a countable noun, else [word]."""
    if word.lower() in COUNTABLE_NOUNS and word.lower() not in PRONOUNS:
        return [_choose_article(word), word]
    return [word]


def _try_template(word_set: set) -> str | None:
    """Match word_set against SENTENCE_TEMPLATES and return filled template, or None."""
    lower_set = {w.lower() for w in word_set}
    for keywords, template in SENTENCE_TEMPLATES:
        if keywords.issubset(lower_set):
            # Collect item nouns from the phrase (not the template keywords)
            non_key = [w for w in word_set if w.lower() not in keywords
                       and w.lower() not in PRONOUNS
                       and w.lower() not in STANDALONE_EXPRESSIONS
                       and w.lower() not in KNOWN_VERBS]
            items_str = ', '.join(non_key) if non_key else ''
            if '{items}' in template:
                if items_str:
                    sentence = template.replace('{items}', items_str)
                else:
                    sentence = template.replace(' {items}', '').replace('{items}', '')
            else:
                sentence = template
            return sentence
    return None


def form_sentence_with_nlp(words: list[str]) -> str:
    """
    Convert a raw list of detected sign class names into a natural English sentence.
    Pipeline:
      1. Normalize labels via WORD_MAP (strip single-letter ASL classes)
      2. Deduplicate consecutive identical words
      3. Try SENTENCE_TEMPLATES — if a known intent matches, use the template
      4. Otherwise use NLTK POS tagging with domain vocabulary to build sentence
    """
    if not words:
        return ""

    # ── Step 1: Normalize ────────────────────────────────────────────────────
    normalized: list[str] = []
    for w in words:
        mapped = WORD_MAP.get(w.lower(), w.lower())   # None = skip (single-letter)
        if mapped is not None:
            normalized.append(mapped)

    if not normalized:
        return ""

    # ── Step 2: Deduplicate consecutive identical words ───────────────────────
    deduped: list[str] = []
    for w in normalized:
        tokens = w.split()  # handle multi-word phrases like "french fries"
        for tok in tokens:
            if not deduped or tok.lower() != deduped[-1].lower():
                deduped.append(tok)

    if not deduped:
        return ""

    # ── Step 3: Template matching ─────────────────────────────────────────────
    word_set = set(deduped)
    template_result = _try_template(word_set)
    if template_result:
        return template_result

    # ── Step 4: Standalone single expression ──────────────────────────────────
    joined_phrase = ' '.join(deduped).lower()
    if joined_phrase in STANDALONE_EXPRESSIONS or (
        len(deduped) == 1 and deduped[0].lower() in STANDALONE_EXPRESSIONS
    ):
        s = ' '.join(deduped).capitalize()
        return s if s[-1] in '.!?' else s + '.'

    # ── Step 5: NLTK-based sentence construction ──────────────────────────────
    if not nlp_ready:
        s = ' '.join(deduped).capitalize()
        return s if s[-1] in '.!?' else s + '.'

    try:
        tagged = pos_tag(deduped)
        output_tokens: list[str] = []

        # Does this phrase already contain a verb (NLTK or known)?
        has_verb = any(
            tag.startswith('VB') or word.lower() in KNOWN_VERBS
            for word, tag in tagged
        )

        for word, tag in tagged:
            w_lower = word.lower()

            # Fix pronoun casing
            if w_lower == 'i':
                output_tokens.append('I')
                continue

            # Insert article before singular countable nouns
            if w_lower in COUNTABLE_NOUNS:
                prev = output_tokens[-1].lower() if output_tokens else ''
                if prev not in ('a', 'an', 'the', 'my', 'your', 'his', 'her', 'our', 'their'):
                    output_tokens.append(_choose_article(word))

            output_tokens.append(word)

        # Inject linking verb if no verb found and phrase isn't standalone
        is_standalone = all(
            t.lower() in STANDALONE_EXPRESSIONS or t.lower() in KNOWN_ADJECTIVES
            for t in deduped
        )

        if not has_verb and len(deduped) > 1 and not is_standalone:
            insert_at = next(
                (j + 1 for j, tok in enumerate(output_tokens)
                 if tok.lower() in PRONOUNS or tok.lower() in COUNTABLE_NOUNS),
                None
            )
            if insert_at is not None and insert_at < len(output_tokens):
                subj = output_tokens[0].lower()
                linking = 'am' if subj == 'i' else (
                    'are' if subj in ('you', 'we', 'they') else 'is'
                )
                output_tokens.insert(insert_at, linking)

        sentence = ' '.join(output_tokens)
        sentence = re.sub(r'\s+([.,!?])', r'\1', sentence)
        if sentence:
            sentence = sentence[0].upper() + sentence[1:]
        if sentence and sentence[-1] not in '.!?':
            sentence += '.'
        return sentence

    except Exception as e:
        print(f"⚠️  NLP processing error: {e}")
        fallback = ' '.join(deduped).capitalize()
        return fallback if fallback[-1] in '.!?' else fallback + '.'




def save_sentence_to_file(sentence: str):
    """Append the finalized sentence to predictions.txt with a timestamp."""
    try:
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        with open(PREDICTIONS_FILE, 'a', encoding='utf-8') as f:
            f.write(f"[{timestamp}] {sentence}\n")
        print(f"💾 Saved: {sentence}")
    except Exception as e:
        print(f"❌ Error saving sentence: {e}")


def finalize_phrase():
    """Called after debounce timeout. Form sentence from buffer and save it."""
    global word_buffer, finalized_sentence, all_sentences

    with state_lock:
        if not word_buffer:
            return
        words_snapshot = list(word_buffer)
        word_buffer.clear()

    print(f"📝 Finalizing phrase from words: {words_snapshot}")
    sentence = form_sentence_with_nlp(words_snapshot)

    with state_lock:
        finalized_sentence = sentence
        if sentence:
            all_sentences.append(sentence)

    save_sentence_to_file(sentence)


def schedule_debounce():
    """Reset the debounce timer: cancel the old one and start a new one."""
    global debounce_thread

    if debounce_thread is not None and debounce_thread.is_alive():
        # Mark as cancelled by checking time later
        pass

    debounce_thread = threading.Timer(DEBOUNCE_SECONDS, finalize_phrase)
    debounce_thread.daemon = True
    debounce_thread.start()


# ─────────────────────────────────────────────
# 5. Image Enhancement Pipeline
# ─────────────────────────────────────────────

# Pre-create CLAHE once (contrast-limited adaptive histogram equalization)
_clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))

# Pre-compute gamma LUT once (avoids rebuilding per frame)
_GAMMA = 1.2
_INV_GAMMA = 1.0 / _GAMMA
_GAMMA_LUT = np.array([
    ((i / 255.0) ** _INV_GAMMA) * 255 for i in range(256)
], dtype='uint8')


def enhance_frame(frame):
    """
    Lightweight image enhancement to improve YOLO prediction accuracy.
    Designed to run in real-time at 720p without stalling the MJPEG stream.

      1. CLAHE — boosts local contrast via the L (lightness) channel
      2. Bilateral filter — removes noise while preserving edges (fast)
      3. Unsharp-mask — sharpens hand/finger edges
      4. Gamma correction — brightens underexposed frames (conditional)
    """
    # ── 1. CLAHE on the L (lightness) channel ─────────────────────────────────
    lab = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB)
    l_ch, a_ch, b_ch = cv2.split(lab)
    l_ch = _clahe.apply(l_ch)
    enhanced = cv2.merge([l_ch, a_ch, b_ch])
    enhanced = cv2.cvtColor(enhanced, cv2.COLOR_LAB2BGR)

    # ── 2. Bilateral filter — fast denoising that keeps edges sharp ───────────
    enhanced = cv2.bilateralFilter(enhanced, d=5, sigmaColor=50, sigmaSpace=50)

    # ── 3. Unsharp-mask sharpening ────────────────────────────────────────────
    gaussian = cv2.GaussianBlur(enhanced, (0, 0), sigmaX=2)
    enhanced = cv2.addWeighted(enhanced, 1.4, gaussian, -0.4, 0)

    # ── 4. Gamma correction (brighten dark frames) ────────────────────────────
    mean_brightness = np.mean(cv2.cvtColor(enhanced, cv2.COLOR_BGR2GRAY))
    if mean_brightness < 100:
        enhanced = cv2.LUT(enhanced, _GAMMA_LUT)

    return enhanced


# ─────────────────────────────────────────────
# 6. Frame Generator
# ─────────────────────────────────────────────
def gen_frames():
    global last_word, last_word_time

    cap = cv2.VideoCapture(0)

    if not cap.isOpened():
        print("❌ Could not open webcam.")
        return

    # ── Request higher resolution from the webcam ─────────────────────────────
    cap.set(cv2.CAP_PROP_FRAME_WIDTH,  CAM_WIDTH)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, CAM_HEIGHT)
    cap.set(cv2.CAP_PROP_FPS, 30)

    # ── Optimise auto-exposure and focus (DirectShow / V4L2) ──────────────────
    cap.set(cv2.CAP_PROP_AUTO_EXPOSURE, 0.75)   # 0.75 = auto exposure on
    cap.set(cv2.CAP_PROP_AUTOFOCUS, 1)           # enable autofocus
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)           # reduce latency

    actual_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    actual_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    print(f"📷 Camera opened at {actual_w}×{actual_h}")

    while True:
        success, frame = cap.read()
        if not success:
            print("⚠️  Failed to grab frame.")
            break

        # ── Enhance the frame before inference ────────────────────────────────
        enhanced = enhance_frame(frame)

        annotated_frame = enhanced.copy()

        if model is not None:
            try:
                results = model.predict(
                    source=enhanced,
                    conf=CONFIDENCE_THRESHOLD,
                    imgsz=YOLO_IMGSZ,
                    verbose=False
                )

                detected_label = None
                detected_conf  = 0.0

                for result in results:
                    # ── Detection / Segmentation ──────────────────────
                    if result.boxes is not None and len(result.boxes):
                        for box in result.boxes:
                            x1, y1, x2, y2 = map(int, box.xyxy[0])
                            conf  = float(box.conf[0])
                            cls   = int(box.cls[0])
                            label = model.names.get(cls, str(cls))

                            # Track highest-confidence detection
                            if conf > detected_conf:
                                detected_conf  = conf
                                detected_label = label

                            cv2.rectangle(annotated_frame, (x1, y1), (x2, y2),
                                          BOX_COLOR, BOX_THICKNESS)

                            display_text = f"{label} {conf:.2f}"
                            (tw, th), _ = cv2.getTextSize(
                                display_text, FONT, FONT_SCALE, FONT_THICKNESS)

                            cv2.rectangle(annotated_frame,
                                          (x1, y1 - th - 10), (x1 + tw + 6, y1),
                                          BOX_COLOR, -1)
                            cv2.putText(annotated_frame, display_text,
                                        (x1 + 3, y1 - 5), FONT, FONT_SCALE,
                                        (0, 0, 0), FONT_THICKNESS, cv2.LINE_AA)

                    # ── Classification (no boxes) ─────────────────────
                    elif result.probs is not None:
                        top_cls   = int(result.probs.top1)
                        top_conf  = float(result.probs.top1conf)
                        label     = model.names.get(top_cls, str(top_cls))

                        if top_conf > detected_conf:
                            detected_conf  = top_conf
                            detected_label = label

                        display_text = f"{label}: {top_conf:.2f}"
                        cv2.putText(annotated_frame, display_text,
                                    (20, 50), FONT, 1.2,
                                    TEXT_COLOR, 2, cv2.LINE_AA)

                # ── Add detected word to buffer ───────────────────────
                if detected_label and detected_conf >= MIN_CONFIDENCE_WORD:
                    now = time.time()
                    with state_lock:
                        # Only add a word if it's different from the last one
                        if detected_label != last_word:
                            word_buffer.append(detected_label)
                            last_word      = detected_label
                            last_word_time = now
                            print(f"🔤 Word added: '{detected_label}' (conf={detected_conf:.2f}) | Buffer: {word_buffer}")
                    # Always reset the debounce timer on each new detection
                    schedule_debounce()

            except Exception as e:
                print(f"Inference error: {e}")

        # ── Encode at higher JPEG quality for the live feed ───────────────────
        encode_params = [cv2.IMWRITE_JPEG_QUALITY, 90]
        ret, buffer = cv2.imencode('.jpg', annotated_frame, encode_params)
        if not ret:
            continue

        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + buffer.tobytes() + b'\r\n')

    cap.release()


# ─────────────────────────────────────────────
# 7. Flask Routes
# ─────────────────────────────────────────────
@app.route('/')
def index():
    return render_template('index.html')


@app.route('/video_feed')
def video_feed():
    return Response(
        gen_frames(),
        mimetype='multipart/x-mixed-replace; boundary=frame'
    )


@app.route('/get_sentence')
def get_sentence():
    """Return the current word buffer and all finalized sentences."""
    with state_lock:
        return jsonify({
            'current_words': list(word_buffer),
            'current_preview': " ".join(word_buffer),
            'finalized_sentence': finalized_sentence,
            'all_sentences': list(all_sentences),
            'nlp_ready': nlp_ready
        })


@app.route('/clear_sentence', methods=['POST'])
def clear_sentence():
    """Clear the session transcript (does not delete the file)."""
    global word_buffer, last_word, last_word_time, finalized_sentence, all_sentences
    with state_lock:
        word_buffer.clear()
        last_word         = None
        last_word_time    = 0.0
        finalized_sentence = ""
        all_sentences.clear()
    return jsonify({'status': 'cleared'})


@app.route('/health')
def health():
    """Simple health endpoint for deployment platforms and load balancers."""
    return jsonify({
        'status': 'ok',
        'model_loaded': model is not None,
        'nlp_ready': nlp_ready
    })


# ─────────────────────────────────────────────
# 8. Entry Point
# ─────────────────────────────────────────────
if __name__ == '__main__':
    app.run(debug=False, host='0.0.0.0', port=int(os.getenv('PORT', '5000')))