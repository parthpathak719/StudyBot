import json
from pathlib import Path
from datetime import date, datetime, timedelta
import random

import streamlit as st

_transformers_available = True
try:
    import torch
    from transformers import T5ForConditionalGeneration, T5Tokenizer, BartForConditionalGeneration, BartTokenizer
except Exception:
    _transformers_available = False

ROOT = Path(".")
DATA_DIR = ROOT / "data"
DATA_DIR.mkdir(exist_ok=True)

SYLLABUS_FILE = DATA_DIR / "syllabus.json"
DB_FILE = DATA_DIR / "user_db.json"
CONTENT_DIR = DATA_DIR / "content"
CONTENT_DIR.mkdir(exist_ok=True)

# seed syllabus if missing
if not SYLLABUS_FILE.exists():
    sample = {
        "Biology": {
            "Cell Biology": ["Cell Structure", "Mitochondria & ATP", "Plasma Membrane"],
            "Genetics": ["Mendelian Genetics", "DNA Replication"]
        },
        "Java": {
            "OOP": ["Classes & Objects", "Inheritance", "Polymorphism"],
            "Collections": ["List", "Map"]
        },
        "Mathematics": {
            "Calculus": ["Limits", "Derivatives", "Integrals"],
            "Algebra": ["Quadratic Equations", "Matrices"]
        }
    }
    SYLLABUS_FILE.write_text(json.dumps(sample, indent=2))

# seed DB if missing
if not DB_FILE.exists():
    db_template = {"cards": [], "attempts": []}
    DB_FILE.write_text(json.dumps(db_template, indent=2))

if not any(CONTENT_DIR.iterdir()):
    samples = {
        "Biology_Cell Structure.txt": "A cell is the basic structural unit of life. Eukaryotic cells contain organelles such as nucleus, mitochondria, and endoplasmic reticulum.",
        "Biology_Mitochondria & ATP.txt": "Mitochondria produce ATP through oxidative phosphorylation and are known as the powerhouse of the cell.",
        "Java_Classes & Objects.txt": "In Java, classes are templates for objects. An object holds state as fields and behavior as methods.",
        "Mathematics_Limits.txt": "A limit describes the value that a function approaches as the input approaches some point."
    }
    for fname, txt in samples.items():
        (CONTENT_DIR / fname).write_text(txt)

def load_db():
    return json.loads(DB_FILE.read_text())

def save_db(db):
    DB_FILE.write_text(json.dumps(db, indent=2))

# SM-2 Spaced Repetition Engine
class SM2Engine:
    def __init__(self):
        self.db = load_db()
        # cards stored as list of dicts under db['cards']
        # card keys: id, user, topic, interval, repetitions, efactor, due_date (ISO)
        if "cards" not in self.db:
            self.db["cards"] = []
            save_db(self.db)

    def _find_card(self, user, topic):
        for i, c in enumerate(self.db["cards"]):
            if c["user"] == user and c["topic"] == topic:
                return i, self.db["cards"][i]
        return None, None

    def initialize_card(self, user, topic):
        # create card with defaults
        card = {
            "id": f"{user}::{topic}",
            "user": user,
            "topic": topic,
            "interval": 1,
            "repetitions": 0,
            "efactor": 2.5,
            "due_date": date.today().isoformat(),
            "created_at": datetime.utcnow().isoformat()
        }
        self.db["cards"].append(card)
        save_db(self.db)
        return card

    def review_card(self, user, topic, quality):
        """
        quality: 0..5 (user self-assessed or auto-graded)
        returns updated card dict
        """
        idx, card = self._find_card(user, topic)
        if card is None:
            card = self.initialize_card(user, topic)
            idx = len(self.db["cards"]) - 1

        if quality < 3:
            card["repetitions"] = 0
            card["interval"] = 1
        else:
            card["repetitions"] += 1
            if card["repetitions"] == 1:
                card["interval"] = 1
            elif card["repetitions"] == 2:
                card["interval"] = 6
            else:
                card["interval"] = int(round(card["interval"] * card["efactor"]))

        # update efactor
        new_ef = card["efactor"] + (0.1 - (5 - quality) * (0.08 + (5 - quality) * 0.02))
        card["efactor"] = max(1.3, new_ef)

        # set next due date
        card["due_date"] = (date.today() + timedelta(days=card["interval"])).isoformat()

        # persist
        self.db["cards"][idx] = card
        save_db(self.db)

        # log attempt
        attempt = {
            "user": user,
            "topic": topic,
            "quality": quality,
            "timestamp": datetime.utcnow().isoformat(),
            "next_due": card["due_date"]
        }
        self.db.setdefault("attempts", []).append(attempt)
        save_db(self.db)

        return card

    def get_due_cards(self, user):
        today_iso = date.today().isoformat()
        return [c for c in self.db.get("cards", []) if c["user"] == user and c["due_date"] <= today_iso]

    def get_all_cards(self, user):
        return [c for c in self.db.get("cards", []) if c["user"] == user]

# Roadmap generator
class RoadmapEngine:
    def __init__(self, syllabus_path=SYLLABUS_FILE):
        self.syllabus_path = Path(syllabus_path)
        self.syllabus = json.loads(self.syllabus_path.read_text())

    def list_subjects(self):
        return list(self.syllabus.keys())

    def flatten_subject_topics(self, subject):
        """Return a flat list of topic strings (Unit :: Topic if needed)"""
        s = self.syllabus.get(subject, {})
        flat = []
        for unit, topics in s.items():
            for t in topics:
                flat.append(f"{unit} :: {t}")
        return flat

    def create_plan(self, subject, days):
        flat = self.flatten_subject_topics(subject)
        if not flat:
            return []
        per_day = max(1, int((len(flat) + days - 1) // days))
        plan = [flat[i:i+per_day] for i in range(0, len(flat), per_day)]
        # attach dates starting today
        out = []
        for i, chunk in enumerate(plan):
            out.append({
                "day_index": i+1,
                "date": (date.today() + timedelta(days=i)).isoformat(),
                "topics": chunk
            })
        return out

# Question generation (template + optional T5)
class QuestionGenerator:
    TEMPLATES = [
        "What is {X}?",
        "Explain {X}.",
        "Describe the role of {X}.",
        "Why is {X} important?",
        "Give a short definition of {X}."
    ]

    def __init__(self):
        self._t5_loaded = False
        self._t5_tokenizer = None
        self._t5_model = None
        self._bart_loaded = False
        self._bart_tokenizer = None
        self._bart_model = None

    def generate_template(self, topic):
        concept = topic.split("::")[-1].strip()
        return random.choice(self.TEMPLATES).format(X=concept)

    # lazy load T5 (small) if user asks and transformers are installed
    def _ensure_t5(self):
        if not _transformers_available:
            raise RuntimeError("Transformers/Torch not installed in env")
        if not self._t5_loaded:
            self._t5_tokenizer = T5Tokenizer.from_pretrained("t5-small")
            self._t5_model = T5ForConditionalGeneration.from_pretrained("t5-small")
            self._t5_loaded = True

    def generate_t5(self, context, num=1):
        try:
            self._ensure_t5()
        except Exception as e:
            return [self.generate_template(context)]  # fallback
        inp = f"generate question: {context}"
        input_ids = self._t5_tokenizer.encode(inp, return_tensors="pt", truncation=True, max_length=512)
        outputs = self._t5_model.generate(input_ids, max_length=64, num_return_sequences=min(3, num), do_sample=True, top_p=0.95, top_k=50)
        qs = [self._t5_tokenizer.decode(o, skip_special_tokens=True) for o in outputs]
        return qs

    # optional summarizer / paraphrase via BART (lazy)
    def _ensure_bart(self):
        if not _transformers_available:
            raise RuntimeError("Transformers/Torch not installed in env")
        if not self._bart_loaded:
            # use smaller distilbart for speed if possible
            try:
                self._bart_tokenizer = BartTokenizer.from_pretrained("sshleifer/distilbart-cnn-12-6")
                self._bart_model = BartForConditionalGeneration.from_pretrained("sshleifer/distilbart-cnn-12-6")
            except Exception:
                self._bart_tokenizer = BartTokenizer.from_pretrained("facebook/bart-large-cnn")
                self._bart_model = BartForConditionalGeneration.from_pretrained("facebook/bart-large-cnn")
            self._bart_loaded = True

    def summarize_bart(self, text, max_len=60):
        try:
            self._ensure_bart()
        except Exception:
            return None
        inputs = self._bart_tokenizer([text], truncation=True, max_length=1024, return_tensors="pt")
        ids = self._bart_model.generate(inputs["input_ids"], max_length=max_len, min_length=20, length_penalty=2.0)
        return self._bart_tokenizer.decode(ids[0], skip_special_tokens=True)

# Helpers: load content for a topic
def load_content_for_topic(topic):
    """
    naive mapping: looks for a file in data/content that contains the topic slug
    topic must be like "Unit :: TopicName" or "TopicName"
    """
    slug = topic.replace(" ", "_").lower()
    for p in CONTENT_DIR.iterdir():
        if slug in p.name.lower():
            try:
                return p.read_text()
            except Exception:
                return ""
    return ""  # no content found

# Streamlit UI
st.set_page_config(page_title="StudyBot Demo", layout="wide")
st.title("StudyBot — Demo (Roadmap + SM-2 spaced repetition)")

# instantiate engines
road = RoadmapEngine()
sm2 = SM2Engine()
qgen = QuestionGenerator()

# Sidebar: user input & toggles
st.sidebar.header("User & Settings")
username = st.sidebar.text_input("Your name", value="student")
enable_t5 = st.sidebar.checkbox("Enable T5 QGen (slow; requires transformers)", value=False)
enable_bart = st.sidebar.checkbox("Enable BART summarizer (slow; optional)", value=False)
st.sidebar.markdown("---")
st.sidebar.markdown("**Data files created in**: `data/`")

# Main UI: select subject and build roadmap
subjects = road.list_subjects()
subject = st.selectbox("Select subject", subjects)

col1, col2 = st.columns(2)
with col1:
    days = st.number_input("Days to distribute topics over", min_value=1, max_value=60, value=7)
    if st.button("Create Study Plan"):
        plan = road.create_plan(subject, days)
        st.success(f"Created plan for {subject} across {len(plan)} day(s)")
        for day in plan:
            st.write(f"**{day['date']}** — {', '.join(day['topics'])}")
with col2:
    st.markdown("### Quick actions")
    if st.button("Show my cards"):
        cards = sm2.get_all_cards(username) if hasattr(sm2, "get_all_cards") else [c for c in load_db().get("cards", []) if c["user"]==username]
        if not cards:
            st.info("No cards yet — generate a quiz to create cards.")
        else:
            st.write(cards)

st.markdown("---")
st.header("Quiz: Generate question & schedule reviews")

# Topic input (suggest topics from syllabus)
flat_topics = road.flatten_subject_topics(subject)
topic_choice = st.selectbox("Pick a topic", [""] + flat_topics)
manual_topic = st.text_input("Or type a custom topic to quiz on", value="")

topic = manual_topic.strip() or (topic_choice if topic_choice else "")

qcol, acol = st.columns([3,2])
with qcol:
    if st.button("Generate question"):
        if not topic:
            st.warning("Choose or type a topic first.")
        else:
            # load content for context
            paragraph = load_content_for_topic(topic)
            # prefer model if toggled and available
            question_text = None
            if enable_t5:
                try:
                    qs = qgen.generate_t5(paragraph or topic, num=1)
                    question_text = qs[0] if qs else None
                except Exception as e:
                    st.warning(f"T5 generation failed, falling back to template. ({e})")
            if not question_text:
                # template fallback
                question_text = qgen.generate_template(topic)
            st.markdown(f"**Q:** {question_text}")
            # show optional BART summary of paragraph if available and toggled
            if enable_bart and paragraph:
                try:
                    summary = qgen.summarize_bart(paragraph, max_len=80)
                    if summary:
                        st.markdown("**Context summary (BART):**")
                        st.write(summary)
                except Exception as e:
                    st.info("BART not available or failed: " + str(e))
            # collect answer & quality
            user_answer = st.text_area("Your answer (type anything)", height=120)
            quality = st.slider("Self-assess recall quality (0=forgot, 5=perfect)", 0, 5, 4)
            if st.button("Submit answer & schedule review"):
                updated = sm2.review_card(username, topic, int(quality))
                st.success(f"Logged. Next review: {updated['due_date']} (interval={updated['interval']}d, efactor={round(updated['efactor'],2)})")

with acol:
    st.markdown("### Quick stats")
    db = load_db()
    attempts = [a for a in db.get("attempts", []) if a.get("user")==username]
    st.write("Attempts:", len(attempts))
    cards_count = len([c for c in db.get("cards", []) if c.get("user")==username])
    st.write("Cards:", cards_count)
    # basic accuracy
    if attempts:
        avg_quality = sum(a.get("quality",0) for a in attempts)/len(attempts)
        st.write("Avg quality:", round(avg_quality,2))

st.markdown("---")
st.header("Due Today — review scheduled cards")

due_cards = sm2.get_due_cards(username)
if not due_cards:
    st.info("No cards due today. Take a quiz to create scheduled cards.")
else:
    for c in due_cards:
        with st.expander(f"Review: {c['topic']} (next due {c['due_date']})"):
            paragraph = load_content_for_topic(c["topic"])
            question_text = None
            # prefer T5 if toggled
            if enable_t5:
                try:
                    qs = qgen.generate_t5(paragraph or c["topic"], num=1)
                    question_text = qs[0] if qs else None
                except Exception:
                    question_text = None
            if not question_text:
                question_text = qgen.generate_template(c["topic"])
            st.write("Q:", question_text)
            answer_text = st.text_area("Your answer", key=f"ans-{c['id']}")
            qval = st.slider("Self-assess (0-5)", 0, 5, 4, key=f"q-{c['id']}")
            if st.button("Submit review", key=f"submit-{c['id']}"):
                updated = sm2.review_card(username, c["topic"], int(qval))
                st.success(f"Updated. Next due: {updated['due_date']} (interval {updated['interval']}d)")

st.markdown("---")
st.header("Attempts log & DB (debug view)")

db = load_db()
st.subheader("Attempts (most recent 20)")
attempts = [a for a in db.get("attempts", []) if a.get("user")==username]
attempts = sorted(attempts, key=lambda x: x.get("timestamp", ""), reverse=True)[:20]
if attempts:
    st.table(attempts)
else:
    st.write("No attempts yet.")

st.subheader("Cards")
cards = [c for c in db.get("cards", []) if c.get("user")==username]
if cards:
    st.table(cards)
else:
    st.write("No cards yet.")

st.markdown("---")
st.write("Demo notes: Use the toggles on the left to enable optional T5/BART (requires transformers & torch). The app defaults to simple template questions for reliable demo speeds.")


