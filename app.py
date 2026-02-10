import json
from pathlib import Path
from datetime import date, datetime, timedelta
import random
import textwrap
import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt

# Optional heavy deps
_transformers_available = True
_pdf_lib_available = True
try:
    import torch
    # Import Auto* classes for flexible model loading
    from transformers import (
        BartForConditionalGeneration,
        BartTokenizer,
        AutoTokenizer,
        AutoModelForSeq2SeqLM,
        AutoModelForSeq2SeqLM as Seq2Seq,  # alias for clarity
    )
except Exception:
    _transformers_available = False

# PDF libs (optional)
try:
    import pdfplumber
except Exception:
    _pdf_lib_available = False

try:
    import PyPDF2
except Exception:
    pass

# basic data layout
ROOT = Path(".")
DATA_DIR = ROOT / "data"
DATA_DIR.mkdir(exist_ok=True)
SYLLABUS_FILE = DATA_DIR / "syllabus.json"
DB_FILE = DATA_DIR / "user_db.json"
CONTENT_DIR = DATA_DIR / "content"
CONTENT_DIR.mkdir(exist_ok=True)

# ------------------ seed syllabus (expanded) ------------------
if not SYLLABUS_FILE.exists():
    sample = {
        "Biology": {
            "Cell Biology": [
                "Cell Structure",
                "Mitochondria & ATP",
                "Plasma Membrane",
                "Lysosomes",
                "Endoplasmic Reticulum",
            ],
            "Genetics": [
                "Mendelian Genetics",
                "DNA Replication",
                "RNA Transcription",
                "Protein Synthesis",
            ],
            "Human Physiology": [
                "Digestive System",
                "Respiratory System",
                "Circulatory System",
            ],
        },
        "Physics": {
            "Mechanics": [
                "Kinematics",
                "Laws of Motion",
                "Work and Energy",
                "Rotational Motion",
            ],
            "Optics": ["Reflection", "Refraction", "Lenses and Mirrors"],
        },
        "Chemistry": {
            "Physical Chemistry": [
                "Mole Concept",
                "Atomic Structure",
                "Chemical Kinetics",
            ],
            "Organic Chemistry": ["Hydrocarbons", "Alcohols", "Amines"],
        },
        "Mathematics": {
            "Calculus": ["Limits", "Derivatives", "Integrals", "Differential Equations"],
            "Algebra": [
                "Quadratic Equations",
                "Matrices",
                "Determinants",
                "Complex Numbers",
            ],
        },
        "Java": {
            "OOP": ["Classes & Objects", "Inheritance", "Polymorphism", "Encapsulation"],
            "Collections": ["List", "Map", "Set", "Streams"],
            "Exceptions": ["Try-Catch", "Custom Exceptions"],
        },
        "History": {
            "World Wars": ["World War I", "World War II", "Cold War"],
            "Indian History": ["Mughal Empire", "Independence Movement"],
        },
    }
    SYLLABUS_FILE.write_text(json.dumps(sample, indent=2))

# ------------------ seed DB ------------------
if not DB_FILE.exists():
    db_template = {"cards": [], "attempts": [], "users": {}}
    DB_FILE.write_text(json.dumps(db_template, indent=2))

# ------------------ seed content samples ------------------
if not any(CONTENT_DIR.iterdir()):
    samples = {
        "Biology_Cell Structure.txt": (
            "The cell structure is the foundation of all living organisms, often referred to as "
            "the basic unit of life. Every organism, from the simplest bacteria to the most "
            "complex human being, is made up of cells that perform essential functions necessary "
            "for survival. Broadly, cells are classified into two main types: prokaryotic and "
            "eukaryotic. Prokaryotic cells, such as those found in bacteria, are simpler in "
            "structure; they lack a true nucleus and membrane-bound organelles. Their genetic "
            "material, called DNA, floats freely in a region known as the nucleoid. Eukaryotic "
            "cells, found in plants, animals, fungi, and protists, are more complex and contain "
            "well-defined, membrane-bound organelles that perform specialized tasks. The plasma "
            "membrane surrounds the cell and controls the movement of substances in and out, "
            "maintaining a stable internal environment. Inside, the cytoplasm is a jelly-like "
            "fluid that holds the organelles. The nucleus acts as the control center of the cell, "
            "housing DNA that regulates cell activities and heredity. Organelles like mitochondria "
            "generate energy through cellular respiration, while ribosomes are responsible for "
            "protein synthesis. The endoplasmic reticulum helps in the synthesis and transport of "
            "proteins and lipids, with the rough ER being covered in ribosomes and the smooth ER "
            "involved in lipid metabolism and detoxification. The Golgi apparatus modifies, sorts, "
            "and packages proteins for secretion or for use within the cell. In plant cells, "
            "additional structures like the cell wall provide rigidity and protection, chloroplasts "
            "carry out photosynthesis to produce food, and a large vacuole maintains cell pressure "
            "and stores nutrients. In contrast, animal cells contain smaller vacuoles and structures "
            "like centrioles that assist in cell division. Together, these components work in harmony "
            "to keep the cell alive, allowing it to grow, reproduce, and perform its specific "
            "functions, making the cell a self-sustaining and highly organized system that lies at "
            "the core of all life."
        ),
        "Biology_Mitochondria & ATP.txt": (
            "Mitochondria produce ATP through oxidative phosphorylation and are known as the powerhouse of the cell."
        ),
        "Java_Classes & Objects.txt": (
            "In Java, classes are templates for objects. An object holds state as fields and behavior as methods."
        ),
        "Mathematics_Limits.txt": (
            "A limit describes the value that a function approaches as the input approaches some point."
        ),
        "Physics_Kinematics.txt": (
            "Kinematics deals with the motion of objects without considering the causes of motion. "
            "It includes displacement, velocity, and acceleration."
        ),
        "Chemistry_Mole Concept.txt": (
            "The mole concept relates mass to the number of particles using Avogadro's number. "
            "It's essential for chemical equations and stoichiometry."
        ),
        "History_World War II.txt": (
            "World War II was a global war from 1939 to 1945 involving most of the world's nations."
        ),
    }
    for fname, txt in samples.items():
        (CONTENT_DIR / fname.replace(" ", "_")).write_text(txt)

# --- UPDATED DB helpers ---
def load_db():
    try:
        # Ensure file exists before reading
        if not DB_FILE.exists():
            save_db({"cards": [], "attempts": [], "users": {}})
        with open(DB_FILE, "r") as f:
            # Handle empty file case
            content = f.read()
            if not content:
                return {"cards": [], "attempts": [], "users": {}}
            return json.loads(content)
    except json.JSONDecodeError:
        st.error("Error: Database file contains invalid JSON. Resetting DB.")
        save_db({"cards": [], "attempts": [], "users": {}})
        return {"cards": [], "attempts": [], "users": {}}
    except Exception as e:
        st.error(f"Error loading DB: {e}. Using default.")
        # Return a default structure instead of raising an error
        return {"cards": [], "attempts": [], "users": {}}

def save_db(db):
    try:
        with open(DB_FILE, "w") as f:
            json.dump(db, f, indent=2)
    except Exception as e:
        st.error(f"Error saving DB: {e}")

# --- SM-2 engine (UPDATED to store detailed stats) ---
class SM2Engine:
    def __init__(self):
        self.db = load_db()
        # Ensure all top-level keys exist
        changed = False
        if "cards" not in self.db:
            self.db["cards"] = []
            changed = True
        if "attempts" not in self.db:
            self.db["attempts"] = []
            changed = True
        if "users" not in self.db:
            self.db["users"] = {}
            changed = True
        if changed:
            save_db(self.db)  # Save immediately if structure was modified

    def _find_card(self, user, topic):
        # Uses the db loaded in __init__ initially, but review_card reloads
        for i, c in enumerate(self.db.get("cards", [])):  # Use .get for safety
            if c.get("user") == user and c.get("topic") == topic:
                return i, c  # Return the actual card dict
        return None, None

    def initialize_card(self, user, topic):
        # This function should add to the currently loaded DB state
        card = {
            "id": f"{user}::{topic}",
            "user": user,
            "topic": topic,
            "interval": 1,
            "repetitions": 0,
            "efactor": 2.5,
            "due_date": date.today().isoformat(),
            "created_at": datetime.utcnow().isoformat(),
        }
        # Append to the list in memory
        self.db.setdefault("cards", []).append(card)
        # Save immediately after initializing a card
        save_db(self.db)
        return card

    def review_card(self, user, topic, quality, details=None):
        """
        Modified to accept 'details' dictionary containing extended stats 
        (keyword_score, matched_keywords, summaries, etc.)
        """
        # Reload DB at the start of the review to get the absolute latest state
        self.db = load_db()
        idx, card = self._find_card(user, topic)
        if card is None:
            # Initialize card adds it to self.db and saves
            card = self.initialize_card(user, topic)
            # Find the index of the newly added card in the reloaded db
            self.db = load_db()  # Reload again after potential save in initialize_card
            idx, card = self._find_card(user, topic)
        if idx is None:  # Should not happen if initialize worked
            st.error("Failed to find card immediately after initialization.")
            return None  # Or handle error appropriately
        # Ensure card is a dictionary before proceeding
        if not isinstance(card, dict):
            st.error(f"Error: Found card is not a dictionary: {card}")
            return None  # Or handle error appropriately
        # --- SM-2 Logic (applied to the card dictionary) ---
        if quality < 3:
            card["repetitions"] = 0
            card["interval"] = 1
        else:
            card["repetitions"] = card.get("repetitions", 0) + 1  # Use .get for safety
            current_interval = card.get("interval", 1)
            current_efactor = card.get("efactor", 2.5)
            if card["repetitions"] == 1:
                card["interval"] = 1
            elif card["repetitions"] == 2:
                card["interval"] = 6
            else:
                card["interval"] = int(round(current_interval * current_efactor))
            # update efactor
            current_efactor = card.get("efactor", 2.5)
            new_ef = current_efactor + (
                0.1 - (5 - quality) * (0.08 + (5 - quality) * 0.02)
            )
            card["efactor"] = max(1.3, new_ef)
        # set next due date
        card["due_date"] = (
            date.today() + timedelta(days=card.get("interval", 1))
        ).isoformat()
        # --- End SM-2 Logic ---
        # Update the card in the list
        self.db["cards"][idx] = card
        
        # Create attempt log
        attempt = {
            "user": user,
            "topic": topic,
            "quality": quality,
            "timestamp": datetime.utcnow().isoformat(),
            "next_due": card["due_date"],
        }
        
        # Merge detailed stats if provided
        if details and isinstance(details, dict):
            attempt.update(details)

        self.db.setdefault("attempts", []).append(attempt)
        # Save the entire updated DB state once at the end
        save_db(self.db)
        return card

    def get_due_cards(self, user):
        self.db = load_db()  # Reload DB before reading
        today_iso = date.today().isoformat()
        # Use .get for safety during list comprehension
        return [
            c
            for c in self.db.get("cards", [])
            if c.get("user") == user and c.get("due_date", "") <= today_iso
        ]

    def get_all_cards(self, user):
        self.db = load_db()  # Reload DB before reading
        # Use .get for safety during list comprehension
        return [c for c in self.db.get("cards", []) if c.get("user") == user]

# ------------------ Roadmap ------------------
class RoadmapEngine:
    def __init__(self, syllabus_path=SYLLABUS_FILE):
        self.syllabus_path = Path(syllabus_path)
        try:
            self.syllabus = json.loads(self.syllabus_path.read_text())
        except Exception as e:
            st.error(f"Error loading syllabus: {e}. Using empty.")
            self.syllabus = {}

    def list_subjects(self):
        return list(self.syllabus.keys())

    def list_units(self, subject):
        return list(self.syllabus.get(subject, {}).keys())

    def flatten_subject_topics(self, subject):
        s = self.syllabus.get(subject, {})
        flat = []
        for unit, topics in s.items():
            for t in topics:
                flat.append(f"{unit} :: {t}")
        return flat

    # --- create_plan method ---
    def create_plan(self, subject, days):
        flat = self.flatten_subject_topics(subject)
        total_topics = len(flat)
        if not total_topics or days <= 0:
            return []
        base_per_day = total_topics // days
        remainder = total_topics % days
        plan_chunks = []
        start_index = 0
        for i in range(days):
            chunk_size = base_per_day + (1 if i < remainder else 0)
            if chunk_size == 0 and start_index >= total_topics:
                break
            end_index = start_index + chunk_size
            plan_chunks.append(flat[start_index:end_index])
            start_index = end_index
            if start_index >= total_topics:
                break
        out = []
        for i, chunk in enumerate(plan_chunks):
            if chunk:
                out.append(
                    {
                        "day_index": i + 1,
                        "date": (date.today() + timedelta(days=i)).isoformat(),
                        "topics": chunk,
                    }
                )
        return out

# ------------------ Question generator & BART summarizer & Answer checker ------------------
class QuestionGenerator:
    TEMPLATES = [
        "Explain in detail what {X} is and how it works in the broader context of the topic.",
        "Describe {X} in depth, including its structure, function, and significance.",
        "Give a comprehensive explanation of {X}, mentioning key concepts and examples.",
        "Why is {X} important? Discuss its role, features, and any related ideas.",
        "Discuss {X} thoroughly, covering definitions, mechanisms, and applications.",
    ]

    def __init__(self):
        self._model_loaded = False
        self._tokenizer = None
        self._model = None
        self._loaded_model_name = None  # Track which model is in memory
        # T5 question generation fine-tuned model (keep your original choice)
        self.qg_model_name = "mrm8488/t5-base-finetuned-question-generation-ap"
        # BART summarizer model (distilbart is smaller; you can change to bart-large if available)
        self.sum_model_name = "sshleifer/distilbart-cnn-12-6"
        # We'll use the summarizer also for BART-based answer-check comparisons
        # Cache for simple runtime checks
        self._last_qg_attempts = {}

    def generate_template(self, topic):
        """Generates a more detailed template-based question."""
        concept = topic.split("::")[-1].strip()
        return random.choice(self.TEMPLATES).format(X=concept)

    def _ensure_model(self, model_name):
        """Loads a specific seq2seq model + tokenizer if not already loaded."""
        if not _transformers_available:
            st.warning(
                "Transformers library not available. Install `torch` and `transformers`."
            )
            return False
        # If already loaded the same model, keep it
        if self._model_loaded and self._loaded_model_name == model_name and self._tokenizer and self._model:
            return True  # Already loaded correct model
        # load
        st.info(f"Loading model: {model_name}...")
        try:
            self._tokenizer = AutoTokenizer.from_pretrained(model_name)
            self._model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
            self._model_loaded = True
            self._loaded_model_name = model_name
            st.info("Model loaded successfully.")
            return True
        except Exception as e:
            st.error(f"Failed to load model {model_name}: {e}")
            # reset
            self._model_loaded = False
            self._loaded_model_name = None
            self._tokenizer = None
            self._model = None
            return False

    # --- generate_model_question with longer, multi-part questions and repetition prevention ---
    def generate_model_question(self, context, answer_text, max_len=120, prev_questions=None):
        """Generates a longer question using a QG-tuned model.
        - prev_questions: list of questions previously generated for this topic (avoid repeats)
        """
        if prev_questions is None:
            prev_questions = []
        if not self._ensure_model(self.qg_model_name):
            return None  # Failed to load model
        try:
            # Craft a stronger prompt so the QG model asks multi-part / exam-style questions
            concept = answer_text
            boosted_answer = (
                f"{concept}. Create a multi-part, exam-style question that requires a multi-line answer. "
                "Make the question challenging and avoid being short or trivial. "
                "If possible, include sub-questions (a, b, c) requiring explanation, examples, or brief calculations."
            )
            input_text = f"answer: {boosted_answer} context: {context} "
            inputs = self._tokenizer(
                [input_text],
                truncation=True,
                max_length=1024,
                return_tensors="pt",
            )
            # Try generation multiple times with different sampling settings to avoid repeats
            for attempt in range(1, 4):
                do_sample = True
                top_k = 40 + attempt * 10
                top_p = 0.92
                min_len = 40 + attempt * 10  # push to longer multi-line output across attempts
                ids = self._model.generate(
                    inputs["input_ids"],
                    max_length=max_len + attempt * 20,
                    min_length=min_len,
                    do_sample=do_sample,
                    top_k=top_k,
                    top_p=top_p,
                    num_return_sequences=1,
                )
                question = self._tokenizer.decode(ids[0], skip_special_tokens=True).strip()
                # remove common prefix if present
                if question.lower().startswith("question:"):
                    question = question.split(":", 1)[1].strip()
                # Avoid trivial short outputs (enforce multi-line)
                if len(question.splitlines()) < 2 or len(question) < 80:
                    # treat as too short, retry
                    continue
                # Avoid repeats by simple string similarity check (exact for now)
                if any(self._is_similar(question, prev) for prev in prev_questions):
                    # try another sampling configuration unless we exhausted attempts
                    continue
                return question
            # If all attempts failed, fallback to template
            return self.generate_template(answer_text)
        except Exception as e:
            st.error(f"Error during question generation: {e}")
            return None

    def _is_similar(self, s1, s2):
        """Simple heuristic similarity check (normalized substring / equality)."""
        if not s1 or not s2:
            return False
        a = " ".join(s1.lower().split())
        b = " ".join(s2.lower().split())
        # exact containment or start similarity
        if a == b or a in b or b in a:
            return True
        # word overlap ratio
        s1_words = set(a.split())
        s2_words = set(b.split())
        if not s1_words or not s2_words:
            return False
        overlap = len(s1_words & s2_words) / max(1, min(len(s1_words), len(s2_words)))
        return overlap > 0.75

    # --------- Dynamic, chunked BART summarization (larger summaries) ----------
    def _generate_summary_once(self, text, max_len=320, min_len=80):
        """Single-pass BART summary helper used internally."""
        if not self._ensure_model(self.sum_model_name):
            return None
        try:
            inputs = self._tokenizer(
                [text],
                truncation=True,
                max_length=1024,
                return_tensors="pt",
            )
            ids = self._model.generate(
                inputs["input_ids"],
                max_length=max_len,
                min_length=min_len,
                length_penalty=1.6,
                num_beams=4,
                early_stopping=True,
            )
            return self._tokenizer.decode(ids[0], skip_special_tokens=True)
        except Exception as e:
            st.error(f"Error during summarization: {e}")
            return None

    def summarize_bart_dynamic(
        self,
        text,
        depth=2,
        base_chunk_tokens=1400,
        overlap_tokens=300,
        chunk_summary_len=320,
        final_summary_len=480,
    ):
        """
        Dynamically summarize arbitrarily long text with larger, richer summaries.
        - Splits into overlapping chunks (approx based on characters).
        - Summarizes each chunk.
        - Optionally summarizes the concatenated chunk summaries again (depth).
        - 'depth' roughly controls how global the final summary is.
        """
        if not text or not text.strip():
            return ""
        if not _transformers_available:
            # Fallback: manually truncate
            return textwrap.shorten(text, width=1600, placeholder=" ...")
        # Heuristic: map chars to approximate "tokens"
        # 1 token ~ 4 characters (very rough)
        chars_per_token = 4
        approx_chunk_chars = base_chunk_tokens * chars_per_token
        approx_overlap_chars = overlap_tokens * chars_per_token
        # If text is small enough, single-shot summarization
        if len(text) <= approx_chunk_chars:
            return (
                self._generate_summary_once(
                    text,
                    max_len=max(chunk_summary_len, 140),
                    min_len=80,
                )
                or text
            )
        # Split into overlapping chunks
        chunks = []
        start = 0
        n = len(text)
        while start < n:
            end = min(start + approx_chunk_chars, n)
            chunk = text[start:end].strip()
            if chunk:
                chunks.append(chunk)
            if end == n:
                break
            start = max(end - approx_overlap_chars, 0)
        st.info(
            f"Content is large ({len(text)} chars). Summarizing in {len(chunks)} chunk(s)..."
        )
        # Summarize each chunk
        chunk_summaries = []
        for i, ch in enumerate(chunks):
            with st.spinner(f"Summarizing chunk {i+1}/{len(chunks)} ..."):
                summary_i = self._generate_summary_once(
                    ch,
                    max_len=chunk_summary_len,
                    min_len=100,
                )
                if not summary_i:
                    summary_i = textwrap.shorten(ch, width=1200, placeholder=" ...")
                chunk_summaries.append(f"Chunk {i+1}:\n{summary_i}")
        # If only one chunk, return its summary
        if len(chunk_summaries) == 1 or depth <= 1:
            # Join with extra spacing for readability
            return "\n\n".join(chunk_summaries)
        # Combine summaries and re-summarize if depth > 1
        combined = "\n\n".join(chunk_summaries)
        with st.spinner("Building high-level summary from all chunks..."):
            final = self._generate_summary_once(
                combined,
                max_len=final_summary_len,
                min_len=160,
            )
        if not final:
            final = combined
        return final

# ------------------ BART-based Answer Checker (keeps the heuristic too) ------------------
class AnswerChecker:
    """Combines simple keyword heuristic with BART summarization overlap checking.
    Stores persistent results so the UI can show them later.
    """

    def __init__(self, qgen: QuestionGenerator):
        self.qgen = qgen
        # We'll reuse the summarizer model in QuestionGenerator for BART steps

    def compute_keyword_score(self, user_answer, reference_text):
        # use the original simple heuristic
        if not user_answer or not reference_text:
            return 0.0, [], []
        ref_norm = _normalize_text_for_matching(reference_text)
        ans_norm = _normalize_text_for_matching(user_answer)
        tokens = [t for t in ref_norm.split() if len(t) > 4]
        if not tokens:
            return 0.0, [], []
        uniq = []
        for t in tokens:
            if t not in uniq:
                uniq.append(t)
            if len(uniq) >= 20:
                break
        matched = [t for t in uniq if t in ans_norm]
        missing = [t for t in uniq if t not in ans_norm]
        score = len(matched) / max(1, len(uniq))
        return score, matched, missing

    def compute_bart_overlap_score(self, user_answer, reference_text):
        """Summarize both reference and answer with BART and compute overlap ratio.
        Returns (score_0_1, ref_summary, ans_summary, matched_terms)
        """
        if not _transformers_available:
            return None, None, None, []
        # create short concise summaries for both
        try:
            ref_summary = self.qgen.summarize_bart_dynamic(
                reference_text,
                depth=1,
                base_chunk_tokens=1200,
                overlap_tokens=200,
                chunk_summary_len=220,
                final_summary_len=320,
            )
            ans_summary = self.qgen.summarize_bart_dynamic(
                user_answer,
                depth=1,
                base_chunk_tokens=600,
                overlap_tokens=100,
                chunk_summary_len=160,
                final_summary_len=200,
            )
            # extract keywords from the summaries (simple token filter)
            if not ref_summary:
                return None, ref_summary, ans_summary, []
            def _extract_keywords(s):
                s_norm = _normalize_text_for_matching(s)
                words = [w for w in s_norm.split() if len(w) > 4]
                uniq = []
                for w in words:
                    if w not in uniq:
                        uniq.append(w)
                    if len(uniq) >= 40:
                        break
                return uniq
            ref_k = _extract_keywords(ref_summary)
            ans_k = _extract_keywords(ans_summary)
            if not ref_k:
                return None, ref_summary, ans_summary, []
            matched = [w for w in ref_k if any(w in ak for ak in ans_k)]
            score = len(matched) / max(1, len(ref_k))
            return score, ref_summary, ans_summary, matched
        except Exception as e:
            st.warning(f"BART overlap scoring failed: {e}")
            return None, None, None, []

# ------------------ content loader ------------------
def load_content_for_topic(topic):
    if "::" in topic:
        base_topic = topic.split("::")[-1].strip()
    else:
        base_topic = topic.strip()
    slug = base_topic.lower().replace(" ", "_")  # Match file naming convention
    if not slug:
        return ""
    # Look for files matching pattern Subject_Topic.txt
    for p in CONTENT_DIR.iterdir():
        normalized_p_name = p.stem.lower().replace(" ", "_")
        if normalized_p_name.endswith(slug):
            try:
                return p.read_text()
            except Exception as e:
                st.error(f"Error reading content file {p.name}: {e}")
                return ""
    return ""

# ------------------ simple answer checking helpers (kept original) ------------------
def _normalize_text_for_matching(text):
    """Lowercase, strip, and remove extra spaces for simple matching."""
    return " ".join(text.lower().strip().split())

def compute_answer_score(user_answer, reference_text):
    """
    Heuristic answer checking:
    - Extract some key tokens from reference.
    - Check how many appear in the user answer.
    - Returns (score_0_1, matched_keywords, missing_keywords).
    """
    if not user_answer or not reference_text:
        return 0.0, [], []
    ref_norm = _normalize_text_for_matching(reference_text)
    ans_norm = _normalize_text_for_matching(user_answer)
    # Pick candidate key words: frequent but not stopwords (very simple)
    tokens = [t for t in ref_norm.split() if len(t) > 4]
    if not tokens:
        return 0.0, [], []
    uniq = []
    for t in tokens:
        if t not in uniq:
            uniq.append(t)
        if len(uniq) >= 20:
            break
    matched = [t for t in uniq if t in ans_norm]
    missing = [t for t in uniq if t not in ans_norm]
    score = len(matched) / max(1, len(uniq))
    return score, matched, missing

# ------------------ PDF import helpers ------------------
def extract_text_from_pdf_bytes(file_bytes):
    """Attempt to extract text from uploaded PDF bytes using pdfplumber or PyPDF2."""
    text = ""
    if not _pdf_lib_available:
        st.warning("PDF libraries not installed. Cannot import PDF.")
        return text
    # Try pdfplumber first
    try:
        import io
        with pdfplumber.open(io.BytesIO(file_bytes)) as pdf:
            pages = [p.extract_text() or "" for p in pdf.pages]
            text = "\n\n".join(pages)
    except Exception as e_plumber:
        st.warning(f"pdfplumber failed ({e_plumber}), trying PyPDF2...")
        try:
            from PyPDF2 import PdfReader
            import io
            reader = PdfReader(io.BytesIO(file_bytes))
            pages = []
            for p in reader.pages:
                try:
                    pages.append(p.extract_text() or "")
                except Exception as e_page:
                    st.warning(
                        f"PyPDF2 failed to extract text from a page: {e_page}"
                    )
                    pages.append("")
            text = "\n\n".join(pages)
        except Exception as e_pypdf:
            st.error(
                f"Both pdfplumber and PyPDF2 failed to extract text. Error: {e_pypdf}"
            )
            text = ""
    return text

# ------------------ Streamlit UI ------------------
st.set_page_config(page_title="StudyBot++", layout="wide")
st.title("StudyBot++ (T5 QG + Dynamic BART summarizer + BART answer-check + PDF import + Answer Check)")

# Initialize engines - Use st.cache_resource for objects that don't change often
@st.cache_resource
def get_roadmap_engine():
    return RoadmapEngine()

@st.cache_resource
def get_qgen_engine():
    return QuestionGenerator()

# instantiate checker (no caching - uses qgen which caches inside)
qgen = get_qgen_engine()
answer_checker = AnswerChecker(qgen)

# SM2 Engine in session state
if "sm2_engine" not in st.session_state:
    st.session_state.sm2_engine = SM2Engine()

road = get_roadmap_engine()
sm2 = st.session_state.sm2_engine

# Ensure persistent structures in session_state for question history and last checks
if "generated_questions" not in st.session_state:
    st.session_state.generated_questions = {}  # structure: {username: {topic: [questions...]}}
if "last_answer_checks" not in st.session_state:
    st.session_state.last_answer_checks = {}  # structure: {username: {topic: {...}}}

# sidebar — user & settings
st.sidebar.header("User & Settings")
username = st.sidebar.text_input("Your name", value="student")

# Load user preferences safely
db = load_db()
user_meta = db.setdefault("users", {}).setdefault(
    username,
    {
        "enable_summary": True,
        "enable_qg": True,
        "summary_depth": 2,
        "enable_answer_check": True,
    },
)
enable_summary = st.sidebar.checkbox(
    "Enable Summary (BART)", value=user_meta.get("enable_summary", True)
)
enable_qg = st.sidebar.checkbox(
    "Enable Question Generation (T5)", value=user_meta.get("enable_qg", True)
)
summary_depth = st.sidebar.slider(
    "Summary depth (higher = more global summary)",
    min_value=1,
    max_value=3,
    value=user_meta.get("summary_depth", 2),
)
enable_answer_check = st.sidebar.checkbox(
    "Enable Answer Checking (keyword + BART)", value=user_meta.get("enable_answer_check", True)
)

# Save preferences
user_meta["enable_summary"] = enable_summary
user_meta["enable_qg"] = enable_qg
user_meta["summary_depth"] = summary_depth
user_meta["enable_answer_check"] = enable_answer_check
save_db(db)

st.sidebar.markdown("---")
st.sidebar.markdown("*Data files created in*: `data/`")
st.sidebar.markdown("Add custom subjects / topics below")
new_subject = st.sidebar.text_input("New subject name", value="")
new_unit = st.sidebar.text_input("New unit name (for the subject)", value="")
new_topics = st.sidebar.text_area("Comma-separated topics (for that unit)")
if st.sidebar.button("Add subject/unit/topics"):
    if new_subject.strip() and new_unit.strip() and new_topics.strip():
        payload = [t.strip() for t in new_topics.split(",") if t.strip()]
        try:
            syllabus = json.loads(SYLLABUS_FILE.read_text())
        except Exception:
            syllabus = {}  # Start fresh if corrupt
        syllabus.setdefault(new_subject.strip(), {})
        syllabus[new_subject.strip()].setdefault(new_unit.strip(), [])
        existing_topics = syllabus[new_subject.strip()][new_unit.strip()]
        for p in payload:
            if p not in existing_topics:
                existing_topics.append(p)
        try:
            SYLLABUS_FILE.write_text(json.dumps(syllabus, indent=2))
            st.sidebar.success("Added — refresh the main view to see changes")
            st.cache_resource.clear()
            st.rerun()
        except Exception as e:
            st.sidebar.error(f"Failed to save syllabus: {e}")
    else:
        st.sidebar.error("Please fill subject, unit and topics")
st.sidebar.markdown("---")
if st.sidebar.button("Export DB (download JSON)"):
    try:
        db_content = DB_FILE.read_text()
        st.sidebar.download_button(
            "Download user_db.json",
            data=db_content,
            file_name="user_db.json",
            mime="application/json",
        )
    except Exception as e:
        st.sidebar.error(f"Failed to read DB for export: {e}")

# main — roadmap & quick actions
subjects = road.list_subjects()
if "selected_subject" not in st.session_state:
    st.session_state.selected_subject = subjects[0] if subjects else None
if st.session_state.selected_subject not in subjects and subjects:
    st.session_state.selected_subject = subjects[0]
subject = st.selectbox(
    "Select subject",
    subjects,
    index=subjects.index(st.session_state.selected_subject)
    if subjects and st.session_state.selected_subject in subjects
    else 0,
)
if subject != st.session_state.selected_subject:
    st.session_state.selected_subject = subject
    if "study_plan" in st.session_state:
        del st.session_state.study_plan

col1, col2 = st.columns(2)
with col1:
    days = st.number_input(
        "Days to distribute topics over",
        min_value=1,
        max_value=365,
        value=st.session_state.get("plan_days", 7),
    )
    st.session_state.plan_days = days
    if st.button("Create Study Plan"):
        with st.spinner("Generating plan..."):
            plan = road.create_plan(subject, days)
            st.session_state.study_plan = plan
            st.session_state.plan_subject = subject
            st.session_state.plan_days_generated = days
    if (
        "study_plan" in st.session_state
        and st.session_state.get("plan_subject") == subject
        and st.session_state.get("plan_days_generated") == days
    ):
        st.success(
            f"Showing plan for {subject} across {len(st.session_state.study_plan)} day(s)"
        )
        for day_plan in st.session_state.study_plan:
            topics_str = ", ".join(day_plan["topics"])
            st.write(f"**{day_plan['date']}** — {topics_str}")

with col2:
    st.markdown("### Quick actions")
    if st.button("Show my cards"):
        cards = sm2.get_all_cards(username)
        if not cards:
            st.info("No cards yet — generate a quiz to create cards.")
        else:
            df_cards = pd.DataFrame(cards)
            st.dataframe(
                df_cards[["topic", "interval", "repetitions", "efactor", "due_date"]]
            )

st.markdown("---")
st.header("Quiz: Generate question & schedule reviews")
flat_topics = road.flatten_subject_topics(subject)
topic_options = [""] + flat_topics
if "selected_topic" not in st.session_state:
    st.session_state.selected_topic = ""
if st.session_state.selected_topic not in topic_options:
    st.session_state.selected_topic = ""
topic_choice = st.selectbox(
    "Pick a topic",
    topic_options,
    index=topic_options.index(st.session_state.selected_topic)
    if st.session_state.selected_topic in topic_options
    else 0,
    key="topic_selector",
)
if topic_choice != st.session_state.selected_topic:
    st.session_state.selected_topic = topic_choice
    st.session_state.manual_topic_input = ""
manual_topic = st.text_input(
    "Or type a custom topic to quiz on",
    value=st.session_state.get("manual_topic_input", ""),
    key="manual_topic_input_widget",
)
st.session_state.manual_topic_input = manual_topic
topic = manual_topic.strip() or (topic_choice if topic_choice else "")
qcol, acol = st.columns([3, 2])
if "current_question_info" not in st.session_state:
    st.session_state.current_question_info = {}

with qcol:
    if st.button("Generate question"):
        st.session_state.current_question_info = {}
        if not topic:
            st.warning("Choose or type a topic first.")
        else:
            st.info(f"Looking for content for topic: '{topic}'")
            paragraph = load_content_for_topic(topic)
            question_text = None
            summary_text = None
            generation_method = "template"
            if not paragraph:
                st.error(
                    f"No content file found that matches '{topic}'. Using template question."
                )
                question_text = qgen.generate_template(topic)
            else:
                st.success(f"Found content file! ({len(paragraph)} characters)")
                if enable_qg and _transformers_available:
                    st.info("T5 is enabled. Generating longer question...")
                    with st.spinner(
                        "🤖 Generating detailed question with T5... (may take a minute)"
                    ):
                        # fetch previous generated Qs for this user+topic to avoid repeats
                        prev_list = st.session_state.generated_questions.get(username, {}).get(topic, [])
                        question_text = qgen.generate_model_question(
                            paragraph, topic.split("::")[-1].strip(), max_len=160, prev_questions=prev_list
                        )
                    if question_text:
                        generation_method = "T5"
                        # save history to avoid repetition
                        st.session_state.generated_questions.setdefault(username, {}).setdefault(topic, [])
                        st.session_state.generated_questions[username][topic].append(question_text)
                    else:
                        st.warning("T5 failed, falling back to template.")
                        question_text = qgen.generate_template(topic)
                else:
                    question_text = qgen.generate_template(topic)
                    if not enable_qg:
                        st.info("T5 disabled. Using template question.")
                    elif not _transformers_available:
                        st.warning("Transformers not available. Using template.")
                # Dynamic summarization with larger summaries
                if enable_summary and _transformers_available:
                    st.info("BART is enabled. Generating expanded summary...")
                    with st.spinner(
                        "🚀 Running BART to summarize content (longer summary)..."
                    ):
                        # use stronger defaults to produce longer, more descriptive summaries
                        summary_text = qgen.summarize_bart_dynamic(
                            paragraph,
                            depth=summary_depth,
                            base_chunk_tokens=1400,
                            overlap_tokens=300,
                            chunk_summary_len=320,
                            final_summary_len=480,
                        )
                    if not summary_text:
                        st.warning("BART ran but produced no summary.")
            st.session_state.current_question_info = {
                "topic": topic,
                "question": question_text,
                "summary": summary_text,
                "method": generation_method,
                "context": paragraph or "",  # store reference text for answer checking
                "generated_at": datetime.utcnow().isoformat(),
            }

q_info = st.session_state.current_question_info
if q_info and q_info.get("question"):
    st.markdown(f"**Q ({q_info.get('method')}):** {q_info.get('question')}")
    if q_info.get("summary"):
        st.markdown("**Context summary (BART, dynamic, extended):**")
        st.info(q_info.get("summary"))

    # If there's a persisted last check for this user+topic, show it prominently
    last_checks_user = st.session_state.last_answer_checks.get(username, {})
    last_for_topic = last_checks_user.get(q_info.get("topic"))
    if last_for_topic:
        st.markdown("---")
        st.markdown("#### Last auto-check (persisted)")
        st.write(f"**Checked at:** {last_for_topic.get('timestamp')}")
        st.write(f"**Keyword heuristic score:** {last_for_topic.get('keyword_score_display')}")
        if last_for_topic.get("bart_score_display") is not None:
            st.write(f"**BART overlap score:** {last_for_topic.get('bart_score_display')}")
        if last_for_topic.get("ref_summary"):
            st.write("**Reference summary (BART):**")
            st.info(last_for_topic.get("ref_summary"))
        if last_for_topic.get("ans_summary"):
            st.write("**Your answer summary (BART):**")
            st.info(last_for_topic.get("ans_summary"))
        if last_for_topic.get("matched_terms"):
            st.write("**Matched key concepts (from BART summaries):**")
            st.write(", ".join(last_for_topic.get("matched_terms")[:15]))

with acol:
    st.markdown("### Quick stats")
    db_stats = load_db()
    attempts_stats = [
        a for a in db_stats.get("attempts", []) if a.get("user") == username
    ]
    cards_stats = [c for c in db_stats.get("cards", []) if c.get("user") == username]
    st.metric("Attempts", len(attempts_stats))
    st.metric("Cards", len(cards_stats))
    if attempts_stats:
        avg_quality_stat = sum(a.get("quality", 0) for a in attempts_stats) / len(
            attempts_stats
        )
        st.metric("Avg quality", f"{avg_quality_stat:.2f}")
    else:
        st.metric("Avg quality", "N/A")

# Answer submission + checking
q_info_for_answer = st.session_state.current_question_info
if q_info_for_answer and q_info_for_answer.get("question"):
    st.markdown("---")
    st.markdown("### Answer the question above:")
    # Encourage multi-line answers and require more than a word
    user_answer = st.text_area(
        "Your answer (please write a multi-line answer with explanations; short one-word responses will be flagged)",
        height=220,
        key=f"answer_input_{username}_{q_info_for_answer.get('topic')}",
        placeholder="Write your multi-line answer here. Explain each point, use examples and sub-parts if the question had them.",
    )

    # if enabled, show automatic answer check
    auto_score = None
    bart_result = None
    
    # Store these variables locally so we can pass them to review_card later
    current_keyword_score = None
    current_bart_score = None
    current_matched_kw = []
    current_missing_kw = []
    current_ref_summary = None
    current_ans_summary = None
    current_matched_terms = []

    if enable_answer_check and user_answer.strip():
        ref_context = q_info_for_answer.get("context", "")
        if ref_context:
            # compute keyword heuristic
            current_keyword_score, current_matched_kw, current_missing_kw = answer_checker.compute_keyword_score(
                user_answer, ref_context
            )
            # compute BART overlap score (if available)
            current_bart_score, current_ref_summary, current_ans_summary, current_matched_terms = answer_checker.compute_bart_overlap_score(
                user_answer, ref_context
            )
            
            # Persist the results so they don't disappear from Session State (immediate UI feedback)
            st.session_state.last_answer_checks.setdefault(username, {})[q_info_for_answer.get("topic")] = {
                "timestamp": datetime.utcnow().isoformat(),
                "keyword_score": current_keyword_score,
                "keyword_score_display": f"{current_keyword_score*100:.1f}%",
                "matched_keywords": current_matched_kw,
                "missing_keywords": current_missing_kw,
                "bart_score": current_bart_score if current_bart_score is not None else None,
                "bart_score_display": f"{current_bart_score*100:.1f}%" if (current_bart_score is not None) else None,
                "ref_summary": current_ref_summary,
                "ans_summary": current_ans_summary,
                "matched_terms": current_matched_terms,
            }
            
            # Show immediate feedback
            st.markdown("#### Automatic answer check (heuristic + BART overlap)")
            st.write(f"Estimated correctness (keyword heuristic): **{current_keyword_score*100:.1f}%**")
            if current_matched_kw:
                st.write("Key ideas you mentioned:")
                st.write(", ".join(current_matched_kw))
            if current_missing_kw:
                st.write("Important ideas you may have missed:")
                st.write(", ".join(current_missing_kw[:10]))
            if current_bart_score is not None:
                st.write(f"Estimated correctness (BART overlap): **{current_bart_score*100:.1f}%**")
                if current_matched_terms:
                    st.write("Matched concepts (from BART summaries):")
                    st.write(", ".join(current_matched_terms[:15]))
                if current_ref_summary:
                    st.write("Reference summary (BART):")
                    st.info(current_ref_summary)
                if current_ans_summary:
                    st.write("Your answer summary (BART):")
                    st.info(current_ans_summary)
            else:
                st.info("BART-based check not available (transformers not loaded).")
        else:
            st.info("No reference context available for automatic checking; showing only keyword heuristic.")
            # Do heuristic against empty content (will be poor)
            current_keyword_score, current_matched_kw, current_missing_kw = answer_checker.compute_keyword_score(user_answer, "")
            st.write(f"Heuristic score: **{current_keyword_score*100:.1f}%**")

    # Provide gentle hint if answer too short
    if user_answer.strip() and len(user_answer.strip().split()) < 25:
        st.warning("Your answer seems short — try writing at least a few sentences or multiple lines so the system can evaluate more robustly.")

    quality = st.slider(
        "Self-assess recall quality (0=forgot, 5=perfect)",
        0,
        5,
        4,
        key=f"quality_slider_{username}_{q_info_for_answer.get('topic')}",
    )

    if st.button("Submit answer & schedule review", key=f"submit_button_{username}"):
        topic_to_submit = q_info_for_answer.get("topic")
        if topic_to_submit:
            # Prepare detailed stats dict to save to DB
            submission_details = {
                "user_answer": user_answer,
                "keyword_score": current_keyword_score,
                "matched_keywords": current_matched_kw,
                "missing_keywords": current_missing_kw,
                "bart_score": current_bart_score,
                "matched_terms": current_matched_terms,
                "ref_summary": current_ref_summary,
                "ans_summary": current_ans_summary,
                "question_text": q_info_for_answer.get("question") # Save question too
            }
            
            # Pass details to review_card
            updated = sm2.review_card(username, topic_to_submit, int(quality), details=submission_details)
            
            if updated:
                st.success(
                    f"✅ Logged! Next review for '{topic_to_submit}': {updated['due_date']} "
                    f"(interval={updated['interval']}d, efactor={round(updated['efactor'],2)})"
                )
                # Clear current question so user can generate a new one if desired
                st.session_state.current_question_info = {}
                # Do not clear generated_questions history; we want to avoid repeats in future
                st.rerun()
            else:
                st.error("Failed to log review. Please check logs or DB file.")
        else:
            st.error("Cannot submit answer: Topic information missing.")

st.markdown("---")
st.header("PDF Import — add content from PDFs")
if not _pdf_lib_available:
    st.warning(
        "No PDF library available. Install `pdfplumber` or `PyPDF2` to enable PDF import."
    )
else:
    with st.expander("Upload PDF and map to subject/unit/topic"):
        uploaded_files = st.file_uploader(
            "Upload PDF file(s)", type=["pdf"], accept_multiple_files=True
        )
        target_subject = st.selectbox(
            "Target subject", [""] + subjects, key="pdf_subject"
        )
        target_unit = None
        if target_subject:
            units = road.list_units(target_subject)
            target_unit = st.selectbox(
                "Target unit (existing)", [""] + units, key="pdf_unit"
            )
        custom_unit = st.text_input(
            "Or type a custom unit name", value="", key="pdf_custom_unit"
        )
        custom_topic = st.text_input(
            "Target topic name (for content)", value="", key="pdf_custom_topic"
        )
        if uploaded_files:
            if st.button("Process Uploaded PDF(s)"):
                num_processed = 0
                for up in uploaded_files:
                    with st.spinner(f"Processing {up.name}..."):
                        bytes_data = up.read()
                        text = extract_text_from_pdf_bytes(bytes_data)
                        if not text or not text.strip():
                            st.error(
                                f"Failed to extract text from {up.name} — PDF may be empty, scanned, or protected."
                            )
                            continue
                        subj = target_subject or "Misc"
                        unit = custom_unit.strip() or target_unit or "Imported"
                        topic_name = (
                            custom_topic.strip()
                            or Path(up.name).stem.replace("_", " ").title()
                        )
                        safe_filename_base = f"{subj}_{topic_name}".replace(" ", "_")
                        counter = 0
                        safe_name = f"{safe_filename_base}.txt"
                        p = CONTENT_DIR / safe_name
                        while p.exists():
                            counter += 1
                            safe_name = f"{safe_filename_base}_{counter}.txt"
                            p = CONTENT_DIR / safe_name
                        try:
                            p.write_text(text)
                            st.success(
                                f"Imported content from '{up.name}' to '{p.name}' (Topic: '{topic_name}')"
                            )
                            num_processed += 1
                        except Exception as e:
                            st.error(f"Failed to save content for {up.name}: {e}")
                            continue
                        # Add to syllabus
                        try:
                            syllabus = json.loads(SYLLABUS_FILE.read_text())
                        except Exception:
                            syllabus = {}
                        syllabus.setdefault(subj, {})
                        syllabus[subj].setdefault(unit, [])
                        if topic_name not in syllabus[subj][unit]:
                            syllabus[subj][unit].append(topic_name)
                        try:
                            SYLLABUS_FILE.write_text(json.dumps(syllabus, indent=2))
                            st.info(
                                f"Syllabus updated: added topic '{topic_name}' under '{subj}' -> '{unit}'"
                            )
                            st.cache_resource.clear()
                        except Exception as e:
                            st.error(f"Failed to save syllabus update: {e}")
                if num_processed > 0:
                    st.info(
                        f"Processed {num_processed} PDF(s). Refresh may be needed to see new topics."
                    )

st.markdown("---")
st.header("Due Today — review scheduled cards")
due_cards = sm2.get_due_cards(username)
if not due_cards:
    st.info("🎉 No cards due today!")
else:
    st.info(f"You have {len(due_cards)} card(s) due for review:")
    with st.form(key="review_form"):
        reviews_to_submit = {}
        for i, c in enumerate(due_cards):
            st.markdown("---")
            st.markdown(f"**Reviewing:** `{c.get('topic', 'Unknown Topic')}`")
            try:
                ef = float(c.get("efactor", 0.0))
                ef_str = f"{ef:.2f}"
            except Exception:
                ef_str = str(c.get("efactor", "N/A"))
            st.caption(
                f"Interval: {c.get('interval', 'N/A')}d | Reps: {c.get('repetitions', 'N/A')} | EFactor: {ef_str}"
            )
            question_text = qgen.generate_template(c.get("topic", ""))
            st.write("**Q:**", question_text)
            answer_key = f"review_ans_{username}_{i}"
            quality_key = f"review_q_{username}_{i}"
            _ = st.text_area("Your answer", key=answer_key, height=100)
            quality_val = st.slider(
                "Self-assess (0-5)", 0, 5, 4, key=quality_key
            )
            reviews_to_submit[i] = {
                "topic": c.get("topic"),
                "quality": quality_val,
            }
        submitted = st.form_submit_button("Submit All Reviews")
        if submitted:
            num_submitted = 0
            with st.spinner("Submitting reviews..."):
                for i, review_data in reviews_to_submit.items():
                    if review_data.get("topic"):
                        updated = sm2.review_card(
                            username,
                            review_data["topic"],
                            int(review_data["quality"]),
                        )
                        if updated:
                            num_submitted += 1
                        else:
                            st.error(
                                f"Failed to submit review for: {review_data['topic']}"
                            )
            if num_submitted > 0:
                st.success(f"Submitted {num_submitted} review(s).")
                st.rerun()
            else:
                st.warning("No reviews were submitted.")

st.markdown("---")
st.header("📊 Progress Dashboard")
db_dash = load_db()
attempts_dash = [
    a for a in db_dash.get("attempts", []) if a.get("user") == username
]
if not attempts_dash:
    st.info(
        "No study data yet — start generating quizzes to build your progress history!"
    )
else:
    df = pd.DataFrame(attempts_dash)
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    df["date"] = df["timestamp"].dt.date
    avg_quality = df["quality"].mean()
    dates = sorted(set(df["date"]))
    streak = 0
    if dates:
        latest_date = max(dates)
        current_date = latest_date
        while current_date in dates:
            streak += 1
            current_date = current_date - timedelta(days=1)
    topics_done = df["topic"].nunique()
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric(
            "Average Recall Quality",
            f"{avg_quality:.2f}" if not pd.isna(avg_quality) else "N/A",
        )
    with col2:
        st.metric("Active Streak (days)", f"{streak}")
    with col3:
        st.metric("Topics Attempted", f"{topics_done}")
    st.markdown("#### 📈 Recall Quality Over Time (all subjects)")
    if not df.empty and "date" in df.columns and "quality" in df.columns:
        daily = df.groupby("date")["quality"].mean().reset_index()
        fig, ax = plt.subplots(figsize=(10, 4))
        ax.plot(
            daily["date"],
            daily["quality"],
            marker="o",
            linewidth=2,
            markersize=5,
            linestyle="-",
        )
        ax.set_xlabel("Date")
        ax.set_ylabel("Average Quality")
        ax.grid(True, linestyle="--", alpha=0.6)
        plt.xticks(rotation=45)
        plt.tight_layout()
        st.pyplot(fig)
        plt.close(fig)
    else:
        st.info("Not enough data to plot quality over time.")
    st.markdown("#### 🔍 Attempts Table (last 40)")
    st.dataframe(df.sort_values("timestamp", ascending=False).head(40))

    # --- NEW SECTION: Detailed Answer History ---
    st.markdown("---")
    st.markdown("#### 📜 Detailed Answer History & Correctness Re-check")
    st.write("Expand an attempt below to see your answer and how it was scored.")
    
    # Sort attempts by latest first
    sorted_attempts = sorted(attempts_dash, key=lambda x: x["timestamp"], reverse=True)
    
    for i, att in enumerate(sorted_attempts[:20]):  # Show last 20 for performance
        ts_str = att.get("timestamp", "").replace("T", " ")[:16]
        topic_str = att.get("topic", "Unknown")
        qual = att.get("quality", "?")
        
        # Check if we have detailed stats saved
        has_details = "keyword_score" in att or "bart_score" in att
        icon = "📝" if has_details else "📅"
        
        with st.expander(f"{icon} {ts_str} | {topic_str} | Quality: {qual}/5"):
            col_d1, col_d2 = st.columns(2)
            with col_d1:
                st.markdown("**Your Answer:**")
                st.info(att.get("user_answer", "No text saved for this attempt."))
                if att.get("question_text"):
                    st.caption(f"Question: {att.get('question_text')}")
            
            with col_d2:
                if has_details:
                    kw_score = att.get("keyword_score")
                    bart_score = att.get("bart_score")
                    
                    if kw_score is not None:
                        st.metric("Keyword Score", f"{kw_score*100:.1f}%")
                    if bart_score is not None:
                        st.metric("BART Score", f"{bart_score*100:.1f}%")
                    
                    matched = att.get("matched_keywords", [])
                    missing = att.get("missing_keywords", [])
                    if matched:
                        st.write("**Matched:**", ", ".join(matched))
                    if missing:
                        st.write("**Missing:**", ", ".join(missing[:10]))
                else:
                    st.warning("Detailed metrics were not captured for this older attempt.")
            
            # Show Summaries if available
            if has_details:
                ref_sum = att.get("ref_summary")
                ans_sum = att.get("ans_summary")
                if ref_sum:
                    st.markdown("**Reference Summary (AI):**")
                    st.success(ref_sum)
                if ans_sum:
                    st.markdown("**Your Answer Summary (AI):**")
                    st.info(ans_sum)
    # -----------------------------------------------------------

st.markdown("---")
st.header("Per-subject dashboards")
subjects_dash = road.list_subjects()
sel_subject = st.selectbox(
    "Choose subject to inspect", [""] + subjects_dash, key="per_subject_select"
)
if sel_subject:
    subject_topics_set = set(road.flatten_subject_topics(sel_subject))
    s_attempts = [
        a for a in attempts_dash if a.get("topic") in subject_topics_set
    ]
    if not s_attempts:
        st.info("No attempts yet for this subject.")
    else:
        sdf = pd.DataFrame(s_attempts)
        sdf["timestamp"] = pd.to_datetime(sdf["timestamp"])
        sdf["date"] = sdf["timestamp"].dt.date
        avg_q = sdf["quality"].mean()
        attempts_count = len(sdf)
        topics_covered = sdf["topic"].nunique()
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric(
                f"{sel_subject} — Avg Quality",
                f"{avg_q:.2f}" if not pd.isna(avg_q) else "N/A",
            )
        with col2:
            st.metric(f"{sel_subject} — Attempts", f"{attempts_count}")
        with col3:
            st.metric(f"{sel_subject} — Topics Covered", f"{topics_covered}")
        if not sdf.empty and "date" in sdf.columns and "quality" in sdf.columns:
            sdaily = sdf.groupby("date")["quality"].mean().reset_index()
            fig2, ax2 = plt.subplots(figsize=(10, 4))
            ax2.plot(
                sdaily["date"],
                sdaily["quality"],
                marker="o",
                linewidth=2,
                markersize=5,
                linestyle="-",
            )
            ax2.set_title(f"{sel_subject} — Recall quality over time")
            ax2.set_xlabel("Date")
            ax2.set_ylabel("Average Quality")
            ax2.grid(True, linestyle="--", alpha=0.6)
            plt.xticks(rotation=45)
            plt.tight_layout()
            st.pyplot(fig2)
            plt.close(fig2)
        else:
            st.info(
                "Not enough data for this subject to plot quality over time."
            )
        st.markdown("#### Per-topic performance")
        topic_stats = (
            sdf.groupby("topic")["quality"]
            .agg(["mean", "count"])
            .reset_index()
            .sort_values("mean", ascending=True)
        )
        st.dataframe(topic_stats)
        cards_dash = db_dash.get("cards", [])
        subj_cards = [
            c
            for c in cards_dash
            if c.get("user") == username and c.get("topic") in subject_topics_set
        ]
        if subj_cards:
            st.markdown("#### Cards for subject")
            df_subj_cards = pd.DataFrame(subj_cards)
            st.dataframe(
                df_subj_cards[
                    ["topic", "interval", "repetitions", "efactor", "due_date"]
                ]
            )

st.markdown("---")
st.subheader("Debug: DB snapshot (read-only)")
if st.checkbox("Show raw database content"):
    try:
        st.code(DB_FILE.read_text())
    except Exception:
        st.write("Failed to read DB file")