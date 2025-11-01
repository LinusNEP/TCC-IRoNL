"""
This script is based on the paper, "The Conversation is the Command: Interacting with Real-World Autonomous Robots Through Natural Language": https://dl.acm.org/doi/abs/10.1145/3610978.3640723
Its usage is subject to the  Creative Commons Attribution International 4.0 License.
"""
#!/usr/bin/env python
import re, difflib, os, yaml
from rapidfuzz import process, fuzz
from sentence_transformers import SentenceTransformer, util

class DestinationResolver:
    # --------------------- normaliser ---------------------
    _strip_punct = re.compile(r"[’'`]")
    _stop_words  = re.compile(r"\b(the|room|office|area|space|building)\b", re.I)
    def _norm(self, txt: str) -> str:
        txt = self._strip_punct.sub("", txt.lower())
        txt = self._stop_words.sub(" ", txt)
        txt = re.sub(r"\s+", " ", txt).strip()
        return txt

    # ---------------------- startup -----------------------
    def __init__(self, yaml_path: str, fuzzy_cut=80, embed_cut=0.55):
        with open(yaml_path) as f:
            dest_raw = yaml.safe_load(f)["destinations"]
            self.raw = dest_raw
        self.coords, self.alias_map = {}, {}
        all_aliases = []
        for key, info in dest_raw.items():
            self.coords[key] = info["coords"]
            alist = info.get("aliases", []) + [info.get("display_name", key), key]
            for a in alist:
                na = self._norm(a)
                self.alias_map[na] = key
                all_aliases.append(na)
        self._all_aliases = list(set(all_aliases))
        self._fuzzy_cut   = fuzzy_cut
        self._embed_cut   = embed_cut

        # Sentence‑transformer embeddings
        self._model   = SentenceTransformer("paraphrase-MiniLM-L6-v2", device="cpu")
        self._alias_emb = self._model.encode(self._all_aliases, normalize_embeddings=True)

    # --------------------- public API ---------------------
    def resolve(self, user_phrase: str):
        """Return (slug, how) or (None, reason)."""
        q = self._norm(user_phrase)

        if q in self.alias_map:
            return self.alias_map[q], "exact"

        # fuzzy token‑sort
        cand, score, _ = process.extractOne(q, self._all_aliases, scorer=fuzz.token_sort_ratio)
        if score >= self._fuzzy_cut:
            return self.alias_map[cand], f"fuzzy:{score}"

        # semantic embedding
        q_emb = self._model.encode(q, normalize_embeddings=True)
        sims  = util.cos_sim(q_emb, self._alias_emb)[0].cpu().numpy()
        best  = sims.argmax()
        if sims[best] >= self._embed_cut:
            return self.alias_map[self._all_aliases[best]], f"embed:{sims[best]:.2f}"
        return None, "not_found"

