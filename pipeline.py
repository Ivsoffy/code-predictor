import re
import sys
from pathlib import Path

import torch
import torch.nn.functional as F
from transformers import AutoModel, AutoTokenizer
from tqdm import tqdm

BASE_DIR = Path(__file__).resolve().parent
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

from LP import function_code, subfunction_code, specialization_code, job_title

from model import FunctionModel  # noqa: E402

WEIGHTS_DIR = BASE_DIR / "function_model_weights"
ckpt = str(WEIGHTS_DIR / "model_weights")
device = "cuda" if torch.cuda.is_available() else "cpu"
tqdm.pandas()

langs = {
    "H":[".net", "c#", "csharp"],
    "L":["c","c++"],
    "M":["go","golang", "го"],
    "N":["delphi","делфи"],
    "O":["java","джава"],
    "R":["perl"],
    "S":["php","backend"],
    "T":["python","питон"],
    "U":["ruby","руби"],
    "V":["scala","скала"]
}

LANG_PATTERNS = {
    key: [
        re.compile(rf"(?<!\w){re.escape(lang.lower())}(?!\w)") for lang in values
    ]
    for key, values in langs.items()
}



class CodeModel:
    def __init__(self, model, name_encode="intfloat/multilingual-e5-base"):
        self.tokenizer = AutoTokenizer.from_pretrained(name_encode)
        self.model_encode = AutoModel.from_pretrained(name_encode).to(device)
        self.model_encode.eval()
        self.model_norm = FunctionModel(model=model)
        self.device = device

        data = torch.load(WEIGHTS_DIR / "code_embeddings.pt", map_location="cpu")
        self.code_ids = data["code_ids"]
        self.normalized_code_desc = data["embeddings"]

    def _clean_text(self, val):
        if val is None:
            return ""
        s = str(val).strip()
        if s.lower() in {"", "nan", "none", "null", "na", "n/a", "nil", "undefined"}:
            return ""
        return s
    
    @staticmethod
    def _normalize_sentence(value):
        if value is None:
            return ""
        text = str(value).strip()
        if text.lower() in {
            "",
            "nan",
            "none",
            "null",
            "na",
            "n/a",
            "nil",
            "undefined",
            "<na>",
        }:
            return ""
        return text

    def _embed(self, sentences, batch_size=64):
        embeddings = []
        for i in range(0, len(sentences), batch_size):
            batch_sents = sentences[i:i + batch_size]
            batch = self.tokenizer(
                batch_sents,
                padding=True,
                truncation=True,
                return_tensors="pt",
                max_length=128
            ).to(self.device)

            with torch.inference_mode():
                outputs = self.model_encode(**batch)
                cls_embeddings = outputs.last_hidden_state[:, 0]
            embeddings.append(cls_embeddings.cpu())
        return torch.cat(embeddings, dim=0)
    
    def _calc_cosine_simularity(self, sentences):
        print("Tokenization..")
        cls_embeddings = self._embed(sentences)
        print("Calculate cosine simularity..")

        emb_norm = F.normalize(cls_embeddings, p=2, dim=1)
        cos_sim = emb_norm @ emb_norm.T
        print(f"Similarity between s1 and s2: {cos_sim[0,1].item():.4f}")
        return cos_sim[0,1].item()
    
    def _detect_ita_lang_suffix(self, title):
        normalized_title = self._normalize_sentence(title).lower()
        if not normalized_title:
            return None
        for key, patterns in LANG_PATTERNS.items():
            if any(pattern.search(normalized_title) for pattern in patterns):
                return key
        return None

    def it_langs(self, df):
        ita_mask = df["predicted_code"] == "ITA"
        if ita_mask.any():
            ita_suffixes = df.loc[ita_mask, job_title].apply(
                self._detect_ita_lang_suffix
            )
            matched_idx = ita_suffixes[ita_suffixes.notna()].index
            df.loc[matched_idx, "predicted_code"] = (
                df.loc[matched_idx, "predicted_code"]
                + "-"
                + ita_suffixes.loc[matched_idx]
            )
        return df
    def predict(self, df, test=False):
        df = df.copy()

        print("Preparing data...")
        df = self.model_norm.predict(df)

        print('Creating embeddings..')
        pred_desc_embeds = self._embed(df['predicted_desc'].tolist())

        print("Calculate cosine simularity..")
        normalized_pred_desc = F.normalize(pred_desc_embeds, dim=1)
        cos_desc = normalized_pred_desc @ self.normalized_code_desc.T

        best_idx = torch.argmax(cos_desc, dim=1)
        best_codes = [self.code_ids[i] for i in best_idx.tolist()]


        df['predicted_code'] = best_codes
        df = self.it_langs(df)
        print("Codes predicted!")

        df[function_code] = df['predicted_code'].str[:2]
        df[subfunction_code] = df['predicted_code'].str[:3]
        df[specialization_code] = df['predicted_code'].apply(lambda x: x[:5] if "-" in x else "")

        if test==True:
            # df['description'] = df['predicted_code'].apply(lambda x: self.codes[x]['description'])
            df['check'] = df['predicted_code'] == df['code']
            print(f"True predictions: {df['check'].sum()}/{df.shape[0]}")
        df = df.drop(columns=['predicted_code','input_text','predicted_desc'])
        return df
