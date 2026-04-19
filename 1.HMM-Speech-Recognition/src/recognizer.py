import numpy as np
from hmmlearn import hmm

# ── EV MODELİ ──────────────────────────────────────────────
# Gizli durumlar: 0=e, 1=v
# Gözlemler:      0=High, 1=Low

model_ev = hmm.CategoricalHMM(n_components=2, n_iter=100)

model_ev.startprob_ = np.array([1.0, 0.0])  # P(e)=1.0, P(v)=0.0

model_ev.transmat_ = np.array([
    [0.6, 0.4],  # e -> e=0.6, e -> v=0.4
    [0.2, 0.8],  # v -> e=0.2, v -> v=0.8
])

model_ev.emissionprob_ = np.array([
    [0.7, 0.3],  # e durumunda: P(High)=0.7, P(Low)=0.3
    [0.1, 0.9],  # v durumunda: P(High)=0.1, P(Low)=0.9
])

# ── OKUL MODELİ ────────────────────────────────────────────
# Gizli durumlar: 0=o, 1=k, 2=u, 3=l
# "OKUL" daha uzun bir kelime, daha düşük High olasılıkları

model_okul = hmm.CategoricalHMM(n_components=4, n_iter=100)

model_okul.startprob_ = np.array([1.0, 0.0, 0.0, 0.0])  # P(o)=1.0

model_okul.transmat_ = np.array([
    [0.5, 0.5, 0.0, 0.0],  # o -> o veya k
    [0.0, 0.4, 0.6, 0.0],  # k -> k veya u
    [0.0, 0.0, 0.3, 0.7],  # u -> u veya l
    [0.0, 0.0, 0.1, 0.9],  # l -> l (son durum)
])

model_okul.emissionprob_ = np.array([
    [0.4, 0.6],  # o: düşük frekanslı ses
    [0.2, 0.8],  # k: daha çok Low
    [0.3, 0.7],  # u: düşük frekans
    [0.1, 0.9],  # l: çok düşük frekans
])

# ── SINIFLANDIRICI FONKSİYONU ───────────────────────────────

def classify_word(observation_sequence):
    """
    Gelen gözlem dizisini EV mi OKUL mu olduğunu sınıflandırır.
    
    Parametreler:
        observation_sequence: list
            Gözlem indeksleri listesi. 0=High, 1=Low
            Örnek: [0, 1] → [High, Low]
    
    Döndürür:
        str: "EV" veya "OKUL"
    """
    obs = np.array(observation_sequence).reshape(-1, 1)

    score_ev   = model_ev.score(obs)
    score_okul = model_okul.score(obs)

    print(f"Gözlem Dizisi : {['High' if x==0 else 'Low' for x in observation_sequence]}")
    print(f"EV   Log-Likelihood : {score_ev:.4f}")
    print(f"OKUL Log-Likelihood : {score_okul:.4f}")
    print(f"Sonuç → {'EV' if score_ev > score_okul else 'OKUL'}")
    print("-" * 45)

    return "EV" if score_ev > score_okul else "OKUL"

# ── TEST ────────────────────────────────────────────────────

if __name__ == "__main__":
    print("=" * 45)
    print("        HMM Kelime Tanıyıcı - Test")
    print("=" * 45)

    # Test 1: [High, Low] → EV beklenir
    classify_word([0, 1])

    # Test 2: [Low, Low, Low, Low] → OKUL beklenir
    classify_word([1, 1, 1, 1])

    # Test 3: [High, High] → EV beklenir
    classify_word([0, 0])

    # Test 4: [Low, Low] → OKUL beklenir
    classify_word([1, 1])