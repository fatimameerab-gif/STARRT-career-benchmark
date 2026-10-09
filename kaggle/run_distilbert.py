import argparse, math, random
import numpy as np, pandas as pd, torch
from torch.utils.data import DataLoader, Dataset
from transformers import AutoTokenizer, AutoModelForSequenceClassification, get_linear_schedule_with_warmup
import common as C


def seed_all(s):
    random.seed(s); np.random.seed(s); torch.manual_seed(s); torch.cuda.manual_seed_all(s)


class DS(Dataset):
    def __init__(self, enc, y=None): self.enc, self.y = enc, y
    def __len__(self): return len(self.enc["input_ids"])
    def __getitem__(self, i):
        item = {k: torch.tensor(v[i]) for k, v in self.enc.items()}
        if self.y is not None: item["labels"] = torch.tensor(self.y[i])
        return item


def train_predict(a, tr_text, tr_y, te_text, classes):
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    tok = AutoTokenizer.from_pretrained(a.model)
    model = AutoModelForSequenceClassification.from_pretrained(a.model, num_labels=len(classes)).to(dev)
    cid = {c: i for i, c in enumerate(classes)}
    enc_tr = tok(list(tr_text), truncation=True, max_length=a.max_len, padding="max_length")
    enc_te = tok(list(te_text), truncation=True, max_length=a.max_len, padding="max_length")
    dl = DataLoader(DS(enc_tr, [cid[y] for y in tr_y]), batch_size=a.batch, shuffle=True)
    opt = torch.optim.AdamW(model.parameters(), lr=a.lr, weight_decay=0.01)
    steps = a.epochs * math.ceil(len(dl))
    sch = get_linear_schedule_with_warmup(opt, int(0.06 * steps), steps)
    scaler = torch.amp.GradScaler(enabled=(dev == "cuda"))
    model.train()
    for ep in range(a.epochs):
        tot = 0.0
        for b in dl:
            b = {k: v.to(dev) for k, v in b.items()}
            with torch.autocast(device_type=dev, dtype=torch.float16, enabled=(dev == "cuda")):
                loss = model(**b).loss
            opt.zero_grad(); scaler.scale(loss).backward(); scaler.unscale_(opt)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(opt); scaler.update(); sch.step(); tot += loss.item()
        print(f"  epoch {ep + 1}/{a.epochs} loss {tot / len(dl):.4f}")
    model.eval(); probs = []
    with torch.no_grad():
        for b in DataLoader(DS(enc_te), batch_size=a.batch * 2):
            b = {k: v.to(dev) for k, v in b.items()}
            with torch.autocast(device_type=dev, dtype=torch.float16, enabled=(dev == "cuda")):
                probs.append(torch.softmax(model(**b).logits.float(), -1).cpu().numpy())
    del model; torch.cuda.empty_cache() if dev == "cuda" else None
    return np.concatenate(probs)


def run(a, dataset, folds):
    df = C.load(dataset); classes = sorted(df.label.unique()); fm, rows = [], []
    for k in folds:
        seed_all(a.seed); print(f"{dataset} fold {k}")
        tr, te = df[df.fold != k], df[df.fold == k]
        if a.limit: tr = tr.sample(min(len(tr), a.limit), random_state=0); te = te.head(a.limit // 4)
        P = train_predict(a, tr.text, tr.label, te.text, classes)
        pred = np.array(classes)[P.argmax(1)]
        fm.append(C.metrics(te.label, pred, P, classes))
        rows.append(pd.DataFrame({"id": te.id, "fold": k, "true": te.label, "pred": pred}))
    p = pd.concat(rows)
    C.save(a.tag, dataset, C.summarize(fm), fm, p, {"top_confusions": C.top_confusions(p.true, p.pred), "config": vars(a)})


def forms(a):
    seed_all(a.seed)
    tr, fo = C.load("djinni"), C.load_forms()
    if a.limit: tr = tr.sample(min(len(tr), a.limit), random_state=0)
    classes = sorted(tr.label.unique()); fo = fo[fo.label.isin(classes)].reset_index(drop=True)
    P = train_predict(a, tr.text, tr.label, fo.text, classes)
    pred = np.array(classes)[P.argmax(1)]
    C.save(a.tag, "forms_from_djinni", C.metrics(fo.label, pred, P, classes),
           preds=pd.DataFrame({"id": fo.id, "true": fo.label, "pred": pred}), extra={"config": vars(a)})


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="distilbert-base-uncased")
    ap.add_argument("--tag", default="distilbert")
    ap.add_argument("--datasets", default="djinni,stackoverflow,faruque")
    ap.add_argument("--folds", default="0,1,2,3,4")
    ap.add_argument("--epochs", type=int, default=3)
    ap.add_argument("--lr", type=float, default=5e-5)
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--max_len", type=int, default=256)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--limit", type=int, default=0, help="small subset for a quick test run")
    ap.add_argument("--skip_forms", action="store_true")
    a = ap.parse_args()
    folds = [int(x) for x in a.folds.split(",")]
    for ds in a.datasets.split(","):
        run(a, ds, folds)
    if not a.skip_forms and C.has_forms():
        forms(a)
