"""Build the architecture files read by the visualiser (visualisation/arch/<name>.json).

    python visualisation/build_arch.py                                   # M.A.I v2
    python visualisation/build_arch.py --config configs/moe_1b_a200m.yaml
    python visualisation/build_arch.py --config configs/moe_1b_a200m.yaml \
        --set model.n_layers=4 --set model.d_model=256 --name moe-demo

The model is built from the config exactly as mllm.train builds it, measured on a
dummy batch by archkit.Probe, then described block by block below. Variants are
read from the instantiated modules (MoE or dense, local or global window), never
recomputed from the config, so the diagram cannot drift from the code.

Every block points at its module, which is how parameters, tensor shapes and the
layer/ metrics logged by mllm.layer_stats attach to it.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(ROOT / "src"))

from archkit import Probe, link, node, write  # noqa: E402

from mllm.config import Config  # noqa: E402
from mllm.layers.ffn import MatFormerMLP  # noqa: E402
from mllm.layers.moe import MoE  # noqa: E402
from mllm.model import Transformer  # noqa: E402
from mllm.train import apply_overrides  # noqa: E402

# dummy batch: values chosen so they never collide with a real dimension
B, T = 5, 37
# main path cards are drawn larger so the overview stays readable
MAIN_SCALE = 3

ATTN_NAMES = {
    "mha": "Multi-Head Attention", "mqa": "Multi-Query Attention",
    "gqa": "Grouped-Query Attention", "mla": "Multi-head Latent Attention",
}
ATTN_REFS = {
    "mha": "Vaswani et al. 2017, arXiv 1706.03762",
    "mqa": "Shazeer 2019, arXiv 1911.02150",
    "gqa": "Ainslie et al. 2023, arXiv 2305.13245",
    "mla": "DeepSeek-V2 2024, arXiv 2405.04434",
}
FFN_NAMES = {"mlp": "MLP", "swiglu": "SwiGLU", "geglu": "GeGLU", "reglu": "ReGLU"}
GATE_ACT = {"swiglu": "SiLU", "geglu": "GELU", "reglu": "ReLU"}
NORM_NAMES = {"rmsnorm": "RMSNorm", "layernorm": "LayerNorm", "dyt": "DyT"}
POS_NAMES = {
    "rope": "RoPE", "alibi": "ALiBi", "sinusoidal": "sinusoïdale",
    "learned": "apprise", "nope": "aucune (NoPE)",
}

# the explanations shown in the detail panel, kept here so the builder stays short
DOCS = {
    "rmsnorm": "x / RMS(x) · w, calculé en fp32. Pas de recentrage ni de biais, plus simple "
               "et aussi bon que LayerNorm (Zhang et Sennrich 2019, arXiv 1910.07467).",
    "layernorm": "La normalisation de 2017 : moyenne et variance, gain et biais appris.",
    "dyt": "Dynamic Tanh : tanh(α·x) remplace la normalisation, sans statistique à calculer "
           "(arXiv 2503.10622).",
    "residual": "Connexion résiduelle : la sortie du sous-bloc s'ajoute au flux principal.",
    "qk_norm": "RMSNorm sur chaque tête de q et k, qui borne les logits d'attention "
               "(arXiv 2010.04245).",
    "rope": "Tourne chaque paire de coordonnées de q et k d'un angle proportionnel à la "
            "position. Le produit q·k ne dépend alors que de la distance entre les tokens. "
            "Tables en fp32.",
    "sdpa": "F.scaled_dot_product_attention (kernel fusionné) sauf avec softcap ou puits "
            "appris, qui passent par le chemin naïf. Le masque causal cache les tokens futurs.",
    "o_proj": "Recombine les têtes et revient dans le flux résiduel.",
    "kv_a": "Une seule projection produit le latent compressé c_KV et la clé rotative partagée.",
    "kv_b": "Décompresse le latent en k_nope et v. À l'inférence, ces poids sont absorbés "
            "dans W_Q et W_O.",
    "mla": "Les clés et valeurs passent par un latent de faible rang : le cache ne garde que "
           "ce latent et la petite clé rotative, au lieu de toutes les têtes K/V.",
    "mlp": "Le bloc de 2017 : une activation entre deux matrices.",
    "gate": "La porte filtre, coordonnée par coordonnée, ce qui passe du chemin valeur.",
    "experts": "Experts fins (DeepSeekMoE) : beaucoup de petits experts multiplient les "
               "combinaisons possibles.",
    "shared": "Toujours actifs : le savoir commun n'est stocké qu'une fois au lieu d'être "
              "dupliqué dans chaque expert.",
    "router_free": "Le biais par expert ne sert qu'à choisir le top-k, jamais à pondérer : "
                   "l'équilibrage ne touche pas au calcul du modèle (DeepSeek-V3, "
                   "arXiv 2412.19437).",
    "router_aux": "Équilibrage par perte auxiliaire (Switch, arXiv 2101.03961).",
    "moe": "Chaque token n'active que quelques experts : les paramètres grandissent, le calcul "
           "par token non. La charge des experts (mesure « Charge ») montre un routeur qui "
           "s'effondre bien avant la loss.",
    "moe_sum": "Σ g_i · E_i(x) sur les experts choisis, plus les experts partagés.",
    "tokens": "Le batch d'entrée, des identifiants de tokens.",
    "embed": "Table de correspondance token → vecteur. Liée à la tête de sortie, elle "
             "économise V × d paramètres.",
    "pos_abs": "Embedding de position absolu ajouté à l'entrée (2017).",
    "trunk": "La pile de blocs décodeurs. Chaque bloc lit et réécrit le flux résiduel de "
             "dimension d_model.",
    "lm_head": "Projection vers le vocabulaire, logits en fp32 (un softmax sur 32k en bf16 "
               "perd la queue).",
    "loss": "Chaque terme est journalisé séparément (loss/ce, loss/z_loss, loss/mtp, loss/moe) : "
            "fusionnés, un routeur qui diverge deviendrait invisible.",
    "mtp": "Chaque profondeur prédit un token de plus, en partant de l'état du tronc et de "
           "l'embedding du token futur. Le signal d'apprentissage se densifie. À l'inférence "
           "les modules sont retirés, ou servent de brouillon.",
    "root": "Le modèle complet, mesuré sur le vrai code de mllm.model.Transformer.",
}


def n(x: int) -> str:
    """Thousands with a thin space, the way the panel shows them."""
    return f"{x:,}".replace(",", " ")


class Builder:
    def __init__(self, model: Transformer, probe: Probe) -> None:
        self.model = model
        self.probe = probe
        self.cfg = model.cfg
        self.d = self.cfg.d_model

    # -- small pieces ---------------------------------------------------------

    def norm(self, module: str, label: str | None = None) -> dict:
        c = self.cfg.norm
        info = [f"ε {c.eps:g}"] + (["poids (1 + w)"] if c.unit_offset else [])
        return node(label or NORM_NAMES[c.kind], "norm", module=module, info=info,
                    doc=DOCS[c.kind])

    def linear(self, module: str, label: str, extra: list[str] | None = None,
               doc: str | None = None) -> dict:
        lin = self.model.get_submodule(module)
        info = [f"{n(lin.in_features)} → {n(lin.out_features)}"] + (extra or [])
        if lin.bias is not None:
            info.append("avec biais")
        return node(label, "linear", module=module, info=info, doc=doc)

    def add(self) -> dict:
        return node("Addition", "add", doc=DOCS["residual"])

    def scaled_init_text(self) -> list[str]:
        return ["init / √(2·L)"] if self.cfg.init.scaled_residual else []

    # -- attention --------------------------------------------------------------

    def attention(self, prefix: str) -> dict:
        attn = self.model.get_submodule(prefix)
        a = self.cfg.attention
        if attn.kind == "mla":
            return self.mla(prefix, attn)
        H, KV, hd = attn.n_heads, attn.n_kv_heads, attn.head_dim
        rope = self.cfg.position.kind == "rope"

        def branch(proj: str, label: str, heads: int, rotate: bool) -> dict:
            plural = "s" if heads > 1 else ""
            steps = [self.linear(f"{prefix}.{proj}", label, [f"{heads} tête{plural} × {hd}"])]
            qk = node(f"QK-norm ({proj[0]})", "norm", module=f"{prefix}.qk_norm.{proj[0]}_norm",
                      shape=f"[B, {heads}, T, {hd}]", doc=DOCS["qk_norm"])
            if attn.qk_norm is not None and rotate and not a.qk_norm_after_rope:
                steps.append(qk)
            if rope and rotate:
                steps.append(self.rope_node(heads, hd))
            if attn.qk_norm is not None and rotate and a.qk_norm_after_rope:
                steps.append(qk)
            return node(label, "group", box=False, children=steps)

        qkv = node("Q, K, V", "group", box=False, layout="row", flow="par", children=[
            branch("q_proj", "W_Q", H, True),
            branch("k_proj", "W_K", KV, True),
            branch("v_proj", "W_V", KV, False),
        ])
        sharing = {
            "mha": "une tête K/V par tête de requête",
            "mqa": f"une seule tête K/V partagée par les {H} têtes",
            "gqa": f"{KV} têtes K/V, chacune partagée par {H // KV} têtes de requête",
        }[attn.kind]
        return node(
            ATTN_NAMES[attn.kind], "attention", module=prefix,
            info=[f"{H} têtes · {KV} K/V · d_head {hd}", self.window_text(attn)],
            badge="local" if attn.window else None, ref=ATTN_REFS[attn.kind],
            doc=f"Attention causale, {sharing}. Le cache KV ne garde que les têtes K/V, "
                f"d'où le choix de {attn.kind.upper()} pour l'inférence.",
            children=[
                qkv,
                self.sdpa(attn, H, hd),
                self.linear(f"{prefix}.o_proj", "W_O", self.scaled_init_text(),
                            doc=DOCS["o_proj"]),
            ],
        )

    def rope_node(self, heads: int, dim: int, *, label: str = "RoPE") -> dict:
        p = self.cfg.position
        info = [f"θ = {p.rope_theta:g} · {p.rope_style}"]
        if p.scaling is not None:
            info.append(f"{p.scaling.kind} ×{p.scaling.factor:g}")
        return node(label, "rope", info=info, shape=f"[B, {heads}, T, {dim}]",
                    ref="Su et al. 2021, arXiv 2104.09864", doc=DOCS["rope"])

    def window_text(self, attn) -> str:
        if attn.window:
            return f"fenêtre glissante {n(attn.window)} tokens"
        return "causale, contexte complet"

    def sdpa(self, attn, H: int, hd: int) -> dict:
        a = self.cfg.attention
        scale = "1/d" if a.scale == "mup" else f"1/√{hd}"
        info = [f"softmax(QKᵀ · {scale} + masque) · V"]
        extras = []
        if self.cfg.position.kind == "alibi":
            extras.append("biais ALiBi linéaire en distance")
        if a.logit_softcap:
            extras.append(f"softcap {a.logit_softcap:g}")
        if a.attn_sinks:
            extras.append(f"{a.attn_sinks} token(s) puits toujours visibles")
        if a.learned_sink:
            extras.append("puits appris par tête")
        info.append(" · ".join(extras) if extras else self.window_text(attn))
        return node("Attention causale", "sdpa", info=info, doc=DOCS["sdpa"],
                    shape=f"[B, {H}, T, {attn.v_head_dim}]")

    def mla(self, prefix: str, attn) -> dict:
        H = attn.n_heads
        heads_qk = f"{H} × ({attn.qk_nope_head_dim}+{attn.qk_rope_head_dim})"
        q_steps = []
        if attn.q_lora_rank is None:
            q_steps.append(self.linear(f"{prefix}.q_proj", "W_Q", [heads_qk]))
        else:
            q_steps += [
                self.linear(f"{prefix}.q_a_proj", "W_DQ (compression)"),
                self.norm(f"{prefix}.q_a_norm", "RMSNorm latent q"),
                self.linear(f"{prefix}.q_b_proj", "W_UQ", [heads_qk]),
            ]
        q_steps.append(self.rope_node(H, attn.qk_rope_head_dim, label="RoPE (partie rope de q)"))
        latent = node("latent", "group", box=False, children=[
            self.norm(f"{prefix}.kv_a_norm", "RMSNorm latent"),
            self.linear(f"{prefix}.kv_b_proj", "W_UK · W_UV",
                        [f"{H} × ({attn.qk_nope_head_dim}+{attn.v_head_dim})"], doc=DOCS["kv_b"]),
        ])
        kv = node("KV", "group", box=False, children=[
            self.linear(f"{prefix}.kv_a_proj", "W_DKV",
                        [f"latent {attn.kv_lora_rank} + k_rope {attn.qk_rope_head_dim}"],
                        doc=DOCS["kv_a"]),
            node("Séparation", "split", doc="c_KV d'un côté, k_rope de l'autre."),
            node("latent | k_rope", "group", box=False, layout="row", flow="par", children=[
                latent,
                self.rope_node(1, attn.qk_rope_head_dim, label="RoPE (k partagée)"),
            ]),
        ])
        q = node("Q", "group", box=False, children=q_steps)
        branches = node("Q | KV", "group", box=False, layout="row", flow="par", children=[q, kv])
        cache = (f"cache : {attn.kv_lora_rank} + {attn.qk_rope_head_dim} "
                 "valeurs par token et par couche")
        return node(
            ATTN_NAMES["mla"], "attention", module=prefix, ref=ATTN_REFS["mla"],
            info=[f"{H} têtes · latent {attn.kv_lora_rank}", cache],
            badge="local" if attn.window else None, doc=DOCS["mla"],
            children=[branches, self.sdpa(attn, H, attn.qk_head_dim),
                      self.linear(f"{prefix}.o_proj", "W_O", self.scaled_init_text())],
        )

    # -- feed-forward -------------------------------------------------------------

    def ffn(self, prefix: str, *, label: str | None = None, kind: str = "ffn") -> dict:
        mlp = self.model.get_submodule(prefix)
        k = self.cfg.ffn.kind
        hidden = f"[B, T, {n(mlp.d_ff)}]"
        info = [f"{n(self.d)} → {n(mlp.d_ff)} → {n(self.d)}"]
        if isinstance(mlp, MatFormerMLP):
            info.append("MatFormer " + " / ".join(f"{g:g}" for g in mlp.granularities))
        down = self.linear(f"{prefix}.down_proj", "W_down", self.scaled_init_text())
        if k == "mlp":
            kids = [self.linear(f"{prefix}.up_proj", "W_up"),
                    node(self.cfg.ffn.activation.upper(), "act", shape=hidden), down]
            doc = DOCS["mlp"]
        else:
            gate = node("porte", "group", box=False, children=[
                self.linear(f"{prefix}.gate_proj", "W_gate"),
                node(GATE_ACT[k], "act", shape=hidden),
            ])
            kids = [
                node("porte | valeur", "group", box=False, layout="row", flow="par",
                     children=[gate, self.linear(f"{prefix}.up_proj", "W_up")]),
                node("Produit", "mul", doc=DOCS["gate"]),
                down,
            ]
            doc = (f"down({GATE_ACT[k]}(gate(x)) ⊙ up(x)). Trois matrices, d_ff ≈ 8/3·d pour "
                   "garder le budget d'un MLP 4d (Shazeer 2020, arXiv 2002.05202).")
        return node(label or f"{FFN_NAMES[k]} feed-forward", kind, module=prefix, info=info,
                    doc=doc, children=kids)

    def moe(self, prefix: str, moe: MoE) -> dict:
        c = self.cfg.moe
        d_ff = moe.experts[0].d_ff
        ffn_name = FFN_NAMES[self.cfg.ffn.kind]
        experts = [
            node(f"Expert {j}", "expert", module=f"{prefix}.experts.{j}",
                 info=[f"{ffn_name} {n(d_ff)}"])
            for j in range(len(moe.experts))
        ]
        routed = node(f"{len(experts)} experts routés", "moe", module=f"{prefix}.experts",
                      layout="grid", flow="none", cols=8 if len(experts) > 16 else 4,
                      children=experts, info=[f"chaque token en active {c.top_k}"],
                      doc=DOCS["experts"])
        parts = [routed]
        if moe.shared_experts:
            shared = [node(f"Partagé {j}", "expert", module=f"{prefix}.shared_experts.{j}",
                           info=[n(d_ff)])
                      for j in range(len(moe.shared_experts))]
            parts.append(node("Experts partagés", "ffn", module=f"{prefix}.shared_experts",
                              flow="none", children=shared, info=["toujours actifs"],
                              doc=DOCS["shared"]))
        free = c.balance == "aux_loss_free"
        router = node(
            "Routeur", "router", module=f"{prefix}.router",
            info=[f"{n(self.d)} → {c.n_experts} scores · {c.gate}",
                  f"top-{c.top_k}" + (" · biais de sélection" if free else "")],
            doc=DOCS["router_free" if free else "router_aux"],
        )
        return node(
            "Mixture of Experts", "moe", module=prefix, badge=f"top-{c.top_k}",
            info=[f"{c.n_experts} routés · top-{c.top_k} · {len(moe.shared_experts)} partagés",
                  f"d_ff expert {n(d_ff)}"],
            ref="DeepSeekMoE 2024, arXiv 2401.06066", doc=DOCS["moe"],
            children=[router,
                      node("experts", "group", box=False, layout="row", flow="par",
                           children=parts),
                      node("Somme pondérée", "add", doc=DOCS["moe_sum"])],
        )

    # -- blocks -----------------------------------------------------------------------

    def residual(self, label: str, inner: dict, norm_prefix: str) -> dict:
        placement = self.cfg.norm.placement
        if placement == "post":
            kids = [inner, self.add(), self.norm(f"{norm_prefix}.norm_in")]
            return node(label, "group", box=False, skip="addition", children=kids)
        kids = [self.norm(f"{norm_prefix}.norm_in"), inner]
        if placement == "sandwich":
            kids.append(self.norm(f"{norm_prefix}.norm_out", "Norm de sortie"))
        kids.append(self.add())
        return node(label, "group", box=False, skip=True, children=kids)

    def block(self, prefix: str, label: str) -> dict:
        blk = self.model.get_submodule(prefix)
        sub = blk.ffn.sublayer
        is_moe = isinstance(sub, MoE)
        attn = self.attention(f"{prefix}.attn.sublayer")
        ffp = f"{prefix}.ffn.sublayer"
        ff = self.moe(ffp, sub) if is_moe else self.ffn(ffp)
        badge = None
        if self.cfg.moe.enabled and not is_moe:
            badge = "dense"
        elif blk.attn.sublayer.window:
            badge = "local"
        kind = "MoE" if is_moe else FFN_NAMES[self.cfg.ffn.kind]
        return node(
            label, "block", module=prefix, badge=badge,
            info=[f"{self.cfg.attention.kind.upper()} + {kind}",
                  f"norm {self.cfg.norm.placement}"],
            children=[self.residual("attention résiduelle", attn, f"{prefix}.attn"),
                      self.residual("feed-forward résiduel", ff, f"{prefix}.ffn")],
        )

    # -- whole model --------------------------------------------------------------------

    def entry(self) -> list[dict]:
        cfg = self.cfg
        pos = cfg.position.kind
        tokens = node("Tokens", "input", info=[f"ids dans [0, {n(cfg.vocab_size)})"],
                      shape="[B, T]", scale=MAIN_SCALE, doc=DOCS["tokens"])
        tied = ["lié à la tête LM"] if cfg.tie_embeddings else []
        embed = node("Embedding", "embed", module="embed", id="embed", scale=MAIN_SCALE,
                     info=[f"{n(cfg.vocab_size)} × {n(self.d)}"] + tied, doc=DOCS["embed"])
        if pos not in ("sinusoidal", "learned"):
            return [tokens, embed]
        pos_node = node(f"Position {POS_NAMES[pos]}", "pos",
                        module="pos.embedding" if pos == "learned" else None,
                        info=[f"{n(cfg.max_seq_len)} positions"], shape=f"[T, {n(self.d)}]",
                        doc=DOCS["pos_abs"])
        return [tokens, node("entrée", "group", box=False, children=[
            node("token | position", "group", box=False, layout="row", flow="par",
                 children=[embed, pos_node]),
            node("Addition", "add", doc="x = emb(token) + pos(t)"),
        ])]

    def exit(self) -> list[dict]:
        cfg = self.cfg
        V = cfg.vocab_size
        head_src = "poids de l'embedding" if cfg.tie_embeddings else "matrice propre"
        return [
            self.norm("norm_f", "Norm finale") | {"id": "norm_f", "scale": MAIN_SCALE},
            node("Tête LM", "head", module="lm_head", id="lm_head", scale=MAIN_SCALE,
                 info=[f"{n(self.d)} → {n(V)}", head_src], doc=DOCS["lm_head"]),
            node("Logits", "output", shape=f"[B, T, {n(V)}]", info=["token suivant"],
                 scale=MAIN_SCALE),
            node("Pertes", "loss", id="loss", info=self.loss_lines(), scale=MAIN_SCALE,
                 doc=DOCS["loss"]),
        ]

    def mtp(self, links: list[dict]) -> dict:
        cfg = self.cfg
        V = cfg.vocab_size
        mods = []
        for k in range(cfg.mtp_depth):
            p = f"mtp.modules_.{k}"
            fut = node(f"emb(t+{k + 1})", "group", box=False, children=[
                node("Embedding partagé", "embed", id=f"mtp_emb_{k}", info=[f"token t+{k + 1}"],
                     shape=f"[B, T−{k + 1}, {n(self.d)}]"),
                self.norm(f"{p}.norm_embed", "RMSNorm emb"),
            ])
            prev_label = "RMSNorm h" if k == 0 else f"RMSNorm h{k}"
            prev = self.norm(f"{p}.norm_hidden", prev_label) | {"id": f"mtp_in_{k}"}
            mods.append(node(
                f"Profondeur {k + 1} · prédit t+{k + 2}", "mtp", module=p, id=f"mtp_{k}",
                info=[f"h_{k} ⊕ emb(t+{k + 1}) → bloc → tête"],
                children=[
                    node("h | emb", "group", box=False, layout="row", flow="par",
                         children=[prev, fut]),
                    node("Concaténation", "concat",
                         doc=f"[norm(h) ; norm(emb)] : 2·{self.d} coordonnées."),
                    self.linear(f"{p}.proj", "Projection",
                                doc="Ramène la concaténation à d_model."),
                    self.block(f"{p}.block", f"Bloc MTP {k + 1}"),
                    node("Tête LM partagée", "head", id=f"mtp_head_{k}", info=["mêmes poids"],
                         shape=f"[B, T−{k + 1}, {n(V)}]"),
                ],
            ))
            links.append(link("embed", f"mtp_emb_{k}", f"emb(t+{k + 1})", kind="embed"))
            links.append(link(f"mtp_head_{k}", "lm_head", "partagée", kind="head"))
        links.insert(0, link("norm_f", "mtp_in_0", "h du tronc", kind="mtp"))
        links.append(link("mtp", "loss", f"λ = {cfg.mtp_lambda:g}", kind="loss"))
        return node(
            "Multi-Token Prediction", "mtp", module="mtp", id="mtp", layout="row",
            info=[f"profondeur {cfg.mtp_depth} · λ = {cfg.mtp_lambda:g}",
                  "retiré à l'inférence ou drafter spéculatif"],
            ref="Gloeckle et al. 2024, arXiv 2404.19737 · DeepSeek-V3", doc=DOCS["mtp"],
            children=mods,
        )

    def build(self, title: str) -> tuple[dict, list[dict]]:
        cfg = self.cfg
        links: list[dict] = []
        blocks = [self.block(f"blocks.{i}", f"Bloc {i}") for i in range(cfg.n_layers)]
        trunk = node(f"Tronc · {cfg.n_layers} blocs", "stack", module="blocks", id="trunk",
                     layout="row", badge=f"×{cfg.n_layers}", children=blocks,
                     info=[f"d_model {n(self.d)}", f"position {POS_NAMES[cfg.position.kind]}"],
                     doc=DOCS["trunk"])
        main = node("chemin principal", "group", box=False, layout="row",
                    children=[*self.entry(), trunk, *self.exit()])
        children = [main]
        if cfg.tie_embeddings:
            links.append(link("embed", "lm_head", "poids partagés", kind="embed"))
        if self.model.mtp is not None:
            children.append(self.mtp(links))
        root = node(title, "stack", layout="col", flow="none", children=children,
                    info=[self.subtitle()], doc=DOCS["root"])
        return root, links

    def loss_lines(self) -> list[str]:
        lines = ["cross-entropy"]
        if self.model.z_loss_coef:
            lines.append(f"z-loss {self.model.z_loss_coef:g}")
        if self.model.mtp is not None:
            lines.append(f"MTP × {self.cfg.mtp_lambda:g}")
        if self.cfg.moe.enabled:
            lines.append("aux MoE + z-loss routeur")
        return lines

    def subtitle(self) -> str:
        c = self.cfg
        ff = "MoE" if c.moe.enabled else FFN_NAMES[c.ffn.kind]
        parts = [f"{c.n_layers} blocs", c.attention.kind.upper(), ff, POS_NAMES[c.position.kind],
                 f"{NORM_NAMES[c.norm.kind]} {c.norm.placement}"]
        if c.mtp_depth:
            parts.append(f"MTP ×{c.mtp_depth}")
        return "décodeur seul · " + " · ".join(parts)

    def summary(self) -> list[tuple[str, str]]:
        c, m = self.cfg, self.model
        att = m.blocks[0].attn.sublayer
        d_ff = getattr(m.blocks[0].ffn.sublayer, "d_ff", None)
        if att.kind == "mla":
            heads = f"{att.n_heads} (MLA)"
        else:
            heads = f"{att.n_heads} / {att.n_kv_heads} KV"
        out = [("d_model", n(c.d_model)), ("blocs", str(c.n_layers)), ("têtes", heads),
               ("d_head", str(att.head_dim))]
        if d_ff:
            out.append(("d_ff", n(d_ff)))
        if c.moe.enabled:
            out.append(("actifs", f"{m.n_active_params() / 1e6:.0f} M"))
        out += [("vocab", n(c.vocab_size)), ("contexte", n(c.max_seq_len))]
        return out


def build_one(config: Path, overrides: list[str], name: str | None) -> None:
    cfg = apply_overrides(Config.from_yaml(config), overrides)
    name = name or cfg.name or config.stem
    title = {"mai-v1": "M.A.I v1", "mai-v2": "M.A.I v2"}.get(name, name)
    torch.manual_seed(0)
    model = Transformer(cfg.model, z_loss_coef=cfg.train.z_loss_coef).eval()
    idx = torch.randint(0, cfg.model.vocab_size, (B, T))
    probe = Probe(model, lambda: model(idx, idx), symbols={"B": B, "T": T})
    builder = Builder(model, probe)
    root, links = builder.build(title)
    source = str(config.as_posix()) + "".join(f" --set {o}" for o in overrides)
    write(HERE / "arch" / f"{name}.json", root, probe, name=name, title=title,
          subtitle=builder.subtitle(), summary=builder.summary(), links=links, source=source)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--config", type=Path, action="append", help="profile in configs/, repeatable")
    p.add_argument("--set", dest="overrides", action="append", default=[], metavar="KEY=VALUE",
                   help="same overrides as mllm.train, e.g. --set model.n_layers=4")
    p.add_argument("--name", default=None,
                   help="file name in visualisation/arch/, defaults to the config name")
    args = p.parse_args()
    configs = args.config or [ROOT / "configs" / "mai_v2.yaml"]
    for config in configs:
        build_one(config, args.overrides, args.name if len(configs) == 1 else None)


if __name__ == "__main__":
    main()
