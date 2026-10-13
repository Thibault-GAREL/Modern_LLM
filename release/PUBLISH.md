# Publishing M.A.I on the Hugging Face Hub

Two repositories, both under the `Thibault-GAREL` account:

| Repo | Type | Content |
|---|---|---|
| `Thibault-GAREL/mai-v2` | model | `mai-v2.pt` (444 MB, weights only) and the model card |
| `Thibault-GAREL/mai` | Space, Gradio, free CPU | `release/space/`, which downloads the weights above at start |

```powershell
& c:\0-Code_py_temp\pytorch_py312\Scripts\Activate.ps1
cd D:\Loisir\Code_python\Modern_LLM
hf auth login                                   # a token with write access, once

# the weights and their card
hf upload Thibault-GAREL/mai-v2 outputs/models/mai-v2/mai-v2.pt mai-v2.pt
hf upload Thibault-GAREL/mai-v2 release/model_card/README.md README.md

# the Space: create it once in the web UI (SDK Gradio, hardware CPU basic),
# or with `hf repo create` (check `hf repo create --help` for the Space flags), then
hf upload Thibault-GAREL/mai release/space . --repo-type space
```

The Space installs this library from GitHub (`release/space/requirements.txt`),
so the commit it needs must be pushed first. It starts in about two minutes,
then a 100 token answer takes a few seconds on the free CPU.

For a later version, change `MAI_REPO` and `MAI_FILE` in the Space settings
(Variables) rather than the code.
