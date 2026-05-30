import torch
from fastapi import FastAPI
from fastapi.responses import HTMLResponse, JSONResponse
from pydantic import BaseModel
from transformers import AutoModelForCausalLM, AutoTokenizer
tok = AutoTokenizer.from_pretrained('./recon_model', trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained('./hf_model', trust_remote_code=True, dtype=torch.bfloat16, device_map='cuda').eval()
app = FastAPI()
PAGE = """<!doctype html><meta name=viewport content="width=device-width,initial-scale=1">
<body style="font-family:sans-serif;max-width:600px;margin:auto;padding:1em">
<h3>Laguna XS.2 dense (my SFT model)</h3>
<textarea id=p rows=3 style="width:100%">Write a Python function that adds two numbers.</textarea>
<button onclick=go()>Generate</button><pre id=o style="white-space:pre-wrap;background:#eee;padding:1em"></pre>
<script>async function go(){o.textContent='...';let r=await fetch('/generate',{method:'POST',headers:{'content-type':'application/json'},body:JSON.stringify({prompt:p.value})});o.textContent=(await r.json()).text}</script>"""
@app.get('/', response_class=HTMLResponse)
def home(): return PAGE
class Req(BaseModel):
    prompt: str
@app.post('/generate')
def gen(r: Req):
    t = tok.apply_chat_template([{'role':'user','content':r.prompt}], tokenize=False, add_generation_prompt=True)
    i = tok(t, return_tensors='pt', add_special_tokens=False).to(model.device)
    with torch.no_grad():
        o = model.generate(**i, max_new_tokens=200, do_sample=False)
    return JSONResponse({'text': tok.decode(o[0][i['input_ids'].shape[1]:], skip_special_tokens=True)})
