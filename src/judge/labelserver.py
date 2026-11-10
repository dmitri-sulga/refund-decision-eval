"""The labelling page: a tiny local web server (standard library, 127.0.0.1 only)."""
from __future__ import annotations

import json
import webbrowser
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

from . import config as C
from . import label as L
from .data import load_records
from .translate import Translations

PAGE = r"""<!doctype html><html lang="ru"><head><meta charset="utf-8"><title>Blind labelling</title>
<style>
body{font:16px/1.5 -apple-system,Segoe UI,Roboto,sans-serif;margin:0;background:#f4f4f1;color:#1b1b1b}
header{background:#1f2a37;color:#fff;padding:10px 20px;display:flex;gap:20px;align-items:center}
header b{font-size:18px} header a{color:#cfe3ff} #bar{flex:1;height:8px;background:#44546a;border-radius:4px}
#fill{height:8px;background:#7fd08a;border-radius:4px;width:0}
main{padding:14px 20px 150px} .aid{background:#fff6d6;border:1px solid #e8d48a;padding:6px 10px;border-radius:6px;font-size:14px}
.warn{background:#ffe1dc;border:1px solid #e09a8f;padding:6px 10px;border-radius:6px;font-size:14px;margin-top:6px}
.cols{display:grid;grid-template-columns:1fr 1fr;gap:16px;margin-top:10px}
.col{background:#fff;border:1px solid #ddd;border-radius:8px;padding:12px 16px}
.col h3{margin:0 0 6px;font-size:13px;letter-spacing:.04em;text-transform:uppercase;color:#666}
.hd{font-size:14px;color:#444;border-bottom:1px solid #eee;padding-bottom:6px;margin-bottom:8px}
.txt{white-space:pre-wrap;word-wrap:break-word}
footer{position:fixed;left:0;right:0;bottom:0;background:#fff;border-top:1px solid #ccc;padding:10px 20px}
.btns{display:flex;gap:10px;flex-wrap:wrap;align-items:center}
button{font:inherit;padding:10px 16px;border-radius:8px;border:1px solid #888;background:#fafafa;cursor:pointer}
button small{display:block;color:#666;font-size:12px}
button.a1{border-color:#2f855a} button.a2{border-color:#b7791f} button.a3{border-color:#718096}
textarea{width:100%;box-sizing:border-box;font:inherit;height:44px;margin-bottom:8px}
.extra{background:#eef3fb;border:1px solid #b9c9e6;border-radius:8px;padding:10px 14px;margin-top:10px;font-size:15px}
.big{font-size:20px;margin:60px auto;max-width:760px;text-align:center}
</style></head><body>
<header><b id="title">Разметка</b><span id="count"></span><div id="bar"><div id="fill"></div></div>
<a href="/rubric" target="_blank">Правила / rubric</a></header>
<main id="main"></main>
<footer id="footer"></footer>
<script>
let state=null, t0=0, mode='label';
const $=id=>document.getElementById(id);
function el(tag,cls,text){const e=document.createElement(tag); if(cls)e.className=cls; if(text!==undefined)e.textContent=text; return e;}
async function api(path,body){const r=await fetch(path,body?{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify(body)}:{});
  const j=await r.json(); if(!r.ok){alert(j.error||'error'); throw new Error(j.error);} return j;}
function columns(s){
  const cols=el('div','cols');
  const ru=el('div','col'); ru.append(el('h3','','Русский (машинный перевод отдельной моделью, для удобства чтения)'));
  ru.append(el('div','hd txt',s.header_ru.join('\n'))); ru.append(el('div','txt',s.narrative_ru||'Перевода нет. Читайте английский оригинал.'));
  const en=el('div','col'); en.append(el('h3','','English (original — the text the models see)'));
  en.append(el('div','hd txt',s.header_en.join('\n'))); en.append(el('div','txt',s.narrative_en));
  cols.append(ru,en); return cols;}
function progress(s,what){ $('count').textContent=what+' '+(s.position||s.done)+' из '+s.total+' · готово '+s.done; $('fill').style.width=(100*s.done/s.total)+'%'; }
async function load(){
  mode='label'; state=await api('/api/next'); const m=$('main'), f=$('footer'); m.innerHTML=''; f.innerHTML='';
  $('title').textContent='Слепая разметка / blind labelling';
  if(!state.position){ progress(state,'Случай');
    m.append(el('div','big','Все '+state.total+' случаев размечены, ответы зафиксированы и больше не меняются. Спасибо!'));
    const b=el('button','','Второй экран: проверить 24 разобранных расхождения'); b.onclick=loadReview; f.append(b); return; }
  progress(state,'Случай');
  m.append(el('div','aid','Русский текст — машинный перевод, сделанный отдельной моделью (не той, которую проверяют), только для удобства чтения; он может содержать ошибки. Если перевод и оригинал расходятся, верен английский оригинал справа.'));
  if(state.narrative_ru && state.translation_problems.length) m.append(el('div','warn','Внимание: перевод этого случая может быть неполным или неточным ('+state.translation_problems.join(', ')+'). Опирайтесь на английский текст.'));
  m.append(columns(state));
  const note=el('textarea'); note.id='note'; note.placeholder='Заметка (необязательно): какой факт решил дело'; f.append(note);
  const row=el('div','btns');
  [['relief','a1','1 · Вернуть деньги','money back'],['no_relief','a2','2 · Не возвращать деньги','no money back'],['cannot_tell','a3','3 · Не могу решить по тексту','cannot tell']]
   .forEach(([v,c,ru,en])=>{const b=el('button',c); b.append(document.createTextNode(ru),el('small','',en)); b.onclick=()=>answer(v); row.append(b);});
  if(state.can_undo){const u=el('button','','Отменить предыдущий ответ'); u.onclick=async()=>{await api('/api/undo',{}); load();}; row.append(u);}
  f.append(row); window.scrollTo(0,0); t0=performance.now();
}
async function answer(v){
  if(state.total-state.done===1 && !confirm('Это последний случай. После ответа все 100 ответов будут зафиксированы, изменить их будет нельзя. Подтвердить?')) return;
  await api('/api/answer',{position:state.position,label:v,note:$('note').value,seconds:(performance.now()-t0)/1000}); load();
}
async function loadReview(){
  mode='review'; const s=await api('/api/review/next'); state=s; const m=$('main'), f=$('footer'); m.innerHTML=''; f.innerHTML='';
  $('title').textContent='Второй экран: проверка разбора расхождений'; progress(s,'Случай');
  if(!s.position){ m.append(el('div','big','Все '+s.total+' случаев проверены. Спасибо! Окно можно закрыть.')); return; }
  const x=el('div','extra');
  x.append(el('div','','Решение компании (записано): '+s.recorded));
  x.append(el('div','','Ответ модели v1: '+(s.judge.relief?'вернуть деньги':'не возвращать')+' — '+s.judge.reason));
  x.append(el('div','','Предложенное прочтение: '+s.drafted_group_ru+' ['+s.drafted_group_en+']'));
  x.append(el('div','','Категория: '+s.category_en+(s.quote?' · цитата: "'+s.quote+'"':'')+(s.note?' · '+s.note:'')));
  m.append(x); m.append(columns(s));
  const note=el('textarea'); note.id='note'; note.placeholder='Заметка (необязательно)'; f.append(note);
  const row=el('div','btns');
  s.groups.forEach(g=>{const b=el('button',g.group===s.drafted_group?'a1':''); b.append(document.createTextNode((g.group===s.drafted_group?'Подтверждаю: ':'Изменить на: ')+g.ru.split(':')[0]),el('small','',g.en.split(':')[0]));
    b.onclick=async()=>{await api('/api/review/answer',{position:s.position,group:g.group,note:$('note').value}); loadReview();}; row.append(b);});
  f.append(row); window.scrollTo(0,0);
}
document.addEventListener('keydown',e=>{ if(mode!=='label'||!state||!state.position||e.target.tagName==='TEXTAREA') return;
  const k={'1':'relief','2':'no_relief','3':'cannot_tell'}[e.key]; if(k) answer(k); });
load();
</script></body></html>"""


def make_handler(annotator: str):
    records, translations = load_records(), Translations()

    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *args):          # keep the terminal quiet
            pass

        def _send(self, code: int, body: bytes, ctype: str) -> None:
            self.send_response(code)
            self.send_header("Content-Type", ctype)
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            self.wfile.write(body)

        def _json(self, obj, code: int = 200) -> None:
            self._send(code, json.dumps(obj, ensure_ascii=False).encode(), "application/json; charset=utf-8")

        def _review(self):
            from . import review as V          # imported only here: it reads outcomes and model answers
            from .llm import Cache
            return V, Cache()

        def do_GET(self):
            try:
                if self.path == "/":
                    self._send(200, PAGE.encode(), "text/html; charset=utf-8")
                elif self.path == "/rubric":
                    text = "\n\n".join((C.ROOT / "docs" / f).read_text(encoding="utf-8")
                                       for f in ("labelling_rubric_ru.md", "labelling_rubric.md"))
                    self._send(200, text.encode(), "text/plain; charset=utf-8")
                elif self.path == "/api/next":
                    self._json(L.next_case(annotator, records, translations))
                elif self.path == "/api/review/next":
                    V, cache = self._review()
                    self._json(V.next_review(annotator, records, cache, translations))
                else:
                    self._json({"error": "not found"}, 404)
            except (ValueError, SystemExit) as e:
                self._json({"error": str(e)}, 400)

        def do_POST(self):
            try:
                body = json.loads(self.rfile.read(int(self.headers.get("Content-Length") or 0)) or b"{}")
                if self.path == "/api/answer":
                    view = L.next_case(annotator, records, translations)
                    L.record_answer(annotator, int(body["position"]), body["label"], body.get("note", ""),
                                    float(body.get("seconds", 0)), "web", view)
                elif self.path == "/api/undo":
                    L.undo_last(annotator)
                elif self.path == "/api/review/answer":
                    V, _ = self._review()
                    V.record_review(annotator, int(body["position"]), body["group"], body.get("note", ""))
                else:
                    return self._json({"error": "not found"}, 404)
                self._json({"ok": True})
            except (ValueError, KeyError, SystemExit) as e:
                self._json({"error": str(e)}, 400)

    return Handler


def serve(annotator: str, port: int = 8765, open_browser: bool = True) -> None:
    handler = make_handler(annotator)
    for p in range(port, port + 20):
        try:
            server = ThreadingHTTPServer(("127.0.0.1", p), handler)
            break
        except OSError:
            continue
    else:
        raise SystemExit("no free local port")
    url = f"http://127.0.0.1:{p}/"
    print(f"Labelling page: {url}\nAnswers are saved after every case. Stop with Ctrl+C; run the same command to continue.")
    if open_browser:
        webbrowser.open(url)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nStopped. Everything answered so far is saved.")
