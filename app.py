"""Gradio 界面(全中文):左侧传图(上传/拍照/粘贴),右侧识别结果与养护建议。

- A 识别按钮处理中由队列禁用,防连点。
- B 结果末尾显示真实耗时与路线。
- C “复制结果”直接复制右侧已渲染结果文字(剪贴板 API + execCommand 降级)。
- 右侧结果以卡片样式排版;隐藏 Gradio 自带脚注。
"""

import glob
import json
import os

import gradio as gr

import agent

_ROUTE = {"大模型识别": "api", "本地模型识别": "local"}

_FALLBACK = ["向日葵", "玫瑰", "月季", "茉莉", "百合", "郁金香", "康乃馨", "栀子花",
             "山茶花", "菊花", "桂花", "杜鹃花", "绿萝", "吊兰", "龟背竹", "富贵竹",
             "发财树", "芦荟", "多肉", "仙人掌"]

_CSS = """
.gradio-container { max-width: 1120px !important; margin: 0 auto !important; }
.gradio-container .gr-row { align-items: flex-start !important; }
#hdr h1, #hdr h2 { margin: .1em 0 .35em; }
#out-box { border: 1px solid #e4e7e4; border-radius: 12px; padding: 16px 20px;
           background: #fcfdfc; min-height: 360px; overflow-y: auto; }
#out-box h2 { font-size: 1.3em; margin: .2em 0 .4em; color: #2f6b3f;
              border-bottom: 2px solid #cfe6d3; padding-bottom: 6px; }
#out-box h3 { margin: .9em 0 .35em; color: #1f4e79; }
#out-box ul { padding-left: 1.3em; margin: .35em 0; }
#out-box li { margin: .22em 0; }
#out-box blockquote { color: #7a7a7a; border-left: 3px solid #c9c9c9;
                      margin: .5em 0; padding: 2px 0 2px 10px; }
#copy-btn { width: auto !important; }
#muted-note p { color: #8a8f8a; font-size: 13px; margin: 4px 0; }
img { border-radius: 8px; }
"""


def supported_classes():
    try:
        with open(os.path.join("models", "classes.json"), encoding="utf-8") as f:
            c = json.load(f)
        return c if c else _FALLBACK
    except Exception:
        return _FALLBACK


def example_images():
    ex = [("向日葵", "data/向日葵/000001.jpeg"),
          ("百合", "data/百合/000001.jpg"),
          ("绿萝", "data/绿萝/000004.jpg"),
          ("发财树", "data/发财树/000005.jpg")]
    return [(lab, os.path.abspath(p)) for lab, p in ex if os.path.exists(p)]


def run(img, route_label, question):
    """界面回调:参数归一 → agent 编排 → 异常转错误文案。"""
    route = _ROUTE.get(route_label, "api")
    try:
        return agent.agent_handle(img, route, question)
    except Exception as e:  # noqa: BLE001
        return f"出错了: {e}"


def build():
    cls = supported_classes()
    samples = example_images()
    choices = samples if samples else [("(无示例)", "")]

    with gr.Blocks(title="植物识别与养护问答助手") as demo:
        gr.Markdown("## 植物识别与养护问答助手\n"
                    "上传照片可识别植物;也可以直接输入养护问题。两种方式可单用,也可“照片 + 追问”。",
                    elem_id="hdr")
        with gr.Row():
            with gr.Column(scale=5):
                img = gr.Image(
                    type="filepath", height=300, label="植物照片(可上传/拍照/粘贴)",
                    sources=["upload", "webcam", "clipboard"], interactive=True)
                question = gr.Textbox(
                    label="想问什么?(可不填)", lines=2,
                    placeholder="例如:叶子发黄怎么办?多久浇一次水?")
                gr.Examples(
                    examples=[["绿萝叶子发黄怎么办"], ["黄金葛怎么养"], ["多肉夏天怎么浇水"]],
                    inputs=[question], label="示例问题")
                route = gr.Radio(list(_ROUTE), value=list(_ROUTE)[0],
                                 label="识别方式",
                                 info="大模型:联网识别+AI 生成建议,消耗少量 API 额度;"
                                      "本地模型:识别与建议全离线,不耗额度。")
                btn = gr.Button("开始识别 / 提问", variant="primary")
                sample = gr.Radio(choices=choices, value=None,
                                  label="或从示例中选择一张")
                clear = gr.Button("清空", size="sm")
            with gr.Column(scale=6):
                out = gr.Markdown(label="结果", elem_id="out-box")
                copy_btn = gr.Button("复制结果", size="sm", elem_id="copy-btn")

        gr.Markdown(f"当前支持识别 {len(cls)} 种:{'、'.join(cls)}", elem_id="muted-note")

        sample.select(lambda v: v if v else None, inputs=[sample], outputs=[img])
        btn.click(run, [img, route, question], out)
        question.submit(run, [img, route, question], out)
        clear.click(lambda: (None, "", ""), outputs=[img, question, out])

        copy_btn.click(None, js="""
        () => {
          const box = document.getElementById('out-box');
          const txt = box ? (box.innerText || box.textContent || '').trim() : '';
          if (!txt) return;
          const btn = document.getElementById('copy-btn');
          const target = btn ? (btn.tagName === 'BUTTON' ? btn : btn.querySelector('button') || btn) : null;
          const orig = target ? target.textContent : '';
          const flash = () => { if (target) { target.textContent = '已复制'; setTimeout(() => { target.textContent = orig; }, 1400); } };
          const fallback = (s) => {
            const t = document.createElement('textarea');
            t.value = s; t.style.position = 'fixed'; t.style.opacity = '0';
            document.body.appendChild(t); t.select();
            try { document.execCommand('copy'); } catch (e) {}
            document.body.removeChild(t); flash();
          };
          if (navigator.clipboard && window.isSecureContext) {
            navigator.clipboard.writeText(txt).then(flash, () => fallback(txt));
          } else { fallback(txt); }
        }""")

        demo.load(None, js="""
        () => {
          setTimeout(() => {
            const walk = document.createTreeWalker(document.body, NodeFilter.SHOW_TEXT);
            const targets = [];
            while (walk.nextNode()) {
              const s = (walk.currentNode.textContent || '').trim();
              if (s === 'Runs' || s === 'Flag' || /Built with Gradio/.test(s)
                  || /使用 Gradio 构建/.test(s) || /通过 API/.test(s)) {
                let el = walk.currentNode.parentElement;
                for (let i = 0; i < 5 && el; i++) {
                  const c = typeof el.className === 'string' ? el.className : '';
                  if (/footer|more|status|overflow|settings/i.test(c)) { targets.push(el); break; }
                  el = el.parentElement;
                }
              }
            }
            targets.forEach(el => el.remove());
          }, 1200);
        }""")
        # 只抹掉“/预估秒”那截,保留实时秒数(Gradio 队列预估时间不准)
        demo.load(None, js="""
        () => {
          const strip = () => {
            const walker = document.createTreeWalker(document.body, NodeFilter.SHOW_TEXT);
            const hits = [];
            while (walker.nextNode()) {
              const n = walker.currentNode, t = n.nodeValue || '';
              const m = t.match(/(\\d+(?:\\.\\d+)?s?)\\s*\\/\\s*\\d+(?:\\.\\d+)?s/);
              if (m) hits.push([n, m[1]]);
              else if (/^\\s*\\/\\s*\\d+(?:\\.\\d+)?s\\s*$/.test(t)) hits.push([n, '']);
            }
            hits.forEach(([n, v]) => { if (n.nodeValue !== v) n.nodeValue = v; });
          };
          strip();
          const obs = new MutationObserver(strip);
          obs.observe(document.body, { subtree: true, childList: true, characterData: true });
        }""")
    return demo


demo = build()

if __name__ == "__main__":
    demo.queue().launch(server_name="127.0.0.1", share=False,
                        footer_links=[], run_history=False, css=_CSS)
